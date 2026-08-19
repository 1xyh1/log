"""Pure helpers for T1-GR E2-E5 evidence-chain tooling v2."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

SCHEMA_CONTRACT_PRIVATE = "t1gr-formal-data-contract-private-v2"
SCHEMA_CONTRACT_PUBLIC = "t1gr-formal-data-contract-public-v2"
SCHEMA_SPLIT_PRIVATE = "t1gr-split-proposal-private-v2"
SCHEMA_SPLIT_FREEZE_PRIVATE = "t1gr-split-freeze-private-v2"
SCHEMA_SPLIT_FREEZE_PUBLIC = "t1gr-split-freeze-public-v2"
SCHEMA_VIEW_MANIFEST = "t1gr-step1-view-manifest-v2"
SCHEMA_STEP1_RECIPE = "t1gr-step1-baseline-recipe-v2"

SPLITS = ("train", "dev", "final_holdout")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return dt.astimezone(timezone.utc)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_ids_sha(ids: Sequence[str]) -> str:
    return sha256_json(sorted(map(str, ids)))


def inside(child: Path, parent: Path) -> bool:
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def require_outside_repo(path: Path, repo_root: Path, code: str) -> None:
    if inside(path, repo_root):
        raise RuntimeError(code)


def sample_id_from_path(path: Path, spec: Mapping) -> str:
    mode = spec.get("mode", "stem")
    if mode == "stem":
        return path.stem
    if mode == "regex":
        rx = spec.get("regex")
        if not rx:
            raise ValueError("sample regex unresolved")
        m = re.search(rx, path.name)
        if not m:
            raise ValueError(f"SAMPLE_REGEX_MISS:{path}")
        g = spec.get("regex_group")
        return m.group(g) if g is not None else m.group(1)
    raise ValueError(f"UNKNOWN_SAMPLE_ID_MODE:{mode}")


def parse_yolo_label(
    path: Path,
    num_classes: int | None = None,
    *,
    exact_fields: int = 5,
    edge_tolerance: float = 1e-6,
) -> dict:
    """Strict detect-label parser. Formal rows must have exactly 5 fields."""
    classes: list[int] = []
    bad: list[dict] = []
    text = Path(path).read_text(encoding="utf-8-sig")
    for ln, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != exact_fields:
            bad.append({"line": ln, "reason": "field_count_not_exact", "count": len(parts), "expected": exact_fields})
            continue
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            bad.append({"line": ln, "reason": "non_numeric", "text": line})
            continue
        cls_f, cx, cy, bw, bh = vals
        cls_i = int(cls_f)
        if cls_f != cls_i:
            bad.append({"line": ln, "reason": "class_not_integer", "value": cls_f})
        if num_classes is not None and not (0 <= cls_i < num_classes):
            bad.append({"line": ln, "reason": "class_out_of_range", "class": cls_i})
        if not all(math.isfinite(v) for v in vals):
            bad.append({"line": ln, "reason": "non_finite"})
        if not all(-edge_tolerance <= v <= 1 + edge_tolerance for v in (cx, cy, bw, bh)):
            bad.append({"line": ln, "reason": "bbox_value_out_of_normalized_range", "bbox": [cx, cy, bw, bh]})
        if bw <= 0 or bh <= 0:
            bad.append({"line": ln, "reason": "nonpositive_wh", "bbox": [cx, cy, bw, bh]})
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        if x1 < -edge_tolerance or y1 < -edge_tolerance or x2 > 1 + edge_tolerance or y2 > 1 + edge_tolerance:
            bad.append({"line": ln, "reason": "bbox_edges_outside_image", "xyxy": [x1, y1, x2, y2]})
        classes.append(cls_i)
    return {"n_boxes": len(classes), "classes": classes, "errors": bad}


def group_map(ids: Sequence[str], paths: Mapping[str, str], rule: Mapping, root: Path) -> dict[str, str]:
    t = rule.get("type")
    if not t:
        raise ValueError("GROUP_RULE_UNRESOLVED")
    if t == "regex":
        rx_text = rule.get("regex")
        if not rx_text:
            raise ValueError("GROUP_REGEX_UNRESOLVED")
        rx = re.compile(rx_text)
        g = rule.get("regex_group", 1)
        out = {}
        for sid in ids:
            m = rx.search(sid)
            if not m:
                raise ValueError(f"GROUP_REGEX_MISS:{sid}")
            out[sid] = str(m.group(g))
        return out
    if t == "parent_directory":
        lvl = int(rule.get("parent_level", 1))
        out = {}
        for sid in ids:
            p = Path(paths[sid])
            parents = p.parents
            if lvl < 1 or lvl > len(parents):
                raise ValueError(f"PARENT_LEVEL_INVALID:{sid}")
            out[sid] = parents[lvl - 1].name
        return out
    if t == "metadata_field":
        mp = Path(rule.get("metadata_file") or "")
        if not mp:
            raise ValueError("GROUP_METADATA_FILE_UNRESOLVED")
        if not mp.is_absolute():
            mp = root / mp
        if not mp.is_file():
            raise ValueError(f"GROUP_METADATA_MISSING:{mp}")
        idf = rule.get("metadata_id_field")
        gf = rule.get("metadata_group_field")
        if not idf or not gf:
            raise ValueError("GROUP_METADATA_FIELDS_UNRESOLVED")
        rows = {}
        if mp.suffix.lower() == ".json":
            obj = json.loads(mp.read_text(encoding="utf-8-sig"))
            seq = obj if isinstance(obj, list) else obj.get("rows", [])
            for r in seq:
                if idf not in r or gf not in r:
                    raise ValueError("GROUP_METADATA_ROW_SCHEMA_FAIL")
                sid = str(r[idf])
                if sid in rows:
                    raise ValueError(f"GROUP_METADATA_DUPLICATE_ID:{sid}")
                rows[sid] = str(r[gf])
        else:
            with mp.open("r", encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    if idf not in r or gf not in r:
                        raise ValueError("GROUP_METADATA_ROW_SCHEMA_FAIL")
                    sid = str(r[idf])
                    if sid in rows:
                        raise ValueError(f"GROUP_METADATA_DUPLICATE_ID:{sid}")
                    rows[sid] = str(r[gf])
        miss = [sid for sid in ids if sid not in rows]
        if miss:
            raise ValueError(f"METADATA_GROUP_MISSING:{miss[:10]}")
        empty = [sid for sid in ids if not rows[sid]]
        if empty:
            raise ValueError(f"METADATA_GROUP_EMPTY:{empty[:10]}")
        return {sid: rows[sid] for sid in ids}
    raise ValueError(f"UNKNOWN_GROUP_RULE:{t}")


def exact_duplicate_groups(hashes: Mapping[str, str]) -> list[list[str]]:
    rev: dict[str, list[str]] = defaultdict(list)
    for sid, h in hashes.items():
        rev[str(h)].append(str(sid))
    return sorted([sorted(v) for v in rev.values() if len(v) > 1])


def triplet_hash(rgb_sha: str, ir_sha: str, depth_sha: str) -> str:
    return sha256_json({"rgb": rgb_sha, "ir": ir_sha, "depth": depth_sha})


def class_stats_for_ids(per_id: Mapping[str, Mapping], ids: Sequence[str], n_classes: int) -> dict:
    boxes = [0] * n_classes
    images = [0] * n_classes
    for sid in ids:
        classes = [int(c) for c in per_id[sid]["classes"]]
        cc = Counter(classes)
        for c, n in cc.items():
            if 0 <= c < n_classes:
                boxes[c] += n
                images[c] += 1
    return {"box_counts": boxes, "image_counts": images, "n_images": len(ids)}


def group_stats_from_contract(group_to_ids: Mapping[str, Sequence[str]], per_id: Mapping[str, Mapping], n_classes: int) -> dict:
    return {g: class_stats_for_ids(per_id, ids, n_classes) for g, ids in group_to_ids.items()}


def _split_cost(
    counts: Mapping[str, dict],
    targets: Mapping[str, dict],
    names: Sequence[str],
    *,
    w_samples: float,
    w_images: float,
    w_boxes: float,
) -> float:
    cost = 0.0
    for s in names:
        c, t = counts[s], targets[s]
        cost += w_samples * ((c["n_images"] - t["n_images"]) / max(t["n_images"], 1.0)) ** 2
        for key, w in (("image_counts", w_images), ("box_counts", w_boxes)):
            for cv, tv in zip(c[key], t[key]):
                if tv <= 0:
                    continue
                cost += w * ((cv - tv) / max(tv, 1.0)) ** 2
    return cost


def group_stratified_split(
    group_to_ids: Mapping[str, Sequence[str]],
    group_stats: Mapping[str, Mapping],
    fractions: Mapping[str, float],
    seed: int,
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, list[str]]:
    """Deterministic group-aware greedy + local move search balancing size/classes."""
    names = SPLITS
    if set(fractions) != set(names):
        raise ValueError("split fraction keys mismatch")
    if abs(sum(float(fractions[k]) for k in names) - 1.0) > 1e-9 or any(float(fractions[k]) <= 0 for k in names):
        raise ValueError("split fractions must be positive and sum to 1")
    groups = list(group_to_ids)
    if len(groups) < len(names):
        raise ValueError("NEED_AT_LEAST_THREE_GROUPS")
    n_classes = len(next(iter(group_stats.values()))["box_counts"])
    totals = {
        "n_images": sum(group_stats[g]["n_images"] for g in groups),
        "image_counts": [sum(group_stats[g]["image_counts"][c] for g in groups) for c in range(n_classes)],
        "box_counts": [sum(group_stats[g]["box_counts"][c] for g in groups) for c in range(n_classes)],
    }
    targets = {
        s: {
            "n_images": totals["n_images"] * float(fractions[s]),
            "image_counts": [v * float(fractions[s]) for v in totals["image_counts"]],
            "box_counts": [v * float(fractions[s]) for v in totals["box_counts"]],
        }
        for s in names
    }
    weights = dict(weights or {})
    ws, wi, wb = float(weights.get("samples", 1.0)), float(weights.get("class_images", 2.0)), float(weights.get("class_boxes", 1.0))
    rng = random.Random(int(seed))
    shuffled = groups[:]
    rng.shuffle(shuffled)
    # Rarer groups first, then larger groups. Shuffle is final deterministic tie-break source.
    rarity = {}
    total_group_presence = [sum(1 for g in groups if group_stats[g]["image_counts"][c] > 0) for c in range(n_classes)]
    for g in groups:
        rarity[g] = sum((1.0 / max(total_group_presence[c], 1)) for c, v in enumerate(group_stats[g]["image_counts"]) if v > 0)
    order_index = {g: i for i, g in enumerate(shuffled)}
    ordered = sorted(groups, key=lambda g: (-rarity[g], -group_stats[g]["n_images"], order_index[g]))

    assigned = {s: [] for s in names}
    counts = {s: {"n_images": 0, "image_counts": [0] * n_classes, "box_counts": [0] * n_classes} for s in names}

    def add_to(s: str, g: str, sign: int = 1):
        st = group_stats[g]
        counts[s]["n_images"] += sign * st["n_images"]
        for c in range(n_classes):
            counts[s]["image_counts"][c] += sign * st["image_counts"][c]
            counts[s]["box_counts"][c] += sign * st["box_counts"][c]

    # Force nonempty splits: seed one group into each split, largest target first.
    split_seed_order = sorted(names, key=lambda s: (-fractions[s], names.index(s)))
    for g, s in zip(ordered[: len(names)], split_seed_order):
        assigned[s].append(g)
        add_to(s, g)

    for g in ordered[len(names):]:
        best = None
        for s in names:
            add_to(s, g)
            cost = _split_cost(counts, targets, names, w_samples=ws, w_images=wi, w_boxes=wb)
            add_to(s, g, -1)
            cand = (cost, names.index(s), s)
            if best is None or cand < best:
                best = cand
        s = best[2]
        assigned[s].append(g)
        add_to(s, g)

    # Deterministic local single-group moves, preserving nonempty splits.
    improved = True
    while improved:
        improved = False
        base_cost = _split_cost(counts, targets, names, w_samples=ws, w_images=wi, w_boxes=wb)
        best_move = None
        for src in names:
            if len(assigned[src]) <= 1:
                continue
            for g in sorted(assigned[src]):
                for dst in names:
                    if dst == src:
                        continue
                    add_to(src, g, -1); add_to(dst, g)
                    cost = _split_cost(counts, targets, names, w_samples=ws, w_images=wi, w_boxes=wb)
                    add_to(dst, g, -1); add_to(src, g)
                    cand = (cost, src, dst, g)
                    if cost + 1e-12 < base_cost and (best_move is None or cand < best_move):
                        best_move = cand
        if best_move is not None:
            _, src, dst, g = best_move
            assigned[src].remove(g); assigned[dst].append(g)
            add_to(src, g, -1); add_to(dst, g)
            improved = True

    return {s: sorted(assigned[s]) for s in names}


def classify_overlap(split_groups: Mapping[str, Sequence[str]]) -> dict:
    names = list(split_groups)
    pairs = {}
    ok = True
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sorted(set(split_groups[a]) & set(split_groups[b]))
            pairs[f"{a}__{b}"] = inter
            ok = ok and not inter
    return {"pairwise_group_overlap": pairs, "passed": bool(ok)}


def split_sample_overlap(samples: Mapping[str, Sequence[str]]) -> dict:
    pairs = {}
    ok = True
    names = list(samples)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sorted(set(samples[a]) & set(samples[b]))
            pairs[f"{a}__{b}"] = inter
            ok = ok and not inter
    return {"pairwise_sample_overlap": pairs, "passed": bool(ok)}


def coverage_audit(
    support: Mapping[str, Mapping],
    policy: Mapping,
    n_classes: int,
) -> dict:
    """Pre-registered class coverage; exemptions must be explicit with reasons."""
    min_img = policy.get("min_image_count_by_split") or {}
    min_box = policy.get("min_box_count_by_split") or {}
    exemptions_raw = policy.get("exempt_classes") or []
    exemptions = {}
    for item in exemptions_raw:
        cid = int(item["class_id"])
        reason = str(item.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"COVERAGE_EXEMPTION_REASON_REQUIRED:{cid}")
        exemptions[cid] = reason
    failures = []
    for split in SPLITS:
        req_i = int(min_img.get(split, 0))
        req_b = int(min_box.get(split, 0))
        for c in range(n_classes):
            if c in exemptions:
                continue
            got_i = int(support[split]["image_counts"][c])
            got_b = int(support[split]["box_counts"][c])
            if got_i < req_i or got_b < req_b:
                failures.append({
                    "split": split, "class_id": c,
                    "image_count": got_i, "required_images": req_i,
                    "box_count": got_b, "required_boxes": req_b,
                })
    return {"passed": not failures, "failures": failures, "exemptions": exemptions}


def cross_split_duplicate_audit(duplicate_groups_by_kind: Mapping[str, Sequence[Sequence[str]]], split_of: Mapping[str, str]) -> dict:
    out = {}
    any_cross = False
    for kind, groups in duplicate_groups_by_kind.items():
        cross = []
        for grp in groups:
            seen = sorted({split_of.get(sid) for sid in grp if sid in split_of})
            seen = [x for x in seen if x is not None]
            if len(seen) > 1:
                cross.append({"ids": list(grp), "splits": seen})
        out[kind] = cross
        any_cross = any_cross or bool(cross)
    return {"by_kind": out, "passed": not any_cross}


def endpoint_label(deltas: Sequence[float]) -> str:
    if all(x > 0 for x in deltas):
        return "STABLE_POSITIVE"
    if all(x < 0 for x in deltas):
        return "STABLE_NEGATIVE"
    if all(x == 0 for x in deltas):
        return "EXACT_TIE"
    return "MIXED"
