#!/usr/bin/env python3
"""Build private/public formal data contracts. Formal mode always full-hashes every file."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multimodal.t1gr_e2e5 import (  # noqa: E402
    SCHEMA_CONTRACT_PRIVATE, SCHEMA_CONTRACT_PUBLIC, canonical_ids_sha, exact_duplicate_groups,
    group_map, parse_yolo_label, require_outside_repo, sample_id_from_path,
    sha256_file, sha256_json, triplet_hash, utc_now_iso,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def files_for(root: Path, entry: dict) -> list[Path]:
    dname = entry.get("dir")
    if not dname:
        raise RuntimeError("MODALITY_DIR_UNRESOLVED")
    d = root / dname
    if not d.is_dir():
        raise RuntimeError(f"MODALITY_DIR_MISSING:{d}")
    ex = {x.lower() for x in entry.get("extensions") or []}
    if not ex:
        raise RuntimeError(f"MODALITY_EXTENSIONS_UNRESOLVED:{d}")
    return sorted(x for x in d.rglob("*") if x.is_file() and x.suffix.lower() in ex)


def observe_array(path: Path) -> dict:
    try:
        if path.suffix.lower() == ".npy":
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            readable = True
            loader = "npy"
        else:
            a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            readable = a is not None
            loader = "cv2"
        if not readable:
            return {"readable": False, "loader": loader}
        shape = list(a.shape)
        channels = 1 if a.ndim == 2 else (int(a.shape[2]) if a.ndim == 3 else None)
        return {
            "readable": True,
            "loader": loader,
            "shape": shape,
            "dtype": str(a.dtype),
            "ndim": int(a.ndim),
            "channels": channels,
            "height": int(a.shape[0]) if a.ndim >= 2 else None,
            "width": int(a.shape[1]) if a.ndim >= 2 else None,
        }
    except Exception as e:
        return {"readable": False, "loader": "error", "error": f"{type(e).__name__}:{e}"}


def expectation_resolved(exp: dict) -> bool:
    return bool(exp.get("allowed_dtypes") and exp.get("allowed_ndim") and exp.get("allowed_channels"))


def check_format(obs: dict, exp: dict) -> list[str]:
    errors = []
    if not obs.get("readable"):
        return ["unreadable"]
    if not expectation_resolved(exp):
        return ["format_expectation_unresolved"]
    if obs.get("dtype") not in set(map(str, exp["allowed_dtypes"])):
        errors.append(f"dtype:{obs.get('dtype')}")
    if int(obs.get("ndim", -1)) not in set(map(int, exp["allowed_ndim"])):
        errors.append(f"ndim:{obs.get('ndim')}")
    if obs.get("channels") not in set(map(int, exp["allowed_channels"])):
        errors.append(f"channels:{obs.get('channels')}")
    if exp.get("height") is not None and obs.get("height") != int(exp["height"]):
        errors.append(f"height:{obs.get('height')}")
    if exp.get("width") is not None and obs.get("width") != int(exp["width"]):
        errors.append(f"width:{obs.get('width')}")
    return errors


def public_group_summary(grouping: dict[str, str]) -> dict:
    counts = Counter(grouping.values())
    sizes = sorted(counts.values())
    return {
        "group_count": len(counts),
        "group_size_min": min(sizes) if sizes else 0,
        "group_size_max": max(sizes) if sizes else 0,
        "group_size_mean": (sum(sizes) / len(sizes)) if sizes else 0,
        "group_assignment_sha256": sha256_json(grouping) if grouping else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout-spec", required=True)
    ap.add_argument("--private-out", required=True, help="Must be outside repo; contains all IDs/hashes/paths.")
    ap.add_argument("--public-out", default="reports/step4_t1gr/data_contract_public.json")
    ap.add_argument("--repo-root", default=str(ROOT))
    args = ap.parse_args()

    spec_path = Path(args.layout_spec)
    private_out = Path(args.private_out)
    public_out = ROOT / args.public_out
    repo_root = Path(args.repo_root)
    require_outside_repo(private_out, repo_root, "PRIVATE_CONTRACT_MUST_BE_OUTSIDE_REPO")
    if private_out.exists() or public_out.exists():
        raise RuntimeError("REFUSE_OVERWRITE_CONTRACT")

    spec = load_json(spec_path)
    if spec.get("schema") != "t1gr-layout-spec-v2":
        raise RuntimeError("LAYOUT_SPEC_SCHEMA_FAIL")
    root = Path(spec["dataset_root"])
    if not root.is_dir():
        raise RuntimeError(f"DATASET_ROOT_NOT_FOUND:{root}")

    num_classes = spec.get("label_format", {}).get("num_classes")
    names = spec.get("label_format", {}).get("class_names")
    class_config_errors = []
    if num_classes is None or int(num_classes) <= 0:
        class_config_errors.append("num_classes_unresolved")
    if not isinstance(names, list):
        class_config_errors.append("class_names_unresolved")
    elif num_classes is not None and len(names) != int(num_classes):
        class_config_errors.append("class_names_length_mismatch")

    sid_spec = spec.get("sample_id") or {}
    maps: dict[str, dict[str, Path]] = {}
    dup_ids = {}
    for mod in ("rgb", "ir", "depth", "labels"):
        m: dict[str, Path] = {}
        dup = []
        for p in files_for(root, spec["modalities"][mod]):
            sid = sample_id_from_path(p, sid_spec)
            if sid in m:
                dup.append(sid)
            m[sid] = p
        maps[mod] = m
        dup_ids[mod] = sorted(set(dup))

    sets = {m: set(v) for m, v in maps.items()}
    all_ids = sorted(set.union(*sets.values())) if sets else []
    common = sorted(set.intersection(*sets.values())) if sets else []
    missing = {mod: sorted(set(all_ids) - sets[mod]) for mod in maps}
    pairing_complete = bool(common) and all(not v for v in missing.values()) and all(not v for v in dup_ids.values())

    expected_count = spec.get("expected_sample_count")
    expected_count_resolved = expected_count is not None and int(expected_count) > 0
    expected_count_match = expected_count_resolved and len(common) == int(expected_count)

    label_stats = {}
    label_errors = {}
    class_box_counts = Counter()
    class_image_counts = Counter()
    exact_fields = int(spec.get("label_format", {}).get("exact_fields", 5))
    edge_tol = float(spec.get("label_format", {}).get("edge_tolerance", 1e-6))
    for sid in common:
        st = parse_yolo_label(maps["labels"][sid], int(num_classes) if num_classes is not None else None,
                              exact_fields=exact_fields, edge_tolerance=edge_tol)
        label_stats[sid] = {"n_boxes": st["n_boxes"], "classes": st["classes"]}
        class_box_counts.update(st["classes"])
        class_image_counts.update(set(st["classes"]))
        if st["errors"]:
            label_errors[sid] = st["errors"]

    exp_all = spec.get("format_expectations") or {}
    exp_resolution = {m: expectation_resolved(exp_all.get(m) or {}) for m in ("rgb", "ir", "depth")}
    format_audit = {}
    format_failures = {}
    spatial_failures = []
    for sid in common:
        obs = {m: observe_array(maps[m][sid]) for m in ("rgb", "ir", "depth")}
        format_audit[sid] = obs
        for m in ("rgb", "ir", "depth"):
            errs = check_format(obs[m], exp_all.get(m) or {})
            if errs:
                format_failures.setdefault(sid, {})[m] = errs
        if exp_all.get("require_cross_modal_hw_match") is not True:
            spatial_failures.append({"sample_id": sid, "reason": "cross_modal_hw_policy_not_true"})
        else:
            hws = [(obs[m].get("height"), obs[m].get("width")) for m in ("rgb", "ir", "depth")]
            if any(None in hw for hw in hws) or len(set(hws)) != 1:
                spatial_failures.append({"sample_id": sid, "observed_hw": dict(zip(("rgb", "ir", "depth"), hws))})

    # Formal contract ALWAYS hashes every paired file. No bypass flag exists.
    file_meta = {}
    hashes_by_mod = {m: {} for m in ("rgb", "ir", "depth")}
    triplet_hashes = {}
    for sid in common:
        entry = {}
        for mod in ("rgb", "ir", "depth", "labels"):
            p = maps[mod][sid]
            h = sha256_file(p)
            entry[mod] = {
                "relative_path": str(p.relative_to(root)).replace("\\", "/"),
                "bytes": p.stat().st_size,
                "sha256": h,
            }
            if mod in hashes_by_mod:
                hashes_by_mod[mod][sid] = h
        triplet_hashes[sid] = triplet_hash(hashes_by_mod["rgb"][sid], hashes_by_mod["ir"][sid], hashes_by_mod["depth"][sid])
        entry["triplet_sha256"] = triplet_hashes[sid]
        file_meta[sid] = entry

    duplicate_groups_by_kind = {m: exact_duplicate_groups(hashes_by_mod[m]) for m in ("rgb", "ir", "depth")}
    duplicate_groups_by_kind["triplet"] = exact_duplicate_groups(triplet_hashes)

    group_rule = spec.get("group_rule") or {}
    group_rule_resolved = bool(group_rule.get("type"))
    grouping = {}
    grouping_error = None
    if group_rule_resolved:
        try:
            rgb_paths = {sid: file_meta[sid]["rgb"]["relative_path"] for sid in common}
            grouping = group_map(common, rgb_paths, group_rule, root)
        except Exception as e:
            grouping_error = f"{type(e).__name__}:{e}"
    group_rule_validation = {
        "resolved": group_rule_resolved,
        "executed": group_rule_resolved,
        "passed": bool(group_rule_resolved and not grouping_error and len(grouping) == len(common) and all(grouping.values())),
        "error": grouping_error,
    }

    gates = {
        "pairing_complete": pairing_complete,
        "expected_sample_count_resolved": expected_count_resolved,
        "expected_sample_count_match": expected_count_match,
        "class_config_valid": not class_config_errors,
        "labels_valid": not label_errors,
        "format_expectations_resolved": all(exp_resolution.values()),
        "format_valid": not format_failures,
        "cross_modal_hw_valid": not spatial_failures,
        "full_hash_complete": all("sha256" in file_meta[sid][m] for sid in common for m in ("rgb", "ir", "depth", "labels")),
    }
    contract_gate_passed = all(gates.values())

    private = {
        "schema": SCHEMA_CONTRACT_PRIVATE,
        "created_at_utc": utc_now_iso(),
        "read_only_dataset_audit": True,
        "dataset_root": str(root.resolve()),
        "layout_spec": spec,
        "layout_spec_sha256": sha256_file(spec_path),
        "counts_by_modality": {m: len(v) for m, v in maps.items()},
        "all_discovered_ids_count": len(all_ids),
        "paired_ids_count": len(common),
        "paired_ids": common,
        "paired_ids_sha256": canonical_ids_sha(common),
        "duplicate_sample_ids_within_modality": dup_ids,
        "missing_ids_by_modality": missing,
        "pairing_complete": pairing_complete,
        "expected_sample_count": expected_count,
        "expected_sample_count_match": expected_count_match,
        "label_errors": label_errors,
        "label_stats": label_stats,
        "class_box_counts": {str(k): v for k, v in sorted(class_box_counts.items())},
        "class_image_counts": {str(k): v for k, v in sorted(class_image_counts.items())},
        "class_config_errors": class_config_errors,
        "format_audit": format_audit,
        "format_failures": format_failures,
        "spatial_failures": spatial_failures,
        "format_expectations_resolved": exp_resolution,
        "file_meta": file_meta,
        "full_hash_mode": True,
        "duplicate_groups_by_kind": duplicate_groups_by_kind,
        "group_rule": group_rule,
        "group_rule_validation": group_rule_validation,
        "group_map": grouping,
        "gates": gates,
        "contract_gate_passed": contract_gate_passed,
    }
    private_out.parent.mkdir(parents=True, exist_ok=True)
    private_out.write_text(json.dumps(private, ensure_ascii=False, indent=2), encoding="utf-8")
    private_sha = sha256_file(private_out)

    public = {
        "schema": SCHEMA_CONTRACT_PUBLIC,
        "created_at_utc": private["created_at_utc"],
        "private_contract_sha256": private_sha,
        "layout_spec_sha256": private["layout_spec_sha256"],
        "dataset_root_basename": root.name,
        "counts_by_modality": private["counts_by_modality"],
        "paired_ids_count": len(common),
        "paired_ids_sha256": private["paired_ids_sha256"],
        "expected_sample_count": expected_count,
        "class_names": names,
        "num_classes": num_classes,
        "class_box_counts": private["class_box_counts"],
        "class_image_counts": private["class_image_counts"],
        "format_expectations": exp_all,
        "format_gate_passed": bool(gates["format_expectations_resolved"] and gates["format_valid"] and gates["cross_modal_hw_valid"]),
        "full_hash_mode": True,
        "duplicate_group_counts_by_kind": {k: len(v) for k, v in duplicate_groups_by_kind.items()},
        "group_rule": group_rule,
        "group_rule_validation": group_rule_validation,
        "group_summary": public_group_summary(grouping),
        "gates": gates,
        "contract_gate_passed": contract_gate_passed,
        "contains_sample_ids": False,
    }
    forbidden_public_keys = {"paired_ids", "all_ids", "file_meta", "label_stats", "group_map"}
    if forbidden_public_keys & set(public):
        raise RuntimeError("PUBLIC_CONTRACT_EXPOSES_PRIVATE_SAMPLE_DATA")
    public_out.parent.mkdir(parents=True, exist_ok=True)
    public_out.write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "private_out": str(private_out), "private_sha256": private_sha,
        "public_out": str(public_out), "paired": len(common),
        "contract_gate_passed": contract_gate_passed,
        "group_rule_validation": group_rule_validation,
        "failed_gates": [k for k, v in gates.items() if not v],
    }, ensure_ascii=False, indent=2))
    if not contract_gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
