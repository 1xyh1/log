"""Step 3-A data contract: raw sample index, exclusion audit, class support, group split.

Frozen facts (reviewer-approved 2026-08-14):
    raw groups = 18; usable metric multimodal groups = 17
    excluded: 00000008 (non-metric depth visualization: 360x640, 3ch uint8 JPG depth)
    excluded group carried ALL 3 UAV GTs -> usable coverage 9/12,
    absent = [1 boat, 7 ball, 10 uav] reported as N/A (NOT 0 AP)
    grouping_rule = first_id_field_proxy_v1 (first numeric field before '_');
    claim: "no cross-group leakage under the current group proxy" (not "true scene
    independence" - official 2000-image metadata will define the real groups).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2

EXCLUDED = {
    "00000008": {
        "reason": "non-metric depth visualization",
        "rgb_format": "360x640 3ch uint8 JPG",
        "ir_format": "360x640 3ch uint8 JPG",
        "depth_format": "3-channel uint8 JPG (not uint16 metric)",
    }
}
GROUPING_RULE = "first_id_field_proxy_v1"
RAW_DEFAULT = "D:/pycharm/Python Develop/YOLO_1/sample_multimodal"
SPLIT_DEFAULT = "D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample"
OUT_DEFAULT = "D:/pycharm/Python Develop/YOLO_1/step3_data_contract.json"


def group_of(sample_id: str) -> str:
    return sample_id.split("_")[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_label_classes(path: Path) -> list[int]:
    out = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            out.append(int(line.split()[0]))
    return out


METRIC_RGB_SHAPE = (1080, 1920)
EXCLUDED_EXPECTED = {"00000008": {"rgb": (360, 640, 3), "ir": (360, 640, 3),
                                  "depth": (360, 640, 3)}}


def audit_raw_format(raw: Path, sid: str) -> dict:
    """G1a: real cv2.imread format audit of one raw group (not a filename whitelist)."""
    def fmt(p):
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if a is None:
            return None
        return {"shape": list(a.shape), "dtype": str(a.dtype)}

    return {
        "rgb": fmt(next((raw / "visible").glob(f"{sid}.*"))),
        "ir": fmt(next((raw / "infrared").glob(f"{sid}.*"))),
        "depth": fmt(next((raw / "depth").glob(f"{sid}.*"))),
    }


def _metric_format_ok(f: dict) -> bool:
    rgb, ir, dep = f["rgb"], f["ir"], f["depth"]
    return bool(
        rgb is not None and ir is not None and dep is not None
        and rgb["dtype"] == "uint8" and len(rgb["shape"]) == 3 and rgb["shape"][2] == 3
        and ir["dtype"] == "uint8" and len(ir["shape"]) == 3 and ir["shape"][2] == 3
        and dep["dtype"] == "uint16" and len(dep["shape"]) == 2
        and rgb["shape"][:2] == list(METRIC_RGB_SHAPE)
        and ir["shape"][:2] == list(METRIC_RGB_SHAPE)
        and dep["shape"] == list(METRIC_RGB_SHAPE))


def build_contract(raw_dir: str = RAW_DEFAULT, split_src: str = SPLIT_DEFAULT,
                   out_path: str = OUT_DEFAULT) -> dict:
    raw = Path(raw_dir)
    split_root = Path(split_src)
    for sub in ("visible", "infrared", "depth", "labels"):
        if not (raw / sub).is_dir():
            raise RuntimeError(f"raw subdir missing: {raw / sub}")

    raw_ids = sorted(p.stem for p in (raw / "visible").glob("*"))
    ids_with_all_modalities = [
        sid for sid in raw_ids
        if all((raw / sub / f"{sid}.png").is_file() or (raw / sub / f"{sid}.jpg").is_file()
               for sub in ("visible", "infrared", "depth"))
        and (raw / "labels" / f"{sid}.txt").is_file()
    ]
    excluded_present = {sid: sid in raw_ids for sid in EXCLUDED}
    # G1a: full format audit of ALL raw groups (observed fact, not filename whitelist)
    raw_format_audit = {sid: audit_raw_format(raw, sid) for sid in raw_ids}
    metric_ok = {sid: _metric_format_ok(raw_format_audit[sid]) for sid in raw_ids}
    for sid in EXCLUDED:
        f = raw_format_audit[sid]
        expected = EXCLUDED_EXPECTED[sid]
        if not (f["rgb"]["shape"] == list(expected["rgb"])
                and f["ir"]["shape"] == list(expected["ir"])
                and f["depth"]["shape"] == list(expected["depth"])):
            raise RuntimeError(f"excluded {sid} observed format changed: {f} vs {expected}")
    bad_metric = sorted(sid for sid, ok in metric_ok.items()
                        if not ok and sid not in EXCLUDED)
    if bad_metric:
        raise RuntimeError(f"non-excluded ids failing metric-format audit: {bad_metric}")
    usable = sorted(sid for sid in ids_with_all_modalities if sid not in EXCLUDED)

    # split: the ACTUAL Step-1 split on disk (v031), not the repo splits/*.txt
    split_ids = {}
    for split in ("train", "val"):
        ids = sorted(p.stem for p in (split_root / "images" / split).glob("*.png"))
        missing = [sid for sid in ids if sid not in usable]
        if missing:
            raise RuntimeError(f"{split} ids not usable: {missing}")
        split_ids[split] = ids
    all17 = sorted(split_ids["train"] + split_ids["val"])

    # class support per split
    def support(ids):
        counts = {}
        for sid in ids:
            for c in read_label_classes(raw / "labels" / f"{sid}.txt"):
                counts[c] = counts.get(c, 0) + 1
        return counts

    class_support = {split: support(split_ids[split]) for split in ("train", "val")}
    class_support["all17"] = support(all17)
    present = sorted(set(class_support["all17"]))
    absent = sorted(set(range(12)) - set(present))

    # G2: sample-level AND group-proxy-level split checks + set relations
    train_groups = sorted(set(group_of(s) for s in split_ids["train"]))
    val_groups = sorted(set(group_of(s) for s in split_ids["val"]))
    intersection = sorted(set(train_groups) & set(val_groups))
    sample_overlap = sorted(set(split_ids["train"]) & set(split_ids["val"]))
    union_equals_usable = set(split_ids["train"]) | set(split_ids["val"]) == set(usable)
    missing_from_split = sorted(set(usable) - (set(split_ids["train"]) | set(split_ids["val"])))
    split_relations = {
        "sample_overlap_empty": len(sample_overlap) == 0,
        "group_proxy_overlap_empty": len(intersection) == 0,
        "split_union_equals_usable": bool(union_equals_usable),
        "missing_from_split": missing_from_split,
        "train_count": len(split_ids["train"]),
        "val_count": len(split_ids["val"]),
        "unique_count": len(set(split_ids["train"]) | set(split_ids["val"])),
    }
    split_gate_passed = bool(split_relations["sample_overlap_empty"]
                             and split_relations["group_proxy_overlap_empty"]
                             and split_relations["split_union_equals_usable"]
                             and split_relations["train_count"] == 11
                             and split_relations["val_count"] == 6
                             and split_relations["unique_count"] == 17)

    # file hashes for all usable raw files
    file_hashes = {}
    for sid in usable:
        entry = {}
        for sub in ("visible", "infrared", "depth"):
            p = next((raw / sub).glob(f"{sid}.*"))
            entry[sub] = {"file": p.name, "sha256": sha256_file(p)}
        entry["label"] = {"file": f"{sid}.txt",
                          "sha256": sha256_file(raw / "labels" / f"{sid}.txt")}
        file_hashes[sid] = entry

    # excluded sample audit evidence (format check)
    excluded_audit = {}
    for sid in EXCLUDED:
        vis = next((raw / "visible").glob(f"{sid}.*"))
        dep = next((raw / "depth").glob(f"{sid}.*"))
        v = cv2.imread(str(vis), cv2.IMREAD_UNCHANGED)
        d = cv2.imread(str(dep), cv2.IMREAD_UNCHANGED)
        excluded_audit[sid] = {
            **EXCLUDED[sid],
            "observed_rgb_shape_dtype": [*v.shape, str(v.dtype)],
            "observed_depth_shape_dtype": [*d.shape, str(d.dtype)],
            "n_gt_boxes": len(read_label_classes(raw / "labels" / f"{sid}.txt")),
        }

    contract = {
        "_raw_dir": str(raw.resolve()),
        "_labels_dir": str((raw / "labels").resolve()),
        "raw_multimodal_groups": len(raw_ids),
        "usable_metric_multimodal_groups": len(usable),
        "excluded_groups": excluded_audit,
        "excluded_present_in_raw": excluded_present,
        "usable_ids": usable,
        "raw_format_audit": raw_format_audit,
        "metric_format_passed": {sid: ok for sid, ok in metric_ok.items() if sid not in EXCLUDED},
        "format_gate_passed": bool(all(ok for sid, ok in metric_ok.items() if sid not in EXCLUDED)
                                   and len(usable) == 17),
        "grouping_rule": GROUPING_RULE,
        "train_ids": split_ids["train"],
        "val_ids": split_ids["val"],
        "all17_ids": all17,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "group_intersection": intersection,
        "split_relations": split_relations,
        "split_gate_passed": split_gate_passed,
        "class_support": class_support,
        "classes_present": present,
        "classes_absent": absent,
        "n_classes": 12,
        "file_hashes": file_hashes,
        "note": "group proxy = first numeric field before '_'. 'No cross-group leakage under "
                "the current group proxy' - NOT claimed as true scene independence; official "
                "2000-image metadata will define real groups. absent classes reported N/A.",
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    return contract


if __name__ == "__main__":
    c = build_contract()
    print(json.dumps({k: c[k] for k in (
        "raw_multimodal_groups", "usable_metric_multimodal_groups", "excluded_groups",
        "grouping_rule", "train_groups", "val_groups", "group_intersection",
        "split_gate_passed", "classes_present", "classes_absent")}, indent=2, ensure_ascii=False))
