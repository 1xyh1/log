#!/usr/bin/env python3
"""Build a copy-only RGB TRAIN/DEV view and a hash-pinned private view manifest."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multimodal.t1gr_e2e5 import (  # noqa: E402
    SCHEMA_CONTRACT_PRIVATE, SCHEMA_SPLIT_FREEZE_PRIVATE, SCHEMA_STEP1_RECIPE,
    SCHEMA_VIEW_MANIFEST, canonical_ids_sha, require_outside_repo, sha256_file, utc_now_iso,
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def copy_verified(src: Path, dst: Path, expected_sha: str) -> dict:
    if sha256_file(src) != expected_sha:
        raise RuntimeError(f"SOURCE_SHA_DRIFT:{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise RuntimeError(f"REFUSE_OVERWRITE:{dst}")
    shutil.copy2(src, dst)
    got = sha256_file(dst)
    if got != expected_sha:
        raise RuntimeError(f"VIEW_COPY_SHA_MISMATCH:{dst}")
    return {"source": str(src), "dest": str(dst), "sha256": got, "bytes": dst.stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private-contract", required=True)
    ap.add_argument("--split-manifest-private", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--repo-root", default=str(ROOT))
    args = ap.parse_args()

    cp, sp, rp = Path(args.private_contract), Path(args.split_manifest_private), Path(args.recipe)
    out = Path(args.out_root)
    require_outside_repo(out, Path(args.repo_root), "STEP1_VIEW_MUST_BE_OUTSIDE_REPO")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"VIEW_OUT_NOT_EMPTY:{out}")

    c, s, r = load(cp), load(sp), load(rp)
    if c.get("schema") != SCHEMA_CONTRACT_PRIVATE or c.get("contract_gate_passed") is not True:
        raise RuntimeError("PRIVATE_CONTRACT_FAIL")
    if s.get("schema") != SCHEMA_SPLIT_FREEZE_PRIVATE or s.get("proposal_gate_passed") is not True:
        raise RuntimeError("PRIVATE_SPLIT_FREEZE_FAIL")
    if r.get("schema") != SCHEMA_STEP1_RECIPE:
        raise RuntimeError("STEP1_RECIPE_SCHEMA_FAIL")
    if r["private_contract_sha256"] != sha256_file(cp):
        raise RuntimeError("RECIPE_PRIVATE_CONTRACT_PIN_FAIL")
    if r["split_manifest_private_sha256"] != sha256_file(sp):
        raise RuntimeError("RECIPE_PRIVATE_SPLIT_PIN_FAIL")
    if r["view_policy"]["mode"] != "copy":
        raise RuntimeError("FORMAL_STEP1_VIEW_MUST_COPY")

    train_ids = sorted(map(str, s["train_ids"]))
    dev_ids = sorted(map(str, s["dev_ids"]))
    holdout_ids = set(map(str, s["final_holdout_ids"]))
    if set(train_ids) & set(dev_ids) or set(train_ids) & holdout_ids or set(dev_ids) & holdout_ids:
        raise RuntimeError("PRIVATE_SPLIT_OVERLAP_FAIL")
    if canonical_ids_sha(train_ids) != s["ids_sha256"]["train"] or canonical_ids_sha(dev_ids) != s["ids_sha256"]["dev"]:
        raise RuntimeError("PRIVATE_SPLIT_ID_HASH_FAIL")

    raw = Path(c["dataset_root"])
    mappings = []
    for split, ids in (("train", train_ids), ("val", dev_ids)):
        for sid in ids:
            meta = c["file_meta"][sid]
            rgb_src = raw / meta["rgb"]["relative_path"]
            lab_src = raw / meta["labels"]["relative_path"]
            rgb_dst = out / "images" / split / (sid + rgb_src.suffix.lower())
            lab_dst = out / "labels" / split / (sid + ".txt")
            rgb_map = copy_verified(rgb_src, rgb_dst, meta["rgb"]["sha256"])
            lab_map = copy_verified(lab_src, lab_dst, meta["labels"]["sha256"])
            mappings.append({"sample_id": sid, "split": split, "rgb": rgb_map, "label": lab_map})

    # Runtime filesystem proof: no filename/sample ID from holdout occurs in the view.
    view_ids = {p.stem for p in (out / "images").rglob("*") if p.is_file()}
    holdout_intersection = sorted(view_ids & holdout_ids)
    if holdout_intersection:
        raise RuntimeError(f"FINAL_HOLDOUT_PRESENT_IN_STEP1_VIEW:{holdout_intersection[:10]}")

    names = c["layout_spec"]["label_format"]["class_names"]
    nc = int(c["layout_spec"]["label_format"]["num_classes"])
    if not isinstance(names, list) or len(names) != nc:
        raise RuntimeError("CLASS_NAMES_COUNT_FAIL")
    dataset_yaml = out / "dataset.yaml"
    yaml_text = (
        "path: " + str(out.resolve()).replace("\\", "/") + "\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {nc}\n"
        "names:\n" + "".join(f"  {i}: {json.dumps(str(n), ensure_ascii=False)}\n" for i, n in enumerate(names))
    )
    dataset_yaml.write_text(yaml_text, encoding="utf-8")

    manifest = {
        "schema": SCHEMA_VIEW_MANIFEST,
        "created_at_utc": utc_now_iso(),
        "view_root": str(out.resolve()),
        "mode": "copy",
        "recipe_sha256": sha256_file(rp),
        "private_contract_sha256": sha256_file(cp),
        "split_manifest_private_sha256": sha256_file(sp),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "dataset_yaml_sha256": sha256_file(dataset_yaml),
        "train_ids": train_ids,
        "dev_ids": dev_ids,
        "train_ids_sha256": canonical_ids_sha(train_ids),
        "dev_ids_sha256": canonical_ids_sha(dev_ids),
        "final_holdout_ids_sha256": s["ids_sha256"]["final_holdout"],
        "final_holdout_intersection_count": len(holdout_intersection),
        "final_holdout_excluded_by_actual_id_set": len(holdout_intersection) == 0,
        "mappings": mappings,
        "mapping_count": len(mappings),
        "all_destination_hashes_verified": True,
    }
    mf = out / "view_manifest.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "view_manifest": str(mf), "view_manifest_sha256": sha256_file(mf),
        "dataset_yaml": str(dataset_yaml), "train": len(train_ids), "dev": len(dev_ids),
        "holdout_intersection": len(holdout_intersection),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
