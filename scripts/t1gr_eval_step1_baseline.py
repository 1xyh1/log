#!/usr/bin/env python3
"""DEV-only Step1 evaluator. Dataset path is derived only from the pinned view manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multimodal.t1gr_e2e5 import (  # noqa: E402
    SCHEMA_STEP1_RECIPE, SCHEMA_VIEW_MANIFEST, canonical_ids_sha, sha256_file, utc_now_iso,
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def actual_dev_ids(view_root: Path) -> list[str]:
    d = view_root / "images" / "val"
    return sorted(p.stem for p in d.iterdir() if p.is_file())


def verify_eval_inputs(recipe_path: Path, view_path: Path, run_dir: Path) -> tuple[dict, dict, dict, Path]:
    r, v = load(recipe_path), load(view_path)
    if r.get("schema") != SCHEMA_STEP1_RECIPE or v.get("schema") != SCHEMA_VIEW_MANIFEST:
        raise RuntimeError("STEP1_EVAL_SCHEMA_FAIL")
    if v.get("recipe_sha256") != sha256_file(recipe_path):
        raise RuntimeError("STEP1_EVAL_VIEW_RECIPE_PIN_FAIL")
    if v["dev_ids_sha256"] != r["split_ids_sha256"]["dev"]:
        raise RuntimeError("STEP1_EVAL_DEV_COMMITMENT_FAIL")
    if v.get("final_holdout_excluded_by_actual_id_set") is not True or int(v.get("final_holdout_intersection_count", -1)) != 0:
        raise RuntimeError("STEP1_EVAL_HOLDOUT_VIEW_EVIDENCE_FAIL")
    dataset_yaml = Path(v["dataset_yaml"])
    if sha256_file(dataset_yaml) != v["dataset_yaml_sha256"]:
        raise RuntimeError("STEP1_EVAL_DATASET_YAML_SHA_DRIFT")
    dev = actual_dev_ids(Path(v["view_root"]))
    if dev != sorted(v["dev_ids"]) or canonical_ids_sha(dev) != r["split_ids_sha256"]["dev"]:
        raise RuntimeError("STEP1_EVAL_ACTUAL_VAL_IDS_NOT_FROZEN_DEV")

    mf = run_dir / "t1gr_step1_manifest.json"
    last = run_dir / "weights" / "last.pt"
    if not mf.is_file() or not last.is_file():
        raise RuntimeError("STEP1_RUN_INCOMPLETE")
    m = load(mf)
    if m.get("schema") != "t1gr-step1-run-manifest-v2":
        raise RuntimeError("STEP1_RUN_MANIFEST_SCHEMA_FAIL")
    if m.get("recipe_sha256") != sha256_file(recipe_path) or m.get("view_manifest_sha256") != sha256_file(view_path):
        raise RuntimeError("STEP1_RUN_PIN_FAIL")
    if m.get("final_holdout_access_derived") != "EXCLUDED_FROM_PINNED_VIEW":
        raise RuntimeError("STEP1_RUN_HOLDOUT_EVIDENCE_FAIL")
    if m.get("last_pt_sha256") != sha256_file(last):
        raise RuntimeError("STEP1_LAST_PT_SHA_DRIFT")
    if m.get("actual_dev_ids_sha256") != r["split_ids_sha256"]["dev"]:
        raise RuntimeError("STEP1_RUN_DEV_IDS_PIN_FAIL")
    return r, v, m, last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--view-manifest", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="reports/step4_t1gr/step1_baseline_report.json")
    args = ap.parse_args()

    recipe_path, view_path, run_dir = Path(args.recipe), Path(args.view_manifest), Path(args.run_dir)
    r, v, m, last = verify_eval_inputs(recipe_path, view_path, run_dir)

    import ultralytics
    from ultralytics import YOLO
    if str(ultralytics.__version__) != str(r["ultralytics_version"]):
        raise RuntimeError("STEP1_EVAL_ULTRALYTICS_VERSION_DRIFT")
    y = YOLO(str(last))
    head = y.model.model[-1]
    physical_nc = int(getattr(head, "nc", -1))
    if physical_nc != int(r["num_classes"]):
        raise RuntimeError(f"STEP1_PHYSICAL_HEAD_NC_FAIL:{physical_nc}!={r['num_classes']}")
    actual_end2end = bool(getattr(head, "end2end", getattr(y.model, "end2end", False)))
    if actual_end2end != bool(r["train_args"]["end2end"]):
        raise RuntimeError("STEP1_EVAL_HEAD_MODE_DRIFT")

    eval_args = dict(r["eval_args"])
    if eval_args.pop("split") != "val":
        raise RuntimeError("STEP1_EVAL_SPLIT_NOT_VAL")
    result = y.val(
        data=v["dataset_yaml"],
        split="val",
        device=args.device,
        verbose=False,
        **eval_args,
    )
    box = getattr(result, "box", None)
    metrics = {}
    if box is not None:
        metrics = {
            "map50_95": float(box.map),
            "map50": float(box.map50),
            "per_class_map50_95": [float(x) for x in getattr(box, "maps", [])],
        }
    actual_dev = actual_dev_ids(Path(v["view_root"]))
    report = {
        "schema": "t1gr-step1-baseline-report-v2",
        "status": "STEP1_BASELINE_EXECUTED",
        "evaluated_at_utc": utc_now_iso(),
        "authority": "DEV_ONLY_DIAGNOSTIC_BASELINE",
        "recipe_sha256": sha256_file(recipe_path),
        "view_manifest_sha256": sha256_file(view_path),
        "run_manifest_sha256": sha256_file(run_dir / "t1gr_step1_manifest.json"),
        "last_pt_sha256": sha256_file(last),
        "dataset_yaml_sha256": v["dataset_yaml_sha256"],
        "actual_eval_ids_sha256": canonical_ids_sha(actual_dev),
        "expected_dev_ids_sha256": r["split_ids_sha256"]["dev"],
        "actual_eval_count": len(actual_dev),
        "physical_head_nc": physical_nc,
        "expected_nc": int(r["num_classes"]),
        "head_end2end": actual_end2end,
        "eval_args": r["eval_args"],
        "dev_metrics": metrics,
        "final_holdout_access_derived": "EXCLUDED_FROM_PINNED_VIEW",
        "final_holdout_commitment_sha256": r["split_ids_sha256"]["final_holdout"],
    }
    out = ROOT / args.out
    if out.exists():
        raise RuntimeError(f"REFUSE_OVERWRITE:{out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
