#!/usr/bin/env python3
"""Formal Step1 RGB baseline runner. Accepts only a hash-pinned view manifest, never arbitrary --data."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multimodal.t1gr_e2e5 import (  # noqa: E402
    SCHEMA_STEP1_RECIPE, SCHEMA_VIEW_MANIFEST, canonical_ids_sha, parse_utc,
    sha256_file, utc_now_iso,
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def actual_ids(view_root: Path, split: str) -> list[str]:
    d = view_root / "images" / split
    if not d.is_dir():
        raise RuntimeError(f"VIEW_SPLIT_DIR_MISSING:{d}")
    return sorted(p.stem for p in d.iterdir() if p.is_file())


def verify_view_files(v: dict) -> None:
    view_root = Path(v["view_root"])
    yaml_path = Path(v["dataset_yaml"])
    if sha256_file(yaml_path) != v["dataset_yaml_sha256"]:
        raise RuntimeError("VIEW_DATASET_YAML_SHA_DRIFT")
    for row in v["mappings"]:
        for k in ("rgb", "label"):
            dst = Path(row[k]["dest"])
            if not dst.is_file() or sha256_file(dst) != row[k]["sha256"]:
                raise RuntimeError(f"VIEW_FILE_SHA_DRIFT:{dst}")
    train = actual_ids(view_root, "train")
    dev = actual_ids(view_root, "val")
    if canonical_ids_sha(train) != v["train_ids_sha256"] or train != sorted(v["train_ids"]):
        raise RuntimeError("VIEW_ACTUAL_TRAIN_IDS_FAIL")
    if canonical_ids_sha(dev) != v["dev_ids_sha256"] or dev != sorted(v["dev_ids"]):
        raise RuntimeError("VIEW_ACTUAL_DEV_IDS_FAIL")
    if set(train) & set(dev):
        raise RuntimeError("VIEW_TRAIN_DEV_OVERLAP")
    if len(train) + len(dev) != int(v["mapping_count"]):
        raise RuntimeError("VIEW_EXTRA_OR_MISSING_SAMPLE_FILES")


def preflight(recipe_path: Path, view_manifest_path: Path) -> tuple[dict, dict, Path]:
    r, v = load(recipe_path), load(view_manifest_path)
    if r.get("schema") != SCHEMA_STEP1_RECIPE:
        raise RuntimeError("STEP1_RECIPE_SCHEMA_FAIL")
    if v.get("schema") != SCHEMA_VIEW_MANIFEST:
        raise RuntimeError("STEP1_VIEW_MANIFEST_SCHEMA_FAIL")
    if v.get("recipe_sha256") != sha256_file(recipe_path):
        raise RuntimeError("VIEW_RECIPE_PIN_FAIL")
    if v.get("private_contract_sha256") != r.get("private_contract_sha256"):
        raise RuntimeError("VIEW_CONTRACT_PIN_FAIL")
    if v.get("split_manifest_private_sha256") != r.get("split_manifest_private_sha256"):
        raise RuntimeError("VIEW_SPLIT_PIN_FAIL")
    if v.get("train_ids_sha256") != r["split_ids_sha256"]["train"]:
        raise RuntimeError("VIEW_TRAIN_IDS_COMMITMENT_FAIL")
    if v.get("dev_ids_sha256") != r["split_ids_sha256"]["dev"]:
        raise RuntimeError("VIEW_DEV_IDS_COMMITMENT_FAIL")
    if v.get("final_holdout_ids_sha256") != r["split_ids_sha256"]["final_holdout"]:
        raise RuntimeError("VIEW_HOLDOUT_COMMITMENT_FAIL")
    if v.get("final_holdout_excluded_by_actual_id_set") is not True or int(v.get("final_holdout_intersection_count", -1)) != 0:
        raise RuntimeError("VIEW_HOLDOUT_EXCLUSION_EVIDENCE_FAIL")
    verify_view_files(v)

    ck = Path(r["base_checkpoint"])
    if not ck.is_file():
        raise RuntimeError("BASE_CHECKPOINT_MISSING_AT_RUNTIME")
    got_ck = sha256_file(ck)
    if got_ck != r["base_checkpoint_sha256"]:
        raise RuntimeError(f"BASE_CHECKPOINT_SHA_DRIFT:{got_ck}")
    return r, v, ck


def normalize_value(v):
    if isinstance(v, Path):
        return str(v)
    return v


def compare_effective_args(effective, expected: dict) -> dict:
    mismatch = {}
    for k, exp in expected.items():
        got = getattr(effective, k, None)
        if normalize_value(got) != normalize_value(exp):
            mismatch[k] = {"expected": exp, "effective": got}
    return mismatch


def build_model(base_checkpoint: Path, nc: int, end2end: bool):
    import torch
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    d = yaml_model_load("yolo26s.yaml")
    d["nc"] = int(nc)
    d["end2end"] = bool(end2end)
    model = DetectionModel(d, ch=3, nc=int(nc))
    ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    src = ckpt["model"].float().state_dict()
    model.load(src)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--view-manifest", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/step4_t1gr_step1")
    ap.add_argument("--name", default="STEP1_RGB_BASELINE")
    args = ap.parse_args()

    recipe_path, view_path = Path(args.recipe), Path(args.view_manifest)
    r, v, ck = preflight(recipe_path, view_path)
    start_time = utc_now_iso()
    freeze_precedes = parse_utc(r["freeze_timestamp_utc"]) < parse_utc(start_time)
    if not freeze_precedes:
        raise RuntimeError("SPLIT_FREEZE_DOES_NOT_PRECEDE_TRAINING")

    import ultralytics
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    if str(ultralytics.__version__) != str(r["ultralytics_version"]):
        raise RuntimeError(f"ULTRALYTICS_VERSION_DRIFT:{ultralytics.__version__}!={r['ultralytics_version']}")

    run = ROOT / args.project / args.name
    if run.exists():
        raise RuntimeError(f"REFUSE_EXISTING_RUN:{run}")
    model = build_model(ck, int(r["num_classes"]), bool(r["train_args"]["end2end"]))
    head = model.model[-1]
    physical_nc = int(getattr(head, "nc", -1))
    if physical_nc != int(r["num_classes"]):
        raise RuntimeError(f"STEP1_PRETRAIN_PHYSICAL_HEAD_NC_FAIL:{physical_nc}")
    actual_end2end = bool(getattr(head, "end2end", getattr(model, "end2end", False)))
    if actual_end2end != bool(r["train_args"]["end2end"]):
        raise RuntimeError(f"STEP1_PRETRAIN_HEAD_MODE_FAIL:{actual_end2end}")

    overrides = dict(r["train_args"])
    overrides.update(r["eval_args"])
    overrides.update({
        "task": "detect",
        "mode": "train",
        "model": "yolo26s.yaml",
        "data": v["dataset_yaml"],
        "device": args.device,
        "project": str(ROOT / args.project),
        "name": args.name,
        "exist_ok": False,
        "pretrained": False,
    })
    trainer = DetectionTrainer(overrides=overrides)
    # Compare the actual Trainer args BEFORE any optimizer/data work.
    expected_effective = dict(r["train_args"])
    expected_effective.update(r["eval_args"])
    mismatch = compare_effective_args(trainer.args, expected_effective)
    if mismatch:
        raise RuntimeError(f"STEP1_EFFECTIVE_ARGS_PREFLIGHT_MISMATCH:{mismatch}")
    trainer.model = model
    trainer.model.args = trainer.args
    trainer.train()

    last = run / "weights" / "last.pt"
    args_yaml = run / "args.yaml"
    if not last.is_file() or not args_yaml.is_file():
        raise RuntimeError("STEP1_RUN_ARTIFACT_MISSING")

    # args.yaml is the complete post-run effective snapshot. Re-parse and verify frozen keys.
    import yaml
    post_effective = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
    post_mismatch = {}
    for k, exp in expected_effective.items():
        got = post_effective.get(k)
        if normalize_value(got) != normalize_value(exp):
            post_mismatch[k] = {"expected": exp, "effective": got}
    if post_mismatch:
        raise RuntimeError(f"STEP1_EFFECTIVE_ARGS_POSTRUN_MISMATCH:{post_mismatch}")

    # args.yaml is preserved as the complete effective snapshot; pin its hash.
    manifest = {
        "schema": "t1gr-step1-run-manifest-v2",
        "status": "STEP1_BASELINE_TRAIN_COMPLETE",
        "training_started_at_utc": start_time,
        "training_finished_at_utc": utc_now_iso(),
        "freeze_timestamp_utc": r["freeze_timestamp_utc"],
        "freeze_precedes_training_derived": freeze_precedes,
        "recipe_sha256": sha256_file(recipe_path),
        "view_manifest_sha256": sha256_file(view_path),
        "dataset_yaml_sha256": v["dataset_yaml_sha256"],
        "actual_train_ids_sha256": canonical_ids_sha(actual_ids(Path(v["view_root"]), "train")),
        "actual_dev_ids_sha256": canonical_ids_sha(actual_ids(Path(v["view_root"]), "val")),
        "expected_train_ids_sha256": r["split_ids_sha256"]["train"],
        "expected_dev_ids_sha256": r["split_ids_sha256"]["dev"],
        "final_holdout_commitment_sha256": r["split_ids_sha256"]["final_holdout"],
        "view_holdout_intersection_count": v["final_holdout_intersection_count"],
        "final_holdout_access_derived": "EXCLUDED_FROM_PINNED_VIEW",
        "base_checkpoint_sha256_runtime": sha256_file(ck),
        "base_checkpoint_sha256_expected": r["base_checkpoint_sha256"],
        "ultralytics_version_runtime": str(ultralytics.__version__),
        "ultralytics_version_expected": r["ultralytics_version"],
        "pretrain_physical_head_nc": physical_nc,
        "pretrain_head_end2end": actual_end2end,
        "effective_args_yaml": str(args_yaml.resolve()),
        "effective_args_yaml_sha256": sha256_file(args_yaml),
        "effective_args_frozen_keys_match": True,
        "effective_args_frozen_key_count": len(expected_effective),
        "last_pt": str(last.resolve()),
        "last_pt_sha256": sha256_file(last),
    }
    (run / "t1gr_step1_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
