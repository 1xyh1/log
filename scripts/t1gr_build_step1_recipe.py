#!/usr/bin/env python3
"""Freeze a fully explicit Step1 RGB baseline recipe. Does not train or expose holdout IDs."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multimodal.t1gr_e2e5 import (  # noqa: E402
    SCHEMA_CONTRACT_PUBLIC, SCHEMA_SPLIT_FREEZE_PUBLIC, SCHEMA_STEP1_RECIPE,
    sha256_file, utc_now_iso,
)

REQUIRED_TRAIN_ARGS = (
    "epochs", "batch", "imgsz", "patience", "optimizer", "lr0", "lrf", "momentum",
    "weight_decay", "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "nbs",
    "amp", "workers", "deterministic", "cache", "close_mosaic", "hsv_h", "hsv_s",
    "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud",
    "fliplr", "mosaic", "mixup", "cutmix", "copy_paste", "erasing", "plots",
    "end2end", "seed",
)
REQUIRED_EVAL_ARGS = ("split", "iou", "max_det", "conf", "half", "plots", "save_json")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def unresolved(d: dict, keys: tuple[str, ...]) -> list[str]:
    return [k for k in keys if k not in d or d[k] is None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-contract", required=True)
    ap.add_argument("--split-freeze-public", required=True)
    ap.add_argument("--training-spec", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--out", default="reports/step4_t1gr/step1_baseline_recipe.json")
    args = ap.parse_args()

    cp, fp, tsp = Path(args.public_contract), Path(args.split_freeze_public), Path(args.training_spec)
    c, f, ts = load(cp), load(fp), load(tsp)
    if c.get("schema") != SCHEMA_CONTRACT_PUBLIC or c.get("contract_gate_passed") is not True:
        raise RuntimeError("PUBLIC_CONTRACT_NOT_FORMAL_PASS")
    if c.get("full_hash_mode") is not True or c.get("format_gate_passed") is not True:
        raise RuntimeError("PUBLIC_CONTRACT_FORMAT_OR_HASH_GATE_FAIL")
    if f.get("schema") != SCHEMA_SPLIT_FREEZE_PUBLIC or f.get("contains_any_sample_ids") is not False:
        raise RuntimeError("PUBLIC_SPLIT_FREEZE_FAIL")
    if f.get("proposal_gate_passed") is not True:
        raise RuntimeError("SPLIT_FREEZE_PROPOSAL_NOT_PASS")
    if ts.get("schema") != "t1gr-step1-training-spec-v2":
        raise RuntimeError("TRAINING_SPEC_SCHEMA_FAIL")
    train_args = dict(ts.get("train_args") or {})
    eval_args = dict(ts.get("eval_args") or {})
    missing = [f"train_args.{x}" for x in unresolved(train_args, REQUIRED_TRAIN_ARGS)]
    missing += [f"eval_args.{x}" for x in unresolved(eval_args, REQUIRED_EVAL_ARGS)]
    if missing:
        raise RuntimeError(f"STEP1_RECIPE_UNRESOLVED:{missing}")
    if eval_args["split"] != "val":
        raise RuntimeError("STEP1_EVAL_SPLIT_MUST_BE_DEV_VIEW_VAL")
    if train_args["deterministic"] is not True:
        raise RuntimeError("STEP1_DETERMINISTIC_MUST_BE_TRUE")

    ck = Path(args.base_checkpoint)
    if not ck.is_file():
        raise RuntimeError("BASE_CHECKPOINT_MISSING")

    try:
        import ultralytics
        import torch
        ultra_version = str(ultralytics.__version__)
        torch_version = str(torch.__version__)
    except Exception as e:
        raise RuntimeError(f"FORMAL_RECIPE_REQUIRES_ULTRALYTICS_TORCH:{e}")

    rec = {
        "schema": SCHEMA_STEP1_RECIPE,
        "created_at_utc": utc_now_iso(),
        "task": "detect",
        "architecture": "yolo26s",
        "model_build": "fresh physical nc from yolo26s.yaml + partial pretrained state transfer",
        "base_checkpoint": str(ck.resolve()),
        "base_checkpoint_sha256": sha256_file(ck),
        "public_contract_sha256": sha256_file(cp),
        "private_contract_sha256": c["private_contract_sha256"],
        "split_freeze_public_sha256": sha256_file(fp),
        "split_manifest_private_sha256": f["split_manifest_private_sha256"],
        "split_ids_sha256": f["ids_sha256"],
        "split_sample_counts": f["sample_counts"],
        "freeze_timestamp_utc": f["freeze_timestamp_utc"],
        "num_classes": int(c["num_classes"]),
        "class_names": c["class_names"],
        "ultralytics_version": ultra_version,
        "torch_version": torch_version,
        "python_version": platform.python_version(),
        "train_args": train_args,
        "eval_args": eval_args,
        "view_policy": {
            "mode": "copy",
            "contains": ["train", "dev"],
            "final_holdout": "MUST_NOT_BE_PRESENT",
            "runner_accepts_arbitrary_dataset_yaml": False,
        },
        "final_holdout_access": "FORBIDDEN_UNTIL_T1GR_FINAL_ADJUDICATION",
        "effective_args_policy": "post-run args.yaml must match all frozen train/eval keys where applicable",
    }
    out = ROOT / args.out
    if out.exists():
        raise RuntimeError(f"REFUSE_OVERWRITE:{out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "recipe": str(out), "ultralytics_version": ultra_version,
        "base_checkpoint_sha256": rec["base_checkpoint_sha256"],
        "frozen_train_arg_count": len(train_args), "frozen_eval_arg_count": len(eval_args),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
