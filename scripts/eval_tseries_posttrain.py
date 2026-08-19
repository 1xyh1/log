#!/usr/bin/env python3
"""Post-training matched performance evaluation for T0/T1/T2 last.pt."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402
from multimodal.tseries_core import RUN_NAMES, TREATMENTS, sha256_file  # noqa: E402
from multimodal.tseries_runtime import (  # noqa: E402
    collect_detection_stats, combine_stats_results, load_checkpoint_model,
    results_csv_metrics,
)

SCHEMA = "step4-tseries-posttrain-performance-v1"

def clean_result(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "_stats"}

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_tseries")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--device", default="0")
    p.add_argument("--out", default="reports/step4_tseries/posttrain_performance.json")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    out = ROOT / a.out
    if out.exists() and not a.overwrite:
        raise RuntimeError(f"T_SERIES_REFUSE_OVERWRITE:{out}")

    contract = load_json(Path(a.contract))
    expected_train = [str(x) for x in contract["train_ids"]]
    expected_val = [str(x) for x in contract["val_ids"]]
    expected_all = [str(x) for x in contract["all17_ids"]]
    if set(expected_train) | set(expected_val) != set(expected_all):
        raise RuntimeError("T_SERIES_ALL17_SPLIT_CLOSURE_FAIL")

    devarg = str(a.device)
    if devarg == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    elif devarg.startswith("cuda:"):
        device = torch.device(devarg)
    else:
        device = torch.device(f"cuda:{devarg}")

    train_ds = TriModalDataset(contract, split="train", group="C1-I", augment=False)
    val_ds = TriModalDataset(contract, split="val", group="C1-I", augment=False)
    if list(train_ds.ids) != expected_train:
        raise RuntimeError("T_SERIES_TRAIN11_DRIFT")
    if list(val_ds.ids) != expected_val:
        raise RuntimeError("T_SERIES_VAL6_DRIFT")

    systems = {}
    for treatment in ("T0-N", "T1-F", "T2-A"):
        run_dir = (ROOT / a.project / RUN_NAMES[treatment]).resolve()
        manifest_path = run_dir / "manifest.json"
        ckpt_path = run_dir / "weights/last.pt"
        results_path = run_dir / "results.csv"
        if not manifest_path.exists() or not ckpt_path.exists():
            raise RuntimeError(f"T_SERIES_FORMAL_ARTIFACT_MISSING:{treatment}")
        manifest = load_json(manifest_path)
        if manifest.get("treatment_id") != treatment or manifest.get("run_kind") != "formal":
            raise RuntimeError(f"T_SERIES_MANIFEST_IDENTITY:{treatment}")
        model, _ = load_checkpoint_model(ckpt_path, device)
        if model.treatment_id != treatment:
            raise RuntimeError(f"T_SERIES_CHECKPOINT_TREATMENT:{treatment}:{model.treatment_id}")

        val = collect_detection_stats(model, val_ds, device)
        train = collect_detection_stats(model, train_ds, device)
        all17 = combine_stats_results([train, val], expected_all)
        curve = results_csv_metrics(results_path)
        systems[treatment] = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "last_pt_sha256": sha256_file(ckpt_path),
            "results_csv_sha256": sha256_file(results_path),
            "last_val6": clean_result(val),
            "train11": {
                "full": clean_result(train)["full"],
                "loo": clean_result(train)["loo"],
            },
            "all17": all17,
            "training_curve": curve,
            "last_pt_val_vs_curve_last_delta": float(
                val["full"]["map50_95"] - curve["last_val_map50_95"]
            ),
        }

    report = {
        "schema": SCHEMA,
        "authority": {
            "last_val6": "re-evaluated last.pt with stock Step3 validator semantics",
            "train11": "re-evaluated last.pt",
            "all17": "union of train11+val6 per-image validator stats",
            "late10": "results.csv median over final 10 epochs",
            "best_epoch": "descriptive_only",
        },
        "systems": systems,
        "provenance": {
            "evaluator_sha256": sha256_file(ROOT / "scripts/eval_tseries_posttrain.py"),
            "model_source_sha256": sha256_file(ROOT / "src/multimodal/tseries_p5_model.py"),
        },
        "protocol": {
            "val_ids": expected_val,
            "train_ids": expected_train,
            "all17_ids": expected_all,
            "checkpoint": "last.pt",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"schema": SCHEMA, "out": str(out)}, indent=2))

if __name__ == "__main__":
    main()
