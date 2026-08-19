#!/usr/bin/env python3
"""Common ZERO-inference evaluation for U0/U1/U2 in T1-TR."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import step3_eval_utils as evu  # noqa: E402
from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402
from multimodal.tseries_core import sha256_file, tensor_sha256  # noqa: E402
from multimodal.tseries_runtime import (  # noqa: E402
    collect_detection_stats, combine_stats_results, load_checkpoint_model,
    results_csv_metrics,
)
from multimodal.t1tr_training_source import (  # noqa: E402
    T0_LAST_SHA256, T1_LAST_SHA256, T1S_ZERO_VAL6_MAP5095, U2_RUN_NAME,
)

SCHEMA = "step4-t1tr-posttrain-zero-eval-v1"

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def clean(result):
    return {k: v for k, v in result.items() if k != "_stats"}

def zero_forward_factory(model, *, require_native_equal=False):
    def forward(sid, sample, batch):
        full, _ = model.p5_full_and_ac_from_input(batch["img"])
        zero = torch.zeros_like(full)
        out = model.predict_with_p5_residual(batch["img"], zero)
        trace = {
            "mode": "ZERO",
            "recipient_id": sid,
            "zero_residual_sha256": tensor_sha256(zero),
        }
        if require_native_equal:
            native = model._predict_once(batch["img"])
            nr = evu.extract_detection_tensor(native).detach()
            zr = evu.extract_detection_tensor(out).detach()
            if not torch.equal(nr, zr):
                raise RuntimeError(f"T1TR_T0_ZERO_NATIVE_ANCHOR_FAIL:{sid}")
            trace["native_detection_sha256"] = tensor_sha256(nr)
            trace["zero_detection_sha256"] = tensor_sha256(zr)
            trace["native_zero_bitwise_equal"] = True
        return out, trace
    return forward

def eval_zero(model, dataset, device, *, require_native_equal=False):
    return collect_detection_stats(
        model, dataset, device,
        zero_forward_factory(model, require_native_equal=require_native_equal),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default=OUT_DEFAULT)
    ap.add_argument("--project", default="runs/step4_t1tr")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="reports/step4_t1tr/posttrain_zero_eval.json")
    a = ap.parse_args()

    out = ROOT / a.out
    if out.exists():
        raise RuntimeError(f"T1TR_REFUSE_OVERWRITE:{out}")

    contract = load(Path(a.contract))
    train_ids = [str(x) for x in contract["train_ids"]]
    val_ids = [str(x) for x in contract["val_ids"]]
    all17 = [str(x) for x in contract["all17_ids"]]
    if set(train_ids) | set(val_ids) != set(all17):
        raise RuntimeError("T1TR_ALL17_SPLIT_CLOSURE_FAIL")

    devarg = str(a.device)
    if devarg == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    elif devarg.startswith("cuda:"):
        device = torch.device(devarg)
    else:
        device = torch.device(f"cuda:{devarg}")

    train_ds = TriModalDataset(contract, split="train", group="C1-I", augment=False)
    val_ds = TriModalDataset(contract, split="val", group="C1-I", augment=False)

    systems = {
        "U0-N": {
            "ckpt": ROOT / "runs/step4_tseries/T0-N_P5_NULL_seed20260812/weights/last.pt",
            "require_native_equal": True,
        },
        "U1-P": {
            "ckpt": ROOT / "runs/step4_tseries/T1-F_P5_FULL_seed20260812/weights/last.pt",
            "require_native_equal": False,
        },
        "U2-S": {
            "ckpt": ROOT / a.project / U2_RUN_NAME / "weights/last.pt",
            "require_native_equal": False,
        },
    }

    rows = {}
    for arm, cfg in systems.items():
        ckpt = cfg["ckpt"]
        if not ckpt.is_file():
            raise RuntimeError(f"T1TR_CHECKPOINT_MISSING:{arm}:{ckpt}")
        if arm == "U0-N" and sha256_file(ckpt) != T0_LAST_SHA256:
            raise RuntimeError("T1TR_U0_CHECKPOINT_SHA_DRIFT")
        if arm == "U1-P" and sha256_file(ckpt) != T1_LAST_SHA256:
            raise RuntimeError("T1TR_U1_CHECKPOINT_SHA_DRIFT")

        model, _ = load_checkpoint_model(ckpt, device)
        if arm in {"U1-P", "U2-S"} and model.treatment_id != "T1-F":
            raise RuntimeError(f"T1TR_MODEL_TREATMENT_FAIL:{arm}:{model.treatment_id}")
        if arm == "U0-N" and model.treatment_id != "T0-N":
            raise RuntimeError(f"T1TR_MODEL_TREATMENT_FAIL:{arm}:{model.treatment_id}")

        zv = eval_zero(
            model, val_ds, device,
            require_native_equal=cfg["require_native_equal"],
        )
        zt = eval_zero(model, train_ds, device)
        all_metric = combine_stats_results([zt, zv], all17)

        row = {
            "checkpoint_path": str(ckpt.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(ckpt),
            "zero_val6": clean(zv),
            "zero_train11": {
                "full": clean(zt)["full"],
                "loo": clean(zt)["loo"],
            },
            "zero_all17": all_metric,
        }
        if arm == "U2-S":
            native = collect_detection_stats(model, val_ds, device)
            row["secondary_u2_native_val6"] = clean(native)
            results_csv = ROOT / a.project / U2_RUN_NAME / "results.csv"
            if results_csv.is_file():
                row["secondary_u2_native_training_curve"] = results_csv_metrics(results_csv)
        rows[arm] = row

    # Frozen semantic anchors.
    u0 = rows["U0-N"]["zero_val6"]["full"]["map50_95"]
    # T0 zero override was bitwise checked per sample; numeric value is inherited from it.
    if abs(float(u0) - 0.26443877551020406) > 1e-12:
        raise RuntimeError(f"T1TR_T0_ZERO_NUMERIC_ANCHOR_FAIL:{u0}")
    u1 = rows["U1-P"]["zero_val6"]["full"]["map50_95"]
    if abs(float(u1) - T1S_ZERO_VAL6_MAP5095) > 1e-12:
        raise RuntimeError(
            f"T1TR_T1_ZERO_NUMERIC_ANCHOR_FAIL:{u1}!={T1S_ZERO_VAL6_MAP5095}"
        )

    report = {
        "schema": SCHEMA,
        "authority": "common last.pt ZERO-inference evaluator; Step3 validator semantics",
        "primary_inference": "ZERO",
        "systems": rows,
        "protocol": {
            "train_ids": train_ids,
            "val_ids": val_ids,
            "all17_ids": all17,
        },
        "anchors": {
            "U0_native_zero_bitwise_per_val_sample": True,
            "U0_zero_val6_map50_95": u0,
            "U1_zero_val6_map50_95": u1,
            "U1_zero_matches_T1S": True,
        },
        "interpretation_discipline": {
            "u2_native_curve_is_secondary_only": True,
            "primary_comparison_uses_zero_inference_for_all_arms": True,
            "no_best_checkpoint_decision": True,
        },
        "provenance": {
            "evaluator_sha256": sha256_file(ROOT / "scripts/eval_t1tr.py"),
            "u2_manifest_sha256": sha256_file(
                ROOT / a.project / U2_RUN_NAME / "manifest.json"
            ),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"schema": SCHEMA, "out": str(out)}, indent=2))

if __name__ == "__main__":
    main()
