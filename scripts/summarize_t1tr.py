#!/usr/bin/env python3
"""Pre-registered T1-TR adjudication from common ZERO-inference endpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1tr_training_source import (  # noqa: E402
    contrast_label, decide_training_source_specificity,
)

SCHEMA = "step4-t1tr-summary-v1"

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def primary_vector(report, arm):
    s = report["systems"][arm]
    return {
        "val6_zero": float(s["zero_val6"]["full"]["map50_95"]),
        "train11_zero": float(s["zero_train11"]["full"]["map50_95"]),
        "all17_zero": float(s["zero_all17"]["map50_95"]),
    }

def loo_contrast(report, new_arm, base_arm):
    n = report["systems"][new_arm]["zero_val6"]["loo"]
    b = report["systems"][base_arm]["zero_val6"]["loo"]
    if set(n) != set(b):
        raise RuntimeError("T1TR_LOO_ID_MISMATCH")
    vals = {
        sid: float(n[sid]["map50_95"]) - float(b[sid]["map50_95"])
        for sid in n
    }
    xs = list(vals.values())
    return {
        "loo": vals,
        "median": float(statistics.median(xs)),
        "positive_folds": sum(v > 0 for v in xs),
        "negative_folds": sum(v < 0 for v in xs),
        "zero_folds": sum(v == 0 for v in xs),
        "authority": "secondary_sensitivity_only",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="reports/step4_t1tr/posttrain_zero_eval.json")
    ap.add_argument("--out", default="reports/step4_t1tr/t1tr_summary.json")
    a = ap.parse_args()

    out = ROOT / a.out
    if out.exists():
        raise RuntimeError(f"T1TR_REFUSE_OVERWRITE:{out}")

    rep = load(ROOT / a.eval)
    if rep.get("schema") != "step4-t1tr-posttrain-zero-eval-v1":
        raise RuntimeError("T1TR_EVAL_SCHEMA_FAIL")
    if rep.get("primary_inference") != "ZERO":
        raise RuntimeError("T1TR_PRIMARY_INFERENCE_FAIL")

    p = {arm: primary_vector(rep, arm) for arm in ("U0-N", "U1-P", "U2-S")}
    decision = decide_training_source_specificity(p["U0-N"], p["U1-P"], p["U2-S"])

    report = {
        "schema": SCHEMA,
        "primary_inference": "ZERO",
        "primary_endpoints": p,
        "contrasts": decision["contrasts"],
        "loo_sensitivity": {
            "U1_minus_U0": loo_contrast(rep, "U1-P", "U0-N"),
            "U2_minus_U0": loo_contrast(rep, "U2-S", "U0-N"),
            "U1_minus_U2": loo_contrast(rep, "U1-P", "U2-S"),
        },
        "decision": {
            "branch": decision["branch"],
            "replication_seed_go": decision["replication_seed_go"],
            "depth_go": False,
            "production_go": False,
        },
        "secondary": {
            "u2_native_val6": rep["systems"]["U2-S"].get("secondary_u2_native_val6"),
            "u2_native_training_curve": rep["systems"]["U2-S"].get(
                "secondary_u2_native_training_curve"
            ),
        },
        "interpretation_discipline": {
            "no_arbitrary_ap_margin": True,
            "all_primary_arms_use_zero_inference": True,
            "val6_loo_is_secondary_only": True,
            "u2_native_curve_is_secondary_only": True,
            "single_seed_is_not_replication": True,
            "depth_remains_hold": True,
            "production_remains_hold": True,
        },
        "provenance": {
            "zero_eval_sha256": sha256_file(ROOT / a.eval),
            "summary_source_sha256": sha256_file(ROOT / "scripts/summarize_t1tr.py"),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "decision": report["decision"],
        "out": str(out),
    }, indent=2))

if __name__ == "__main__":
    main()
