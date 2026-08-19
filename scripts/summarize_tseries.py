#!/usr/bin/env python3
"""Summarize matched T-series performance and retrained paired causality."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCHEMA = "step4-tseries-summary-v1"

def sha256_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def primary_vector(perf: dict, treatment: str) -> dict:
    s = perf["systems"][treatment]
    return {
        "last_val6": float(s["last_val6"]["full"]["map50_95"]),
        "late10_median_val6": float(s["training_curve"]["late10_median_val_map50_95"]),
        "train11": float(s["train11"]["full"]["map50_95"]),
    }

def contrast(a: dict, b: dict) -> dict:
    d = {k: float(a[k] - b[k]) for k in a}
    vals = list(d.values())
    if all(v > 0 for v in vals):
        label = "STABLE_POSITIVE"
    elif all(v < 0 for v in vals):
        label = "STABLE_NEGATIVE"
    elif all(v == 0 for v in vals):
        label = "EXACT_TIE"
    else:
        label = "MIXED"
    return {"deltas": d, "label": label}

def loo_contrast(perf: dict, new_t: str, base_t: str) -> dict:
    n = perf["systems"][new_t]["last_val6"]["loo"]
    b = perf["systems"][base_t]["last_val6"]["loo"]
    if set(n) != set(b):
        raise RuntimeError("T_SERIES_PERFORMANCE_LOO_ID_MISMATCH")
    vals = {
        sid: float(n[sid]["map50_95"]) - float(b[sid]["map50_95"])
        for sid in n
    }
    xs = list(vals.values())
    import statistics
    return {
        "loo": vals,
        "median": float(statistics.median(xs)),
        "positive_folds": sum(v > 0 for v in xs),
        "negative_folds": sum(v < 0 for v in xs),
        "zero_folds": sum(v == 0 for v in xs),
        "authority": "secondary_sensitivity_only",
    }

def decision(c10: dict, c21: dict, c20: dict, paired_t1: str, paired_t2: str) -> dict:
    t1_paired_positive = paired_t1 == "SEED20260812_POSITIVE_PAIRED_EVIDENCE"
    t2_paired_positive = paired_t2 == "SEED20260812_POSITIVE_PAIRED_EVIDENCE"
    replication = False
    if c20["label"] == "STABLE_POSITIVE" and t2_paired_positive:
        replication = True
        if c21["label"] == "STABLE_POSITIVE":
            branch = "T2_SINGLE_SEED_CANDIDATE_GO_TO_REPLICATION"
            reason = "T2 beats matched NULL and T1 on all primary endpoints and retains positive paired IR evidence."
        elif c10["label"] == "STABLE_POSITIVE" and c21["label"] == "STABLE_NEGATIVE":
            if t1_paired_positive:
                branch = "T1_FULL_LEADS_T2_CENTERING_MAY_HURT_GO_TO_REPLICATION"
                reason = "Both P5-only arms establish paired IR evidence, but T1 is stably better than T2; forced centering may remove useful trainable content."
            else:
                branch = "T2_PAIRED_NET_GAIN_BUT_T1_PERFORMANCE_LEADS_CAUSAL_SPLIT_UNRESOLVED"
                reason = "T2 has net paired gain over NULL, but T1 performs better without establishing paired causality; replicate before architecture choice."
        elif c10["label"] == "STABLE_POSITIVE":
            branch = "P5_ONLY_TOPOLOGY_CANDIDATE_CENTERING_INCREMENT_UNPROVEN"
            reason = "P5-only FULL already beats NULL; T2 also beats NULL with paired IR evidence, but centering increment over T1 is not stable."
        else:
            branch = "T2_NET_GAIN_WITH_PAIRED_IR_CENTERING_CAUSAL_SPLIT_UNRESOLVED"
            reason = "T2 beats NULL and retains paired IR evidence, but T1 contrast does not isolate a stable centering increment."
    elif c20["label"] == "STABLE_POSITIVE" and not t2_paired_positive:
        branch = "ARCHITECTURAL_GAIN_PAIRED_COMPLEMENTARITY_UNPROVEN"
        reason = "T2 performance improves, but retrained recipient-vs-donor paired causality is not stably positive."
    elif c10["label"] == "STABLE_POSITIVE" and t1_paired_positive:
        replication = True
        branch = "T1_P5_FULL_SINGLE_SEED_CANDIDATE_GO_TO_REPLICATION"
        reason = "T1 has stable gain over NULL and retains positive paired IR evidence while T2 net gain is not established."
    elif c10["label"] == "STABLE_POSITIVE":
        branch = "T1_ARCHITECTURAL_GAIN_PAIRED_COMPLEMENTARITY_UNPROVEN"
        reason = "T1 improves performance over NULL, but correct-pair IR dependence is not stably positive."
    elif c10["label"] == "STABLE_NEGATIVE" and c20["label"] == "STABLE_NEGATIVE":
        branch = "STOP_P5_DIRECT_IR_TRAINING_ROUTE"
        reason = "Both IR treatments are stably worse than architecture-matched NULL."
    else:
        branch = "T_SERIES_FIRST_SEED_INCONCLUSIVE"
        reason = "Primary matched contrasts are mixed; no threshold or post-hoc margin is introduced."
    return {
        "branch": branch,
        "replication_seed_go": replication,
        "depth_go": False,
        "production_go": False,
        "reason": reason,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--performance", default="reports/step4_tseries/posttrain_performance.json")
    p.add_argument("--paired", default="reports/step4_tseries/posttrain_paired.json")
    p.add_argument("--out", default="reports/step4_tseries/tseries_summary.json")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    out = ROOT / a.out
    if out.exists() and not a.overwrite:
        raise RuntimeError(f"T_SERIES_REFUSE_OVERWRITE:{out}")
    perf = load(ROOT / a.performance)
    paired = load(ROOT / a.paired)
    if perf.get("schema") != "step4-tseries-posttrain-performance-v1":
        raise RuntimeError("T_SERIES_PERFORMANCE_SCHEMA")
    if paired.get("schema") != "step4-tseries-posttrain-paired-v1":
        raise RuntimeError("T_SERIES_PAIRED_SCHEMA")

    primary = {t: primary_vector(perf, t) for t in ("T0-N", "T1-F", "T2-A")}
    c10 = contrast(primary["T1-F"], primary["T0-N"])
    c21 = contrast(primary["T2-A"], primary["T1-F"])
    c20 = contrast(primary["T2-A"], primary["T0-N"])
    t1_paired = paired["systems"]["T1-F"]["single_seed_label"]
    t2_paired = paired["systems"]["T2-A"]["single_seed_label"]

    report = {
        "schema": SCHEMA,
        "primary_endpoints": primary,
        "contrasts": {
            "T1_minus_T0": c10,
            "T2_minus_T1": c21,
            "T2_minus_T0": c20,
        },
        "paired_causality": {
            "T1-F": t1_paired,
            "T2-A": t2_paired,
        },
        "loo_sensitivity": {
            "T1_minus_T0": loo_contrast(perf, "T1-F", "T0-N"),
            "T2_minus_T1": loo_contrast(perf, "T2-A", "T1-F"),
            "T2_minus_T0": loo_contrast(perf, "T2-A", "T0-N"),
        },
        "secondary_all17": {
            t: float(perf["systems"][t]["all17"]["map50_95"])
            for t in ("T0-N", "T1-F", "T2-A")
        },
        "decision": decision(c10, c21, c20, t1_paired, t2_paired),
        "provenance": {
            "performance_sha256": sha256_file(ROOT / a.performance),
            "paired_sha256": sha256_file(ROOT / a.paired),
            "summary_source_sha256": sha256_file(ROOT / "scripts/summarize_tseries.py"),
        },
        "interpretation_discipline": {
            "no_arbitrary_ap_margin": True,
            "best_epoch_is_descriptive_only": True,
            "single_seed_is_not_replication": True,
            "performance_gain_is_not_paired_causality": True,
            "depth_remains_hold": True,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"schema": SCHEMA, "decision": report["decision"], "out": str(out)}, indent=2))

if __name__ == "__main__":
    main()
