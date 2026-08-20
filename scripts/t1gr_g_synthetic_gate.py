#!/usr/bin/env python3
"""Exercise all frozen schedule and decision branches without private data."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_g_core import (  # noqa: E402
    ARMS, SEEDS, payload_sha256, schedule_summary, summarize_results, validate_design,
)


def read_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("T1GR_G_JSON_OBJECT_REQUIRED")
    return obj


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(obj, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def result_matrix(g0: dict, g1: dict, g2: dict) -> dict:
    values = {"G0-N": g0, "G1-P": g1, "G2-S": g2}
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            value = float(values[arm][seed])
            rows.append({
                "seed": seed,
                "arm": arm,
                "dev_map50_95": value,
                "lofo_map50_95": {f"fold_{i}": value for i in range(5)},
                "run_manifest_sha256": "a" * 64,
                "last_checkpoint_sha256": "b" * 64,
            })
    return {
        "schema": "t1gr-g-per-seed-results-v1",
        "final_holdout_accessed": False,
        "metric": "mAP50-95",
        "checkpoint": "last.pt",
        "max_det": 100,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    design = read_json(Path(args.design))
    validate_design(design)

    synthetic_ids = [f"synthetic_{i:04d}" for i in range(1504)]
    schedules = [schedule_summary(synthetic_ids, seed, 80) for seed in SEEDS]

    g0 = {seed: 0.30 for seed in SEEDS}
    branch_inputs = {
        "PAIRED_TRAINING_GENERALIZATION_SUPPORTED": (
            g0,
            {seed: 0.33 for seed in SEEDS},
            {seed: 0.31 for seed in SEEDS},
        ),
        "GENERIC_TRAINING_BENEFIT_SOURCE_IDENTITY_NOT_ESTABLISHED": (
            g0,
            {SEEDS[0]: 0.32, SEEDS[1]: 0.33, SEEDS[2]: 0.32},
            {SEEDS[0]: 0.33, SEEDS[1]: 0.32, SEEDS[2]: 0.32},
        ),
        "PAIRED_SOURCE_SPECIFICITY_FAILED": (
            g0,
            {seed: 0.32 for seed in SEEDS},
            {seed: 0.33 for seed in SEEDS},
        ),
        "SMALL_SAMPLE_SIGNAL_DID_NOT_TRANSFER": (
            {seed: 0.33 for seed in SEEDS},
            {seed: 0.31 for seed in SEEDS},
            {seed: 0.32 for seed in SEEDS},
        ),
    }
    branches = {}
    for expected, (b0, b1, b2) in branch_inputs.items():
        _, summary = summarize_results(result_matrix(b0, b1, b2))
        branches[expected] = {
            "actual": summary["decision"],
            "passed": summary["decision"] == expected,
        }

    checks = {
        "all_three_seed_schedules_pass": all(row["passed"] for row in schedules),
        "all_schedules_no_self": all(row["self_pair_count"] == 0 for row in schedules),
        "all_schedules_one_donor_use_per_epoch": all(
            row["donor_use_per_epoch_min"] == row["donor_use_per_epoch_max"] == 1
            for row in schedules
        ),
        "all_recipients_have_80_distinct_donors": all(
            row["recipient_distinct_donor_min"]
            == row["recipient_distinct_donor_max"]
            == 80
            for row in schedules
        ),
        "all_frozen_decision_branches_reachable": all(row["passed"] for row in branches.values()),
        "multiseed_training_remains_blocked": design["authority"]["multiseed_training_authorized"] is False,
        "final_holdout_remains_blocked": design["authority"]["final_holdout_open_authorized"] is False,
    }
    report = {
        "schema": "t1gr-g-synthetic-gate-public-v1",
        "synthetic_gate_passed": all(checks.values()),
        "checks": checks,
        "schedule_summaries": schedules,
        "decision_branch_probes": branches,
        "private_data_accessed": False,
        "smoke_training_authorized": False,
        "multiseed_training_authorized": False,
        "final_holdout_open_authorized": False,
    }
    report["payload_sha256"] = payload_sha256(report)
    atomic_json(Path(args.out), report)
    print(json.dumps({
        "status": "PASS" if report["synthetic_gate_passed"] else "FAIL",
        "out": str(Path(args.out).resolve()),
        "multiseed_training_authorized": False,
        "final_holdout_open_authorized": False,
    }, indent=2))
    if not report["synthetic_gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

