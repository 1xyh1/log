#!/usr/bin/env python3
"""Post-run integrity verifier for the single U2-S formal arm."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1tr_training_source import (  # noqa: E402
    FORMAL_EPOCHS, U2_RUN_NAME, schedule_balance,
)

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def fail(msg):
    raise SystemExit(f"INVALID:{msg}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="runs/step4_t1tr")
    a = ap.parse_args()

    run = ROOT / a.project / U2_RUN_NAME
    required = [
        run / "manifest.json",
        run / "optimizer_manifest.json",
        run / "t1tr_source_schedule.jsonl",
        run / "t1tr_mechanism.jsonl",
        run / "results.csv",
        run / "weights/last.pt",
    ]
    for p in required:
        if not p.is_file() or p.stat().st_size == 0:
            fail(f"missing_or_empty:{p}")

    m = load(run / "manifest.json")
    if m.get("schema") != "step4-t1tr-u2-run-manifest-v1":
        fail("manifest_schema")
    if m.get("run_kind") != "formal":
        fail("run_kind")
    if m.get("arm") != "U2-S":
        fail("arm")
    if m.get("completed_epochs") != FORMAL_EPOCHS:
        fail("completed_epochs")
    if m.get("optimizer_exact_t1") is not True:
        fail("optimizer_exact_t1")
    if m.get("runtime_all_epochs_no_self") is not True:
        fail("runtime_all_epochs_no_self")
    if m.get("runtime_schedule_exact") is not True:
        fail("runtime_schedule_exact")

    rows = [
        json.loads(x) for x in (run / "t1tr_source_schedule.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if x.strip()
    ]
    if len(rows) != FORMAL_EPOCHS:
        fail(f"schedule_rows:{len(rows)}")
    shifts = Counter()
    for i, r in enumerate(rows):
        if r.get("epoch") != i:
            fail(f"epoch_index:{i}")
        expected_shift = 1 + (i % 10)
        if r.get("shift") != expected_shift:
            fail(f"shift:{i}:{r.get('shift')}")
        if r.get("self_matches") != 0:
            fail(f"self_match:{i}")
        if r.get("n_samples") != 11:
            fail(f"n_samples:{i}")
        if r.get("actual_mapping_matches_schedule") is not True:
            fail(f"mapping:{i}")
        shifts[expected_shift] += 1
    if set(shifts) != set(range(1, 11)) or any(shifts[s] != 8 for s in shifts):
        fail(f"shift_balance:{dict(shifts)}")

    with (run / "results.csv").open("r", encoding="utf-8", newline="") as f:
        result_rows = list(csv.DictReader(f))
    if len(result_rows) != FORMAL_EPOCHS:
        fail(f"results_rows:{len(result_rows)}")

    mech_rows = [
        x for x in (run / "t1tr_mechanism.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if x.strip()
    ]
    if len(mech_rows) != FORMAL_EPOCHS:
        fail(f"mechanism_rows:{len(mech_rows)}")

    bal = m.get("schedule_balance_80ep") or {}
    if bal.get("passed") is not True or bal.get("expected_each_nonself_pair") != 8:
        fail("manifest_schedule_balance")

    print(json.dumps({
        "status": "T1TR_U2_FORMAL_RUN_PASS",
        "run_dir": str(run),
        "epochs": FORMAL_EPOCHS,
        "results_rows": len(result_rows),
        "schedule_rows": len(rows),
        "mechanism_rows": len(mech_rows),
        "shift_counts": dict(sorted(shifts.items())),
        "last_pt_bytes": (run / "weights/last.pt").stat().st_size,
    }, indent=2))

if __name__ == "__main__":
    main()
