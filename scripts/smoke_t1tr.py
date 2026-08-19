#!/usr/bin/env python3
"""Run one real epoch of U2-S and materialize dynamic pretraining evidence."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.t1tr_training_source import sha256_file  # noqa: E402

SCHEMA = "step4-t1tr-pretraining-smoke-v1"

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default=OUT_DEFAULT)
    ap.add_argument("--data", default="D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml")
    ap.add_argument("--base-checkpoint", default="E:/odin/yolo26s.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/step4_t1tr_smoke")
    ap.add_argument("--out", default="reports/step4_t1tr/pretraining_smoke.json")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    project = (ROOT / a.project).resolve()
    project.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in project.iterdir() if p.is_dir()}

    cmd = [
        a.python, str(ROOT / "scripts/run_t1tr.py"),
        "--run-kind", "smoke",
        "--epochs", "1",
        "--batch", "4",
        "--seed", "20260812",
        "--project", str(project),
        "--contract", str(a.contract),
        "--data", str(a.data),
        "--base-checkpoint", str(a.base_checkpoint),
        "--device", str(a.device),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)

    created = [p for p in project.iterdir() if p.is_dir() and p.name not in before]
    if len(created) != 1:
        raise RuntimeError(f"T1TR_SMOKE_RUN_DISCOVERY_FAIL:{created}")
    run = created[0]
    manifest = load(run / "manifest.json")
    rows = [
        json.loads(x) for x in (run / "t1tr_source_schedule.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if x.strip()
    ]
    if len(rows) != 1:
        raise RuntimeError("T1TR_SMOKE_EXPECTED_ONE_EPOCH")

    gates = {
        "initial_identity_exact_t1": manifest.get("initial_identity_exact_t1") is True,
        "optimizer_exact_t1": manifest.get("optimizer_exact_t1") is True,
        "completed_one_epoch": manifest.get("completed_epochs") == 1,
        "no_self_match": rows[0].get("self_matches") == 0,
        "actual_mapping_matches_schedule": rows[0].get("actual_mapping_matches_schedule") is True,
        "n_train_samples": rows[0].get("n_samples") == 11,
        "shift_epoch0_is_1": rows[0].get("shift") == 1,
    }
    report = {
        "schema": SCHEMA,
        "all_dynamic_gates_passed": all(gates.values()),
        "gates": gates,
        "run_dir": str(run.relative_to(ROOT)),
        "manifest_sha256": sha256_file(run / "manifest.json"),
        "schedule_sha256": sha256_file(run / "t1tr_source_schedule.jsonl"),
        "source_hashes": {
            "runner": sha256_file(ROOT / "scripts/run_t1tr.py"),
            "core": sha256_file(ROOT / "src/multimodal/t1tr_training_source.py"),
            "design": sha256_file(ROOT / "docs/step4_t1tr/DESIGN_FREEZE.md"),
        },
    }
    if not report["all_dynamic_gates_passed"]:
        raise RuntimeError(f"T1TR_SMOKE_FAIL:{gates}")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise RuntimeError(f"T1TR_REFUSE_OVERWRITE:{out}")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "T1TR_SMOKE_PASS", "out": str(out)}, indent=2))

if __name__ == "__main__":
    main()
