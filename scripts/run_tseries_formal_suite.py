#!/usr/bin/env python3
"""Sequentially run the three formal 80-epoch T-series arms after G1-G18 audit."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.tseries_core import RUN_NAMES  # noqa: E402

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_tseries")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--data", default="D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml")
    p.add_argument("--base-checkpoint", default="E:/odin/yolo26s.pt")
    p.add_argument("--audit", default="reports/step4_tseries/pretraining_audit.json")
    p.add_argument("--device", default="0")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--with-posttrain-eval", action="store_true")
    a = p.parse_args()

    audit_path = ROOT / a.audit
    if not audit_path.exists():
        raise RuntimeError(f"T_SERIES_FORMAL_AUDIT_MISSING:{audit_path}")
    audit = load(audit_path)
    if (
        audit.get("schema") != "step4-tseries-pretraining-audit-v1"
        or audit.get("phase") != "formal"
        or audit.get("all_passed") is not True
    ):
        raise RuntimeError("T_SERIES_FORMAL_AUDIT_NOT_PASSING")
    gates = audit.get("gates") or {}
    if set(gates) != {f"G{i}" for i in range(1, 19)} or not all(gates.values()):
        raise RuntimeError(f"T_SERIES_FORMAL_GATES_NOT_ALL_PASS:{gates}")

    project = ROOT / a.project
    existing = [str(project / RUN_NAMES[t]) for t in ("T0-N", "T1-F", "T2-A")
                if (project / RUN_NAMES[t]).exists()]
    if existing:
        raise RuntimeError(f"T_SERIES_FORMAL_RUN_EXISTS:{existing}")

    completed = []
    for treatment in ("T0-N", "T1-F", "T2-A"):
        cmd = [
            a.python, str(ROOT / "scripts/run_tseries.py"),
            "--treatment", treatment,
            "--run-kind", "formal",
            "--project", str(project),
            "--contract", str(a.contract),
            "--data", str(a.data),
            "--base-checkpoint", str(a.base_checkpoint),
            "--formal-audit", str(a.audit),
            "--device", str(a.device),
            "--seed", "20260812",
            "--epochs", "80",
            "--batch", "4",
        ]
        print("FORMAL", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        completed.append(treatment)

    result = {"formal_runs_complete": completed}
    if a.with_posttrain_eval:
        for script in ("eval_tseries_posttrain.py", "eval_tseries_paired.py", "summarize_tseries.py"):
            cmd = [a.python, str(ROOT / "scripts" / script)]
            if script != "summarize_tseries.py":
                cmd += ["--project", str(a.project), "--contract", str(a.contract), "--device", str(a.device)]
            print("POST", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
        result["posttrain_eval_complete"] = True

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
