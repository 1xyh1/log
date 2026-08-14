#!/usr/bin/env python3
"""Diagnose Step-3 training curves without relying on post-hoc detection evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_curve(run_dir: Path) -> dict:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {"status": "MISSING_RESULTS", "run": str(run_dir)}
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        return {"status": "EMPTY_RESULTS", "run": str(run_dir)}

    def vals(key):
        out = []
        for row in rows:
            try:
                out.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                out.append(float("nan"))
        return out

    maps = vals("metrics/mAP50-95(B)")
    # Ultralytics 8.4.56 can emit named losses.  Sum whichever train/*loss columns exist.
    loss_keys = [k for k in rows[0] if k.startswith("train/") and "loss" in k.lower()]
    losses = []
    for row in rows:
        total = 0.0
        valid = False
        for key in loss_keys:
            try:
                total += float(row[key])
                valid = True
            except (TypeError, ValueError):
                pass
        losses.append(total if valid else None)

    best_i = max(range(len(maps)), key=lambda i: maps[i])
    tail = maps[-10:]
    report = {
        "run": str(run_dir),
        "epochs": len(rows),
        "map_epoch1": maps[0],
        "map_last": maps[-1],
        "map_best": maps[best_i],
        "map_best_epoch": best_i + 1,
        "late10_mean": sum(tail) / len(tail),
        "loss_epoch1": losses[0],
        "loss_last": losses[-1],
    }
    if len(rows) < 80:
        report["status"] = "LIKELY_SHORT_OR_OVERWRITTEN_RUN"
    elif maps[-1] < 0.5 * maps[best_i] and maps[best_i] > 0.05:
        report["status"] = "LATE_GENERALIZATION_COLLAPSE_OR_INSTABILITY"
    elif losses[0] is not None and losses[-1] is not None and losses[-1] < losses[0] and maps[-1] > maps[0]:
        report["status"] = "HEALTHY_LEARNING_SIGNAL"
    else:
        report["status"] = "MIXED_NEEDS_REVIEW"
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step3_earlyfusion")
    p.add_argument("--runs", nargs="+", default=["C0-N", "C1-I", "C2-D"])
    p.add_argument("--out", default=None)
    a = p.parse_args()
    project = Path(a.project)
    report = {name: read_curve(project / name) for name in a.runs}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
