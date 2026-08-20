#!/usr/bin/env python3
"""Generate the frozen nine-run order without starting a training process."""
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
    SCHEMA_DESIGN_AUDIT, SCHEMA_RUN_PLAN, payload_ok, payload_sha256, validate_design,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--design-audit", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    design = read_json(Path(args.design))
    audit = read_json(Path(args.design_audit))
    validate_design(design)
    if (
        audit.get("schema") != SCHEMA_DESIGN_AUDIT
        or not payload_ok(audit)
        or audit.get("design_freeze_passed") is not True
        or audit.get("implementation_entry_authorized") is not True
        or audit.get("multiseed_training_authorized") is not False
        or audit.get("final_holdout_open_authorized") is not False
    ):
        raise RuntimeError("T1GR_G_DESIGN_AUDIT_NOT_PASSING")

    rows = []
    position = 0
    for seed_row in design["launch_order"]:
        seed = int(seed_row["seed"])
        for arm in seed_row["arms"]:
            position += 1
            rows.append({
                "position": position,
                "seed": seed,
                "arm": arm,
                "run_name": f"{arm}_seed{seed}",
                "expected_epochs": 80,
                "primary_checkpoint": "last.pt",
                "execution_status": "BLOCKED_PENDING_IMPLEMENTATION_SMOKE_AUDIT",
            })
    report = {
        "schema": SCHEMA_RUN_PLAN,
        "experiment": "T1-GR",
        "rows": rows,
        "n_runs": len(rows),
        "order_frozen": True,
        "adaptive_reordering_forbidden": True,
        "selective_rerun_forbidden": True,
        "implementation_entry_authorized": True,
        "smoke_training_authorized": False,
        "multiseed_training_authorized": False,
        "final_holdout_open_authorized": False,
        "next_action": "implementation and matched-arm smoke audit",
    }
    report["payload_sha256"] = payload_sha256(report)
    atomic_json(Path(args.out), report)
    print(json.dumps({
        "status": "PASS",
        "n_runs": len(rows),
        "execution_authorized": False,
        "out": str(Path(args.out).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

