#!/usr/bin/env python3
"""Produce the pre-registered DEV-only T1-GR cross-seed decision."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_g_core import summarize_results, validate_design  # noqa: E402


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
    ap.add_argument("--results", required=True)
    ap.add_argument("--cross-seed-out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()
    validate_design(read_json(Path(args.design)))
    cross, summary = summarize_results(read_json(Path(args.results)))
    atomic_json(Path(args.cross_seed_out), cross)
    atomic_json(Path(args.summary_out), summary)
    print(json.dumps({
        "status": "PASS",
        "decision": summary["decision"],
        "final_holdout_open_authorized": False,
        "cross_seed_out": str(Path(args.cross_seed_out).resolve()),
        "summary_out": str(Path(args.summary_out).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

