#!/usr/bin/env python3
"""CLI for Step-3 run provenance/integrity checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.run_integrity import inspect_step3_run  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--expected-epochs", type=int, default=80)
    p.add_argument("--no-weights", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    report = inspect_step3_run(
        args.run_dir,
        expected_epochs=args.expected_epochs,
        require_weights=not args.no_weights,
    )
    payload = report.to_dict()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    raise SystemExit(0 if report.passed else 2)


if __name__ == "__main__":
    main()
