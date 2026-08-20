#!/usr/bin/env python3
"""Dependency-light regression gate for the E5 v2 bundle."""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "tests" / "test_t1gr_e5_hardened.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("t1gr_e5_v2_tests", TEST_FILE)
    if spec is None or spec.loader is None:
        print(json.dumps({"status": "FAIL", "error": "TEST_MODULE_LOAD_FAIL"}))
        return 2
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tests = [
        (name, fn) for name, fn in vars(module).items()
        if name.startswith("test_") and callable(fn) and len(inspect.signature(fn).parameters) == 0
    ]
    failures = []
    for name, fn in sorted(tests):
        try:
            fn()
        except BaseException as exc:
            failures.append({
                "test": name,
                "exception_type": type(exc).__name__,
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            })
    result = {
        "schema": "t1gr-e5-v2-regression-gate-v1",
        "status": "PASS" if not failures else "FAIL",
        "passed": len(tests) - len(failures),
        "total": len(tests),
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

