#!/usr/bin/env python3
"""Run or resume the frozen authorized nine-run 80-epoch formal suite."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_g_suite import run_suite  # noqa: E402
from multimodal.t1gr_secure_io import safe_error_message  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--suite-state", required=True)
    args = parser.parse_args()
    try:
        result = run_suite(
            repo=ROOT,
            mode="formal",
            design_path=ROOT / "config/t1gr_g_design.frozen.json",
            preflight_path=ROOT / "reports/step4_t1gr/t1gr_g_implementation_preflight_public.json",
            view_manifest=Path(args.view_manifest),
            base_checkpoint=Path(args.base_checkpoint),
            run_root=Path(args.run_root),
            suite_state_path=Path(args.suite_state),
            smoke_audit_path=ROOT / "reports/step4_t1gr/t1gr_g_smoke_audit_public.json",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": safe_error_message(exc)}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
