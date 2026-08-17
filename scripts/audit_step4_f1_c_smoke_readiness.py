#!/usr/bin/env python3
"""Build the formal F1-C smoke-readiness report from raw smoke artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.step4_f1_c_readiness import (  # noqa: E402
    APPROVED_FORMAL_GROUPS,
    READINESS_SCHEMA,
    evaluate_smoke_readiness,
)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_f1_c")
    p.add_argument("--c0-smoke", required=True)
    p.add_argument("--fixed-smoke", required=True)
    p.add_argument("--magsoft-smoke", required=True)
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument(
        "--out", default="reports/step4_f1_c/smoke_readiness.json"
    )
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    project = (ROOT / a.project).resolve()
    contract_path = Path(a.contract)
    if not contract_path.is_absolute():
        contract_path = (ROOT / contract_path).resolve()
    out = Path(a.out)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    if out.exists() and not a.overwrite:
        raise RuntimeError(f"REFUSE_OVERWRITE_F1C_READINESS: {out}")

    smoke_runs = {
        "C0": project / a.c0_smoke,
        "FIXED": project / a.fixed_smoke,
        "MAGSOFT": project / a.magsoft_smoke,
    }
    result = evaluate_smoke_readiness(ROOT, smoke_runs, contract_path)
    report = {
        "schema": READINESS_SCHEMA,
        "smoke_runs": {
            tag: path.resolve().relative_to(ROOT.resolve()).as_posix()
            for tag, path in smoke_runs.items()
        },
        "approved_formal_groups": list(APPROVED_FORMAL_GROUPS),
        "original_soft_smoke_required": False,
        "evidence": result["evidence"],
        "evidence_sha256": result["evidence_sha256"],
        "provenance": result["provenance"],
        "errors": result["errors"],
        "all_passed": result["all_passed"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("->", out)
    if not report["all_passed"]:
        raise RuntimeError("F1C_SMOKE_READINESS_FAIL")

if __name__ == "__main__":
    main()
