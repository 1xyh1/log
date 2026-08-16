#!/usr/bin/env python3
"""F1-B pre-training audit: frozen corruption schedule + G9 logic + references.

Static checks only (torch-free where possible): the frozen schedule values,
SHA256-driven randomness (Python hash() forbidden), epoch-dependent noise,
G9 evidence requirements, F1 v4 frozen status, and the B1 docs.  Dynamic
model gates are re-checked by the runner's G1-G8 and the smoke runs.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.step4_f1_b_corruption import (  # noqa: E402
    KIND_PROBS, NOISE_SIGMA, SEVERITIES, ZERO_SEVERITY, apply_schedule_to_plane,
    sample_schedule, schedule_sha256)

EXPECTED_KIND_PROBS = {"clean": 0.50, "zero": 0.125, "noise": 0.125,
                       "blur": 0.125, "contrast": 0.125}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_schedule_frozen() -> dict:
    checks = {
        "kind_probs_exact": dict(KIND_PROBS) == EXPECTED_KIND_PROBS,
        "severities_exact": SEVERITIES == (0.25, 0.50, 0.75, 1.00),
        "zero_severity_exact": ZERO_SEVERITY == 1.0,
        "noise_sigma_matches_eval": NOISE_SIGMA == 0.20,
        "shift_not_in_training": "shift" not in dict(KIND_PROBS),
    }
    return {"checks": checks, "passed": all(checks.values())}


def audit_sha_driven() -> dict:
    """Randomness must come from SHA256 digest bytes, not Python hash().
    The no-builtin-hash check is AST-based so docstring mentions don't trip."""
    import ast

    corruption_path = ROOT / "src" / "multimodal" / "step4_f1_b_corruption.py"
    corruption_src = corruption_path.read_text(encoding="utf-8")
    tree = ast.parse(corruption_src)
    hash_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "hash":
            hash_calls.append(node.lineno)
    checks = {
        "no_builtin_hash_calls": len(hash_calls) == 0,
        "sha256_digest_used": "hashlib.sha256" in corruption_src
                               and "_digest" in corruption_src,
        "epoch_in_payload": '|{int(epoch)}|' in corruption_src
                            or '|{int(epoch)}' in corruption_src,
        "noise_uses_digest": 'field="noise"' in corruption_src
                             or '"noise"' in corruption_src,
        "numpy_not_python_rng": "np.random.default_rng" in corruption_src,
    }
    diagnostics = {"hash_call_lines": hash_calls}
    # behavioral checks
    s0 = sample_schedule(20260812, 0, "000001")
    s0_again = sample_schedule(20260812, 0, "000001")
    s1 = sample_schedule(20260812, 1, "000001")
    checks["deterministic_same_input"] = s0 == s0_again
    checks["epoch_affects_schedule"] = any(
        sample_schedule(20260812, e, "000001") != s0 for e in range(1, 8))
    checks["schedule_sha_repeatable"] = (
        schedule_sha256(20260812, 0, ["000001"]) ==
        schedule_sha256(20260812, 0, ["000001"]))
    # noise field depends on epoch (behavioral)
    import numpy as np
    plane = np.full((16, 16), 0.5, dtype=np.float32)
    sched = {"sample_id": "000001", "kind": "noise", "severity": 0.5}
    out0 = apply_schedule_to_plane(plane, sched, seed=20260812, epoch=0)
    out1 = apply_schedule_to_plane(plane, sched, seed=20260812, epoch=1)
    checks["noise_epoch_dependent"] = bool((out0 != out1).any())
    return {"checks": checks, "passed": all(checks.values())}


def audit_g9_logic_present() -> dict:
    runner = (ROOT / "scripts" / "run_step4_f1_b.py").read_text(encoding="utf-8")
    checks = {
        "g9_trace_written": "step4_b1_g9_trace.jsonl" in runner,
        "expected_schedule_sha_recorded": "expected_schedule_sha256" in runner,
        "actual_schedule_sha_recorded": "actual_schedule_sha256" in runner,
        "ir_sha_before_after": ("ir_sha_before" in runner
                                and "ir_sha_after" in runner),
        "rgb_depth_labels_bbox_assert": ("rgb_depth_labels_bboxes_unchanged"
                                         in runner),
        "kind_counts": "kind_counts" in runner,
        "c0_noop_semantics": "apply_corruption=(spec[\"aux_mode\"] != \"zero\")" in runner,
        "fail_fast": "G9_CORRUPTION_TRACE_FAIL" in runner,
    }
    return {"checks": checks, "passed": all(checks.values())}


def audit_f1_v4_frozen() -> dict:
    summary_path = ROOT / "runs" / "step4_f1_ir_gate" / "_summary_step4_f1.json"
    checks = {"f1_summary_exists": summary_path.exists()}
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        checks["f1_schema_v4"] = s.get("schema") == "step4-f1-summary-v4"
        checks["f1_verdict_frozen"] = s.get("verdict_frozen") is True
        checks["f1_a0_conclusion"] = s.get("a0_constant_vs_adaptive", {}).get(
            "conclusion") == "CONSTANT ATTENUATION DOMINATES; ADAPTIVITY IS NOT PROVEN"
    return {"checks": checks, "passed": all(checks.values())}


def audit_docs_present() -> dict:
    docs_dir = ROOT / "docs" / "step4_f1_b_corruption"
    checks = {
        "design_freeze": (docs_dir / "DESIGN_FREEZE.md").exists(),
        "execution_guide": (docs_dir / "EXECUTION_GUIDE.md").exists(),
        "feedback_readme": (docs_dir / "feedback" / "README.md").exists(),
    }
    return {"checks": checks, "passed": all(checks.values())}


def main() -> None:
    sections = {
        "schedule_frozen": audit_schedule_frozen(),
        "sha256_driven_randomness": audit_sha_driven(),
        "g9_logic_present": audit_g9_logic_present(),
        "f1_v4_frozen_reference": audit_f1_v4_frozen(),
        "docs_present": audit_docs_present(),
    }
    report = {
        "schema": "step4-f1-b-audit-v1",
        "sections": sections,
        "provenance": {
            "corruption_source_sha256": _sha_file(
                ROOT / "src" / "multimodal" / "step4_f1_b_corruption.py"),
            "runner_source_sha256": _sha_file(
                ROOT / "scripts" / "run_step4_f1_b.py"),
            "audit_source_sha256": _sha_file(Path(__file__)),
            "f1_summary_sha256": _sha_file(
                ROOT / "runs" / "step4_f1_ir_gate" / "_summary_step4_f1.json"),
        },
        "all_passed": all(s["passed"] for s in sections.values()),
    }
    out_dir = ROOT / "reports" / "step4_f1_b_corruption"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "pretrain_audit.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("->", out)
    if not report["all_passed"]:
        raise RuntimeError("B1_PRETRAIN_AUDIT_FAIL")


if __name__ == "__main__":
    main()
