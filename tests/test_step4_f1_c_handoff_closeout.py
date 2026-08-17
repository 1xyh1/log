"""Regression tests for the F1-C handoff/closeout patch."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_quality_original_dir_is_defined_before_integrity_loop():
    path = ROOT / "scripts" / "eval_step4_f1_c_quality.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assign_lines = []
    loop_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "original_dir":
                    assign_lines.append(node.lineno)
        if isinstance(node, ast.For):
            text = ast.get_source_segment(path.read_text(encoding="utf-8"), node.iter) or ""
            if "original_dir" in text and "fixed_dir" in text:
                loop_lines.append(node.lineno)
    assert assign_lines and loop_lines
    assert min(assign_lines) < min(loop_lines)

def test_runner_has_formal_readiness_hard_gate():
    src = (ROOT / "scripts" / "run_step4_f1_c.py").read_text(encoding="utf-8")
    assert "verify_readiness_report" in src
    assert "F1C_FORMAL_READINESS_FAIL" in src
    assert "FORMAL_PROTOCOL_DRIFT" in src

def test_frozen_quality_protocol_does_not_posthoc_require_beating_fixed_macro():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from multimodal.step4_f1_c_closeout import frozen_promotion_decision

    # Registered axes all pass.  Deliberately let diagnostic macro comparison
    # versus FIXED/ORIGSOFT be worse; it must not become an unregistered hard gate.
    out = frozen_promotion_decision(
        c0=0.30, fixed=0.305, normal=0.32, zero=0.29, shuffle=0.28,
        origsoft=0.31, loo_c0_ok=True, loo_fixed_ok=True, loo_orig_ok=True,
        macro_soft=0.25, macro_qclean=0.24,
        worst4_soft=0.20, worst4_qclean=0.19,
        learned_minus_qclean_pos=9,
    )
    assert out["full_promotion_pass"] is True
    assert out["decision"] == "PROMOTE_F1C_MAGNITUDE_GATE_CONFIRM_ONE_SEED"

def test_readiness_contract_has_three_smokes_four_formals():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from multimodal.step4_f1_c_readiness import (
        APPROVED_FORMAL_GROUPS, SMOKE_SPECS
    )
    assert len(SMOKE_SPECS) == 3
    assert set(APPROVED_FORMAL_GROUPS) == {
        "F1C-C0", "F1C-I-fixed", "F1C-I-magsoft", "F1C-I-soft"
    }
    assert "F1C-I-soft" not in {x["group"] for x in SMOKE_SPECS.values()}
