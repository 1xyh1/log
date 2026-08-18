from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.step4_a4_content_mask import (  # noqa: E402
    content_rectangle, feature_content_coverage, input_content_mask, sample_meta,
)
from multimodal.step4_a4_dc_ac import (  # noqa: E402
    decompose_all, decompose_content, full_map_ac, full_map_dc,
    validate_component_trace, weighted_ac, weighted_dc,
)
from multimodal.step4_a4_decision import (  # noqa: E402
    FACTORIAL_CELLS, apply_content_diagnostic_veto, classify_ap_effect,
    content_diagnostic_interpretation, factorial_effects, joint_p5_decision,
    linear_contrast, summarize_effect,
)


def _meta_16x9():
    return {
        "ori_shape": (1080, 1920),
        "ratio_pad": ((1/3, 1/3), (0, 140)),
        "input_hw": (640, 640),
    }


def _result(v, loo=None):
    if loo is None:
        loo = {f"s{i}": v for i in range(6)}
    return {
        "full": {"map50_95": float(v)},
        "loo": {k: {"map50_95": float(x)} for k, x in loo.items()},
    }


# ---- DC/AC math -------------------------------------------------------------
def test_full_dc_spatial_constant():
    x = torch.randn(1, 3, 5, 7)
    dc = full_map_dc(x)
    assert dc.shape == x.shape
    assert torch.allclose(dc, dc[..., :1, :1].expand_as(dc))


def test_full_ac_zero_channel_mean():
    x = torch.randn(1, 4, 9, 11)
    ac = full_map_ac(x)
    assert torch.allclose(ac.mean((-2, -1)), torch.zeros(1, 4), atol=1e-6)


def test_full_reconstruction():
    x = torch.randn(1, 2, 4, 6)
    assert torch.allclose(full_map_ac(x) + full_map_dc(x), x, atol=1e-7)


def test_full_dc_requires_bchw():
    with pytest.raises(ValueError):
        full_map_dc(torch.zeros(3, 4, 5))


def test_weighted_dc_simple():
    x = torch.tensor([[[[1., 3.], [10., 20.]]]])
    w = torch.tensor([[[[1., 1.], [0., 0.]]]])
    dc = weighted_dc(x, w)
    assert torch.allclose(dc, torch.full_like(x, 2.0))


def test_weighted_ac_zero_weighted_mean():
    x = torch.randn(1, 3, 4, 6)
    w = torch.rand(1, 1, 4, 6)
    ac = weighted_ac(x, w)
    m = (ac * w).sum((-2, -1)) / w.sum((-2, -1))
    assert torch.allclose(m, torch.zeros_like(m), atol=1e-6)


def test_weighted_dc_zero_coverage_fails():
    with pytest.raises(RuntimeError, match="A4_CONTENT_DC_COVERAGE_FAIL"):
        weighted_dc(torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))


def test_weighted_dc_shape_mismatch_fails():
    with pytest.raises(RuntimeError, match="SHAPE_MISMATCH"):
        weighted_dc(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 3, 3))


def test_decompose_all_trace_sources_same():
    _, tr = decompose_all(torch.randn(1, 2, 4, 4), source_id="donor")
    assert tr["residual_source_id"] == "donor"
    assert tr["mean_source_id"] == "donor"
    assert tr["content_mask_source_id"] is None


def test_decompose_all_validates():
    _, tr = decompose_all(torch.randn(1, 2, 4, 4), source_id="x")
    assert validate_component_trace(tr)


def test_decompose_all_source_contamination_fails_validation():
    _, tr = decompose_all(torch.randn(1, 2, 4, 4), source_id="donor")
    tr["mean_source_id"] = "recipient"
    assert not validate_component_trace(tr)


# ---- content mask -----------------------------------------------------------
def test_content_rectangle_16x9():
    r = content_rectangle((1080, 1920), ((1/3, 1/3), (0, 140)), (640, 640))
    assert r == {"top": 140, "bottom": 500, "left": 0, "right": 640, "height": 360, "width": 640}


def test_input_content_fraction_16x9():
    _, ev = input_content_mask((1080, 1920), ((1/3, 1/3), (0, 140)), (640, 640))
    assert ev["input_content_fraction"] == pytest.approx(360/640)


def test_input_content_mask_binary():
    m, _ = input_content_mask((100, 100), ((2, 2), (20, 10)), (220, 240))
    assert set(torch.unique(m).tolist()) <= {0.0, 1.0}


def test_invalid_content_rectangle_fails():
    with pytest.raises(RuntimeError, match="RECT_INVALID"):
        content_rectangle((100, 100), ((3, 3), (0, 0)), (200, 200))


def test_feature_coverage_shape_p3():
    m = _meta_16x9()
    c, ev = feature_content_coverage(m["ori_shape"], m["ratio_pad"], m["input_hw"], (80, 80))
    assert c.shape == (1, 1, 80, 80)
    assert ev["feature_hw"] == [80, 80]


def test_feature_coverage_shape_p4():
    m = _meta_16x9()
    c, _ = feature_content_coverage(m["ori_shape"], m["ratio_pad"], m["input_hw"], (40, 40))
    assert c.shape == (1, 1, 40, 40)


def test_feature_coverage_shape_p5():
    m = _meta_16x9()
    c, _ = feature_content_coverage(m["ori_shape"], m["ratio_pad"], m["input_hw"], (20, 20))
    assert c.shape == (1, 1, 20, 20)


def test_feature_coverage_fraction_sanity_16x9():
    m = _meta_16x9()
    c, ev = feature_content_coverage(m["ori_shape"], m["ratio_pad"], m["input_hw"], (80, 80))
    assert ev["coverage_mean"] == pytest.approx(360/640, abs=1e-6)
    assert 0 <= c.min() <= c.max() <= 1
    assert bool(((c > 0) & (c < 1)).any())


def test_full_content_coverage_explicit():
    _, ev = feature_content_coverage((640, 640), ((1, 1), (0, 0)), (640, 640), (20, 20))
    assert ev["full_content_coverage"] is True
    assert ev["coverage_mean"] == 1.0


def test_content_mask_source_is_geometry_only():
    _, ev = input_content_mask((100, 100), ((1, 1), (0, 0)), (100, 100))
    assert ev["source"] == "ori_shape+ratio_pad"


def test_sample_meta_ignores_labels():
    s = {"img": torch.zeros(6, 32, 48), "ori_shape": (20, 30), "ratio_pad": ((1, 1), (9, 6)), "bboxes": torch.ones(2, 4)}
    m = sample_meta(s)
    assert set(m) == {"ori_shape", "ratio_pad", "input_hw"}


def test_decompose_content_uses_own_mask_source():
    x = torch.randn(1, 2, 20, 20)
    ac, tr = decompose_content(x, source_id="donor", content_mask_source_id="donor", meta=_meta_16x9())
    assert ac.shape == x.shape
    assert tr["content_mask_source_id"] == "donor"
    assert validate_component_trace(tr)


def test_decompose_content_recipient_mask_contamination_fails():
    x = torch.randn(1, 2, 20, 20)
    _, tr = decompose_content(x, source_id="donor", content_mask_source_id="donor", meta=_meta_16x9())
    tr["content_mask_source_id"] = "recipient"
    assert not validate_component_trace(tr)


def test_decompose_content_reconstruction():
    x = torch.randn(1, 2, 20, 20)
    ac, tr = decompose_content(x, source_id="x", content_mask_source_id="x", meta=_meta_16x9())
    assert tr["reconstruction_max_abs_error"] <= 1e-6
    assert tr["ac_content_weighted_channel_mean_abs_max"] <= 1e-6


# ---- effect labels ----------------------------------------------------------
def test_summarize_effect_counts():
    s = summarize_effect(0.1, {"a": 1, "b": -1, "c": 0})
    assert s["positive_folds"] == 1 and s["negative_folds"] == 1 and s["zero_folds"] == 1


def test_classify_strong_positive():
    p = summarize_effect(.1, {str(i): v for i, v in enumerate([1, 1, 1, 1, -1, 0])})
    r = summarize_effect(.01, {str(i): 0 for i in range(6)})
    assert classify_ap_effect(p, r) == "STRONG_POSITIVE"


def test_classify_strong_negative():
    p = summarize_effect(-.1, {str(i): v for i, v in enumerate([-1, -1, -1, -1, 1, 0])})
    r = summarize_effect(-.01, {str(i): 0 for i in range(6)})
    assert classify_ap_effect(p, r) == "STRONG_NEGATIVE"


def test_cross_system_sign_conflict_inconclusive():
    p = summarize_effect(.1, {str(i): 1 for i in range(6)})
    r = summarize_effect(-.01, {str(i): -1 for i in range(6)})
    assert classify_ap_effect(p, r) == "INCONCLUSIVE"


def test_custom_rescue_labels():
    p = summarize_effect(.1, {str(i): 1 for i in range(6)})
    r = summarize_effect(.01, {str(i): 1 for i in range(6)})
    assert classify_ap_effect(p, r, "STRONG_POSITIVE_RESCUE", "STRONG_NEGATIVE_RESCUE") == "STRONG_POSITIVE_RESCUE"


# ---- factorial --------------------------------------------------------------
def test_factorial_has_exact_eight_cells():
    assert len(FACTORIAL_CELLS) == 8 and set(FACTORIAL_CELLS) == {"C000","C100","C010","C001","C110","C101","C011","C111"}


def test_linear_contrast_simple():
    cells = {"A": _result(.3), "B": _result(.2)}
    c = linear_contrast(cells, {"A": 1, "B": -1})
    assert c["full"] == pytest.approx(.1)


def test_factorial_main_r3():
    cells = {k: _result(0.0) for k in FACTORIAL_CELLS}
    cells["C100"] = _result(.2)
    e = factorial_effects(cells)
    assert e["R3"]["full"] == pytest.approx(.2)


def test_factorial_pair_interaction_i34():
    cells = {k: _result(0.0) for k in FACTORIAL_CELLS}
    cells["C110"] = _result(.3)
    cells["C100"] = _result(.1)
    cells["C010"] = _result(.1)
    assert factorial_effects(cells)["I34"]["full"] == pytest.approx(.1)


def test_factorial_three_way_interaction():
    cells = {k: _result(0.0) for k in FACTORIAL_CELLS}
    cells["C111"] = _result(.5)
    assert factorial_effects(cells)["I345"]["full"] == pytest.approx(.5)


def test_factorial_missing_cell_fails():
    cells = {k: _result(0.0) for k in FACTORIAL_CELLS[:-1]}
    with pytest.raises(RuntimeError, match="A4_FACTORIAL_INCOMPLETE"):
        factorial_effects(cells)


# ---- joint decision ---------------------------------------------------------
def test_joint_go_requires_paired_and_rescue_same_context():
    d = joint_p5_decision(
        {"standalone": "STRONG_POSITIVE", "conditional": "INCONCLUSIVE"},
        {"standalone": "STRONG_POSITIVE_RESCUE", "conditional": "INCONCLUSIVE"},
    )
    assert d["training_go"] is True and d["branch"] == "CENTERING_TRAINING_GO"


def test_paired_positive_without_rescue_no_go():
    d = joint_p5_decision(
        {"standalone": "STRONG_POSITIVE", "conditional": "INCONCLUSIVE"},
        {"standalone": "INCONCLUSIVE", "conditional": "INCONCLUSIVE"},
    )
    assert d["training_go"] is False and d["branch"] == "PAIRED_RESTORED_NO_PERFORMANCE_RESCUE"


def test_rescue_positive_without_paired_no_go():
    d = joint_p5_decision(
        {"standalone": "INCONCLUSIVE", "conditional": "INCONCLUSIVE"},
        {"standalone": "STRONG_POSITIVE_RESCUE", "conditional": "INCONCLUSIVE"},
    )
    assert d["training_go"] is False and d["branch"] == "PERFORMANCE_RESCUE_WITHOUT_PAIRED_RESTORATION"


def test_paired_negative_stops_centering():
    d = joint_p5_decision(
        {"standalone": "STRONG_NEGATIVE", "conditional": "INCONCLUSIVE"},
        {"standalone": "INCONCLUSIVE", "conditional": "INCONCLUSIVE"},
    )
    assert d["branch"] == "STOP_CENTERING_ROUTE" and not d["training_go"]



def test_mixed_paired_context_never_goes():
    d = joint_p5_decision(
        {"standalone": "STRONG_POSITIVE", "conditional": "STRONG_NEGATIVE"},
        {"standalone": "INCONCLUSIVE", "conditional": "INCONCLUSIVE"},
    )
    assert d["branch"] == "MIXED_PAIRED_CONTEXT_NO_GO" and not d["training_go"]


def test_mixed_context_sign_conflict_beats_same_context_go():
    # Reviewer adjudication 2026-08-19 regression: the executed A4 hit exactly
    # this combination (standalone paired-positive + rescue-positive, conditional
    # paired STRONG_NEGATIVE). The old implementation returned CENTERING_TRAINING_GO
    # from the go_contexts check before MIXED_PAIRED_CONTEXT_NO_GO could veto.
    # Corrected precedence: cross-context sign conflict MUST win.
    d = joint_p5_decision(
        {"standalone": "STRONG_POSITIVE", "conditional": "STRONG_NEGATIVE"},
        {"standalone": "STRONG_POSITIVE_RESCUE", "conditional": "STRONG_POSITIVE_RESCUE"},
    )
    assert d["branch"] == "MIXED_PAIRED_CONTEXT_NO_GO"
    assert d["training_go"] is False

def test_inconclusive_joint_no_go():
    d = joint_p5_decision(
        {"standalone": "INCONCLUSIVE", "conditional": "INCONCLUSIVE"},
        {"standalone": "INCONCLUSIVE", "conditional": "INCONCLUSIVE"},
    )
    assert d["branch"] == "A4_DIAGNOSIS_INCONCLUSIVE" and not d["training_go"]


def test_context_mismatch_does_not_go():
    d = joint_p5_decision(
        {"standalone": "STRONG_POSITIVE", "conditional": "INCONCLUSIVE"},
        {"standalone": "INCONCLUSIVE", "conditional": "STRONG_POSITIVE_RESCUE"},
    )
    assert not d["training_go"]


# ---- diagnostic interpretation ---------------------------------------------

def test_content_diagnostic_can_veto_but_not_create_go():
    primary = {"branch": "CENTERING_TRAINING_GO", "training_go": True, "contexts": ["conditional"]}
    blocked = apply_content_diagnostic_veto(primary, {"conditional": "INCONCLUSIVE"})
    assert blocked["branch"] == "PADDING_GLOBAL_STATISTICS_AUDIT_BEFORE_TRAINING"
    assert blocked["training_go"] is False and blocked["content_diagnostic_veto_applied"] is True


def test_content_diagnostic_support_keeps_primary_go():
    primary = {"branch": "CENTERING_TRAINING_GO", "training_go": True, "contexts": ["conditional"]}
    kept = apply_content_diagnostic_veto(primary, {"conditional": "STRONG_POSITIVE_RESCUE"})
    assert kept["training_go"] is True and kept["content_diagnostic_veto_applied"] is False


def test_content_diagnostic_never_creates_go():
    primary = {"branch": "A4_DIAGNOSIS_INCONCLUSIVE", "training_go": False, "contexts": []}
    out = apply_content_diagnostic_veto(primary, {"conditional": "STRONG_POSITIVE_RESCUE"})
    assert out["training_go"] is False

def test_all_rescue_content_not_marks_padding_possible():
    assert content_diagnostic_interpretation("STRONG_POSITIVE_RESCUE", "INCONCLUSIVE") == "PADDING_OR_GLOBAL_STATISTICS_MAY_CONTRIBUTE"


def test_all_and_content_rescue_supports_content_or_global_dc():
    assert content_diagnostic_interpretation("STRONG_POSITIVE_RESCUE", "STRONG_POSITIVE_RESCUE") == "CONTENT_OR_GLOBAL_POSTPROJECTION_DC_SUPPORTED"


def test_content_only_rescue_diagnostic():
    assert content_diagnostic_interpretation("INCONCLUSIVE", "STRONG_POSITIVE_RESCUE") == "CONTENT_SPECIFIC_DC_MAY_BE_MASKED_BY_FULL_MAP_MEAN"


def test_no_rescue_diagnostic():
    assert content_diagnostic_interpretation("INCONCLUSIVE", "INCONCLUSIVE") == "NO_POSITIVE_CONTENT_DIAGNOSTIC_RESCUE"


# ---- source-contract tests --------------------------------------------------
def test_evaluator_has_native_vs_donor_endpoint():
    t = (ROOT / "scripts/eval_step4_a4.py").read_text(encoding="utf-8")
    assert "paired_effect_native_minus_donor" in t


def test_evaluator_has_separate_centering_rescue():
    t = (ROOT / "scripts/eval_step4_a4.py").read_text(encoding="utf-8")
    assert "centering_rescue_native_ac_minus_full_native" in t


def test_evaluator_pins_a3_summary_both_hashes():
    t = (ROOT / "scripts/eval_step4_a4.py").read_text(encoding="utf-8")
    assert "121dacc0ed50" in t and "3523cb526d7a" in t


def test_evaluator_uses_a3_post_projection_forward():
    t = (ROOT / "scripts/eval_step4_a4.py").read_text(encoding="utf-8")
    assert "forward_with_custom_residuals" in t


def test_evaluator_donor_cache_no_gate():
    t = (ROOT / "scripts/eval_step4_a4.py").read_text(encoding="utf-8")
    assert "build_residual_cache_no_gate" in t and "DONOR_CACHE_GATE" in t


def test_evaluator_c000_bitwise_gate():
    t = (ROOT / "scripts/eval_step4_a4.py").read_text(encoding="utf-8")
    assert "C000_NATIVE" in t and "detection_sha256" in t
    assert "C000_A2_M111_FULL" in t and "C000_A2_M111_LOO" in t


def test_evaluator_content_cannot_drive_joint_decision():
    t = (ROOT / "scripts/eval_step4_a4.py").read_text(encoding="utf-8")
    assert 'joint_p5_decision(paired_labels["P5"], rescue_labels["P5"])' in t
    assert '"training_go_allowed": False' in t


def test_design_forbids_projection_bias_claim():
    t = (ROOT / "docs/step4_a4/DESIGN_FREEZE.md").read_text(encoding="utf-8")
    assert 'NO calling DC harm "projection bias parameter harm"' in t


def test_design_primary_secondary_control_frozen():
    t = (ROOT / "docs/step4_a4/DESIGN_FREEZE.md").read_text(encoding="utf-8")
    assert "PRIMARY:\n  P5" in t and "SECONDARY:\n  P3" in t and "CONTROL:\n  P4" in t


def test_audit_requires_dynamic_g1_g14():
    t = (ROOT / "scripts/audit_step4_a4.py").read_text(encoding="utf-8")
    assert "all_g1_g14_present" in t and "g14_dynamic" in t
