from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Bundle-only test harness: A5 reuses this already-frozen A4 helper in the real repo.
# Stub exactly the AC_ALL contract so pure A5 logic can be regression-tested here.
stub = types.ModuleType("multimodal.step4_a4_dc_ac")


def _sha(t):
    import hashlib
    x = t.detach().cpu().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def _decompose_all(residual, *, source_id):
    dc = residual.mean((-2, -1), keepdim=True).expand_as(residual)
    ac = residual - dc
    ev = {
        "mode": "AC_ALL",
        "residual_source_id": str(source_id),
        "mean_source_id": str(source_id),
        "content_mask_source_id": None,
        "source_residual_sha256": _sha(residual),
        "dc_sha256": _sha(dc),
        "ac_sha256": _sha(ac),
        "reconstruction_max_abs_error": float((ac + dc - residual).abs().max()),
        "ac_full_map_channel_mean_abs_max": float(ac.mean((-2, -1)).abs().max()),
        "dc_spatial_variation_abs_max": float((dc - dc[..., :1, :1]).abs().max()),
        "definition": "residual - mean_HW(residual)",
    }
    return ac, ev


def _validate_component_trace(tr, *, tol=1e-6):
    return (
        tr.get("mode") == "AC_ALL"
        and tr.get("residual_source_id") == tr.get("mean_source_id")
        and float(tr.get("reconstruction_max_abs_error", 1)) <= tol
        and float(tr.get("ac_full_map_channel_mean_abs_max", 1)) <= tol
        and float(tr.get("dc_spatial_variation_abs_max", 1)) <= tol
    )


stub.decompose_all = _decompose_all
stub.validate_component_trace = _validate_component_trace
sys.modules["multimodal.step4_a4_dc_ac"] = stub

from multimodal.step4_a5_context import (  # noqa: E402
    CONTEXT_ORDER,
    CONTEXT_STATES,
    active_scales_for_context,
    build_context,
    context_states,
    validate_context_id,
    validate_context_trace,
    validate_pair_isolation,
)
from multimodal.step4_a5_effects import (  # noqa: E402
    INTERACTION_COEFFICIENTS,
    classify_paired_effect,
    classify_shift,
    context_shifts,
    difference_of_effects,
    effect_from_results,
    interaction_effects,
    linear_effect_contrast,
    mechanism_flags,
    route_decision,
    summarize_effect,
)


def _cache():
    torch.manual_seed(7)
    return {
        sid: {
            "P3": torch.randn(1, 4, 8, 8),
            "P4": torch.randn(1, 5, 4, 4),
            "P5": torch.randn(1, 6, 2, 2),
        }
        for sid in ("r", "d", "x")
    }


def _result(v, loo=None):
    if loo is None:
        loo = {f"s{i}": v for i in range(6)}
    return {
        "full": {"map50": float(v), "map50_95": float(v), "n_images": 6},
        "loo": {sid: {"map50": float(x), "map50_95": float(x), "n_images": 5} for sid, x in loo.items()},
    }


def _effect(v, loo=None):
    if loo is None:
        loo = {f"s{i}": v for i in range(6)}
    return summarize_effect(v, loo)


def _trace_from_build(built, recipient="r", donor="d", q=None):
    if q is None:
        q = [0.5]
    sources = {"P3": recipient, "P4": recipient, "P5": recipient}
    sources.update(built.source_ids)
    alpha = {"P3": [0.0], "P4": [0.0], "P5": [0.0]}
    for s in built.active_scales:
        alpha[s] = list(q)
    return {
        "recipient_id": recipient,
        "condition": f"A5_{built.context_id}_P5_{built.p5_role.upper()}",
        "q_native": list(q),
        "active_scales": sorted(built.active_scales),
        "residual_source_ids": sources,
        "alpha": alpha,
        "feature_strides": {"P3": 8, "P4": 16, "P5": 32},
        "a5_context": built.context_id,
        "a5_context_states": dict(built.states),
        "a5_p5_role": built.p5_role,
        "a5_p5_source_id": built.p5_source_id,
        "a5_component_trace": built.component_trace,
    }


# ---- exact context matrix ---------------------------------------------------
def test_context_order_exact():
    assert CONTEXT_ORDER == ("OO", "FO", "OF", "FF", "AO", "OA", "AF", "FA", "AA")


@pytest.mark.parametrize("ctx,expected", [
    ("OO", {"P3": "O", "P4": "O"}),
    ("FO", {"P3": "F", "P4": "O"}),
    ("OF", {"P3": "O", "P4": "F"}),
    ("FF", {"P3": "F", "P4": "F"}),
    ("AO", {"P3": "A", "P4": "O"}),
    ("OA", {"P3": "O", "P4": "A"}),
    ("AF", {"P3": "A", "P4": "F"}),
    ("FA", {"P3": "F", "P4": "A"}),
    ("AA", {"P3": "A", "P4": "A"}),
])
def test_context_states_exact(ctx, expected):
    assert context_states(ctx) == expected


@pytest.mark.parametrize("ctx,expected", [
    ("OO", {"P5"}),
    ("FO", {"P3", "P5"}),
    ("OF", {"P4", "P5"}),
    ("FF", {"P3", "P4", "P5"}),
    ("AO", {"P3", "P5"}),
    ("OA", {"P4", "P5"}),
    ("AF", {"P3", "P4", "P5"}),
    ("FA", {"P3", "P4", "P5"}),
    ("AA", {"P3", "P4", "P5"}),
])
def test_active_scales_exact(ctx, expected):
    assert set(active_scales_for_context(ctx)) == expected


def test_bad_context_fails():
    with pytest.raises(ValueError, match="A5_UNKNOWN_CONTEXT"):
        validate_context_id("ZZ")


# ---- context construction --------------------------------------------------
@pytest.mark.parametrize("ctx,expected_components", [
    ("OO", {"P5"}),
    ("FO", {"P5"}),
    ("OF", {"P5"}),
    ("FF", {"P5"}),
    ("AO", {"P3", "P5"}),
    ("OA", {"P4", "P5"}),
    ("AF", {"P3", "P5"}),
    ("FA", {"P4", "P5"}),
    ("AA", {"P3", "P4", "P5"}),
])
def test_build_context_component_set(ctx, expected_components):
    b = build_context(_cache(), recipient_id="r", donor_id="d", context_id=ctx, p5_role="native")
    assert set(b.component_trace) == expected_components


def test_native_p5_source_is_recipient():
    b = build_context(_cache(), recipient_id="r", donor_id="d", context_id="OO", p5_role="native")
    assert b.p5_source_id == "r"
    assert b.source_ids["P5"] == "AC_ALL[r]"
    assert b.component_trace["P5"]["mean_source_id"] == "r"


def test_donor_p5_source_and_mean_are_donor():
    b = build_context(_cache(), recipient_id="r", donor_id="d", context_id="OO", p5_role="donor")
    assert b.p5_source_id == "d"
    assert b.source_ids["P5"] == "AC_ALL[d]"
    assert b.component_trace["P5"]["residual_source_id"] == "d"
    assert b.component_trace["P5"]["mean_source_id"] == "d"


def test_p3_ac_is_always_recipient_owned():
    b = build_context(_cache(), recipient_id="r", donor_id="d", context_id="AF", p5_role="donor")
    assert b.component_trace["P3"]["residual_source_id"] == "r"
    assert b.component_trace["P3"]["mean_source_id"] == "r"
    assert b.source_ids["P3"] == "AC_ALL[r]"


def test_p4_ac_is_always_recipient_owned():
    b = build_context(_cache(), recipient_id="r", donor_id="d", context_id="FA", p5_role="donor")
    assert b.component_trace["P4"]["residual_source_id"] == "r"
    assert b.component_trace["P4"]["mean_source_id"] == "r"


def test_bad_role_fails():
    with pytest.raises(ValueError, match="A5_BAD_P5_ROLE"):
        build_context(_cache(), recipient_id="r", donor_id="d", context_id="OO", p5_role="shuffle")


def test_self_donor_fails():
    with pytest.raises(RuntimeError, match="A5_DONOR_SELF_MATCH"):
        build_context(_cache(), recipient_id="r", donor_id="r", context_id="OO", p5_role="donor")


def test_missing_cache_fails():
    with pytest.raises(RuntimeError, match="A5_RESIDUAL_CACHE_MISSING"):
        build_context(_cache(), recipient_id="missing", donor_id="d", context_id="OO", p5_role="native")


# ---- runtime trace semantics -----------------------------------------------
@pytest.mark.parametrize("ctx", CONTEXT_ORDER)
def test_validate_native_context_trace(ctx):
    b = build_context(_cache(), recipient_id="r", donor_id="d", context_id=ctx, p5_role="native")
    tr = _trace_from_build(b)
    assert validate_context_trace(tr, recipient_id="r", donor_id="d", context_id=ctx, p5_role="native")["passed"]


@pytest.mark.parametrize("ctx", CONTEXT_ORDER)
def test_validate_donor_context_trace(ctx):
    b = build_context(_cache(), recipient_id="r", donor_id="d", context_id=ctx, p5_role="donor")
    tr = _trace_from_build(b)
    assert validate_context_trace(tr, recipient_id="r", donor_id="d", context_id=ctx, p5_role="donor")["passed"]


def test_donor_p3_contamination_fails_trace():
    b = build_context(_cache(), recipient_id="r", donor_id="d", context_id="FO", p5_role="donor")
    tr = _trace_from_build(b)
    tr["residual_source_ids"]["P3"] = "d"
    assert not validate_context_trace(tr, recipient_id="r", donor_id="d", context_id="FO", p5_role="donor")["passed"]


def test_donor_p4_contamination_fails_trace():
    b = build_context(_cache(), recipient_id="r", donor_id="d", context_id="OF", p5_role="donor")
    tr = _trace_from_build(b)
    tr["residual_source_ids"]["P4"] = "d"
    assert not validate_context_trace(tr, recipient_id="r", donor_id="d", context_id="OF", p5_role="donor")["passed"]


def test_p5_donor_mean_recipient_contamination_fails():
    b = build_context(_cache(), recipient_id="r", donor_id="d", context_id="FF", p5_role="donor")
    tr = _trace_from_build(b)
    tr["a5_component_trace"]["P5"]["mean_source_id"] = "r"
    assert not validate_context_trace(tr, recipient_id="r", donor_id="d", context_id="FF", p5_role="donor")["passed"]


# ---- pair isolation ---------------------------------------------------------
def _native_donor_pair(ctx="AF"):
    cache = _cache()
    nb = build_context(cache, recipient_id="r", donor_id="d", context_id=ctx, p5_role="native")
    db = build_context(cache, recipient_id="r", donor_id="d", context_id=ctx, p5_role="donor")
    return _trace_from_build(nb), _trace_from_build(db)


def test_pair_isolation_passes():
    n, d = _native_donor_pair("AF")
    assert validate_pair_isolation(n, d)["passed"]


def test_pair_q_drift_fails_isolation():
    n, d = _native_donor_pair()
    d["q_native"] = [0.4]
    assert not validate_pair_isolation(n, d)["passed"]


def test_pair_alpha_drift_fails_isolation():
    n, d = _native_donor_pair()
    d["alpha"]["P3"] = [0.4]
    assert not validate_pair_isolation(n, d)["passed"]


def test_pair_p3_source_drift_fails_isolation():
    n, d = _native_donor_pair("FO")
    d["residual_source_ids"]["P3"] = "d"
    assert not validate_pair_isolation(n, d)["passed"]


def test_pair_p5_must_change_identity():
    n, d = _native_donor_pair()
    d["a5_p5_source_id"] = n["a5_p5_source_id"]
    assert not validate_pair_isolation(n, d)["passed"]


# ---- effect math ------------------------------------------------------------
def test_summarize_effect_counts():
    x = summarize_effect(.1, {"a": 1, "b": -1, "c": 0})
    assert (x["positive_folds"], x["negative_folds"], x["zero_folds"]) == (1, 1, 1)


def test_effect_from_results_native_minus_donor():
    e = effect_from_results(_result(.3), _result(.2))
    assert e["full"] == pytest.approx(.1)
    assert e["loo_median"] == pytest.approx(.1)


def test_effect_from_results_loo_id_mismatch_fails():
    with pytest.raises(RuntimeError, match="LOO_ID_SET"):
        effect_from_results(_result(.3, {"a": .3}), _result(.2, {"b": .2}))


def test_classify_paired_positive():
    assert classify_paired_effect(_effect(.1), _effect(.01)) == "STRONG_POSITIVE"


def test_classify_paired_negative():
    assert classify_paired_effect(_effect(-.1), _effect(-.01)) == "STRONG_NEGATIVE"


def test_classify_paired_cross_system_conflict_inconclusive():
    assert classify_paired_effect(_effect(.1), _effect(-.01)) == "INCONCLUSIVE"


def test_classify_shift_antagonistic():
    assert classify_shift(_effect(-.1), _effect(-.01)) == "STRONG_ANTAGONISTIC_SHIFT"


def test_classify_shift_rescuing():
    assert classify_shift(_effect(.1), _effect(.01)) == "STRONG_RESCUING_SHIFT"


def test_difference_effects():
    d = difference_of_effects(_effect(.3), _effect(.1))
    assert d["full"] == pytest.approx(.2)


# ---- interaction formulas --------------------------------------------------
def _all_effects(base=0.0):
    return {c: _effect(base) for c in CONTEXT_ORDER}


@pytest.mark.parametrize("name,context,expected", [
    ("D3F", "FO", 0.2),
    ("D4F", "OF", 0.2),
    ("D3A", "AO", 0.2),
    ("D4A", "OA", 0.2),
])
def test_main_contrast_formula(name, context, expected):
    e = _all_effects()
    e[context] = _effect(.2)
    assert interaction_effects(e)[name]["full"] == pytest.approx(expected)


def test_iff_formula():
    e = _all_effects()
    e["FF"] = _effect(.5)
    e["FO"] = _effect(.1)
    e["OF"] = _effect(.2)
    assert interaction_effects(e)["IFF"]["full"] == pytest.approx(.2)


def test_iaa_formula():
    e = _all_effects()
    e["AA"] = _effect(.5)
    e["AO"] = _effect(.1)
    e["OA"] = _effect(.2)
    assert interaction_effects(e)["IAA"]["full"] == pytest.approx(.2)


def test_iaf_formula():
    e = _all_effects()
    e["AF"] = _effect(.5)
    e["AO"] = _effect(.1)
    e["OF"] = _effect(.2)
    assert interaction_effects(e)["IAF"]["full"] == pytest.approx(.2)


def test_ifa_formula():
    e = _all_effects()
    e["FA"] = _effect(.5)
    e["FO"] = _effect(.1)
    e["OA"] = _effect(.2)
    assert interaction_effects(e)["IFA"]["full"] == pytest.approx(.2)


def test_interaction_coefficients_exact_keys():
    assert set(INTERACTION_COEFFICIENTS) == {"D3F", "D4F", "IFF", "D3A", "D4A", "IAA", "IAF", "IFA"}


def test_missing_context_fails_interactions():
    e = _all_effects()
    e.pop("AA")
    with pytest.raises(RuntimeError, match="A5_CONTEXT_MATRIX_INCOMPLETE"):
        interaction_effects(e)


def test_context_shift_oo_is_zero():
    e = _all_effects(.1)
    shifts = context_shifts(e)
    assert shifts["OO"]["full"] == 0
    assert shifts["OO"]["zero_folds"] == 6


def test_context_shift_fo_relative_oo():
    e = _all_effects(.1)
    e["FO"] = _effect(-.2)
    assert context_shifts(e)["FO"]["full"] == pytest.approx(-.3)


# ---- mechanism flags --------------------------------------------------------
def _labels(default="INCONCLUSIVE"):
    return {c: default for c in CONTEXT_ORDER}


def _shift_labels(default="INCONCLUSIVE_SHIFT"):
    return {c: default for c in CONTEXT_ORDER}


def _interaction_labels(default="INCONCLUSIVE_SHIFT"):
    return {k: default for k in INTERACTION_COEFFICIENTS}


def test_p3_full_sufficient_flip():
    labels = _labels(); shifts = _shift_labels(); inter = _interaction_labels()
    labels["OO"] = "STRONG_POSITIVE"; labels["FO"] = "STRONG_NEGATIVE"
    shifts["FO"] = "STRONG_ANTAGONISTIC_SHIFT"
    f = mechanism_flags(labels, shifts, inter, _all_effects(), _all_effects())
    assert f["P3_FULL_SUFFICIENT_FLIP"] is True
    assert f["P4_FULL_SUFFICIENT_FLIP"] is False


def test_p4_full_sufficient_flip():
    labels = _labels(); shifts = _shift_labels(); inter = _interaction_labels()
    labels["OO"] = "STRONG_POSITIVE"; labels["OF"] = "STRONG_NEGATIVE"
    shifts["OF"] = "STRONG_ANTAGONISTIC_SHIFT"
    f = mechanism_flags(labels, shifts, inter, _all_effects(), _all_effects())
    assert f["P4_FULL_SUFFICIENT_FLIP"] is True


def test_both_full_individually_sufficient():
    labels = _labels(); shifts = _shift_labels(); inter = _interaction_labels()
    labels["OO"] = "STRONG_POSITIVE"; labels["FO"] = labels["OF"] = "STRONG_NEGATIVE"
    shifts["FO"] = shifts["OF"] = "STRONG_ANTAGONISTIC_SHIFT"
    f = mechanism_flags(labels, shifts, inter, _all_effects(), _all_effects())
    assert f["BOTH_FULL_INDIVIDUALLY_SUFFICIENT"] is True


def test_joint_full_required():
    labels = _labels(); shifts = _shift_labels(); inter = _interaction_labels()
    labels["OO"] = "STRONG_POSITIVE"; labels["FF"] = "STRONG_NEGATIVE"
    inter["IFF"] = "STRONG_ANTAGONISTIC_SHIFT"
    f = mechanism_flags(labels, shifts, inter, _all_effects(), _all_effects())
    assert f["JOINT_FULL_CONTEXT_REQUIRED"] is True


def test_joint_flip_unresolved_when_iff_not_stable():
    labels = _labels(); shifts = _shift_labels(); inter = _interaction_labels()
    labels["OO"] = "STRONG_POSITIVE"; labels["FF"] = "STRONG_NEGATIVE"
    f = mechanism_flags(labels, shifts, inter, _all_effects(), _all_effects())
    assert f["FULL_CONTEXT_FLIP_WITH_UNRESOLVED_INTERACTION"] is True


def test_negative_shift_without_sign_flip_not_sufficient_flip():
    labels = _labels(); shifts = _shift_labels(); inter = _interaction_labels()
    labels["OO"] = "STRONG_POSITIVE"; labels["FO"] = "INCONCLUSIVE"
    shifts["FO"] = "STRONG_ANTAGONISTIC_SHIFT"
    f = mechanism_flags(labels, shifts, inter, _all_effects(), _all_effects())
    assert f["P3_FULL_SUFFICIENT_FLIP"] is False


def test_p3_centering_rescue_ff_to_af():
    labels = _labels(); shifts = _shift_labels(); inter = _interaction_labels()
    labels["FF"] = "STRONG_NEGATIVE"; labels["AF"] = "STRONG_POSITIVE"
    pf = _all_effects(); ps = _all_effects()
    pf["FF"] = _effect(-.2); pf["AF"] = _effect(.2)
    ps["FF"] = _effect(-.01); ps["AF"] = _effect(.01)
    f = mechanism_flags(labels, shifts, inter, pf, ps)
    assert f["P3_CENTERING_RESCUES_WITH_P4_FULL"] is True


def test_p4_centering_rescue_ff_to_fa():
    labels = _labels(); shifts = _shift_labels(); inter = _interaction_labels()
    labels["FF"] = "STRONG_NEGATIVE"; labels["FA"] = "STRONG_POSITIVE"
    pf = _all_effects(); ps = _all_effects()
    pf["FF"] = _effect(-.2); pf["FA"] = _effect(.2)
    ps["FF"] = _effect(-.01); ps["FA"] = _effect(.01)
    f = mechanism_flags(labels, shifts, inter, pf, ps)
    assert f["P4_CENTERING_RESCUES_WITH_P3_FULL"] is True


def test_aa_positive_sets_both_centered_restore():
    labels = _labels(); labels["AA"] = "STRONG_POSITIVE"
    f = mechanism_flags(labels, _shift_labels(), _interaction_labels(), _all_effects(), _all_effects())
    assert f["BOTH_CENTERED_RESTORE"] is True


def test_aa_negative_sets_centering_fails():
    labels = _labels(); labels["AA"] = "STRONG_NEGATIVE"
    f = mechanism_flags(labels, _shift_labels(), _interaction_labels(), _all_effects(), _all_effects())
    assert f["CENTERING_FAILS_TO_RESTORE"] is True


# ---- route discipline -------------------------------------------------------
def test_route_single_antagonist_no_training():
    r = route_decision({"P3_FULL_SUFFICIENT_FLIP": True, "P4_FULL_SUFFICIENT_FLIP": False})
    assert r["branch"] == "SINGLE_SCALE_FULL_ANTAGONIST" and r["training_go"] is False


def test_route_both_antagonists_no_training():
    r = route_decision({"P3_FULL_SUFFICIENT_FLIP": True, "P4_FULL_SUFFICIENT_FLIP": True})
    assert r["branch"] == "BOTH_SCALES_INDIVIDUALLY_ANTAGONISTIC" and not r["training_go"]


def test_route_joint_no_training():
    r = route_decision({"JOINT_FULL_CONTEXT_REQUIRED": True})
    assert r["branch"] == "JOINT_ONLY_ANTAGONISM" and not r["training_go"]


def test_route_centered_candidate_never_training_go():
    r = route_decision({"BOTH_CENTERED_RESTORE": True})
    assert r["branch"] == "CENTERED_CONTEXT_RESTORES_P5"
    assert r["training_go"] is False
    assert any("REQUIRES_SEPARATE_TRAINING_FREEZE" in x for x in r["route_candidates"])


def test_route_centering_failure():
    r = route_decision({"CENTERING_FAILS_TO_RESTORE": True})
    assert r["branch"] == "CENTERING_DOES_NOT_RESTORE" and not r["training_go"]


def test_route_inconclusive_no_training():
    r = route_decision({})
    assert r["branch"] == "A5_DIAGNOSIS_INCONCLUSIVE" and r["training_go"] is False


# ---- source / contract regression ------------------------------------------
def _text(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_design_freeze_says_a5_never_training_go():
    t = _text("docs/step4_a5/DESIGN_FREEZE.md")
    assert "A5 无论结果如何，都不直接发放 training GO" in t
    assert "NO A5 training_go=true under any result" in t


def test_evaluator_has_oo_and_ff_anchor_codes():
    t = _text("scripts/eval_step4_a5.py")
    assert "A5_OO_ANCHOR_FAIL" in t and "A5_FF_ANCHOR_FAIL" in t


def test_evaluator_pins_a4_adjudication_false_go():
    t = _text("scripts/eval_step4_a5.py")
    assert '"corrected_training_go": False' in t
    assert '"a4t_status": "HOLD"' in t


def test_evaluator_has_exact_condition_counts():
    t = _text("scripts/eval_step4_a5.py")
    assert '"condition_count_per_system": 18' in t
    assert '"total_condition_instances": 36' in t


def test_evaluator_forbids_training_go():
    t = _text("scripts/eval_step4_a5.py")
    assert "A5_TRAINING_GO_FORBIDDEN" in t
    assert '"training_go": False' in t


def test_evaluator_uses_step3_authoritative_validator():
    t = _text("scripts/eval_step4_a5.py")
    assert "evu.make_detection_validator" in t
    assert "evu.move_step3_batch_to_device" in t


def test_evaluator_uses_no_gate_donor_cache():
    t = _text("scripts/eval_step4_a5.py")
    assert "build_residual_cache_no_gate" in t
    assert "residual_cache_with_gate_guard" in t


def test_no_optimizer_or_backward_in_evaluator():
    t = _text("scripts/eval_step4_a5.py")
    assert ".backward(" not in t
    assert "torch.optim" not in t
    assert "optimizer" not in t.lower()


def test_context_module_never_builds_p3_p4_from_donor():
    t = _text("src/multimodal/step4_a5_context.py")
    assert "Recipient-only P3/P4 context" in t
    assert "p5_source = recipient_id if role == \"native\" else donor_id" in t


def test_effect_route_hardcodes_training_false():
    t = _text("src/multimodal/step4_a5_effects.py")
    assert '"training_go": False' in t
    assert "A5 never grants training GO" in t


def test_a4_mixed_context_regression_acknowledged_in_a5_design():
    t = _text("docs/step4_a5/DESIGN_FREEZE.md")
    assert "MIXED_PAIRED_CONTEXT_NO_GO" in t
    assert "A4T = HOLD" in t


def test_exact_nine_contexts_no_extra_result_driven_condition():
    assert len(CONTEXT_ORDER) == 9
    assert set(CONTEXT_STATES) == set(CONTEXT_ORDER)
