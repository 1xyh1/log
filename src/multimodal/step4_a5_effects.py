"""A5 paired-effect summaries, cross-scale contrasts, and no-training route logic."""
from __future__ import annotations

from typing import Mapping

import numpy as np

CONTEXT_ORDER = ("OO", "FO", "OF", "FF", "AO", "OA", "AF", "FA", "AA")
INTERACTION_COEFFICIENTS = {
    "D3F": {"FO": 1.0, "OO": -1.0},
    "D4F": {"OF": 1.0, "OO": -1.0},
    "IFF": {"FF": 1.0, "FO": -1.0, "OF": -1.0, "OO": 1.0},
    "D3A": {"AO": 1.0, "OO": -1.0},
    "D4A": {"OA": 1.0, "OO": -1.0},
    "IAA": {"AA": 1.0, "AO": -1.0, "OA": -1.0, "OO": 1.0},
    "IAF": {"AF": 1.0, "AO": -1.0, "OF": -1.0, "OO": 1.0},
    "IFA": {"FA": 1.0, "FO": -1.0, "OA": -1.0, "OO": 1.0},
}


def summarize_effect(full: float, loo: Mapping[str, float]) -> dict:
    values = [float(v) for v in loo.values()]
    if not values:
        raise ValueError("A5_EFFECT_REQUIRES_LOO")
    return {
        "full": float(full),
        "loo": {str(k): float(v) for k, v in loo.items()},
        "loo_median": float(np.median(values)),
        "positive_folds": int(sum(v > 0 for v in values)),
        "negative_folds": int(sum(v < 0 for v in values)),
        "zero_folds": int(sum(v == 0 for v in values)),
    }


def effect_from_results(native: Mapping, donor: Mapping) -> dict:
    nloo, dloo = native["loo"], donor["loo"]
    if set(nloo) != set(dloo):
        raise RuntimeError("A5_PAIRED_EFFECT_SEMANTICS_FAIL:LOO_ID_SET")
    full = float(native["full"]["map50_95"]) - float(donor["full"]["map50_95"])
    loo = {
        sid: float(nloo[sid]["map50_95"]) - float(dloo[sid]["map50_95"])
        for sid in nloo
    }
    return summarize_effect(full, loo)


def difference_of_effects(new_effect: Mapping, baseline_effect: Mapping) -> dict:
    if set(new_effect["loo"]) != set(baseline_effect["loo"]):
        raise RuntimeError("A5_CONTEXT_SHIFT_LOO_ID_SET")
    full = float(new_effect["full"]) - float(baseline_effect["full"])
    loo = {
        sid: float(new_effect["loo"][sid]) - float(baseline_effect["loo"][sid])
        for sid in new_effect["loo"]
    }
    return summarize_effect(full, loo)


def classify_paired_effect(primary: Mapping, replication: Mapping) -> str:
    if (
        primary["full"] > 0
        and primary["loo_median"] > 0
        and primary["positive_folds"] >= 4
        and replication["full"] > 0
    ):
        return "STRONG_POSITIVE"
    if (
        primary["full"] < 0
        and primary["loo_median"] < 0
        and primary["negative_folds"] >= 4
        and replication["full"] < 0
    ):
        return "STRONG_NEGATIVE"
    return "INCONCLUSIVE"


def classify_shift(primary: Mapping, replication: Mapping) -> str:
    if (
        primary["full"] < 0
        and primary["loo_median"] < 0
        and primary["negative_folds"] >= 4
        and replication["full"] < 0
    ):
        return "STRONG_ANTAGONISTIC_SHIFT"
    if (
        primary["full"] > 0
        and primary["loo_median"] > 0
        and primary["positive_folds"] >= 4
        and replication["full"] > 0
    ):
        return "STRONG_RESCUING_SHIFT"
    return "INCONCLUSIVE_SHIFT"


def linear_effect_contrast(effects: Mapping[str, Mapping], coefficients: Mapping[str, float]) -> dict:
    missing = [c for c in coefficients if c not in effects]
    if missing:
        raise RuntimeError(f"A5_INTERACTION_CONTRAST_MISSING:{missing}")
    id_sets = [set(effects[c]["loo"]) for c in coefficients]
    if not id_sets or any(s != id_sets[0] for s in id_sets[1:]):
        raise RuntimeError("A5_INTERACTION_CONTRAST_FAIL:LOO_ID_SET")
    ids = list(effects[next(iter(coefficients))]["loo"].keys())
    full = sum(float(w) * float(effects[c]["full"]) for c, w in coefficients.items())
    loo = {
        sid: sum(float(w) * float(effects[c]["loo"][sid]) for c, w in coefficients.items())
        for sid in ids
    }
    return summarize_effect(full, loo)


def interaction_effects(effects: Mapping[str, Mapping]) -> dict:
    if set(effects) != set(CONTEXT_ORDER):
        raise RuntimeError(f"A5_CONTEXT_MATRIX_INCOMPLETE:{sorted(effects)}")
    return {
        name: linear_effect_contrast(effects, coeffs)
        for name, coeffs in INTERACTION_COEFFICIENTS.items()
    }


def context_shifts(effects: Mapping[str, Mapping]) -> dict:
    if set(effects) != set(CONTEXT_ORDER):
        raise RuntimeError(f"A5_CONTEXT_MATRIX_INCOMPLETE:{sorted(effects)}")
    base = effects["OO"]
    return {c: difference_of_effects(effects[c], base) for c in CONTEXT_ORDER}


def _is_rescuing_shift(primary_effects: Mapping, replication_effects: Mapping, new_ctx: str, old_ctx: str) -> tuple[str, dict, dict]:
    pf = difference_of_effects(primary_effects[new_ctx], primary_effects[old_ctx])
    pr = difference_of_effects(replication_effects[new_ctx], replication_effects[old_ctx])
    return classify_shift(pf, pr), pf, pr


def mechanism_flags(
    context_labels: Mapping[str, str],
    shift_labels: Mapping[str, str],
    interaction_labels: Mapping[str, str],
    primary_effects: Mapping[str, Mapping],
    replication_effects: Mapping[str, Mapping],
) -> dict:
    """Apply preregistered A5 mechanism labels.

    No return path from this function can authorize training.
    """
    if set(context_labels) != set(CONTEXT_ORDER):
        raise RuntimeError("A5_MECHANISM_CONTEXT_LABELS_INCOMPLETE")

    p3_sufficient = (
        context_labels["OO"] == "STRONG_POSITIVE"
        and context_labels["FO"] == "STRONG_NEGATIVE"
        and shift_labels["FO"] == "STRONG_ANTAGONISTIC_SHIFT"
    )
    p4_sufficient = (
        context_labels["OO"] == "STRONG_POSITIVE"
        and context_labels["OF"] == "STRONG_NEGATIVE"
        and shift_labels["OF"] == "STRONG_ANTAGONISTIC_SHIFT"
    )
    joint_required = (
        context_labels["OO"] == "STRONG_POSITIVE"
        and context_labels["FF"] == "STRONG_NEGATIVE"
        and context_labels["FO"] != "STRONG_NEGATIVE"
        and context_labels["OF"] != "STRONG_NEGATIVE"
        and interaction_labels.get("IFF") == "STRONG_ANTAGONISTIC_SHIFT"
    )
    unresolved_joint = (
        context_labels["OO"] == "STRONG_POSITIVE"
        and context_labels["FF"] == "STRONG_NEGATIVE"
        and context_labels["FO"] != "STRONG_NEGATIVE"
        and context_labels["OF"] != "STRONG_NEGATIVE"
        and interaction_labels.get("IFF") != "STRONG_ANTAGONISTIC_SHIFT"
    )

    p3_rescue_label, p3_rescue_fixed, p3_rescue_soft = _is_rescuing_shift(
        primary_effects, replication_effects, "AF", "FF"
    )
    p4_rescue_label, p4_rescue_fixed, p4_rescue_soft = _is_rescuing_shift(
        primary_effects, replication_effects, "FA", "FF"
    )
    p3_center_rescue = (
        context_labels["FF"] == "STRONG_NEGATIVE"
        and context_labels["AF"] == "STRONG_POSITIVE"
        and p3_rescue_label == "STRONG_RESCUING_SHIFT"
    )
    p4_center_rescue = (
        context_labels["FF"] == "STRONG_NEGATIVE"
        and context_labels["FA"] == "STRONG_POSITIVE"
        and p4_rescue_label == "STRONG_RESCUING_SHIFT"
    )
    both_centered_restore = context_labels["AA"] == "STRONG_POSITIVE"
    centering_fails = context_labels["AA"] == "STRONG_NEGATIVE"

    return {
        "P3_FULL_SUFFICIENT_FLIP": bool(p3_sufficient),
        "P4_FULL_SUFFICIENT_FLIP": bool(p4_sufficient),
        "BOTH_FULL_INDIVIDUALLY_SUFFICIENT": bool(p3_sufficient and p4_sufficient),
        "JOINT_FULL_CONTEXT_REQUIRED": bool(joint_required),
        "FULL_CONTEXT_FLIP_WITH_UNRESOLVED_INTERACTION": bool(unresolved_joint),
        "P3_CENTERING_RESCUES_WITH_P4_FULL": bool(p3_center_rescue),
        "P4_CENTERING_RESCUES_WITH_P3_FULL": bool(p4_center_rescue),
        "BOTH_CENTERED_RESTORE": bool(both_centered_restore),
        "CENTERING_FAILS_TO_RESTORE": bool(centering_fails),
        "paired_shift_FF_to_AF": {
            "label": p3_rescue_label,
            "FIXED": p3_rescue_fixed,
            "SOFT": p3_rescue_soft,
        },
        "paired_shift_FF_to_FA": {
            "label": p4_rescue_label,
            "FIXED": p4_rescue_fixed,
            "SOFT": p4_rescue_soft,
        },
    }


def route_decision(flags: Mapping[str, object]) -> dict:
    """Deterministic diagnosis route. A5 never grants training GO."""
    p3 = bool(flags.get("P3_FULL_SUFFICIENT_FLIP"))
    p4 = bool(flags.get("P4_FULL_SUFFICIENT_FLIP"))
    centered = any(
        bool(flags.get(k))
        for k in (
            "P3_CENTERING_RESCUES_WITH_P4_FULL",
            "P4_CENTERING_RESCUES_WITH_P3_FULL",
            "BOTH_CENTERED_RESTORE",
        )
    )

    if p3 and p4:
        branch = "BOTH_SCALES_INDIVIDUALLY_ANTAGONISTIC"
    elif p3 ^ p4:
        branch = "SINGLE_SCALE_FULL_ANTAGONIST"
    elif bool(flags.get("JOINT_FULL_CONTEXT_REQUIRED")):
        branch = "JOINT_ONLY_ANTAGONISM"
    elif bool(flags.get("FULL_CONTEXT_FLIP_WITH_UNRESOLVED_INTERACTION")):
        branch = "FULL_CONTEXT_FLIP_WITH_UNRESOLVED_INTERACTION"
    elif centered:
        branch = "CENTERED_CONTEXT_RESTORES_P5"
    elif bool(flags.get("CENTERING_FAILS_TO_RESTORE")):
        branch = "CENTERING_DOES_NOT_RESTORE"
    else:
        branch = "A5_DIAGNOSIS_INCONCLUSIVE"

    route_candidates = []
    if p3 or p4:
        route_candidates.append("A5b_IMPLICATED_SCALE_SOURCE_AUDIT")
    if bool(flags.get("JOINT_FULL_CONTEXT_REQUIRED")):
        route_candidates.append("A6_JOINT_CROSS_SCALE_INTERACTION_MECHANISM")
    if centered:
        route_candidates.append("SELECTIVE_CENTERING_CANDIDATE_REQUIRES_SEPARATE_TRAINING_FREEZE")
    if bool(flags.get("CENTERING_FAILS_TO_RESTORE")):
        route_candidates.append("DEEPER_REPRESENTATION_CHANNEL_SEMANTICS_AUDIT")
    if not route_candidates:
        route_candidates.append("EXPAND_DIAGNOSTIC_SAMPLE_OR_DATA_CONTRACT_AUDIT")

    return {
        "branch": branch,
        "training_go": False,
        "route_candidates": route_candidates,
        "reason": "A5 is evaluation-only and cannot authorize training",
    }
