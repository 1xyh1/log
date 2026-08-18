"""A4 effect summaries, factorial contrasts, and preregistered route decisions."""
from __future__ import annotations

from typing import Mapping

import numpy as np

SCALES = ("P3", "P4", "P5")
FACTORIAL_CELLS = ("C000", "C100", "C010", "C001", "C110", "C101", "C011", "C111")


def summarize_effect(full: float, loo: Mapping[str, float]) -> dict:
    vals = [float(v) for v in loo.values()]
    return {
        "full": float(full),
        "loo": {str(k): float(v) for k, v in loo.items()},
        "loo_median": float(np.median(vals)),
        "positive_folds": int(sum(v > 0 for v in vals)),
        "negative_folds": int(sum(v < 0 for v in vals)),
        "zero_folds": int(sum(v == 0 for v in vals)),
    }


def classify_ap_effect(primary: Mapping, replication: Mapping,
                       pos_label: str = "STRONG_POSITIVE",
                       neg_label: str = "STRONG_NEGATIVE") -> str:
    if (primary["full"] > 0 and primary["loo_median"] > 0
            and primary["positive_folds"] >= 4 and replication["full"] > 0):
        return pos_label
    if (primary["full"] < 0 and primary["loo_median"] < 0
            and primary["negative_folds"] >= 4 and replication["full"] < 0):
        return neg_label
    return "INCONCLUSIVE"


def linear_contrast(results: Mapping[str, Mapping], coefficients: Mapping[str, float]) -> dict:
    missing = [k for k in coefficients if k not in results]
    if missing:
        raise KeyError(f"A4_FACTORIAL_CONTRAST_MISSING:{missing}")
    full = sum(float(c) * float(results[k]["full"]["map50_95"])
               for k, c in coefficients.items())
    ids = list(next(iter(results.values()))["loo"].keys())
    loo = {
        sid: sum(float(c) * float(results[k]["loo"][sid]["map50_95"])
                 for k, c in coefficients.items())
        for sid in ids
    }
    return summarize_effect(full, loo)


def factorial_effects(cells: Mapping[str, Mapping]) -> dict:
    if set(cells) != set(FACTORIAL_CELLS):
        raise RuntimeError(f"A4_FACTORIAL_INCOMPLETE:{sorted(cells)}")
    contrasts = {
        "R3": {"C100": 1, "C000": -1},
        "R4": {"C010": 1, "C000": -1},
        "R5": {"C001": 1, "C000": -1},
        "I34": {"C110": 1, "C100": -1, "C010": -1, "C000": 1},
        "I35": {"C101": 1, "C100": -1, "C001": -1, "C000": 1},
        "I45": {"C011": 1, "C010": -1, "C001": -1, "C000": 1},
        "I345": {
            "C111": 1, "C110": -1, "C101": -1, "C011": -1,
            "C100": 1, "C010": 1, "C001": 1, "C000": -1,
        },
    }
    return {name: linear_contrast(cells, coeff) for name, coeff in contrasts.items()}


def joint_p5_decision(
    paired_labels: Mapping[str, str],
    rescue_labels: Mapping[str, str],
) -> dict:
    """Primary decision consumes AC_ALL only; AC_CONTENT is intentionally absent.

    Reviewer adjudication 2026-08-19 (feedback/2026-08-19_formal-review.md):
    cross-context sign conflict (MIXED_PAIRED_CONTEXT_NO_GO) MUST be evaluated
    BEFORE any same-context GO. The executed A4 (commit 36221d2) checked
    go_contexts first, so conditional STRONG_NEGATIVE could never veto a
    standalone GO; this corrected precedence replaces that behavior.
    """
    contexts = ("standalone", "conditional")
    positive_pair = [c for c in contexts if paired_labels.get(c) == "STRONG_POSITIVE"]
    negative_pair = [c for c in contexts if paired_labels.get(c) == "STRONG_NEGATIVE"]

    if positive_pair and negative_pair:
        return {
            "branch": "MIXED_PAIRED_CONTEXT_NO_GO",
            "training_go": False,
            "contexts": {"positive": positive_pair, "negative": negative_pair},
            "reason": "paired AC changes sign across standalone/conditional contexts",
        }

    go_contexts = [c for c in contexts
                   if paired_labels.get(c) == "STRONG_POSITIVE"
                   and rescue_labels.get(c) == "STRONG_POSITIVE_RESCUE"]
    if go_contexts:
        return {
            "branch": "CENTERING_TRAINING_GO",
            "training_go": True,
            "contexts": go_contexts,
            "reason": "paired AC restored and native performance rescued in the same context",
        }

    positive_rescue = [c for c in contexts if rescue_labels.get(c) == "STRONG_POSITIVE_RESCUE"]

    if negative_pair:
        return {
            "branch": "STOP_CENTERING_ROUTE",
            "training_go": False,
            "contexts": negative_pair,
            "reason": "AC remains paired-negative in a preregistered context",
        }
    if positive_pair:
        return {
            "branch": "PAIRED_RESTORED_NO_PERFORMANCE_RESCUE",
            "training_go": False,
            "contexts": positive_pair,
            "reason": "paired specificity restored without matched positive centering rescue",
        }
    if positive_rescue:
        return {
            "branch": "PERFORMANCE_RESCUE_WITHOUT_PAIRED_RESTORATION",
            "training_go": False,
            "contexts": positive_rescue,
            "reason": "centering may regularize architecture but paired IR value remains unproven",
        }
    return {
        "branch": "A4_DIAGNOSIS_INCONCLUSIVE",
        "training_go": False,
        "contexts": [],
        "reason": "no preregistered primary branch reached",
    }


def content_diagnostic_interpretation(all_rescue_label: str, content_rescue_label: str) -> str:
    if all_rescue_label == "STRONG_POSITIVE_RESCUE" and content_rescue_label != "STRONG_POSITIVE_RESCUE":
        return "PADDING_OR_GLOBAL_STATISTICS_MAY_CONTRIBUTE"
    if all_rescue_label == "STRONG_POSITIVE_RESCUE" and content_rescue_label == "STRONG_POSITIVE_RESCUE":
        return "CONTENT_OR_GLOBAL_POSTPROJECTION_DC_SUPPORTED"
    if all_rescue_label != "STRONG_POSITIVE_RESCUE" and content_rescue_label == "STRONG_POSITIVE_RESCUE":
        return "CONTENT_SPECIFIC_DC_MAY_BE_MASKED_BY_FULL_MAP_MEAN"
    return "NO_POSITIVE_CONTENT_DIAGNOSTIC_RESCUE"


def apply_content_diagnostic_veto(primary_decision: Mapping, content_rescue_labels: Mapping[str, str]) -> dict:
    """AC_CONTENT can never create GO; it may conservatively veto a premature GO.

    This implements Branch E: if AC_ALL says GO but the same-context content-aware
    centering rescue is absent, inspect padding/global statistics before training.
    """
    out = dict(primary_decision)
    out["primary_decision_before_content_diagnostic"] = dict(primary_decision)
    if not bool(primary_decision.get("training_go")):
        out["content_diagnostic_veto_applied"] = False
        return out
    contexts = list(primary_decision.get("contexts") or [])
    supported = [c for c in contexts if content_rescue_labels.get(c) == "STRONG_POSITIVE_RESCUE"]
    if supported:
        out["content_diagnostic_veto_applied"] = False
        out["content_supported_go_contexts"] = supported
        return out
    out.update({
        "branch": "PADDING_GLOBAL_STATISTICS_AUDIT_BEFORE_TRAINING",
        "training_go": False,
        "content_diagnostic_veto_applied": True,
        "contexts": contexts,
        "reason": "AC_ALL primary GO was not reproduced by AC_CONTENT rescue in the same context",
    })
    return out
