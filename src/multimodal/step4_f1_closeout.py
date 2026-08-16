"""Torch-free closeout contracts shared by the F1 LOO producer/consumer."""
from __future__ import annotations

import math
import statistics


LOO_SCHEMA = "step4-f1-loo-v1"

DELTA_SPECS = {
    "SOFT_minus_C0": ("SOFT", "NORMAL", "C0", "NORMAL"),
    "FIXED_minus_C0": ("FIXED", "NORMAL", "C0", "NORMAL"),
    "SOFT_minus_FIXED": ("SOFT", "NORMAL", "FIXED", "NORMAL"),
    "SOFT_N_minus_Z": ("SOFT", "NORMAL", "SOFT", "ZERO-AUX"),
    "SOFT_N_minus_S": ("SOFT", "NORMAL", "SOFT", "SHUFFLE"),
    "FIXED_N_minus_Z": ("FIXED", "NORMAL", "FIXED", "ZERO-AUX"),
    "FIXED_N_minus_S": ("FIXED", "NORMAL", "FIXED", "SHUFFLE"),
}


def compute_f1_deltas(folds: dict, val_ids: list[str]) -> dict:
    out = {}
    for key, (left, left_variant, right, right_variant) in DELTA_SPECS.items():
        full = round(
            folds["full"][left][left_variant]
            - folds["full"][right][right_variant],
            6,
        )
        per_fold = {
            sample_id: round(
                folds[sample_id][left][left_variant]
                - folds[sample_id][right][right_variant],
                6,
            )
            for sample_id in val_ids
        }
        values = list(per_fold.values())
        out[key] = {
            "full": full,
            "per_fold": per_fold,
            "positive_folds": sum(value > 0 for value in values),
            "n_folds": len(values),
            "median": round(statistics.median(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
        }
    return out


def _valid_metric(value) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def validate_f1_loo_payload(loo: dict) -> dict:
    errors: list[str] = []
    if loo.get("schema") != LOO_SCHEMA:
        errors.append("SCHEMA_MISMATCH")
    if loo.get("checkpoint") != "last.pt":
        errors.append("CHECKPOINT_MISMATCH")
    val_ids = list(loo.get("val_ids") or [])
    folds = loo.get("folds") or {}
    if len(val_ids) != 6 or len(set(val_ids)) != 6:
        errors.append("VAL6_IDS_INVALID")
    if set(folds) != ({"full"} | set(val_ids)):
        errors.append("FOLD_KEYS_DO_NOT_MATCH_VAL_IDS")

    for fold_key, rows in folds.items():
        if not isinstance(rows, dict) or set(rows) != {"C0", "FIXED", "SOFT"}:
            errors.append(f"FOLD_GROUPS_INVALID:{fold_key}")
            continue
        c0 = rows["C0"]
        if c0.get("copy_of_normal") is not True or not _valid_metric(c0.get("NORMAL")):
            errors.append(f"C0_INVALID:{fold_key}")
        for tag in ("FIXED", "SOFT"):
            row = rows[tag]
            if row.get("copy_of_normal") is not False:
                errors.append(f"{tag}_COPY_FLAG_INVALID:{fold_key}")
            for variant in ("NORMAL", "ZERO-AUX", "SHUFFLE"):
                if not _valid_metric(row.get(variant)):
                    errors.append(f"VALUE_INVALID:{fold_key}/{tag}/{variant}")

    recomputed = None
    if not errors:
        try:
            recomputed = compute_f1_deltas(folds, val_ids)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"DELTA_RECOMPUTE_FAILED:{type(exc).__name__}:{exc}")
    if recomputed is not None and loo.get("deltas") != recomputed:
        errors.append("DELTA_BLOCK_MISMATCH")
    return {"errors": errors, "passed": not errors, "recomputed": recomputed}
