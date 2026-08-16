from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.step4_f1_closeout import (  # noqa: E402
    LOO_SCHEMA,
    compute_f1_deltas,
    f1_b_checkpoint_sha_key,
    f1_b_loo_output_name,
    validate_effective_q_stats,
    validate_f1_loo_payload,
)


def _payload() -> dict:
    val_ids = [f"v{i}" for i in range(6)]
    folds = {}
    for index, fold in enumerate(["full", *val_ids]):
        offset = index * 0.001
        folds[fold] = {
            "C0": {"NORMAL": 0.30 + offset, "copy_of_normal": True},
            "FIXED": {
                "NORMAL": 0.31 + offset,
                "ZERO-AUX": 0.27 + offset,
                "SHUFFLE": 0.29 + offset,
                "copy_of_normal": False,
            },
            "SOFT": {
                "NORMAL": 0.32 + offset,
                "ZERO-AUX": 0.28 + offset,
                "SHUFFLE": 0.30 + offset,
                "copy_of_normal": False,
            },
        }
    payload = {
        "schema": LOO_SCHEMA,
        "checkpoint": "last.pt",
        "val_ids": val_ids,
        "folds": folds,
    }
    payload["deltas"] = compute_f1_deltas(folds, val_ids)
    return payload


def test_valid_payload_recomputes_exactly():
    report = validate_f1_loo_payload(_payload())
    assert report["passed"]
    assert report["recomputed"]["SOFT_minus_C0"]["positive_folds"] == 6


def test_best_payload_uses_same_full_validator():
    payload = _payload()
    payload["checkpoint"] = "best.pt"
    report = validate_f1_loo_payload(payload, expected_checkpoint="best.pt")
    assert report["passed"]


def test_checkpoint_role_mismatch_is_rejected():
    payload = _payload()
    payload["checkpoint"] = "best.pt"
    report = validate_f1_loo_payload(payload)
    assert not report["passed"]
    assert "CHECKPOINT_MISMATCH" in report["errors"]


def test_loo_names_and_provenance_keys_preserve_frozen_last_contract():
    assert f1_b_loo_output_name("last.pt") == "step4_f1_b_loo.json"
    assert f1_b_loo_output_name("best.pt") == "step4_f1_b_loo_best.json"
    assert f1_b_checkpoint_sha_key("SOFT", "last.pt") == \
        "SOFT_last_pt_sha256"
    assert f1_b_checkpoint_sha_key("SOFT", "best.pt") == \
        "SOFT_best_pt_sha256"


def test_tampered_delta_is_rejected():
    payload = _payload()
    payload["deltas"]["SOFT_minus_C0"]["median"] += 0.000001
    report = validate_f1_loo_payload(payload)
    assert not report["passed"]
    assert "DELTA_BLOCK_MISMATCH" in report["errors"]


def test_bad_fold_value_and_copy_flag_are_rejected():
    payload = copy.deepcopy(_payload())
    payload["folds"]["v0"]["SOFT"]["NORMAL"] = float("nan")
    payload["folds"]["v1"]["FIXED"]["copy_of_normal"] = True
    report = validate_f1_loo_payload(payload)
    assert not report["passed"]
    assert any(error.startswith("VALUE_INVALID:v0/SOFT") for error in report["errors"])
    assert "FIXED_COPY_FLAG_INVALID:v1" in report["errors"]


def test_q_stats_require_exact_count_and_mean_between_extrema():
    assert validate_effective_q_stats(
        {"count": 11, "mean": 0.5, "min": 0.4, "max": 0.6}, 11
    )["passed"]
    bad_count = validate_effective_q_stats(
        {"count": 10, "mean": 0.5, "min": 0.4, "max": 0.6}, 11
    )
    assert not bad_count["passed"]
    assert bad_count["errors"] == ["Q_COUNT_MISMATCH:10!=11"]
    bad_order = validate_effective_q_stats(
        {"count": 11, "mean": 0.7, "min": 0.4, "max": 0.6}, 11
    )
    assert not bad_order["passed"]
    assert "Q_STATS_ORDER_OR_BOUNDS_INVALID" in bad_order["errors"]
