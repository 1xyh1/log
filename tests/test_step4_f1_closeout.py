from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.step4_f1_closeout import (  # noqa: E402
    LOO_SCHEMA,
    compute_f1_deltas,
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
