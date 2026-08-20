"""Pure helpers for the T1-GR G0/G1/G2 formal design.

This module has no filesystem, torch, dataset, or checkpoint dependency.  It is
safe to unit-test before private TRAIN/DEV access is mounted.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence

SCHEMA_DESIGN = "t1gr-g-design-freeze-v1"
SCHEMA_DESIGN_AUDIT = "t1gr-g-design-audit-public-v1"
SCHEMA_RUN_PLAN = "t1gr-g-run-plan-public-v1"
SCHEMA_RESULTS = "t1gr-g-per-seed-results-v1"
SCHEMA_CROSS_SEED = "t1gr-g-cross-seed-summary-v1"
SCHEMA_SUMMARY = "t1gr-g-summary-v1"

ARMS = ("G0-N", "G1-P", "G2-S")
SEEDS = (20260812, 20260813, 20260814)
EPOCHS = 80
TRAIN_COUNT = 1504
DEV_COUNT = 198
HOLDOUT_COUNT = 298
FOLD_COUNT = 5


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def payload_sha256(obj: Mapping[str, Any]) -> str:
    base = dict(obj)
    base.pop("payload_sha256", None)
    base.pop("request_fingerprint", None)
    return sha256_json(base)


def payload_ok(obj: Mapping[str, Any]) -> bool:
    claimed = obj.get("payload_sha256")
    return isinstance(claimed, str) and claimed == payload_sha256(obj)


def _fail(code: str, detail: Any | None = None) -> None:
    if detail is None:
        raise ValueError(code)
    raise ValueError(f"{code}:{detail}")


def _finite_metric(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(code)
    out = float(value)
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        _fail(code)
    return out


def validate_design(design: Mapping[str, Any], *, require_payload: bool = True) -> dict:
    if design.get("schema") != SCHEMA_DESIGN:
        _fail("T1GR_G_DESIGN_SCHEMA_FAIL")
    if design.get("status") != "FROZEN_FOR_IMPLEMENTATION_REVIEW":
        _fail("T1GR_G_DESIGN_STATUS_FAIL")
    if require_payload and not payload_ok(design):
        _fail("T1GR_G_DESIGN_PAYLOAD_FAIL")

    seeds = tuple(design.get("seeds") or ())
    if seeds != SEEDS or len(set(seeds)) != len(SEEDS):
        _fail("T1GR_G_SEED_FREEZE_DRIFT")
    arms = design.get("arms") or {}
    if tuple(sorted(arms)) != tuple(sorted(ARMS)):
        _fail("T1GR_G_ARM_FREEZE_DRIFT")
    if arms["G0-N"].get("model_treatment_id") != "T0-N":
        _fail("T1GR_G_G0_MODEL_MODE_DRIFT")
    if any(arms[a].get("model_treatment_id") != "T1-F" for a in ("G1-P", "G2-S")):
        _fail("T1GR_G_G1_G2_MODEL_MODE_DRIFT")

    authority = design.get("authority") or {}
    expected_false = (
        "smoke_training_authorized",
        "multiseed_training_authorized",
        "final_holdout_open_authorized",
        "depth_go",
        "production_go",
    )
    if authority.get("design_entry_authorized") is not True:
        _fail("T1GR_G_DESIGN_ENTRY_NOT_AUTHORIZED")
    if any(authority.get(k) is not False for k in expected_false):
        _fail("T1GR_G_AUTHORITY_EXPANSION")

    counts = design.get("sample_counts") or {}
    if counts != {"train": TRAIN_COUNT, "dev": DEV_COUNT, "final_holdout": HOLDOUT_COUNT}:
        _fail("T1GR_G_COUNT_DRIFT")

    training = design.get("training") or {}
    exact = {
        "epochs": EPOCHS,
        "batch": 4,
        "nbs": 64,
        "imgsz": 640,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "amp": True,
        "deterministic": True,
        "workers": 8,
        "checkpoint_primary": "last.pt",
        "checkpoint_diagnostic_only": "best.pt",
        "validation_cadence": "EVERY_EPOCH_DEV_ONLY",
    }
    drift = {k: {"expected": v, "actual": training.get(k)} for k, v in exact.items()
             if training.get(k) != v}
    if drift:
        _fail("T1GR_G_E5_RECIPE_DRIFT", drift)

    model = design.get("model") or {}
    if (
        model.get("architecture") != "yolo26s"
        or int(model.get("num_classes", -1)) != 12
        or model.get("end2end") is not True
        or model.get("rgb_backbone_frozen") is not False
        or int(model.get("p5_tap", -1)) != 10
        or int(model.get("p5_channels", -1)) != 512
    ):
        _fail("T1GR_G_MODEL_DRIFT")

    evaluation = design.get("evaluation") or {}
    if (
        evaluation.get("selection_split") != "DEV"
        or evaluation.get("final_holdout") != "SEALED"
        or evaluation.get("metric") != "mAP50-95"
        or int(evaluation.get("max_det", -1)) != 100
        or evaluation.get("primary_checkpoint") != "last.pt"
        or int(evaluation.get("fold_count", -1)) != FOLD_COUNT
    ):
        _fail("T1GR_G_EVALUATION_DRIFT")

    order = design.get("launch_order") or []
    expected_order = {
        20260812: ("G0-N", "G1-P", "G2-S"),
        20260813: ("G1-P", "G2-S", "G0-N"),
        20260814: ("G2-S", "G0-N", "G1-P"),
    }
    got = {int(row.get("seed")): tuple(row.get("arms") or ()) for row in order}
    if got != expected_order:
        _fail("T1GR_G_LAUNCH_ORDER_DRIFT")
    positions = {arm: [] for arm in ARMS}
    for seed in SEEDS:
        for i, arm in enumerate(got[seed]):
            positions[arm].append(i)
    if any(sorted(v) != [0, 1, 2] for v in positions.values()):
        _fail("T1GR_G_LAUNCH_ORDER_NOT_LATIN")

    return {
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "n_runs": len(SEEDS) * len(ARMS),
        "launch_positions": positions,
    }


def seed_keyed_ids(ids: Sequence[str], seed: int) -> tuple[str, ...]:
    normalized = tuple(str(x) for x in ids)
    if len(normalized) < 2:
        _fail("T1GR_G_NEED_AT_LEAST_TWO_IDS")
    if len(set(normalized)) != len(normalized):
        _fail("T1GR_G_DUPLICATE_IDS")

    def key(sid: str) -> tuple[str, str]:
        raw = f"{int(seed)}\0{sid}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest(), sid

    return tuple(sorted(normalized, key=key))


def epoch_shift(epoch: int, n: int) -> int:
    if n < 2:
        _fail("T1GR_G_NEED_AT_LEAST_TWO_IDS")
    if int(epoch) < 0:
        _fail("T1GR_G_NEGATIVE_EPOCH")
    return 1 + (int(epoch) % (int(n) - 1))


def balanced_wrong_map(ids: Sequence[str], seed: int, epoch: int) -> dict[str, str]:
    ordered = seed_keyed_ids(ids, seed)
    shift = epoch_shift(epoch, len(ordered))
    return {sid: ordered[(i + shift) % len(ordered)] for i, sid in enumerate(ordered)}


def verify_epoch_map(
    ids: Sequence[str], seed: int, epoch: int, mapping: Mapping[str, str]
) -> dict:
    normalized = tuple(str(x) for x in ids)
    expected = balanced_wrong_map(normalized, seed, epoch)
    got = {str(k): str(v) for k, v in mapping.items()}
    values = list(got.values())
    passed = (
        got == expected
        and set(got) == set(normalized)
        and set(values) == set(normalized)
        and len(values) == len(set(values))
        and all(got[sid] != sid for sid in normalized)
    )
    return {
        "seed": int(seed),
        "epoch_zero_based": int(epoch),
        "shift": epoch_shift(epoch, len(normalized)),
        "mapping_sha256": sha256_json(got),
        "self_matches": sum(got.get(sid) == sid for sid in normalized),
        "donor_min_uses": min(Counter(values).values()) if values else 0,
        "donor_max_uses": max(Counter(values).values()) if values else 0,
        "exact_expected": got == expected,
        "passed": passed,
    }


def schedule_summary(ids: Sequence[str], seed: int, epochs: int = EPOCHS) -> dict:
    ordered = seed_keyed_ids(ids, seed)
    pair_counts: Counter[tuple[str, str]] = Counter()
    donor_epoch_min = []
    donor_epoch_max = []
    mapping_hashes = []
    for epoch in range(int(epochs)):
        mapping = balanced_wrong_map(ordered, seed, epoch)
        check = verify_epoch_map(ordered, seed, epoch, mapping)
        if not check["passed"]:
            _fail("T1GR_G_INTERNAL_SCHEDULE_FAIL", check)
        donor_epoch_min.append(check["donor_min_uses"])
        donor_epoch_max.append(check["donor_max_uses"])
        mapping_hashes.append(check["mapping_sha256"])
        pair_counts.update(mapping.items())

    distinct_by_recipient = {
        rec: sum(1 for (r, _), n in pair_counts.items() if r == rec and n > 0)
        for rec in ordered
    }
    max_possible_unique = min(int(epochs), len(ordered) - 1)
    return {
        "seed": int(seed),
        "n_ids": len(ordered),
        "epochs": int(epochs),
        "self_pair_count": sum(n for (r, d), n in pair_counts.items() if r == d),
        "donor_use_per_epoch_min": min(donor_epoch_min),
        "donor_use_per_epoch_max": max(donor_epoch_max),
        "recipient_distinct_donor_min": min(distinct_by_recipient.values()),
        "recipient_distinct_donor_max": max(distinct_by_recipient.values()),
        "expected_distinct_donors": max_possible_unique,
        "mapping_hashes_sha256": sha256_json(mapping_hashes),
        "passed": bool(
            all(v == 1 for v in donor_epoch_min)
            and all(v == 1 for v in donor_epoch_max)
            and all(v == max_possible_unique for v in distinct_by_recipient.values())
            and all(r != d for r, d in pair_counts)
        ),
    }


def balanced_component_folds(
    component_by_id: Mapping[str, str], fold_count: int = FOLD_COUNT
) -> dict[str, int]:
    if int(fold_count) < 2:
        _fail("T1GR_G_FOLD_COUNT_INVALID")
    normalized = {str(sid): str(component) for sid, component in component_by_id.items()}
    if not normalized:
        _fail("T1GR_G_COMPONENT_MAP_EMPTY")
    if any(not sid or not component for sid, component in normalized.items()):
        _fail("T1GR_G_COMPONENT_MAP_INVALID")
    components = sorted(
        set(normalized.values()),
        key=lambda c: (hashlib.sha256(f"T1GR_FOLD_V1\0{c}".encode()).hexdigest(), c),
    )
    if len(components) < int(fold_count):
        _fail("T1GR_G_TOO_FEW_COMPONENTS_FOR_FOLDS")
    component_fold = {component: i % int(fold_count) for i, component in enumerate(components)}
    return {sid: component_fold[component] for sid, component in normalized.items()}


def contrast_label(new: Mapping[int, float], base: Mapping[int, float]) -> dict:
    if set(new) != set(base) or set(new) != set(SEEDS):
        _fail("T1GR_G_CONTRAST_SEED_SET_FAIL")
    deltas = {str(seed): float(new[seed]) - float(base[seed]) for seed in SEEDS}
    values = list(deltas.values())
    if all(v > 0.0 for v in values):
        label = "STABLE_POSITIVE"
    elif all(v < 0.0 for v in values):
        label = "STABLE_NEGATIVE"
    elif all(v == 0.0 for v in values):
        label = "EXACT_TIE"
    else:
        label = "MIXED"
    return {
        "label": label,
        "deltas_by_seed": deltas,
        "mean_delta": sum(values) / len(values),
        "median_delta": sorted(values)[len(values) // 2],
    }


def _validate_results(results: Mapping[str, Any]) -> dict[tuple[int, str], dict]:
    if results.get("schema") != SCHEMA_RESULTS:
        _fail("T1GR_G_RESULTS_SCHEMA_FAIL")
    if results.get("final_holdout_accessed") is not False:
        _fail("T1GR_G_HOLDOUT_ACCESS_CLAIM")
    if results.get("metric") != "mAP50-95":
        _fail("T1GR_G_RESULTS_METRIC_DRIFT")
    if results.get("checkpoint") != "last.pt":
        _fail("T1GR_G_RESULTS_CHECKPOINT_DRIFT")
    if int(results.get("max_det", -1)) != 100:
        _fail("T1GR_G_RESULTS_MAX_DET_DRIFT")

    rows = results.get("rows")
    if not isinstance(rows, list) or len(rows) != len(SEEDS) * len(ARMS):
        _fail("T1GR_G_RESULTS_ROW_COUNT_FAIL")
    indexed: dict[tuple[int, str], dict] = {}
    required_folds = {f"fold_{i}" for i in range(FOLD_COUNT)}
    for row in rows:
        if not isinstance(row, dict):
            _fail("T1GR_G_RESULT_ROW_TYPE_FAIL")
        seed = int(row.get("seed", -1))
        arm = str(row.get("arm", ""))
        key = (seed, arm)
        if seed not in SEEDS or arm not in ARMS or key in indexed:
            _fail("T1GR_G_RESULT_KEY_FAIL", key)
        value = _finite_metric(row.get("dev_map50_95"), "T1GR_G_DEV_METRIC_FAIL")
        folds = row.get("lofo_map50_95")
        if not isinstance(folds, dict) or set(folds) != required_folds:
            _fail("T1GR_G_FOLD_KEYS_FAIL", key)
        clean_folds = {
            name: _finite_metric(v, "T1GR_G_FOLD_METRIC_FAIL")
            for name, v in folds.items()
        }
        for hash_key in ("run_manifest_sha256", "last_checkpoint_sha256"):
            h = row.get(hash_key)
            if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                _fail("T1GR_G_RESULT_HASH_FAIL", {"key": key, "field": hash_key})
        indexed[key] = {**row, "dev_map50_95": value, "lofo_map50_95": clean_folds}
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if set(indexed) != expected:
        _fail("T1GR_G_RESULT_MATRIX_FAIL")
    return indexed


def _fold_gate(indexed: Mapping[tuple[int, str], Mapping[str, Any]], new: str, base: str) -> dict:
    rows = {}
    all_pass = True
    for seed in SEEDS:
        new_folds = indexed[(seed, new)]["lofo_map50_95"]
        base_folds = indexed[(seed, base)]["lofo_map50_95"]
        deltas = {fold: float(new_folds[fold]) - float(base_folds[fold])
                  for fold in sorted(new_folds)}
        positive = sum(v > 0.0 for v in deltas.values())
        passed = positive >= 4
        all_pass = all_pass and passed
        rows[str(seed)] = {
            "positive_lofo_count": positive,
            "required_positive_lofo_count": 4,
            "deltas": deltas,
            "passed": passed,
        }
    return {"by_seed": rows, "all_seeds_passed": all_pass}


def summarize_results(results: Mapping[str, Any]) -> tuple[dict, dict]:
    indexed = _validate_results(results)
    metric = {
        arm: {seed: indexed[(seed, arm)]["dev_map50_95"] for seed in SEEDS}
        for arm in ARMS
    }
    g1_g0 = contrast_label(metric["G1-P"], metric["G0-N"])
    g1_g2 = contrast_label(metric["G1-P"], metric["G2-S"])
    g2_g0 = contrast_label(metric["G2-S"], metric["G0-N"])
    fold_g1_g0 = _fold_gate(indexed, "G1-P", "G0-N")
    fold_g1_g2 = _fold_gate(indexed, "G1-P", "G2-S")

    if (
        g1_g0["label"] == "STABLE_POSITIVE"
        and g1_g2["label"] == "STABLE_POSITIVE"
        and fold_g1_g0["all_seeds_passed"]
        and fold_g1_g2["all_seeds_passed"]
    ):
        decision = "PAIRED_TRAINING_GENERALIZATION_SUPPORTED"
    elif (
        g1_g0["label"] == "STABLE_POSITIVE"
        and g2_g0["label"] == "STABLE_POSITIVE"
        and g1_g2["label"] in {"MIXED", "EXACT_TIE"}
    ):
        decision = "GENERIC_TRAINING_BENEFIT_SOURCE_IDENTITY_NOT_ESTABLISHED"
    elif (
        g1_g0["label"] in {"STABLE_NEGATIVE", "EXACT_TIE"}
        and g2_g0["label"] in {"STABLE_NEGATIVE", "EXACT_TIE"}
    ):
        decision = "SMALL_SAMPLE_SIGNAL_DID_NOT_TRANSFER"
    elif g1_g2["label"] in {"STABLE_NEGATIVE", "EXACT_TIE"}:
        decision = "PAIRED_SOURCE_SPECIFICITY_FAILED"
    else:
        decision = "INCONCLUSIVE_REPLICATION"

    cross = {
        "schema": SCHEMA_CROSS_SEED,
        "metric": "DEV_mAP50-95",
        "checkpoint": "last.pt",
        "max_det": 100,
        "seeds": list(SEEDS),
        "arm_metrics": {arm: {str(k): v for k, v in values.items()} for arm, values in metric.items()},
        "contrasts": {
            "G1-P_minus_G0-N": g1_g0,
            "G1-P_minus_G2-S": g1_g2,
            "G2-S_minus_G0-N": g2_g0,
        },
        "fold_sensitivity": {
            "G1-P_minus_G0-N": fold_g1_g0,
            "G1-P_minus_G2-S": fold_g1_g2,
        },
        "final_holdout_accessed": False,
    }
    cross["payload_sha256"] = payload_sha256(cross)
    summary = {
        "schema": SCHEMA_SUMMARY,
        "decision": decision,
        "paired_training_generalization_supported": decision == "PAIRED_TRAINING_GENERALIZATION_SUPPORTED",
        "cross_seed_summary_payload_sha256": cross["payload_sha256"],
        "dev_only": True,
        "final_holdout_accessed": False,
        "final_holdout_open_authorized": False,
        "depth_go": False,
        "production_go": False,
        "next_action": (
            "independent final-holdout opening adjudication required"
            if decision == "PAIRED_TRAINING_GENERALIZATION_SUPPORTED"
            else "do not open FINAL HOLDOUT"
        ),
    }
    summary["payload_sha256"] = payload_sha256(summary)
    return cross, summary
