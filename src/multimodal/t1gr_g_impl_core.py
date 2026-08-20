"""Pure implementation helpers for the T1-GR G experiment."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence

from .t1gr_g_core import ARMS, EPOCHS, SEEDS, balanced_wrong_map, sha256_json

SCHEMA_IMPL = "t1gr-g-implementation-freeze-v1"
SCHEMA_VIEW_PRIVATE = "t1gr-g-multimodal-view-private-v1"
SCHEMA_VIEW_PUBLIC = "t1gr-g-multimodal-view-public-v1"
SCHEMA_PREFLIGHT = "t1gr-g-implementation-preflight-public-v1"
SCHEMA_RUN = "t1gr-g-run-public-v1"
SCHEMA_SMOKE_AUDIT = "t1gr-g-smoke-audit-public-v1"
SCHEMA_SUITE_STATE = "t1gr-g-suite-state-private-v1"
SCHEMA_EVAL = "t1gr-g-eval-public-v1"

ZERO_IR = "__ZERO_IR__"
HEX = frozenset("0123456789abcdef")


def payload_sha256(obj: Mapping[str, Any]) -> str:
    base = dict(obj)
    base.pop("payload_sha256", None)
    base.pop("request_fingerprint", None)
    return sha256_json(base)


def payload_ok(obj: Mapping[str, Any]) -> bool:
    value = obj.get("payload_sha256")
    return isinstance(value, str) and value == payload_sha256(obj)


def hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def validate_impl_spec(spec: Mapping[str, Any]) -> dict:
    if spec.get("schema") != SCHEMA_IMPL:
        raise ValueError("T1GR_G_IMPL_SCHEMA_FAIL")
    if spec.get("status") != "FROZEN_BEFORE_IMPLEMENTATION_SMOKE":
        raise ValueError("T1GR_G_IMPL_STATUS_FAIL")
    if not payload_ok(spec):
        raise ValueError("T1GR_G_IMPL_PAYLOAD_FAIL")
    train = spec.get("training") or {}
    exact = {
        "optimizer": "MuSGD",
        "epochs": 80,
        "smoke_epochs": 1,
        "batch": 4,
        "nbs": 64,
        "imgsz": 640,
        "lr0": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "primary_checkpoint": "last.pt",
        "best_checkpoint_role": "DIAGNOSTIC_ONLY",
    }
    drift = {k: (train.get(k), v) for k, v in exact.items() if train.get(k) != v}
    if drift:
        raise ValueError(f"T1GR_G_IMPL_TRAIN_DRIFT:{drift}")
    runtime = spec.get("runtime") or {}
    if (
        runtime.get("ultralytics") != "8.4.56"
        or int(runtime.get("train_workers", -1)) != 8
        or int(runtime.get("validation_workers", -1)) != 16
        or runtime.get("single_gpu_only") is not True
        or runtime.get("epoch_fresh_worker_iterators") is not True
    ):
        raise ValueError("T1GR_G_IMPL_RUNTIME_DRIFT")
    tensor = spec.get("input_tensor") or {}
    if int(tensor.get("channels", -1)) != 4 or tensor.get("depth_used") is not False:
        raise ValueError("T1GR_G_IMPL_INPUT_DRIFT")
    augmentation = spec.get("augmentation") or {}
    augmentation_exact = {
        "mosaic": 1.0,
        "translate": 0.1,
        "scale": 0.5,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 10,
        "rgb_reference_parity_required": True,
        "same_seed_cross_arm_visible_bitwise_identity_required": True,
    }
    if any(augmentation.get(key) != value for key, value in augmentation_exact.items()):
        raise ValueError("T1GR_G_IMPL_AUGMENTATION_DRIFT")
    model = spec.get("model") or {}
    if (
        model.get("class") != "T1GRP5Model"
        or model.get("rgb_backbone_frozen") is not False
        or model.get("end2end") is not True
        or model.get("loss") != "Ultralytics E2ELoss"
        or model.get("same_seed_complete_initial_state_required") is not True
        or model.get("initialization_claim") != "AUDITABLE_INITIALIZATION_ONLY"
        or model.get("numerical_repeatability_claim") is not False
    ):
        raise ValueError("T1GR_G_IMPL_MODEL_DRIFT")
    eva = spec.get("evaluation") or {}
    if (
        eva.get("primary_inference_ir") != "ZERO_IR"
        or eva.get("split") != "DEV"
        or eva.get("metric") != "mAP50-95"
        or int(eva.get("max_det", -1)) != 100
        or int(eva.get("fold_count", -1)) != 5
    ):
        raise ValueError("T1GR_G_IMPL_EVAL_DRIFT")
    auth = spec.get("authority") or {}
    if auth.get("implementation_entry_authorized") is not True:
        raise ValueError("T1GR_G_IMPL_ENTRY_BLOCKED")
    for key in (
        "smoke_training_authorized_before_preflight",
        "multiseed_training_authorized_before_smoke_audit",
        "final_holdout_open_authorized",
        "depth_go",
        "production_go",
    ):
        if auth.get(key) is not False:
            raise ValueError("T1GR_G_IMPL_AUTHORITY_EXPANSION")
    smoke = spec.get("smoke") or {}
    formal = spec.get("formal") or {}
    if (
        int(smoke.get("run_count", -1)) != 9
        or smoke.get("formal_authorization_requires_all_nine") is not True
        or int(formal.get("run_count", -1)) != 9
        or formal.get("suite_only_entry") is not True
        or formal.get("selective_rerun_forbidden") is not True
    ):
        raise ValueError("T1GR_G_IMPL_SUITE_DRIFT")
    return {"arms": list(ARMS), "seeds": list(SEEDS), "formal_runs": 9}


def source_for_recipient(
    ids: Sequence[str], *, arm: str, seed: int, epoch: int, recipient: str, split: str
) -> str:
    values = tuple(str(x) for x in ids)
    recipient = str(recipient)
    if recipient not in set(values):
        raise ValueError("T1GR_G_RECIPIENT_NOT_IN_SPLIT")
    if split != "train":
        return ZERO_IR
    if arm in {"G0-N", "G1-P"}:
        return recipient
    if arm == "G2-S":
        return balanced_wrong_map(values, int(seed), int(epoch))[recipient]
    raise ValueError("T1GR_G_UNKNOWN_ARM")


def source_schedule_index(ids: Sequence[str], *, seed: int) -> tuple[tuple[str, ...], dict[str, int]]:
    from .t1gr_g_core import seed_keyed_ids

    ordered = seed_keyed_ids(ids, seed)
    return ordered, {sid: i for i, sid in enumerate(ordered)}


def fast_source_for_recipient(
    ordered: Sequence[str], position: Mapping[str, int], *, epoch: int, recipient: str
) -> str:
    n = len(ordered)
    if recipient not in position or n < 2:
        raise ValueError("T1GR_G_FAST_SOURCE_INPUT_FAIL")
    shift = 1 + (int(epoch) % (n - 1))
    return str(ordered[(int(position[recipient]) + shift) % n])


def trace_epoch_summary(
    rows: Sequence[Mapping[str, Any]], ids: Sequence[str], *, arm: str, seed: int, epoch: int
) -> dict:
    expected_ids = tuple(str(x) for x in ids)
    expected_set = set(expected_ids)
    anchors = [r for r in rows if r.get("role") == "anchor"]
    recipients = [str(r.get("recipient")) for r in anchors]
    recipient_counts = Counter(recipients)
    exact_anchor_coverage = (
        set(recipient_counts) == expected_set
        and len(anchors) == len(expected_ids)
        and all(recipient_counts[sid] == 1 for sid in expected_ids)
    )
    expected_map = (
        balanced_wrong_map(expected_ids, int(seed), int(epoch))
        if arm == "G2-S"
        else {sid: sid for sid in expected_ids}
    )
    mapping_exact = exact_anchor_coverage and all(
        str(row.get("donor")) == expected_map[str(row.get("recipient"))] for row in anchors
    )
    donors = [str(r.get("donor")) for r in anchors]
    donor_counts = Counter(donors)
    if arm == "G2-S":
        source_condition_passed = bool(
            mapping_exact
            and all(r != d for r, d in zip(recipients, donors))
            and set(donor_counts) == expected_set
            and all(donor_counts[sid] == 1 for sid in expected_ids)
        )
    else:
        source_condition_passed = mapping_exact and all(r == d for r, d in zip(recipients, donors))
    normalized = [
        {
            "recipient": str(r.get("recipient")),
            "donor": str(r.get("donor")),
            "role": str(r.get("role")),
            "epoch": int(r.get("epoch", -1)),
        }
        for r in rows
    ]
    epoch_exact = all(row["epoch"] == int(epoch) for row in normalized)
    return {
        "epoch_zero_based": int(epoch),
        "arm": arm,
        "seed": int(seed),
        "anchor_count": len(anchors),
        "all_pair_count": len(rows),
        "anchor_coverage_exact": exact_anchor_coverage,
        "anchor_mapping_exact": mapping_exact,
        "anchor_self_match_count": sum(r == d for r, d in zip(recipients, donors)),
        "anchor_donor_min_uses": min(donor_counts.values()) if donor_counts else 0,
        "anchor_donor_max_uses": max(donor_counts.values()) if donor_counts else 0,
        "all_rows_epoch_exact": epoch_exact,
        "source_condition_passed": source_condition_passed and epoch_exact,
        "pair_trace_commitment": sha256_json(normalized),
    }


def optimizer_contract_snapshot(model: Any, optimizer: Any) -> dict:
    names = {id(p): name for name, p in model.named_parameters()}
    groups = []
    assignments = {}
    for index, group in enumerate(optimizer.param_groups):
        group_names = []
        for parameter in group.get("params", []):
            name = names.get(id(parameter))
            if name is None:
                raise ValueError("T1GR_G_OPTIMIZER_UNKNOWN_PARAMETER")
            if name in assignments:
                raise ValueError("T1GR_G_OPTIMIZER_DUPLICATE_PARAMETER")
            assignments[name] = index
            group_names.append(name)
        groups.append({
            "index": index,
            "parameter_names": sorted(group_names),
            "lr": float(group.get("lr", 0.0)),
            "initial_lr": float(group.get("initial_lr", group.get("lr", 0.0))),
            "momentum": None if "momentum" not in group else float(group["momentum"]),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "param_group": group.get("param_group"),
        })
    all_parameters = sorted(name for name, _ in model.named_parameters())
    trainable = sorted(name for name, p in model.named_parameters() if p.requires_grad)
    if set(assignments) != set(all_parameters):
        missing = sorted(set(all_parameters) - set(assignments))
        extra = sorted(set(assignments) - set(all_parameters))
        raise ValueError(f"T1GR_G_OPTIMIZER_MEMBERSHIP_FAIL:missing={missing}:extra={extra}")
    contract = {
        "class_name": type(optimizer).__name__,
        "groups": groups,
        "assignments": dict(sorted(assignments.items())),
        "all_parameter_names": all_parameters,
        "trainable_parameter_names": trainable,
    }
    contract["contract_sha256"] = sha256_json(contract)
    return contract


def parse_component_map(obj: Mapping[str, Any], dev_ids: Sequence[str]) -> dict[str, str]:
    dev = set(str(x) for x in dev_ids)
    out: dict[str, str] = {}
    observed: set[str] = set()
    direct = obj.get("component_by_id")
    if isinstance(direct, dict):
        observed = {str(k) for k in direct}
        out = {str(k): str(v) for k, v in direct.items()}
    elif isinstance(obj.get("assignments"), list):
        for row in obj["assignments"]:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("sample_id", ""))
            component = str(row.get("component_id", row.get("component", "")))
            observed.add(sid)
            out[sid] = component
    elif isinstance(obj.get("components"), list):
        for row in obj["components"]:
            if not isinstance(row, dict):
                continue
            component = str(row.get("component_id", row.get("id", "")))
            members = row.get("member_ids", row.get("members", []))
            if not isinstance(members, list):
                continue
            for value in members:
                sid = str(value)
                observed.add(sid)
                out[sid] = component
    if observed != dev or set(out) != dev or any(not value for value in out.values()):
        raise ValueError("T1GR_G_DEV_COMPONENT_MAP_INCOMPLETE")
    return out


def finite_metric(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("T1GR_G_METRIC_TYPE_FAIL")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("T1GR_G_METRIC_RANGE_FAIL")
    return value
