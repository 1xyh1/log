"""Pure helpers for the T-series P5-only direct IR injection experiment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Iterable
import hashlib
import json
import math

import numpy as np
import torch

TREATMENTS = {
    "T0-N": "NULL",
    "T1-F": "FULL",
    "T2-A": "AC_ALL",
}
RUN_NAMES = {
    "T0-N": "T0-N_P5_NULL_seed20260812",
    "T1-F": "T1-F_P5_FULL_seed20260812",
    "T2-A": "T2-A_P5_ACALL_seed20260812",
}
FORMAL_SEED = 20260812
FORMAL_EPOCHS = 80
FORMAL_BATCH = 4
A5_ACCEPTED_COMMIT = "f154c1ff9af6d31e60bc2c9a2c4fd5baafc3d8b8"
A5_SUMMARY_RAW_SHA256 = "f1dbd1bc828b55674406337a12add25dc0a1cdd3ee96ad46c3c9976014cb7950"
A5_SUMMARY_CANONICAL_LF_SHA256 = "0e3ebb5cc64362ee44a0de68899885f36842ac8f8c40556d26cd541002347915"
A5_SUMMARY_GIT_BLOB_SHA1 = "e5136b4e9680d54aa040f4afba3fc6585d6ae13f"
A4_ORIGINAL_FEEDBACK_SHA256 = "3bd2331d3e618f280b6c8a67699a93780aef1806a09c86bdad2b88ece8dd434a"

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical_lf_sha256(path) -> str:
    return sha256_bytes(open(path, "rb").read().replace(b"\r\n", b"\n"))

def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))

def tensor_sha256(t: torch.Tensor) -> str:
    x = t.detach().cpu().contiguous()
    return sha256_bytes(x.numpy().tobytes())

def state_sha256(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def parameter_sha256(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(module.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def center_full_map(delta: torch.Tensor) -> torch.Tensor:
    if delta.ndim != 4:
        raise ValueError("center_full_map expects BCHW")
    return delta - delta.mean(dim=(-2, -1), keepdim=True)

def full_map_dc(delta: torch.Tensor) -> torch.Tensor:
    if delta.ndim != 4:
        raise ValueError("full_map_dc expects BCHW")
    return delta.mean(dim=(-2, -1), keepdim=True).expand_as(delta)

def rms(t: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(t.detach().float() ** 2)).item())

def p5_mechanism_metrics(delta: torch.Tensor, used: torch.Tensor) -> dict:
    dc = full_map_dc(delta)
    ac = delta - dc
    e_full = rms(delta)
    e_dc = rms(dc)
    e_ac = rms(ac)
    return {
        "full_rms": e_full,
        "dc_rms": e_dc,
        "ac_rms": e_ac,
        "dc_over_full": 0.0 if e_full == 0 else e_dc / e_full,
        "ac_over_full": 0.0 if e_full == 0 else e_ac / e_full,
        "used_rms": rms(used),
        "post_center_channel_mean_abs_max": float(
            used.detach().float().mean(dim=(-2, -1)).abs().max().item()
        ),
    }

def apply_treatment(r5: torch.Tensor, delta: torch.Tensor, treatment_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    if treatment_id == "T0-N":
        # Important: callers must ensure delta is detached/no_grad from the loss graph.
        used = torch.zeros_like(delta)
        return r5, used
    if treatment_id == "T1-F":
        return r5 + delta, delta
    if treatment_id == "T2-A":
        used = center_full_map(delta)
        return r5 + used, used
    raise ValueError(f"unknown treatment: {treatment_id}")

def signed_summary(full: float, loo: Mapping[str, float]) -> dict:
    vals = [float(v) for v in loo.values()]
    if not vals:
        raise ValueError("LOO cannot be empty")
    return {
        "full": float(full),
        "loo": {str(k): float(v) for k, v in loo.items()},
        "loo_median": float(np.median(vals)),
        "positive_folds": int(sum(v > 0 for v in vals)),
        "negative_folds": int(sum(v < 0 for v in vals)),
        "zero_folds": int(sum(v == 0 for v in vals)),
    }

def effect_from_results(native: Mapping, donor: Mapping) -> dict:
    if set(native["loo"]) != set(donor["loo"]):
        raise RuntimeError("T_SERIES_PAIRED_LOO_ID_MISMATCH")
    full = float(native["full"]["map50_95"]) - float(donor["full"]["map50_95"])
    loo = {
        sid: float(native["loo"][sid]["map50_95"]) - float(donor["loo"][sid]["map50_95"])
        for sid in native["loo"]
    }
    return signed_summary(full, loo)

def single_seed_paired_label(effect: Mapping) -> str:
    if (
        effect["full"] > 0
        and effect["loo_median"] > 0
        and effect["positive_folds"] >= 4
    ):
        return "SEED20260812_POSITIVE_PAIRED_EVIDENCE"
    if (
        effect["full"] < 0
        and effect["loo_median"] < 0
        and effect["negative_folds"] >= 4
    ):
        return "SEED20260812_NEGATIVE_PAIRED_EVIDENCE"
    return "SEED20260812_INCONCLUSIVE_PAIRED_EVIDENCE"

def optimizer_group_snapshot(model: torch.nn.Module, optimizer) -> dict:
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    rows = []
    assignment = {}
    for idx, group in enumerate(optimizer.param_groups):
        names = []
        for p in group["params"]:
            n = name_by_id.get(id(p))
            if n is None:
                raise RuntimeError("T_SERIES_OPTIMIZER_UNKNOWN_PARAMETER")
            names.append(n)
            if n in assignment:
                raise RuntimeError(f"T_SERIES_OPTIMIZER_DUP_PARAMETER:{n}")
            assignment[n] = idx
        rows.append({
            "index": idx,
            "names": sorted(names),
            "lr": float(group.get("lr", 0.0)),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "momentum": None if "momentum" not in group else float(group["momentum"]),
        })
    return {
        "groups": rows,
        "assignment": assignment,
        "all_names": sorted(assignment),
    }

def optimizer_snapshots_equivalent(a: Mapping, b: Mapping) -> bool:
    return a["groups"] == b["groups"] and a["assignment"] == b["assignment"]

def grad_abs_max(param: torch.nn.Parameter) -> float | None:
    if param.grad is None:
        return None
    return float(param.grad.detach().abs().max().item())

def grad_norm(param: torch.nn.Parameter) -> float | None:
    if param.grad is None:
        return None
    return float(param.grad.detach().float().norm().item())

def assert_treatment_set(values: Iterable[str]) -> None:
    if set(values) != set(TREATMENTS):
        raise RuntimeError(f"T_SERIES_TREATMENT_SET_MISMATCH:{sorted(values)}")
