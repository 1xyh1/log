"""A4 post-projection DC/AC decomposition helpers."""
from __future__ import annotations

import hashlib
from typing import Mapping

import torch

from multimodal.step4_a4_content_mask import feature_content_coverage


def tensor_sha256(t: torch.Tensor) -> str:
    x = t.detach().cpu().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def full_map_dc(residual: torch.Tensor) -> torch.Tensor:
    if residual.ndim != 4:
        raise ValueError("full_map_dc expects BCHW")
    return residual.mean(dim=(-2, -1), keepdim=True).expand_as(residual)


def full_map_ac(residual: torch.Tensor) -> torch.Tensor:
    return residual - full_map_dc(residual)


def weighted_dc(residual: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
    """Per-channel DC using fractional spatial coverage weights."""
    if residual.ndim != 4 or coverage.ndim != 4:
        raise ValueError("weighted_dc expects BCHW residual and B/1x1xHxW coverage")
    if coverage.shape[-2:] != residual.shape[-2:]:
        raise RuntimeError(
            f"A4_CONTENT_MASK_SHAPE_MISMATCH:{tuple(coverage.shape)}:{tuple(residual.shape)}"
        )
    if coverage.shape[1] != 1:
        raise RuntimeError("A4_CONTENT_MASK_CHANNELS_NOT_ONE")
    if coverage.shape[0] not in (1, residual.shape[0]):
        raise RuntimeError("A4_CONTENT_MASK_BATCH_MISMATCH")
    w = coverage.to(device=residual.device, dtype=residual.dtype)
    denom = w.sum(dim=(-2, -1), keepdim=True)
    if bool((denom <= 0).any()):
        raise RuntimeError("A4_CONTENT_DC_COVERAGE_FAIL:ZERO")
    mean = (residual * w).sum(dim=(-2, -1), keepdim=True) / denom
    return mean.expand_as(residual)


def weighted_ac(residual: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
    return residual - weighted_dc(residual, coverage)


def _max_abs(t: torch.Tensor) -> float:
    return float(t.detach().abs().max().item()) if t.numel() else 0.0


def decompose_all(
    residual: torch.Tensor,
    *,
    source_id: str,
) -> tuple[torch.Tensor, dict]:
    dc = full_map_dc(residual)
    ac = residual - dc
    reconstruct = ac + dc
    evidence = {
        "mode": "AC_ALL",
        "residual_source_id": str(source_id),
        "mean_source_id": str(source_id),
        "content_mask_source_id": None,
        "source_residual_sha256": tensor_sha256(residual),
        "dc_sha256": tensor_sha256(dc),
        "ac_sha256": tensor_sha256(ac),
        "reconstruction_max_abs_error": _max_abs(reconstruct - residual),
        "ac_full_map_channel_mean_abs_max": _max_abs(ac.mean(dim=(-2, -1))),
        "dc_spatial_variation_abs_max": _max_abs(dc - dc[..., :1, :1]),
        "definition": "residual - mean_HW(residual)",
    }
    return ac, evidence


def decompose_content(
    residual: torch.Tensor,
    *,
    source_id: str,
    content_mask_source_id: str,
    meta: Mapping,
) -> tuple[torch.Tensor, dict]:
    coverage, mask_evidence = feature_content_coverage(
        meta["ori_shape"], meta["ratio_pad"], meta["input_hw"], residual.shape[-2:],
        device=residual.device, dtype=residual.dtype,
    )
    dc = weighted_dc(residual, coverage)
    ac = residual - dc
    reconstruct = ac + dc
    # Weighted content mean of AC should be ~0 by construction.
    w = coverage.to(device=ac.device, dtype=ac.dtype)
    ac_weighted_mean = (ac * w).sum(dim=(-2, -1)) / w.sum(dim=(-2, -1))
    evidence = {
        "mode": "AC_CONTENT",
        "residual_source_id": str(source_id),
        "mean_source_id": str(source_id),
        "content_mask_source_id": str(content_mask_source_id),
        "source_residual_sha256": tensor_sha256(residual),
        "dc_sha256": tensor_sha256(dc),
        "ac_sha256": tensor_sha256(ac),
        "reconstruction_max_abs_error": _max_abs(reconstruct - residual),
        "ac_content_weighted_channel_mean_abs_max": _max_abs(ac_weighted_mean),
        "dc_spatial_variation_abs_max": _max_abs(dc - dc[..., :1, :1]),
        "content_mask": mask_evidence,
        "definition": "residual - weighted_content_mean(residual, coverage)",
    }
    return ac, evidence


def validate_component_trace(trace: Mapping, *, tol: float = 1e-6) -> bool:
    if trace.get("mode") == "AC_ALL":
        return (
            trace.get("residual_source_id") == trace.get("mean_source_id")
            and float(trace.get("reconstruction_max_abs_error", 1)) <= tol
            and float(trace.get("ac_full_map_channel_mean_abs_max", 1)) <= tol
            and float(trace.get("dc_spatial_variation_abs_max", 1)) <= tol
        )
    if trace.get("mode") == "AC_CONTENT":
        mask = trace.get("content_mask") or {}
        return (
            trace.get("residual_source_id") == trace.get("mean_source_id")
            and trace.get("content_mask_source_id") == trace.get("residual_source_id")
            and mask.get("source") == "ori_shape+ratio_pad"
            and float(mask.get("coverage_sum", 0)) > 0
            and float(trace.get("reconstruction_max_abs_error", 1)) <= tol
            and float(trace.get("ac_content_weighted_channel_mean_abs_max", 1)) <= tol
            and float(trace.get("dc_spatial_variation_abs_max", 1)) <= tol
        )
    return False
