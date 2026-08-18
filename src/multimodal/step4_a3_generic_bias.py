"""A3-D generic residual bias decomposition."""
from __future__ import annotations

from typing import Mapping
import torch


def native_dc(residual: torch.Tensor) -> torch.Tensor:
    if residual.ndim != 4:
        raise ValueError("native_dc expects BCHW")
    return residual.mean(dim=(-2, -1), keepdim=True).expand_as(residual)


def native_ac(residual: torch.Tensor) -> torch.Tensor:
    return residual - native_dc(residual)


def loo_mean_residual(
    residuals_by_id: Mapping[str, torch.Tensor],
    held_out: str,
) -> tuple[torch.Tensor, list[str]]:
    ids = [sid for sid in residuals_by_id if sid != held_out]
    if held_out not in residuals_by_id:
        raise KeyError(held_out)
    if not ids:
        raise ValueError("LOO mean needs donors")
    tensors = [residuals_by_id[sid] for sid in ids]
    shape = tuple(tensors[0].shape)
    if any(tuple(t.shape) != shape for t in tensors):
        raise RuntimeError("A3_LOO_MEAN_SHAPE_MISMATCH")
    return torch.stack(tensors, dim=0).mean(dim=0), ids


def loo_mean_dc(
    residuals_by_id: Mapping[str, torch.Tensor],
    held_out: str,
) -> tuple[torch.Tensor, list[str]]:
    mean, ids = loo_mean_residual(residuals_by_id, held_out)
    # Definition in freeze is average donor per-channel DC. Linearity makes this
    # exactly equal to DC(mean), while preserving the no-self donor set.
    return native_dc(mean), ids


def build_generic_components(
    residuals_by_id: Mapping[str, torch.Tensor],
    held_out: str,
) -> dict[str, tuple[torch.Tensor, list[str] | None]]:
    native = residuals_by_id[held_out]
    mean, ids = loo_mean_residual(residuals_by_id, held_out)
    mean_dc, dc_ids = loo_mean_dc(residuals_by_id, held_out)
    if ids != dc_ids:
        raise RuntimeError("A3_LOO_MEAN_DC_DONOR_SET_MISMATCH")
    return {
        "LOO_MEAN": (mean, ids),
        "NATIVE_DC": (native_dc(native), None),
        "NATIVE_AC": (native_ac(native), None),
        "LOO_MEAN_DC": (mean_dc, ids),
    }


def classify_generic_labels(effects: dict) -> dict:
    """Consume already cross-system-classified AP effects."""
    mean_label = effects["U_mean"]["label"]
    native_mean_label = effects["native_minus_mean"]["label"]
    dc_native = effects["U_dc"]["label"]
    dc_mean = effects["U_meanDC"]["label"]
    ac = effects["U_ac"]["label"]

    generic = (
        "GENERIC_COMPONENT_SUPPORTED"
        if mean_label == "STRONG_POSITIVE" and native_mean_label != "STRONG_POSITIVE"
        else "NOT_ESTABLISHED"
    )
    generic_dc = (
        "GENERIC_DC_SUPPORTED"
        if dc_native == "STRONG_POSITIVE" or dc_mean == "STRONG_POSITIVE"
        else "NOT_ESTABLISHED"
    )
    spatial_ac = (
        "SPATIAL_AC_SUPPORTED"
        if ac == "STRONG_POSITIVE"
        and dc_native != "STRONG_POSITIVE"
        and dc_mean != "STRONG_POSITIVE"
        else "NOT_ESTABLISHED"
    )
    return {
        "generic_component": generic,
        "generic_dc": generic_dc,
        "spatial_ac": spatial_ac,
    }
