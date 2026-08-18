"""A3-B feature spatial correspondence metrics."""
from __future__ import annotations

import torch

from multimodal.step4_a3_common import energy_map, pearson_corr, summarize_signed
from multimodal.step4_a3_registration import shift_surface


def spatial_row(rgb_feature: torch.Tensor, native_residual: torch.Tensor,
                donor_residual: torch.Tensor) -> dict:
    er = energy_map(rgb_feature)
    en = energy_map(native_residual)
    ed = energy_map(donor_residual)
    native_corr = pearson_corr(er, en)
    donor_corr = pearson_corr(er, ed)
    surf = shift_surface(er, en, radius=2)
    return {
        "corr_native": native_corr,
        "corr_donor": donor_corr,
        "delta_native_minus_donor": float(native_corr - donor_corr),
        "native_shift_surface": surf,
    }


def summarize_spatial_rows(rows: dict[str, dict]) -> dict:
    delta = {sid: float(row["delta_native_minus_donor"]) for sid, row in rows.items()}
    return {"per_sample": rows, "delta_summary": summarize_signed(delta)}
