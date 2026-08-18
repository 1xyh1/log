"""A3-C object-region semantic agreement metrics."""
from __future__ import annotations

import math
import numpy as np
import torch

from multimodal.step4_a3_common import energy_map, summarize_signed


def boxes_xywhn_to_mask(
    boxes: torch.Tensor | np.ndarray,
    feature_hw: tuple[int, int],
) -> torch.Tensor:
    """Normalized xywh boxes in final letterbox space -> union feature-cell mask."""
    b = torch.as_tensor(boxes, dtype=torch.float32)
    fh, fw = map(int, feature_hw)
    mask = torch.zeros((fh, fw), dtype=torch.bool)
    if b.numel() == 0:
        return mask
    b = b.reshape(-1, 4)
    for cx, cy, bw, bh in b.tolist():
        x0 = max(0.0, min(1.0, cx - bw / 2.0))
        y0 = max(0.0, min(1.0, cy - bh / 2.0))
        x1 = max(0.0, min(1.0, cx + bw / 2.0))
        y1 = max(0.0, min(1.0, cy + bh / 2.0))
        ix0 = max(0, min(fw, int(math.floor(x0 * fw))))
        iy0 = max(0, min(fh, int(math.floor(y0 * fh))))
        ix1 = max(0, min(fw, int(math.ceil(x1 * fw))))
        iy1 = max(0, min(fh, int(math.ceil(y1 * fh))))
        if ix1 > ix0 and iy1 > iy0:
            mask[iy0:iy1, ix0:ix1] = True
    return mask


def binary_auroc(scores: torch.Tensor, target: torch.Tensor) -> float:
    """Tie-corrected AUROC via average ranks / Mann-Whitney U."""
    s = scores.detach().cpu().float().reshape(-1).numpy()
    y = target.detach().cpu().bool().reshape(-1).numpy()
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError("SEMANTIC_MASK_DEGENERATE")

    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos = ranks[y].sum()
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def enrichment(energy: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> float:
    if mask.sum() == 0 or (~mask).sum() == 0:
        raise RuntimeError("SEMANTIC_MASK_DEGENERATE")
    obj = float(energy[mask].mean())
    bg = float(energy[~mask].mean())
    return float(obj / (bg + eps))


def semantic_row(boxes, native_residual: torch.Tensor, donor_residual: torch.Tensor) -> dict:
    en = energy_map(native_residual)
    ed = energy_map(donor_residual)
    mask = boxes_xywhn_to_mask(boxes, tuple(en.shape))
    if int(mask.sum()) == 0 or int((~mask).sum()) == 0:
        raise RuntimeError("SEMANTIC_MASK_DEGENERATE")
    an = binary_auroc(en, mask)
    ad = binary_auroc(ed, mask)
    ern = enrichment(en, mask)
    erd = enrichment(ed, mask)
    return {
        "valid": True,
        "object_cells": int(mask.sum()),
        "background_cells": int((~mask).sum()),
        "auroc_native": an,
        "auroc_donor": ad,
        "delta_native_minus_donor": float(an - ad),
        "enrichment_native": ern,
        "enrichment_donor": erd,
        "log_enrichment_delta": float(math.log(max(ern, 1e-12)) - math.log(max(erd, 1e-12))),
    }


def summarize_semantic_rows(rows: dict[str, dict], expected_n: int = 6) -> dict:
    valid = {sid: row for sid, row in rows.items() if row.get("valid") is True}
    if len(valid) < max(1, expected_n - 1):
        raise RuntimeError(f"A3_SEMANTIC_COVERAGE_FAIL:{len(valid)}/{expected_n}")
    delta = {sid: float(row["delta_native_minus_donor"]) for sid, row in valid.items()}
    return {
        "valid_count": len(valid),
        "expected_count": expected_n,
        "per_sample": rows,
        "delta_summary": summarize_signed(delta),
    }
