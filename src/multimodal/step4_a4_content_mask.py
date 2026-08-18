"""A4 padding-aware content-mask helpers.

The content mask is geometric metadata only: ori_shape + ratio_pad.  It never reads
GT, predictions, AP, or feature-correlation results.  Feature-grid coverage uses
area-preserving adaptive average pooling so boundary cells retain fractional support.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def content_rectangle(
    ori_shape: tuple[int, int] | list[int],
    ratio_pad,
    input_hw: tuple[int, int] | list[int],
) -> dict[str, int]:
    """Recover the non-letterbox rectangle in the final model input grid."""
    h, w = int(ori_shape[0]), int(ori_shape[1])
    ih, iw = int(input_hw[0]), int(input_hw[1])
    (rh, rw), (left, top) = ratio_pad
    # Frozen dataset stores identical gains, but consume both explicitly.
    new_h = int(round(h * float(rh)))
    new_w = int(round(w * float(rw)))
    left, top = int(left), int(top)
    bottom = top + new_h
    right = left + new_w
    if not (0 <= top < bottom <= ih and 0 <= left < right <= iw):
        raise RuntimeError(
            f"A4_CONTENT_MASK_RECT_INVALID:{(top,bottom,left,right)}:{(ih,iw)}"
        )
    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "height": new_h,
        "width": new_w,
    }


def input_content_mask(
    ori_shape,
    ratio_pad,
    input_hw,
    *,
    device=None,
    dtype=torch.float32,
) -> tuple[torch.Tensor, dict]:
    """Return 1x1xH xW binary mask and provenance evidence."""
    rect = content_rectangle(ori_shape, ratio_pad, input_hw)
    ih, iw = int(input_hw[0]), int(input_hw[1])
    mask = torch.zeros((1, 1, ih, iw), device=device, dtype=dtype)
    mask[..., rect["top"]:rect["bottom"], rect["left"]:rect["right"]] = 1
    evidence = {
        "source": "ori_shape+ratio_pad",
        "ori_shape": [int(ori_shape[0]), int(ori_shape[1])],
        "input_hw": [ih, iw],
        "ratio_pad": [
            [float(ratio_pad[0][0]), float(ratio_pad[0][1])],
            [int(ratio_pad[1][0]), int(ratio_pad[1][1])],
        ],
        "content_rect": rect,
        "input_content_fraction": float(mask.mean().item()),
    }
    return mask, evidence


def feature_content_coverage(
    ori_shape,
    ratio_pad,
    input_hw,
    feature_hw,
    *,
    device=None,
    dtype=torch.float32,
) -> tuple[torch.Tensor, dict]:
    """Area-weighted content coverage in a feature grid, shape 1x1xHf xWf."""
    input_mask, evidence = input_content_mask(
        ori_shape, ratio_pad, input_hw, device=device, dtype=dtype
    )
    fh, fw = int(feature_hw[0]), int(feature_hw[1])
    if fh <= 0 or fw <= 0:
        raise RuntimeError(f"A4_CONTENT_MASK_FEATURE_SHAPE_INVALID:{(fh,fw)}")
    coverage = F.adaptive_avg_pool2d(input_mask, (fh, fw))
    total = float(coverage.sum().item())
    if total <= 0:
        raise RuntimeError("A4_CONTENT_DC_COVERAGE_FAIL:ZERO")
    if bool((coverage < 0).any()) or bool((coverage > 1).any()):
        raise RuntimeError("A4_CONTENT_DC_COVERAGE_FAIL:RANGE")
    full = bool(torch.all(coverage == 1).item())
    evidence = {
        **evidence,
        "feature_hw": [fh, fw],
        "coverage_sum": total,
        "coverage_mean": float(coverage.mean().item()),
        "coverage_min": float(coverage.min().item()),
        "coverage_max": float(coverage.max().item()),
        "full_content_coverage": full,
        "definition": "adaptive_avg_pool2d(binary_letterbox_content_mask)",
    }
    return coverage, evidence


def sample_meta(sample: Mapping) -> dict:
    """Extract only the metadata A4 is allowed to use for content masks."""
    img = sample["img"]
    return {
        "ori_shape": tuple(int(v) for v in sample["ori_shape"]),
        "ratio_pad": sample["ratio_pad"],
        "input_hw": (int(img.shape[-2]), int(img.shape[-1])),
    }
