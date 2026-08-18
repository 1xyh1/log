"""A3-A registration diagnostics: raw-space phase correlation + LOO cross-fit shift."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from multimodal.step4_a3_common import zero_fill_translate


@dataclass(frozen=True)
class RawShift:
    dx: float
    dy: float
    phase_response: float


def rgb_gray(rgb_chw: np.ndarray) -> np.ndarray:
    if rgb_chw.shape[0] != 3:
        raise ValueError("rgb_gray expects 3xHxW")
    r, g, b = rgb_chw.astype(np.float64)
    return 0.299 * r + 0.587 * g + 0.114 * b


def sobel_magnitude(x: np.ndarray) -> np.ndarray:
    """Small dependency-free Sobel magnitude, same-size with edge replication."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("sobel_magnitude expects HxW")
    p = np.pad(x, 1, mode="edge")
    gx = (
        -p[:-2, :-2] + p[:-2, 2:]
        -2*p[1:-1, :-2] + 2*p[1:-1, 2:]
        -p[2:, :-2] + p[2:, 2:]
    )
    gy = (
        -p[:-2, :-2] - 2*p[:-2, 1:-1] - p[:-2, 2:]
        +p[2:, :-2] + 2*p[2:, 1:-1] + p[2:, 2:]
    )
    return np.sqrt(gx*gx + gy*gy)


def standardize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + eps)


def valid_content_slices(sample: Mapping) -> tuple[slice, slice]:
    """Crop the common non-letterbox region to prevent pad edges driving registration."""
    ratio_pad = sample["ratio_pad"]
    # Dataset sample stores ((r,r),(left,top)).
    (r, _), (left, top) = ratio_pad
    h, w = sample["ori_shape"]
    new_h = int(round(float(h) * float(r)))
    new_w = int(round(float(w) * float(r)))
    return slice(int(top), int(top) + new_h), slice(int(left), int(left) + new_w)


def _phase_peak(ref: np.ndarray, moving: np.ndarray) -> tuple[int, int, float]:
    """Return integer shift to APPLY to moving so it aligns with ref."""
    if ref.shape != moving.shape:
        raise ValueError("phase correlation shape mismatch")
    f_ref = np.fft.fft2(ref)
    f_mov = np.fft.fft2(moving)
    cross = f_ref * np.conj(f_mov)
    denom = np.abs(cross)
    cross = cross / np.maximum(denom, 1e-12)
    corr = np.fft.ifft2(cross).real
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = corr.shape
    # Circular correlation index -> signed translation to apply to moving.
    dy = int(iy if iy <= h // 2 else iy - h)
    dx = int(ix if ix <= w // 2 else ix - w)
    response = float(corr[iy, ix] / (np.mean(np.abs(corr)) + 1e-12))
    return dx, dy, response


def estimate_registration_shift(sample: Mapping) -> RawShift:
    """Estimate raw RGB<-IR translation from frozen preprocessed recipient planes.

    No GT, prediction, AP, model feature, or donor information is consumed.
    """
    img = sample["img"]
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[0] < 4:
        raise ValueError("sample img must be >=4xHxW")
    ys, xs = valid_content_slices(sample)
    rgb = img[:3, ys, xs]
    ir = img[3, ys, xs]
    g_rgb = standardize(sobel_magnitude(rgb_gray(rgb)))
    g_ir = standardize(sobel_magnitude(ir))
    dx, dy, response = _phase_peak(g_rgb, g_ir)
    return RawShift(dx=float(dx), dy=float(dy), phase_response=response)


def cross_fitted_median_shifts(raw: Mapping[str, RawShift]) -> dict[str, dict]:
    ids = list(raw.keys())
    if len(ids) < 2:
        raise ValueError("cross-fitting needs >=2 samples")
    out = {}
    for held_out in ids:
        train_ids = [sid for sid in ids if sid != held_out]
        dx = float(np.median([raw[sid].dx for sid in train_ids]))
        dy = float(np.median([raw[sid].dy for sid in train_ids]))
        out[held_out] = {
            "held_out": held_out,
            "train_ids_for_shift": train_ids,
            "median_shift": {"dx": dx, "dy": dy},
        }
    return out


def raw_shift_to_feature_cells(dx: float, dy: float, stride: int) -> tuple[int, int]:
    if int(stride) <= 0:
        raise ValueError("stride must be positive")
    return int(round(float(dx) / int(stride))), int(round(float(dy) / int(stride)))


def shift_surface(
    reference_hw: torch.Tensor,
    moving_hw: torch.Tensor,
    radius: int = 2,
) -> dict:
    """Feature-space descriptive correlation surface. Never use this for AP rescue."""
    if reference_hw.ndim != 2 or moving_hw.ndim != 2:
        raise ValueError("shift_surface expects HxW")
    if tuple(reference_hw.shape) != tuple(moving_hw.shape):
        raise ValueError("shift_surface shape mismatch")
    h, w = reference_hw.shape
    rows = {}
    best = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # Correlate only the overlapping support, not zero-filled padding.
            y0r, y1r = max(0, dy), min(h, h + dy)
            x0r, x1r = max(0, dx), min(w, w + dx)
            y0m, y1m = max(0, -dy), min(h, h - dy)
            x0m, x1m = max(0, -dx), min(w, w - dx)
            a = reference_hw[y0r:y1r, x0r:x1r].float().reshape(-1)
            b = moving_hw[y0m:y1m, x0m:x1m].float().reshape(-1)
            aa, bb = a-a.mean(), b-b.mean()
            den = torch.sqrt(torch.sum(aa*aa) * torch.sum(bb*bb))
            corr = 0.0 if float(den) <= 1e-12 else float(torch.sum(aa*bb)/den)
            key = f"{dx},{dy}"
            rows[key] = corr
            if best is None or corr > best[2]:
                best = (dx, dy, corr)
    return {
        "radius": int(radius),
        "corr_at_zero": float(rows["0,0"]),
        "best_feature_shift": {"dx": int(best[0]), "dy": int(best[1])},
        "corr_at_best": float(best[2]),
        "best_minus_zero": float(best[2] - rows["0,0"]),
        "surface": rows,
    }
