"""Independent modality-quality diagnostics for the frozen 6-channel contract.

These measurements borrow EvaNet's *evaluation discipline* (separate information
preservation/quality assessment from the fusion model), not its fused-image network.
They never alter training inputs and do not produce a claimed perceptual quality
score.  The output is descriptive evidence for missing, flat, clipped, noisy, or
invalid competition inputs.
"""
from __future__ import annotations

import numpy as np


def content_mask_from_sample(sample: dict, imgsz: int | None = None) -> np.ndarray:
    """Rebuild the real-image region from the existing LetterBox metadata."""
    plane = np.asarray(sample["img"])[0]
    size = int(imgsz or plane.shape[-1])
    h, w = (int(x) for x in sample["ori_shape"])
    ratio_pad = sample["ratio_pad"]
    ratio = float(ratio_pad[0][0])
    left, top = (int(x) for x in ratio_pad[1])
    new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
    mask = np.zeros((size, size), dtype=bool)
    mask[top:min(size, top + new_h), left:min(size, left + new_w)] = True
    return mask


def describe_plane(plane: np.ndarray, content_mask: np.ndarray | None = None,
                   valid_mask: np.ndarray | None = None) -> dict:
    """Return robust, interpretable scalar statistics for one normalized plane."""
    x = np.asarray(plane, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected HxW plane, got {x.shape}")
    mask = np.ones(x.shape, dtype=bool) if content_mask is None else np.asarray(
        content_mask, dtype=bool
    ).copy()
    if mask.shape != x.shape:
        raise ValueError("content mask shape mismatch")
    if valid_mask is not None:
        vm = np.asarray(valid_mask) >= 0.5
        if vm.shape != x.shape:
            raise ValueError("valid mask shape mismatch")
        mask &= vm

    content_count = int(mask.sum())
    finite = np.isfinite(x)
    finite_fraction = float(finite[mask].mean()) if content_count else 0.0
    usable = mask & finite
    values = x[usable]
    if values.size == 0:
        return {
            "content_pixels": content_count,
            "usable_pixels": 0,
            "finite_fraction": finite_fraction,
            "nonzero_fraction": 0.0,
            "mean": None,
            "std": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "dynamic_range_p99_p01": None,
            "low_clip_fraction": None,
            "high_clip_fraction": None,
            "gradient_abs_mean": None,
        }

    p01, p50, p99 = (float(v) for v in np.quantile(values, (0.01, 0.5, 0.99)))
    horizontal_valid = usable[:, 1:] & usable[:, :-1]
    vertical_valid = usable[1:, :] & usable[:-1, :]
    gradients = []
    if horizontal_valid.any():
        gradients.append(np.abs(x[:, 1:] - x[:, :-1])[horizontal_valid])
    if vertical_valid.any():
        gradients.append(np.abs(x[1:, :] - x[:-1, :])[vertical_valid])
    gradient_abs_mean = float(np.concatenate(gradients).mean()) if gradients else 0.0
    return {
        "content_pixels": content_count,
        "usable_pixels": int(values.size),
        "finite_fraction": finite_fraction,
        "nonzero_fraction": float(np.count_nonzero(values) / values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": p01,
        "p50": p50,
        "p99": p99,
        "dynamic_range_p99_p01": p99 - p01,
        "low_clip_fraction": float(np.mean(values <= 0.01)),
        "high_clip_fraction": float(np.mean(values >= 0.99)),
        "gradient_abs_mean": gradient_abs_mean,
    }


def describe_trimodal_sample(sample: dict) -> dict:
    """Describe RGB luminance, IR, Depth, and depth validity independently."""
    img = np.asarray(sample["img"], dtype=np.float32)
    if img.ndim != 3 or img.shape[0] != 6:
        raise ValueError(f"expected 6xHxW frozen tensor, got {img.shape}")
    content = content_mask_from_sample(sample, imgsz=img.shape[-1])
    rgb_luma = (0.2126 * img[0] + 0.7152 * img[1] + 0.0722 * img[2])
    depth_valid = img[5] >= 0.5
    valid_content = content & depth_valid
    return {
        "sample_id": str(sample["sample_id"]),
        "aux_sample_id": str(sample.get("aux_sample_id", sample["sample_id"])),
        "rgb_luma": describe_plane(rgb_luma, content),
        "ir": describe_plane(img[3], content),
        "depth": describe_plane(img[4], content, depth_valid),
        "depth_valid_ratio": float(valid_content.sum() / max(1, int(content.sum()))),
    }


def diagnostic_flags(stats: dict) -> list[str]:
    """Conservative flags; these are diagnostics, never labels or gate targets."""
    flags = []
    for name in ("rgb_luma", "ir"):
        s = stats[name]
        if s["finite_fraction"] < 1.0:
            flags.append(f"{name}:NONFINITE")
        if s["nonzero_fraction"] < 0.01:
            flags.append(f"{name}:MISSING_OR_ZERO")
        dynamic = s["dynamic_range_p99_p01"]
        if dynamic is not None and dynamic < 0.03:
            flags.append(f"{name}:LOW_DYNAMIC_RANGE")
        if s["low_clip_fraction"] is not None and s["low_clip_fraction"] > 0.95:
            flags.append(f"{name}:MOSTLY_LOW_CLIPPED")
        if s["high_clip_fraction"] is not None and s["high_clip_fraction"] > 0.95:
            flags.append(f"{name}:MOSTLY_HIGH_CLIPPED")
    if stats["depth_valid_ratio"] < 0.01:
        flags.append("depth:MISSING_OR_INVALID")
    return flags
