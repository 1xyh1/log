from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.modality_quality import (  # noqa: E402
    content_mask_from_sample,
    describe_plane,
    describe_trimodal_sample,
    diagnostic_flags,
)


def _sample() -> dict:
    img = np.zeros((6, 8, 8), dtype=np.float32)
    img[3] = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    img[4] = 0.5
    img[5] = 1.0
    return {
        "img": img,
        "ori_shape": (4, 8),
        "ratio_pad": ((1.0, 1.0), (0, 2)),
        "sample_id": "quality-test",
    }


def test_content_mask_uses_letterbox_metadata():
    mask = content_mask_from_sample(_sample())
    assert mask.shape == (8, 8)
    assert int(mask.sum()) == 32
    assert not mask[:2].any() and not mask[6:].any()


def test_describe_trimodal_sample_and_flags_are_diagnostic_only():
    stats = describe_trimodal_sample(_sample())
    assert stats["sample_id"] == "quality-test"
    assert stats["depth_valid_ratio"] == 1.0
    assert stats["ir"]["finite_fraction"] == 1.0
    assert "rgb_luma:MISSING_OR_ZERO" in diagnostic_flags(stats)


def test_describe_plane_respects_valid_mask_and_nonfinite_values():
    plane = np.array([[1.0, np.nan], [0.0, 0.5]], dtype=np.float32)
    valid = np.array([[1, 1], [0, 1]], dtype=np.float32)
    stats = describe_plane(plane, valid_mask=valid)
    assert stats["content_pixels"] == 3
    assert stats["usable_pixels"] == 2
    assert stats["finite_fraction"] == 2 / 3
