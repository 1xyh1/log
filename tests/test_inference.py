import cv2
import numpy as np
import pytest
import torch

from mmod_qaf.data import DataConfig, TriModalDataset
from mmod_qaf.inference import (
    TriModalInferenceDataset,
    preprocess_config_from_checkpoint,
    restore_boxes_from_letterbox,
)


def _checkpoint_with_data(train: dict, val: dict | None = None) -> dict:
    data = {"train": train}
    if val is not None:
        data["val"] = val
    return {"train_cfg": {"data": data}}


def test_preprocess_config_is_restored_from_checkpoint_and_imgsz_can_be_overridden():
    data = {
        "root": "data/train",
        "imgsz": 704,
        "ir_mode": "first_channel",
        "depth_min_mm": 450,
        "depth_max_mm": 17500,
        "depth_resize": "nearest",
    }
    checkpoint = _checkpoint_with_data(data, data.copy())

    restored = preprocess_config_from_checkpoint(checkpoint)
    assert restored.imgsz == 704
    assert restored.ir_mode == "first_channel"
    assert restored.depth_min_mm == 450
    assert restored.depth_max_mm == 17500
    assert restored.depth_resize == "nearest"

    overridden = preprocess_config_from_checkpoint(checkpoint, imgsz_override=896)
    assert overridden.imgsz == 896
    assert overridden.ir_mode == restored.ir_mode
    assert overridden.depth_min_mm == restored.depth_min_mm
    assert overridden.depth_max_mm == restored.depth_max_mm
    assert overridden.depth_resize == restored.depth_resize


def test_implicit_training_defaults_are_visible_instead_of_silent():
    data = {"root": "data/train", "imgsz": 768, "depth_resize": "valid_bilinear"}
    checkpoint = _checkpoint_with_data(data, data.copy())

    with pytest.warns(RuntimeWarning, match="does not explicitly record"):
        restored = preprocess_config_from_checkpoint(checkpoint)

    assert restored.ir_mode == "gray"
    assert restored.depth_min_mm == 300
    assert restored.depth_max_mm == 19999


def test_critical_train_val_preprocess_mismatch_is_rejected():
    train = {
        "root": "data/train",
        "imgsz": 768,
        "ir_mode": "gray",
        "depth_min_mm": 300,
        "depth_max_mm": 19999,
        "depth_resize": "valid_bilinear",
    }
    val = {**train, "depth_resize": "nearest"}

    with pytest.raises(ValueError, match="train/val preprocessing mismatch for depth_resize"):
        preprocess_config_from_checkpoint(_checkpoint_with_data(train, val))


def test_different_train_val_imgsz_requires_explicit_override():
    train = {
        "root": "data/train",
        "imgsz": 768,
        "ir_mode": "gray",
        "depth_min_mm": 300,
        "depth_max_mm": 19999,
        "depth_resize": "valid_bilinear",
    }
    val = {**train, "imgsz": 960}
    checkpoint = _checkpoint_with_data(train, val)

    with pytest.raises(ValueError, match="pass --imgsz explicitly"):
        preprocess_config_from_checkpoint(checkpoint)
    assert preprocess_config_from_checkpoint(checkpoint, imgsz_override=832).imgsz == 832


def test_inference_pixels_match_training_preprocess_without_augmentation(tmp_path):
    for name in ("visible", "infrared", "depth", "labels"):
        (tmp_path / name).mkdir()

    height, width = 4, 6
    rgb_bgr = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    ir_bgr = np.stack(
        (
            np.full((height, width), 20, dtype=np.uint8),
            np.full((height, width), 60, dtype=np.uint8),
            np.full((height, width), 100, dtype=np.uint8),
        ),
        axis=2,
    )
    depth = np.array(
        [
            [0, 600, 900, 1200, 1800, 2200],
            [400, 650, 950, 1250, 1850, 2250],
            [450, 700, 1000, 1300, 1900, 2300],
            [500, 750, 1050, 1350, 1950, 65535],
        ],
        dtype=np.uint16,
    )
    assert cv2.imwrite(str(tmp_path / "visible" / "sample.png"), rgb_bgr)
    assert cv2.imwrite(str(tmp_path / "infrared" / "sample.png"), ir_bgr)
    assert cv2.imwrite(str(tmp_path / "depth" / "sample.png"), depth)
    (tmp_path / "labels" / "sample.txt").write_text("", encoding="utf-8")

    common = {
        "imgsz": 8,
        "ir_mode": "first_channel",
        "depth_min_mm": 500,
        "depth_max_mm": 2300,
        "depth_resize": "nearest",
    }
    training_item = TriModalDataset(DataConfig(root=tmp_path, augment=False, **common))[0]
    inference_item = TriModalInferenceDataset(tmp_path, **common)[0]

    torch.testing.assert_close(inference_item["img"], training_item["img"], rtol=0, atol=0)
    torch.testing.assert_close(inference_item["quality"], training_item["quality"], rtol=0, atol=0)
    assert inference_item["ori_shape"] == training_item["ori_shape"]
    assert inference_item["ratio_pad"] == training_item["ratio_pad"]


def test_restore_boxes_from_landscape_letterbox_and_clip_boundaries():
    # Original 200x100 -> 320x160 on a 320 square canvas, with 80 px top/bottom padding.
    boxes = np.array(
        [
            [32.0, 96.0, 288.0, 224.0],
            [-50.0, 20.0, 370.0, 300.0],
        ],
        dtype=np.float32,
    )
    restored = restore_boxes_from_letterbox(boxes, (100, 200), (1.6, (0, 80, 0, 80)))
    expected = np.array([[20.0, 10.0, 180.0, 90.0], [0.0, 0.0, 200.0, 100.0]], dtype=np.float32)

    np.testing.assert_allclose(restored, expected, atol=1e-5)
    np.testing.assert_array_equal(boxes[0], [32.0, 96.0, 288.0, 224.0])


def test_restore_boxes_from_portrait_letterbox_and_empty_input():
    # Original 100x200 -> 160x320 on a 320 square canvas, with 80 px left/right padding.
    boxes = np.array([[96.0, 32.0, 224.0, 288.0]], dtype=np.float32)
    restored = restore_boxes_from_letterbox(boxes, (200, 100), (1.6, (80, 0, 80, 0)))
    np.testing.assert_allclose(restored, [[10.0, 20.0, 90.0, 180.0]], atol=1e-5)

    empty = restore_boxes_from_letterbox(np.empty((0, 4), dtype=np.float32), (200, 100), (1.6, (80, 0, 80, 0)))
    assert empty.shape == (0, 4)
