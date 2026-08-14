from pathlib import Path
import os

import cv2
import numpy as np
import torch

from mmod_qaf.data import (
    DataConfig,
    SampleRecord,
    TriModalDataset,
    _ir_quality,
    _resize_depth,
    _rgb_quality,
    collate_detection_batch,
    read_metric_depth,
)

SAMPLE_ROOT = Path(os.environ.get('MMOD_SAMPLE_ROOT', '/mnt/data/mmod_work/data/sample'))


def test_sample_dataset_shapes():
    if not SAMPLE_ROOT.exists():
        import pytest; pytest.skip('Set MMOD_SAMPLE_ROOT to run integration data tests')
    ds = TriModalDataset(DataConfig(root=SAMPLE_ROOT, imgsz=256, invalid_depth_policy='skip'))
    assert len(ds) == 17
    item = ds[0]
    assert item['img'].shape == (6, 256, 256)
    assert item['img'].dtype == torch.float32
    assert item['img'][5].min() >= 0 and item['img'][5].max() <= 1
    batch = collate_detection_batch([ds[0], ds[1]])
    assert batch['img'].shape == (2, 6, 256, 256)
    assert batch['bboxes'].shape[1] == 4


def test_invalid_depth_sample_is_rejected():
    if not SAMPLE_ROOT.exists():
        import pytest; pytest.skip('Set MMOD_SAMPLE_ROOT to run integration data tests')
    try:
        TriModalDataset(DataConfig(root=SAMPLE_ROOT, imgsz=256, invalid_depth_policy='error'))
    except ValueError as exc:
        assert '00000008' in str(exc)
    else:
        raise AssertionError('Expected invalid visual depth sample to fail')


def test_valid_weighted_resize_does_not_make_invalid_pixels_near():
    depth = np.array([[1000, 0], [1000, 0]], dtype=np.uint16)
    valid = depth > 0
    resized, mask = _resize_depth(depth, valid, (8, 8), 'valid_bilinear')
    assert np.all(resized[mask < 0.5] == 0)
    assert resized[mask > 0.5].min() > 900


def test_metric_depth_normalizes_singleton_channel_from_ultralytics_patch(monkeypatch, tmp_path):
    path = tmp_path / "depth.png"
    singleton = np.full((3, 4, 1), 1200, dtype=np.uint16)
    monkeypatch.setattr(cv2, "imread", lambda *_args, **_kwargs: singleton.copy())
    result = read_metric_depth(path)
    assert result.shape == (3, 4)
    assert result.dtype == np.uint16

    invalid = np.full((3, 4, 3), 1200, dtype=np.uint16)
    monkeypatch.setattr(cv2, "imread", lambda *_args, **_kwargs: invalid.copy())
    import pytest

    with pytest.raises(ValueError, match="single-channel uint16"):
        read_metric_depth(path)


def test_quality_helpers_exclude_padding_and_keep_default_behavior():
    rgb = np.zeros((2, 4, 3), dtype=np.float32)
    ir = np.ones((2, 4), dtype=np.float32)
    content = np.zeros((2, 4), dtype=bool)
    content[:, :2] = True
    rgb[:, 2:] = 114 / 255.0
    ir[:, 2:] = 114 / 255.0

    np.testing.assert_array_equal(_rgb_quality(rgb, content), np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(_ir_quality(ir, content), np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert _rgb_quality(rgb)[0] == 0.5
    assert _ir_quality(ir)[2] == 0.5


def test_dataset_quality_uses_flipped_content_mask(monkeypatch, tmp_path):
    """An asymmetric right pad catches a content mask that was not flipped with the image."""
    label = tmp_path / 'sample.txt'
    label.write_text('', encoding='utf-8')
    record = SampleRecord(
        sample_id='sample',
        rgb=tmp_path / 'rgb.png',
        infrared=tmp_path / 'ir.png',
        depth=tmp_path / 'depth.png',
        label=label,
    )
    rgb_bgr = np.zeros((3, 2, 3), dtype=np.uint8)
    ir_bgr = np.full((3, 2, 3), 255, dtype=np.uint8)
    depth = np.full((3, 2), 1000, dtype=np.uint16)

    def fake_imread(path, flags):
        if Path(path) == record.rgb:
            return rgb_bgr.copy()
        if Path(path) == record.infrared:
            return ir_bgr.copy()
        if Path(path) == record.depth:
            return depth.copy()
        raise AssertionError(f'unexpected path: {path}')

    monkeypatch.setattr(cv2, 'imread', fake_imread)
    ds = TriModalDataset.__new__(TriModalDataset)
    ds.cfg = DataConfig(root=tmp_path, imgsz=4, augment=True, hflip_prob=1.0)
    ds.records = [record]
    ds._epoch = 0

    item = ds[0]
    np.testing.assert_array_equal(
        item['quality'].numpy(),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )
    # Original 3x2 becomes 4x3 with a one-column right pad; after flip, content is columns 1..3.
    np.testing.assert_array_equal(item['img'][5, :, 0].numpy(), np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(item['img'][5, :, 1:].numpy(), np.ones((4, 3), dtype=np.float32))
