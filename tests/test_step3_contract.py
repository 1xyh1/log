"""Step 3-A interface regression tests (reviewer-mandated).

test_trainer_dataset_contract: batch shapes must match stock validator/loss contract,
proven by actually calling DetectionValidator._prepare_batch on a real batch.
test_head_nc12_physical_shape / test_ratio_pad_roundtrip / test_bbox_roundtrip_overlay:
see individual tests.
test_matched_augmentation_schedule: cross-group identical sampler perms + flip decisions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import modality_preprocess as mp  # noqa: E402
from multimodal.raw_sample_index import build_contract, OUT_DEFAULT  # noqa: E402
from multimodal.trimodal_dataset import GROUPS, TriModalDataset  # noqa: E402


@pytest.fixture(scope="module")
def contract():
    return build_contract(out_path=OUT_DEFAULT)


def test_trainer_dataset_contract(contract):
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
    ds = TriModalDataset(contract, split="train", group="C1-I", augment=False)
    batch = TriModalDataset.collate_fn([ds[0], ds[1], ds[2], ds[3]])
    assert batch["img"].shape == (4, 6, 640, 640)
    assert batch["img"].dtype == torch.float32
    assert batch["img"].max() <= 1.0 and batch["img"].min() >= 0.0
    assert batch["cls"].ndim == 2 and batch["cls"].shape[1] == 1
    assert batch["bboxes"].ndim == 2 and batch["bboxes"].shape[1] == 4
    assert batch["batch_idx"].ndim == 1
    v = DetectionValidator(dataloader=None, save_dir=ROOT / "runs",
                           args=IterableSimpleNamespace(**vars(DEFAULT_CFG)))
    v.device = torch.device("cpu")
    for si in range(4):
        pbatch = v._prepare_batch(si, batch)  # proves the batch contract end-to-end
        assert pbatch["cls"].ndim == 1
        assert pbatch["bboxes"].ndim == 2 and pbatch["bboxes"].shape[1] == 4


def test_head_nc12_physical_shape():
    from multimodal.early_fusion_yolo26 import _head_nc_physical, load_snapshot
    m = load_snapshot(str(ROOT / "step3_6ch_rgb_equiv_init.pt"))
    assert m.model[0].conv.in_channels == 6
    assert m.nc == 12
    assert m.model[-1].nc == 12
    assert _head_nc_physical(m) == [12, 12, 12]
    assert m.yaml.get("channels") == 6
    assert m.names == {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign",
                       5: "bicycle", 6: "car", 7: "ball", 8: "light",
                       9: "garbage can", 10: "uav", 11: "tricycle"}


def test_ratio_pad_roundtrip(contract):
    from ultralytics.utils.ops import scale_boxes
    for sid in contract["all17_ids"][:4]:
        ds = TriModalDataset(contract, split="all17", group="C0-N", augment=False)
        idx = ds.ids.index(sid)
        s = ds[idx]
        rp = s["ratio_pad"]
        assert isinstance(rp[0], tuple) and rp[0][0] == rp[0][1]
        assert isinstance(rp[1], tuple) and len(rp[1]) == 2
        xywh = torch.as_tensor(s["bboxes"], dtype=torch.float32).clone()
        xyxy = xywh.clone()
        xyxy[:, [0, 1]] -= xyxy[:, [2, 3]] / 2
        xyxy[:, [2, 3]] += xyxy[:, [0, 1]]
        xyxy = xyxy * 640.0
        back = scale_boxes((640, 640), xyxy, s["ori_shape"], ratio_pad=rp)
        # compare with raw label in original-image pixels
        raw = np.loadtxt(contract["_labels_dir"] + f"/{sid}.txt", ndmin=2).reshape(-1, 5)
        expect = np.zeros_like(raw[:, 1:])
        expect[:, 0] = (raw[:, 1] - raw[:, 3] / 2) * s["ori_shape"][1]
        expect[:, 1] = (raw[:, 2] - raw[:, 4] / 2) * s["ori_shape"][0]
        expect[:, 2] = (raw[:, 1] + raw[:, 3] / 2) * s["ori_shape"][1]
        expect[:, 3] = (raw[:, 2] + raw[:, 4] / 2) * s["ori_shape"][0]
        back_np = back.cpu().numpy() if torch.is_tensor(back) else back
        assert float(np.abs(back_np - expect).max()) < 1.0


def test_bbox_roundtrip_overlay(contract, tmp_path):
    import cv2
    for sid in contract["all17_ids"][:6]:
        ds = TriModalDataset(contract, split="all17", group="C0-N", augment=False)
        idx = ds.ids.index(sid)
        s = ds[idx]
        canvas = ((s["img"][0:3].transpose(1, 2, 0) * 255).astype(np.uint8))[:, :, ::-1].copy()
        for (cx, cy, w, h), c in zip(s["bboxes"], s["cls"].squeeze(-1)):
            x1, y1 = int((cx - w / 2) * 640), int((cy - h / 2) * 640)
            x2, y2 = int((cx + w / 2) * 640), int((cy + h / 2) * 640)
            assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 640
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(tmp_path / f"{sid}.png"), canvas)


def test_matched_augmentation_schedule(contract):
    seeds_ok = True
    for epoch in range(3):
        samplers, flips = [], []
        for g in GROUPS:
            ds = TriModalDataset(contract, split="train", group=g, augment=True)
            ds.set_epoch(epoch)
            samplers.append(ds.sampler.perm)
            flips.append([mp.should_flip(ds.seed, epoch, sid, ds.fliplr) for sid in ds.ids])
        assert all(s == samplers[0] for s in samplers), f"sampler mismatch at epoch {epoch}"
        assert all(f == flips[0] for f in flips), f"flip mismatch at epoch {epoch}"
        seeds_ok = seeds_ok and True
    assert seeds_ok
