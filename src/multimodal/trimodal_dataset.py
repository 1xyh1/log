"""Step 3-A TriModalDataset: float32 6ch [R,G,B,I,D,M] + labels + validator metadata.

Group channel masking (the ONLY content difference between groups):
    C0-N [R,G,B,0,0,0]   C1-I [R,G,B,I,0,0]   C2-D [R,G,B,0,D,M]
Sample dict contract (validator-compatible):
    img (6,H,W) float32 [0,1] / cls (N,) / bboxes (N,4) normalized xywh
    [cx,cy,w,h] in the FINAL 640x640 letterboxed space (loss/validator apply
    xywh2xyxy internally) / batch_idx / im_file / ori_shape (h,w) /
    ratio_pad ((r,r),(left,top)) / sample_id
Deterministic epoch sampler: permutation = PRNG(seed + epoch), precomputed per epoch
by set_epoch so the trainer can record the actual yielded order hash (G8).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from multimodal import modality_preprocess as mp
from multimodal import raw_sample_index as rsi

GROUPS = {
    "C0-N": {"I": False, "D": False, "M": False},
    "C1-I": {"I": True, "D": False, "M": False},
    "C2-D": {"I": False, "D": True, "M": True},
}


class DeterministicEpochSampler(Sampler):
    """permutation = torch.randperm(generator=manual_seed(seed+epoch)); precomputed in set_epoch."""

    def __init__(self, n: int, seed: int, shuffle: bool = True):
        self.n = n
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.perm = list(range(n))

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + epoch)
            self.perm = torch.randperm(self.n, generator=g).tolist()
        else:
            self.perm = list(range(self.n))

    def order_sha256(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(self.perm).encode("utf-8"))
        return h.hexdigest()

    def __iter__(self):
        return iter(self.perm)

    def __len__(self):
        return self.n


def _read_labels(path: Path, r: float, left: int, top: int, new_unpad: tuple,
                 imgsz: int) -> tuple[np.ndarray, np.ndarray]:
    """Label txt -> cls (N,) int64 + bboxes (N,4) normalized xywh [cx,cy,w,h] in the
    FINAL letterboxed 640x640 space (what v8DetectionLoss / validator expect: they
    apply xywh2xyxy internally). Flip later only maps cx -> 1 - cx."""
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    cls, boxes = [], []
    new_h, new_w = new_unpad
    for line in lines:
        if not line.strip():
            continue
        c, cx, cy, bw, bh = (float(x) for x in line.split()[:5])
        cx_lb = (cx * new_w + left) / imgsz
        cy_lb = (cy * new_h + top) / imgsz
        w_lb = bw * new_w / imgsz
        h_lb = bh * new_h / imgsz
        cls.append(int(c))
        boxes.append([cx_lb, cy_lb, w_lb, h_lb])
    if not cls:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    return np.asarray(cls, dtype=np.int64), np.asarray(boxes, dtype=np.float32)


class TriModalDataset(Dataset):
    def __init__(self, contract: dict, split: str, group: str = "C1-I",
                 imgsz: int = 640, seed: int = 20260812, fliplr: float = 0.30393,
                 augment: bool = True):
        if group not in GROUPS:
            raise ValueError(f"unknown group {group!r}")
        self.contract = contract
        self.split = split
        self.group = group
        self.mask = GROUPS[group]
        self.imgsz = imgsz
        self.seed = seed
        self.fliplr = fliplr
        self.augment = augment
        self.epoch = 0
        self.raw_dir = Path(contract["_raw_dir"])
        self.ids = contract[f"{split}_ids"]
        self.label_files = {sid: Path(contract["_labels_dir"]) / f"{sid}.txt"
                            for sid in self.ids}
        # epoch sampler (train only)
        self.sampler = DeterministicEpochSampler(len(self.ids), seed, shuffle=augment)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.sampler.set_epoch(epoch)

    def __len__(self):
        return len(self.ids)

    def _paths(self, sid: str) -> tuple[Path, Path, Path]:
        raw = self.raw_dir
        rgb = next((raw / "visible").glob(f"{sid}.*"))
        ir = next((raw / "infrared").glob(f"{sid}.*"))
        dep = next((raw / "depth").glob(f"{sid}.*"))
        return rgb, ir, dep

    def __getitem__(self, index: int):
        sid = self.ids[index]
        rgb_p, ir_p, dep_p = self._paths(sid)
        lab_p = self.label_files[sid]
        ori = mp.load_rgb_rgb(str(rgb_p))
        h, w = ori.shape[:2]
        r, left, top, new_unpad = mp.letterbox_geometry(h, w, self.imgsz)
        cls, bboxes = _read_labels(lab_p, r, left, top, new_unpad, self.imgsz)

        rgb, ratio_pad = mp.letterbox_rgb(ori, self.imgsz)
        i_plane = mp.letterbox_scalar(mp.ir_median(str(ir_p)), self.imgsz)
        d_raw, m_raw = mp.depth_physical(str(dep_p))
        d_plane, m_plane = mp.valid_aware_resize(d_raw, m_raw, self.imgsz)

        if self.augment and mp.should_flip(self.seed, self.epoch, sid, self.fliplr):
            rgb, i_plane, d_plane, m_plane = mp.apply_flip(
                [rgb, i_plane, d_plane, m_plane], bboxes)

        # group channel masking (the ONLY content difference between groups)
        if not self.mask["I"]:
            i_plane = np.zeros_like(i_plane)
        if not self.mask["D"]:
            d_plane = np.zeros_like(d_plane)
        if not self.mask["M"]:
            m_plane = np.zeros_like(m_plane)

        img = np.stack([rgb[..., 0], rgb[..., 1], rgb[..., 2],
                        i_plane, d_plane, m_plane], axis=0)  # (6, H, W) float32 [0,1]
        img = np.ascontiguousarray(img, dtype=np.float32)
        return {
            "img": img,
            "cls": cls.reshape(-1, 1),  # (N,1): validator indexes with a 2D bool mask
            "bboxes": bboxes,
            "batch_idx": np.zeros(len(cls), dtype=np.float32),  # 1D (N,): validator masks with it
            "im_file": str(rgb_p),
            "ori_shape": (h, w),
            "ratio_pad": ratio_pad,
            "sample_id": sid,
        }

    @staticmethod
    def collate_fn(batch):
        """Locked copy of stock YOLODataset.collate_fn semantics (img stack, cls/bboxes cat,
        unknown keys -> tuple), adapted for numpy sample tensors. img is float32 [0,1];
        stock preprocess /255 is NOT applied to Step-3 batches (Trainer/Validator override)."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            v = values[i]
            if k == "img":
                v = torch.stack([torch.as_tensor(x) for x in v], 0)
            elif k in {"cls", "bboxes"}:
                v = torch.cat([torch.as_tensor(x) for x in v], 0)
            new_batch[k] = v
        bidx = [torch.as_tensor(x) + i for i, x in enumerate(new_batch["batch_idx"])]
        new_batch["batch_idx"] = torch.cat(bidx, 0)
        return new_batch
