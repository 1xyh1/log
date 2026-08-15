"""Step 3-A TriModalDataset: float32 6ch [R,G,B,I,D,M] + labels + audit metadata.

This patch keeps the frozen representation unchanged.  The only protocol addition is
`flip_applied`, which lets G8 hash the transform actually seen by the training loader.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from multimodal import modality_preprocess as mp

GROUPS = {
    "C0-N": {"I": False, "D": False, "M": False},
    "C1-I": {"I": True, "D": False, "M": False},
    "C2-D": {"I": False, "D": True, "M": True},
}


class DeterministicEpochSampler(Sampler):
    """Permutation = torch.randperm(generator=manual_seed(seed + epoch))."""

    def __init__(self, n: int, seed: int, shuffle: bool = True):
        self.n = int(n)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.perm = list(range(self.n))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            self.perm = torch.randperm(self.n, generator=g).tolist()
        else:
            self.perm = list(range(self.n))

    def order_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.perm).encode("utf-8")).hexdigest()

    def __iter__(self):
        return iter(self.perm)

    def __len__(self):
        return self.n


def _read_labels(path: Path, r: float, left: int, top: int, new_unpad: tuple,
                 imgsz: int) -> tuple[np.ndarray, np.ndarray]:
    """YOLO label txt -> cls (N,1), bboxes normalized xywh in final letterbox space."""
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
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    return np.asarray(cls, dtype=np.float32).reshape(-1, 1), np.asarray(boxes, dtype=np.float32)


class TriModalDataset(Dataset):
    def __init__(self, contract: dict, split: str, group: str = "C1-I",
                 imgsz: int = 640, seed: int = 20260812, fliplr: float = 0.30393,
                 augment: bool = True, aux_id_map: dict | None = None,
                 aux_zero: bool = False, exclude_ids: set[str] | None = None):
        if group not in GROUPS:
            raise ValueError(f"unknown group {group!r}")
        self.contract = contract
        self.split = split
        self.group = group
        self.mask = GROUPS[group]
        self.imgsz = int(imgsz)
        self.seed = int(seed)
        self.fliplr = float(fliplr)
        self.augment = bool(augment)
        self.aux_id_map = aux_id_map or {}
        self.aux_zero = bool(aux_zero)
        self.epoch = 0
        self.raw_dir = Path(contract["_raw_dir"])
        self.ids = list(contract[f"{split}_ids"])
        if exclude_ids:
            # Step-4 LOO: leave-one-out folds reuse the frozen contract; excluded
            # ids drop out of the ANCHOR set only (donors stay available on disk).
            self.ids = [sid for sid in self.ids if sid not in exclude_ids]
        self.label_files = {
            sid: Path(contract["_labels_dir"]) / f"{sid}.txt" for sid in self.ids
        }
        self.sampler = DeterministicEpochSampler(len(self.ids), self.seed, shuffle=self.augment)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
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
        rgb_p, _, _ = self._paths(sid)
        lab_p = self.label_files[sid]

        # SHUFFLE causality: auxiliary planes come from donor; geometry/labels stay sid.
        aux_sid = self.aux_id_map.get(sid, sid)
        _, ir_aux_p, dep_aux_p = self._paths(aux_sid)

        ori = mp.load_rgb_rgb(str(rgb_p))
        h, w = ori.shape[:2]
        r, left, top, new_unpad = mp.letterbox_geometry(h, w, self.imgsz)
        cls, bboxes = _read_labels(lab_p, r, left, top, new_unpad, self.imgsz)

        rgb, ratio_pad = mp.letterbox_rgb(ori, self.imgsz)
        i_plane = mp.letterbox_scalar(mp.ir_median(str(ir_aux_p)), self.imgsz)
        d_raw, m_raw = mp.depth_physical(str(dep_aux_p))
        d_plane, m_plane = mp.valid_aware_resize(d_raw, m_raw, self.imgsz)

        flip_applied = bool(
            self.augment and mp.should_flip(self.seed, self.epoch, sid, self.fliplr)
        )
        if flip_applied:
            rgb, i_plane, d_plane, m_plane = mp.apply_flip(
                [rgb, i_plane, d_plane, m_plane], bboxes
            )

        if not self.mask["I"]:
            i_plane = np.zeros_like(i_plane)
        if not self.mask["D"]:
            d_plane = np.zeros_like(d_plane)
        if not self.mask["M"]:
            m_plane = np.zeros_like(m_plane)

        if self.aux_zero:
            if self.mask["I"]:
                i_plane = np.zeros_like(i_plane)
            if self.mask["D"]:
                d_plane = np.zeros_like(d_plane)
            if self.mask["M"]:
                m_plane = np.zeros_like(m_plane)

        img = np.stack(
            [rgb[..., 0], rgb[..., 1], rgb[..., 2], i_plane, d_plane, m_plane],
            axis=0,
        )
        img = np.ascontiguousarray(img, dtype=np.float32)
        return {
            "img": img,
            "cls": cls,  # (N,1)
            "bboxes": bboxes,  # normalized xywh in final letterbox space
            "batch_idx": np.zeros(len(cls), dtype=np.float32),  # stock contract: (N,)
            "im_file": str(rgb_p),
            "ori_shape": (h, w),
            "ratio_pad": ratio_pad,
            "sample_id": sid,
            "aux_sample_id": aux_sid,
            "flip_applied": flip_applied,
        }

    @staticmethod
    def collate_fn(batch):
        """Stock-like YOLO collate semantics, preserving custom metadata as tuples."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, key in enumerate(keys):
            value = values[i]
            if key == "img":
                value = torch.stack([torch.as_tensor(x) for x in value], 0)
            elif key in {"cls", "bboxes"}:
                value = torch.cat([torch.as_tensor(x) for x in value], 0)
            new_batch[key] = value

        bidx = [torch.as_tensor(x) + i for i, x in enumerate(new_batch["batch_idx"])]
        new_batch["batch_idx"] = torch.cat(bidx, 0)
        return new_batch
