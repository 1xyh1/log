"""Deterministic evaluation-only IR corruptions for the F1 reliability probe.

This is a read-only dataset *view*: it reuses ``TriModalDataset`` and changes only
channel I after the frozen preprocessing/LetterBox path.  RGB, labels, boxes, sample
order, and the primary NORMAL/ZERO/SHUFFLE protocol are untouched.
"""
from __future__ import annotations

import zlib

import numpy as np
from torch.utils.data import Dataset

from multimodal.modality_quality import content_mask_from_sample


KINDS = {"identity", "zero", "blur", "contrast", "noise", "shift"}


def _validate(kind: str, severity: float) -> tuple[str, float]:
    kind = str(kind).lower()
    severity = float(severity)
    if kind not in KINDS:
        raise ValueError(f"unknown IR corruption {kind!r}; expected {sorted(KINDS)}")
    if not np.isfinite(severity) or not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be finite in [0,1], got {severity}")
    return kind, severity


def corrupt_ir_plane(plane: np.ndarray, *, kind: str, severity: float,
                     sample_id: str, content_mask: np.ndarray | None = None) -> np.ndarray:
    kind, severity = _validate(kind, severity)
    x = np.asarray(plane, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"IR plane must be HxW, got {x.shape}")
    out = x.copy()
    mask = np.ones(x.shape, dtype=bool) if content_mask is None else np.asarray(
        content_mask, dtype=bool
    )
    if mask.shape != x.shape:
        raise ValueError("content mask shape mismatch")
    if kind == "identity" or severity == 0.0:
        return out
    if kind == "zero":
        out[mask] = 0.0
    elif kind == "contrast":
        values = out[mask]
        center = float(np.median(values)) if values.size else 0.0
        out[mask] = center + (values - center) * (1.0 - severity)
    elif kind == "noise":
        seed = zlib.crc32(f"f1-ir:{sample_id}:{severity:.6f}".encode("utf-8"))
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 0.20 * severity, size=x.shape).astype(np.float32)
        out[mask] += noise[mask]
    elif kind == "shift":
        pixels = max(1, int(round(16 * severity)))
        shifted = np.zeros_like(out)
        shifted[:, pixels:] = out[:, :-pixels]
        out = shifted
    elif kind == "blur":
        import cv2
        kernel = max(3, int(round(2 + 20 * severity)))
        kernel += 1 - kernel % 2
        out = cv2.GaussianBlur(out, (kernel, kernel), sigmaX=0)
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    out[~mask] = 0.0
    return out


class IRCorruptionDatasetView(Dataset):
    """Evaluation-only wrapper around the authoritative TriModalDataset."""

    def __init__(self, dataset, *, kind: str, severity: float):
        self.dataset = dataset
        self.kind, self.severity = _validate(kind, severity)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        base = self.dataset[index]
        sample = dict(base)
        sample["img"] = np.asarray(base["img"], dtype=np.float32).copy()
        content = content_mask_from_sample(sample, imgsz=sample["img"].shape[-1])
        sample["img"][3] = corrupt_ir_plane(
            sample["img"][3], kind=self.kind, severity=self.severity,
            sample_id=str(sample["sample_id"]), content_mask=content,
        )
        sample["ir_corruption"] = self.kind
        sample["ir_corruption_severity"] = self.severity
        return sample

    @property
    def collate_fn(self):
        return self.dataset.collate_fn
