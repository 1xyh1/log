"""Step 3-A modality preprocessing: physical-safe float32 pipeline.

Frozen decisions:
    RGB: explicit BGR->RGB; letterbox pad 114 in uint8 domain, then /255 -> [0,1]
    IR : per-pixel median collapse of the 3 near-redundant channels -> [0,1]
    Depth: uint16 mm -> fixed physical log map (300..19999) -> valid-aware resize
           (num/den + eps), mask kept BINARY via NEAREST; M=0 -> D=0 forced
    Pad: RGB=114/255, I/D/M strictly 0.0 (all-AUX-outside-image == 0)
    Flip: stateless crc32(f"{seed}:{epoch}:{sample_id}")/2^32 < p  (matched across groups)
    Letterbox: single ratio/pad computed once from the RGB shape (all modalities share
    the same 1080x1920 geometry), replicating ultralytics letterbox rounding.
"""
from __future__ import annotations

import math
import zlib

import cv2
import numpy as np

D_MIN, D_MAX = 300.0, 19999.0
RGB_PAD_U8 = 114
AUX_PAD = 0.0


def load_rgb_rgb(path: str) -> np.ndarray:
    """BGR file -> RGB uint8 array (explicit conversion, no hidden channel flips)."""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None or im.ndim != 3:
        raise RuntimeError(f"RGB read failed: {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def ir_median(path: str) -> np.ndarray:
    """3-channel IR -> per-pixel median scalar -> float32 [0,1]."""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None or im.ndim != 3:
        raise RuntimeError(f"IR read failed: {path}")
    return np.median(im.astype(np.float32), axis=2) / 255.0


def depth_physical(path: str) -> tuple[np.ndarray, np.ndarray]:
    """uint16 metric depth -> (D float [0,1], M float {0,1}). Fixed physical log map."""
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is not None and d.ndim == 3 and d.shape[2] == 1:
        d = d[..., 0]  # ultralytics-patched cv2 may return (H,W,1)
    if d is None or d.ndim != 2 or d.dtype != np.uint16:
        raise RuntimeError(f"depth not uint16 HxW: {path} ({d.dtype if d is not None else None})")
    valid = (d >= D_MIN) & (d <= D_MAX)
    vals = np.clip(d.astype(np.float32), D_MIN, D_MAX)
    logd = np.zeros(d.shape, dtype=np.float32)
    logd[valid] = (np.log(vals[valid]) - math.log(D_MIN)) / (math.log(D_MAX) - math.log(D_MIN))
    return logd, valid.astype(np.float32)


def letterbox_geometry(h: int, w: int, imgsz: int = 640) -> tuple[float, int, int, tuple[int, int]]:
    """Replicates ultralytics letterbox rounding.
    Returns (ratio, left, top, new_unpad (h,w)); right/bottom implied by imgsz-new_unpad."""
    shape = (imgsz, imgsz)
    r = min(shape[0] / h, shape[1] / w)
    new_unpad = (int(round(h * r)), int(round(w * r)))
    dh, dw = shape[0] - new_unpad[0], shape[1] - new_unpad[1]
    left, top = int(round(dw / 2 - 0.1)), int(round(dh / 2 - 0.1))
    return r, left, top, new_unpad


def letterbox_rgb(rgb_u8: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, tuple[tuple, tuple]]:
    """RGB uint8 -> letterboxed float32 [0,1] (pad 114/255).

    ratio_pad = ((r, r), (left, top)) — the exact format ultralytics scale_boxes
    expects (gain = ratio_pad[0][0], pad = ratio_pad[1] = (left, top)).
    """
    h, w = rgb_u8.shape[:2]
    r, left, top, new_unpad = letterbox_geometry(h, w, imgsz)
    resized = cv2.resize(rgb_u8, (new_unpad[1], new_unpad[0]), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), RGB_PAD_U8, dtype=np.uint8)
    canvas[top:top + new_unpad[0], left:left + new_unpad[1]] = resized
    ratio_pad = ((r, r), (left, top))
    return canvas.astype(np.float32) / 255.0, ratio_pad


def letterbox_scalar(x: np.ndarray, imgsz: int = 640) -> np.ndarray:
    """float [0,1] single-channel plane -> letterboxed float (pad 0.0)."""
    h, w = x.shape[:2]
    r, left, top, new_unpad = letterbox_geometry(h, w, imgsz)
    resized = cv2.resize(x, (new_unpad[1], new_unpad[0]), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((imgsz, imgsz), dtype=np.float32)
    canvas[top:top + new_unpad[0], left:left + new_unpad[1]] = resized
    return canvas


def valid_aware_resize(d: np.ndarray, valid: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, np.ndarray]:
    """Depth planes: D = num/den (linear, validity-weighted), M binary (NEAREST).

    Then letterbox both with pad 0; enforce M=0 -> D=0 afterwards.
    """
    h, w = d.shape[:2]
    r, left, top, new_unpad = letterbox_geometry(h, w, imgsz)
    size = (new_unpad[1], new_unpad[0])
    num = cv2.resize(d * valid, size, interpolation=cv2.INTER_LINEAR)
    den = cv2.resize(valid, size, interpolation=cv2.INTER_LINEAR)
    d_r = num / np.maximum(den, 1e-6)
    m_r = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)
    m_r = (m_r >= 0.5).astype(np.float32)
    d_r[m_r < 0.5] = 0.0
    d_canvas = np.zeros((imgsz, imgsz), dtype=np.float32)
    m_canvas = np.zeros((imgsz, imgsz), dtype=np.float32)
    d_canvas[top:top + new_unpad[0], left:left + new_unpad[1]] = d_r
    m_canvas[top:top + new_unpad[0], left:left + new_unpad[1]] = m_r
    d_canvas[m_canvas < 0.5] = 0.0
    return d_canvas, m_canvas


def should_flip(seed: int, epoch: int, sample_id: str, p: float) -> bool:
    """Stateless horizontal flip decision; identical across groups for the same (epoch, sample)."""
    return zlib.crc32(f"{seed}:{epoch}:{sample_id}".encode("utf-8")) / 2 ** 32 < p


def apply_flip(planes: list[np.ndarray], bboxes: np.ndarray) -> list[np.ndarray]:
    """Horizontal flip of all planes; bboxes are normalized xywh [cx,cy,w,h] -> cx = 1-cx."""
    out = [np.ascontiguousarray(p[:, ::-1]) for p in planes]
    bboxes[:, 0] = 1.0 - bboxes[:, 0]
    return out
