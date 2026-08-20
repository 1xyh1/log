"""Recipient-keyed, geometry-locked RGB/IR dataset for T1-GR."""
from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from ultralytics.data.augment import (
    Albumentations,
    Compose,
    Format,
    LetterBox,
    Mosaic,
    RandomHSV,
    RandomPerspective,
    v8_transforms,
)
from ultralytics.data.dataset import YOLODataset

from .t1gr_g_core import ARMS
from .t1gr_g_impl_core import ZERO_IR, fast_source_for_recipient, source_schedule_index


def _raw_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def recipient_draw_seed(seed: int, epoch: int, recipient: str) -> int:
    raw = f"T1GR_AUG_V1\0{int(seed)}\0{int(epoch)}\0{recipient}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


@contextmanager
def scoped_rng(seed: int):
    """Temporarily key Python/NumPy/Torch CPU RNGs without touching trainer RNG."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        torch.manual_seed(int(seed) % (2**63 - 1))
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


def set_albumentations_seed(obj: Any, seed: int, seen: set[int] | None = None) -> None:
    """Key every nested Albumentations private RNG to the same recipient draw."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    transform = getattr(obj, "transform", None)
    if transform is not None and hasattr(transform, "set_random_seed"):
        transform.set_random_seed(int(seed) % (2**32))
    for child in getattr(obj, "transforms", []):
        set_albumentations_seed(child, seed, seen)
    pre = getattr(obj, "pre_transform", None)
    if pre is not None:
        set_albumentations_seed(pre, seed, seen)


class T1GRMosaic(Mosaic):
    """Stock mosaic with a private trace of every actual recipient/donor load."""

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        pairs = []
        for patch_index, item in enumerate(params["layout"]):
            for row in item["labels_patch"].get("source_pairs", []):
                copied = dict(row)
                copied["role"] = "anchor" if patch_index == 0 else "mosaic_aux"
                pairs.append(copied)
        layout = params["layout"]
        if self.n == 4:
            img = np.empty((self.imgsz * 2, self.imgsz * 2, 4), dtype=np.uint8)
            img[:, :, :3] = 114
            img[:, :, 3] = 0
            for item in layout:
                patch = item["labels_patch"]["img"]
                img[item["y1a"] : item["y2a"], item["x1a"] : item["x2a"]] = patch[
                    item["y1b"] : item["y2b"], item["x1b"] : item["x2b"]
                ]
            labels["img"] = img
        elif self.n == 9:
            img = np.empty((self.imgsz * 3, self.imgsz * 3, 4), dtype=np.uint8)
            img[:, :, :3] = 114
            img[:, :, 3] = 0
            for item in layout:
                patch = item["labels_patch"]["img"]
                x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
                x1b, y1b = x1 - item["padw"], y1 - item["padh"]
                x2b, y2b = x1b + (x2 - x1), y1b + (y2 - y1)
                img[y1:y2, x1:x2] = patch[y1b:y2b, x1b:x2b]
            labels["img"] = img[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        else:  # pragma: no cover - Ultralytics constructor already rejects this
            raise RuntimeError("T1GR_G_MOSAIC_GRID_FAIL")
        labels["source_pairs"] = pairs
        return labels


class T1GRLetterBox(LetterBox):
    """Visible padding stays 114 while the IR padding plane stays exactly zero."""

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        img = labels["img"]
        if img.ndim != 3 or img.shape[2] != 4:
            return super().apply_image(labels, params)
        new_unpad = params["new_unpad"]
        if img.shape[:2][::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=self.interpolation)
            if img.ndim == 2:
                img = img[..., None]
        h, w, _ = img.shape
        top, bottom = params["top"], params["bottom"]
        left, right = params["left"], params["right"]
        padded = np.empty((h + top + bottom, w + left + right, 4), dtype=img.dtype)
        padded[:, :, :3] = self.padding_value
        padded[:, :, 3] = 0
        padded[top : top + h, left : left + w] = img
        labels["img"] = padded
        labels["resized_shape"] = params["new_shape"]
        return labels


class T1GRRandomPerspective(RandomPerspective):
    """Stock affine parameters with an explicit four-channel border scalar."""

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        img = labels["img"]
        matrix = params["M"]
        size = params["size"]
        if (size[0] != img.shape[1] or size[1] != img.shape[0]) or (matrix != np.eye(3)).any():
            border = (114, 114, 114, 0) if int(img.shape[2]) == 4 else (114,) * int(img.shape[2])
            if self.perspective:
                img = cv2.warpPerspective(img, matrix, dsize=size, borderValue=border)
            else:
                img = cv2.warpAffine(img, matrix[:2], dsize=size, borderValue=border)
            if img.ndim == 2:
                img = img[..., None]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels


class T1GRVisibleAlbumentations:
    """Use the already-built stock non-spatial transform on visible BGR only."""

    def __init__(self, stock: Albumentations):
        self.p = stock.p
        self.transform = stock.transform
        self.contains_spatial = stock.contains_spatial if stock.transform is not None else False
        if self.contains_spatial:
            raise RuntimeError("T1GR_G_SPATIAL_ALBUMENTATIONS_FORBIDDEN")

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.transform is None or random.random() > self.p:
            return labels
        img = labels["img"]
        if img.ndim != 3 or img.shape[2] != 4:
            raise RuntimeError("T1GR_G_ALBUMENTATIONS_EXPECTED_FOUR_CHANNELS")
        visible = np.ascontiguousarray(img[:, :, :3])
        visible = self.transform(image=visible)["image"]
        img[:, :, :3] = visible
        labels["img"] = img
        return labels


class T1GRVisibleHSV(RandomHSV):
    """Apply the exact stock HSV algorithm and RNG draw to visible BGR only."""

    def apply_image(self, labels, params: dict[str, Any] | None = None):
        img = labels["img"]
        if img.ndim != 3 or img.shape[2] != 4:
            return super().apply_image(labels, params)
        visible = np.ascontiguousarray(img[:, :, :3])
        tmp = {"img": visible}
        super().apply_image(tmp, params)
        img[:, :, :3] = tmp["img"]
        labels["img"] = img
        return labels


class T1GRFormat(Format):
    """Match E5 BGR-to-RGB formatting for visible channels; keep IR scalar."""

    def _format_img(self, img: np.ndarray) -> torch.Tensor:
        if img.ndim == 3 and img.shape[2] == 4:
            reverse_visible = random.uniform(0, 1) > self.bgr
            visible = img[:, :, :3]
            if reverse_visible:
                visible = visible[:, :, ::-1]
            merged = np.concatenate((visible, img[:, :, 3:4]), axis=2)
            return torch.from_numpy(np.ascontiguousarray(merged.transpose(2, 0, 1)))
        return super()._format_img(img)


def _replace_transforms(compose: Compose) -> None:
    for index, transform in enumerate(list(compose.transforms)):
        replacement = transform
        if type(transform) is Mosaic:
            replacement = T1GRMosaic(transform.dataset, imgsz=transform.imgsz, p=transform.p, n=transform.n)
        elif type(transform) is RandomPerspective:
            replacement = T1GRRandomPerspective(
                degrees=transform.degrees,
                translate=transform.translate,
                scale=transform.scale,
                shear=transform.shear,
                perspective=transform.perspective,
                size=transform.size,
            )
        elif type(transform) is LetterBox:
            replacement = T1GRLetterBox(
                new_shape=transform.new_shape,
                auto=transform.auto,
                scale_fill=transform.scale_fill,
                scaleup=transform.scaleup,
                center=transform.center,
                stride=transform.stride,
                padding_value=transform.padding_value,
                interpolation=transform.interpolation,
            )
        elif type(transform) is Albumentations:
            replacement = T1GRVisibleAlbumentations(transform)
        elif type(transform) is RandomHSV:
            replacement = T1GRVisibleHSV(transform.hgain, transform.sgain, transform.vgain)
        compose.transforms[index] = replacement
        if isinstance(replacement, Compose):
            _replace_transforms(replacement)


def transform_graph(obj: Any) -> list[str]:
    out = [type(obj).__name__]
    for child in getattr(obj, "transforms", []):
        out.extend(transform_graph(child))
    pre = getattr(obj, "pre_transform", None)
    if pre is not None and pre is not obj and id(pre) not in {id(x) for x in getattr(obj, "transforms", [])}:
        out.extend(transform_graph(pre))
    return out


class T1GRDataset(YOLODataset):
    """YOLODataset that changes only the training-time IR source condition."""

    def __init__(
        self,
        *args,
        ir_by_sid: Mapping[str, str | Path],
        arm: str,
        seed: int,
        split: str,
        **kwargs,
    ):
        if arm not in ARMS:
            raise ValueError(f"T1GR_G_UNKNOWN_ARM:{arm}")
        if split not in {"train", "dev"}:
            raise ValueError(f"T1GR_G_UNKNOWN_SPLIT:{split}")
        self.t1gr_arm = arm
        self.t1gr_seed = int(seed)
        self.t1gr_split = split
        self.t1gr_epoch = 0
        self.ir_by_sid = {str(k): str(Path(v)) for k, v in ir_by_sid.items()}
        data = dict(kwargs.get("data") or {})
        data["channels"] = 4
        kwargs["data"] = data
        super().__init__(*args, **kwargs)
        self.ids = tuple(Path(path).stem for path in self.im_files)
        if len(self.ids) != len(set(self.ids)):
            raise RuntimeError("T1GR_G_DUPLICATE_VISIBLE_IDS")
        if not set(self.ids) <= set(self.ir_by_sid):
            raise RuntimeError("T1GR_G_IR_MAPPING_INCOMPLETE")
        self._ordered, self._position = source_schedule_index(self.ids, seed=self.t1gr_seed)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("T1GR_G_NEGATIVE_EPOCH")
        self.t1gr_epoch = int(epoch)

    def draw_seed(self, index: int) -> int:
        return recipient_draw_seed(self.t1gr_seed, self.t1gr_epoch, self.ids[index])

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Each top-level recipient owns its complete transform stream, including
        # mosaic selection and geometry.  Independent arms therefore receive
        # bitwise-matched visible inputs under the same seed/epoch.
        draw_seed = self.draw_seed(index)
        set_albumentations_seed(self.transforms, draw_seed)
        with scoped_rng(draw_seed):
            return super().__getitem__(index)

    def source_sid(self, recipient: str) -> str:
        if self.t1gr_split != "train":
            return ZERO_IR
        if self.t1gr_arm in {"G0-N", "G1-P"}:
            return recipient
        return fast_source_for_recipient(
            self._ordered, self._position, epoch=self.t1gr_epoch, recipient=recipient
        )

    def _read_ir(self, donor: str, recipient_hw: tuple[int, int], resized_hw: tuple[int, int]) -> np.ndarray:
        if donor == ZERO_IR:
            return np.zeros(resized_hw, dtype=np.uint8)
        path = self.ir_by_sid.get(donor)
        if path is None:
            raise RuntimeError("T1GR_G_IR_DONOR_PATH_MISSING")
        ir = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if ir is None:
            raise FileNotFoundError("T1GR_G_IR_IMAGE_READ_FAIL")
        h0, w0 = recipient_hw
        if tuple(ir.shape[:2]) != (h0, w0):
            ir = cv2.resize(ir, (w0, h0), interpolation=cv2.INTER_LINEAR)
        hr, wr = resized_hw
        if tuple(ir.shape[:2]) != (hr, wr):
            ir = cv2.resize(ir, (wr, hr), interpolation=cv2.INTER_LINEAR)
        return ir

    def load_image(self, i: int, rect_mode: bool = True, resize_short: bool = False):
        visible, original_hw, resized_hw = super().load_image(i, rect_mode=rect_mode, resize_short=resize_short)
        recipient = Path(self.im_files[i]).stem
        donor = self.source_sid(recipient)
        ir = self._read_ir(donor, original_hw, resized_hw)
        if visible.ndim != 3 or visible.shape[2] != 3 or visible.shape[:2] != ir.shape[:2]:
            raise RuntimeError("T1GR_G_MODALITY_SHAPE_FAIL")
        # Four-channel contract: visible BGR + selected grayscale IR.
        # cv2.imread(IMREAD_GRAYSCALE) under ultralytics patching returns (H,W,1).
        if ir.ndim == 3:
            ir = ir[:, :, 0] if ir.shape[2] == 1 else cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        merged = np.concatenate((visible, ir[:, :, None]), axis=2)
        return merged, original_hw, resized_hw

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        label = super().get_image_and_label(index)
        recipient = Path(self.im_files[index]).stem
        donor = self.source_sid(recipient)
        label["source_pairs"] = [{
            "recipient": recipient,
            "donor": donor,
            "epoch": int(self.t1gr_epoch),
            "role": "anchor",
        }]
        return label

    def build_transforms(self, hyp=None) -> Compose:
        if self.augment:
            if getattr(hyp, "augmentations", None) is not None:
                raise RuntimeError("T1GR_G_CUSTOM_ALBUMENTATIONS_FORBIDDEN")
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
            _replace_transforms(transforms)
        else:
            transforms = Compose([T1GRLetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            T1GRFormat(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms

    def raw_pair_probe(self, index: int) -> dict:
        visible = cv2.imread(self.im_files[index], cv2.IMREAD_COLOR)
        if visible is None:
            raise FileNotFoundError("T1GR_G_VISIBLE_IMAGE_READ_FAIL")
        recipient = Path(self.im_files[index]).stem
        donor = self.source_sid(recipient)
        ir = self._read_ir(donor, visible.shape[:2], visible.shape[:2])
        return {
            "recipient": recipient,
            "donor": donor,
            "visible_sha256": _raw_sha(visible),
            "ir_sha256": _raw_sha(ir),
            "visible_shape": list(visible.shape),
            "ir_shape": list(ir.shape),
            "epoch": int(self.t1gr_epoch),
        }
