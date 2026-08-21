"""Recipient-keyed six-channel RGB/IR/Depth dataset for T1-U6."""
from __future__ import annotations

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

from .t1gr_u6_core import ARMS, ZERO_IR, arm_policy, encode_depth_array, raw_array_sha256
from .t1gr_g_dataset import recipient_draw_seed, scoped_rng, set_albumentations_seed
from .t1gr_g_impl_core import fast_source_for_recipient, source_schedule_index


def _two_dimensional(array: np.ndarray, code: str) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 1:
        return array[:, :, 0]
    raise RuntimeError(code)


def resize_depth_valid(depth: np.ndarray, mask: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Validity-weighted linear resize plus a nearest, binary validity mask."""
    depth = _two_dimensional(depth, "T1GR_U6_DEPTH_DIMENSION_FAIL")
    mask = _two_dimensional(mask, "T1GR_U6_MASK_DIMENSION_FAIL")
    if depth.shape != mask.shape:
        raise RuntimeError("T1GR_U6_DEPTH_MASK_SHAPE_FAIL")
    valid = (mask > 0).astype(np.float32)
    numerator = cv2.resize(depth.astype(np.float32) * valid, size, interpolation=cv2.INTER_LINEAR)
    denominator = cv2.resize(valid, size, interpolation=cv2.INTER_LINEAR)
    nearest = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST) > 0.5
    out = np.zeros(nearest.shape, dtype=np.uint8)
    usable = nearest & (denominator > 1e-6)
    out[usable] = np.rint(np.clip(numerator[usable] / denominator[usable], 0.0, 255.0)).astype(np.uint8)
    binary = nearest.astype(np.uint8) * np.uint8(255)
    out[binary == 0] = 0
    return out, binary


def warp_depth_valid(
    depth: np.ndarray,
    mask: np.ndarray,
    matrix: np.ndarray,
    size: tuple[int, int],
    *,
    perspective: bool,
) -> tuple[np.ndarray, np.ndarray]:
    depth = _two_dimensional(depth, "T1GR_U6_DEPTH_DIMENSION_FAIL")
    mask = _two_dimensional(mask, "T1GR_U6_MASK_DIMENSION_FAIL")
    valid = (mask > 0).astype(np.float32)
    numerator = depth.astype(np.float32) * valid
    if perspective:
        warp = lambda x, interpolation: cv2.warpPerspective(
            x, matrix, dsize=size, flags=interpolation, borderValue=0
        )
    else:
        warp = lambda x, interpolation: cv2.warpAffine(
            x, matrix[:2], dsize=size, flags=interpolation, borderValue=0
        )
    num_warp = warp(numerator, cv2.INTER_LINEAR)
    den_warp = warp(valid, cv2.INTER_LINEAR)
    nearest = warp(valid, cv2.INTER_NEAREST) > 0.5
    out = np.zeros(nearest.shape, dtype=np.uint8)
    usable = nearest & (den_warp > 1e-6)
    out[usable] = np.rint(np.clip(num_warp[usable] / den_warp[usable], 0.0, 255.0)).astype(np.uint8)
    binary = nearest.astype(np.uint8) * np.uint8(255)
    out[binary == 0] = 0
    return out, binary


class U6Mosaic(Mosaic):
    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        pairs, depths = [], []
        for patch_index, item in enumerate(params["layout"]):
            patch_labels = item["labels_patch"]
            for row in patch_labels.get("source_pairs", []):
                copied = dict(row)
                copied["role"] = "anchor" if patch_index == 0 else "mosaic_aux"
                pairs.append(copied)
            for row in patch_labels.get("depth_records", []):
                copied = dict(row)
                copied["role"] = "anchor" if patch_index == 0 else "mosaic_aux"
                depths.append(copied)
        layout = params["layout"]
        if self.n == 4:
            image = np.empty((self.imgsz * 2, self.imgsz * 2, 6), dtype=np.uint8)
            image[:, :, :3] = 114
            image[:, :, 3:] = 0
            for item in layout:
                patch = item["labels_patch"]["img"]
                image[item["y1a"]:item["y2a"], item["x1a"]:item["x2a"]] = patch[
                    item["y1b"]:item["y2b"], item["x1b"]:item["x2b"]
                ]
            labels["img"] = image
        elif self.n == 9:
            image = np.empty((self.imgsz * 3, self.imgsz * 3, 6), dtype=np.uint8)
            image[:, :, :3] = 114
            image[:, :, 3:] = 0
            for item in layout:
                patch = item["labels_patch"]["img"]
                x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
                x1b, y1b = x1 - item["padw"], y1 - item["padh"]
                x2b, y2b = x1b + (x2 - x1), y1b + (y2 - y1)
                image[y1:y2, x1:x2] = patch[y1b:y2b, x1b:x2b]
            labels["img"] = image[-self.border[0]:self.border[0], -self.border[1]:self.border[1]]
        else:
            raise RuntimeError("T1GR_U6_MOSAIC_GRID_FAIL")
        labels["source_pairs"] = pairs
        labels["depth_records"] = depths
        return labels


class U6LetterBox(LetterBox):
    def apply_image(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = labels["img"]
        if image.ndim != 3 or image.shape[2] != 6:
            return super().apply_image(labels, params)
        new_unpad = params["new_unpad"]
        if image.shape[:2][::-1] != new_unpad:
            rgb_ir = cv2.resize(image[:, :, :4], new_unpad, interpolation=self.interpolation)
            depth, mask = resize_depth_valid(image[:, :, 4], image[:, :, 5], new_unpad)
            image = np.concatenate((rgb_ir, depth[:, :, None], mask[:, :, None]), axis=2)
        h, w = image.shape[:2]
        top, bottom, left, right = params["top"], params["bottom"], params["left"], params["right"]
        padded = np.empty((h + top + bottom, w + left + right, 6), dtype=np.uint8)
        padded[:, :, :3] = self.padding_value
        padded[:, :, 3:] = 0
        padded[top:top + h, left:left + w] = image
        labels["img"] = padded
        labels["resized_shape"] = params["new_shape"]
        return labels


class U6RandomPerspective(RandomPerspective):
    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        image, matrix, size = labels["img"], params["M"], params["size"]
        if image.ndim != 3 or image.shape[2] != 6:
            return super().apply_image(labels, params)
        if (size[0] != image.shape[1] or size[1] != image.shape[0]) or (matrix != np.eye(3)).any():
            if self.perspective:
                rgb_ir = cv2.warpPerspective(image[:, :, :4], matrix, dsize=size, borderValue=(114, 114, 114, 0))
            else:
                rgb_ir = cv2.warpAffine(image[:, :, :4], matrix[:2], dsize=size, borderValue=(114, 114, 114, 0))
            depth, mask = warp_depth_valid(
                image[:, :, 4], image[:, :, 5], matrix, size, perspective=bool(self.perspective)
            )
            image = np.concatenate((rgb_ir, depth[:, :, None], mask[:, :, None]), axis=2)
        labels["img"] = image
        labels["resized_shape"] = image.shape[:2]
        return labels


class U6VisibleAlbumentations:
    def __init__(self, stock: Albumentations):
        self.p = stock.p
        self.transform = stock.transform
        self.contains_spatial = stock.contains_spatial if stock.transform is not None else False
        if self.contains_spatial:
            raise RuntimeError("T1GR_U6_SPATIAL_ALBUMENTATIONS_FORBIDDEN")

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        import random

        if self.transform is None or random.random() > self.p:
            return labels
        image = labels["img"]
        if image.ndim != 3 or image.shape[2] != 6:
            raise RuntimeError("T1GR_U6_ALBUMENTATIONS_EXPECTED_SIX_CHANNELS")
        image[:, :, :3] = self.transform(image=np.ascontiguousarray(image[:, :, :3]))["image"]
        labels["img"] = image
        return labels


class U6VisibleHSV(RandomHSV):
    def apply_image(self, labels, params: dict[str, Any] | None = None):
        image = labels["img"]
        if image.ndim != 3 or image.shape[2] != 6:
            return super().apply_image(labels, params)
        tmp = {"img": np.ascontiguousarray(image[:, :, :3])}
        super().apply_image(tmp, params)
        image[:, :, :3] = tmp["img"]
        labels["img"] = image
        return labels


class U6Format(Format):
    def _format_img(self, img: np.ndarray) -> torch.Tensor:
        if img.ndim == 3 and img.shape[2] == 6:
            import random

            visible = img[:, :, :3]
            if random.uniform(0, 1) > self.bgr:
                visible = visible[:, :, ::-1]
            merged = np.concatenate((visible, img[:, :, 3:]), axis=2)
            return torch.from_numpy(np.ascontiguousarray(merged.transpose(2, 0, 1)))
        return super()._format_img(img)


def _replace_transforms(compose: Compose) -> None:
    for index, transform in enumerate(list(compose.transforms)):
        replacement = transform
        if type(transform) is Mosaic:
            replacement = U6Mosaic(transform.dataset, imgsz=transform.imgsz, p=transform.p, n=transform.n)
        elif type(transform) is RandomPerspective:
            replacement = U6RandomPerspective(
                degrees=transform.degrees,
                translate=transform.translate,
                scale=transform.scale,
                shear=transform.shear,
                perspective=transform.perspective,
                size=transform.size,
            )
        elif type(transform) is LetterBox:
            replacement = U6LetterBox(
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
            replacement = U6VisibleAlbumentations(transform)
        elif type(transform) is RandomHSV:
            replacement = U6VisibleHSV(transform.hgain, transform.sgain, transform.vgain)
        compose.transforms[index] = replacement
        if isinstance(replacement, Compose):
            _replace_transforms(replacement)


class T1GRU6Dataset(YOLODataset):
    def __init__(
        self,
        *args,
        ir_by_sid: Mapping[str, str | Path],
        depth_by_sid: Mapping[str, str | Path],
        depth_kind_by_sid: Mapping[str, str],
        arm: str,
        seed: int,
        split: str,
        ir_condition: str = "ARM_NATIVE",
        depth_condition: str = "NATIVE",
        **kwargs,
    ):
        if (
            arm not in ARMS
            or split not in {"train", "dev"}
            or ir_condition not in {"ARM_NATIVE", "ZERO", "PAIRED", "WRONG"}
            or depth_condition not in {"NATIVE", "ZERO"}
        ):
            raise ValueError("T1GR_U6_DATASET_REQUEST_FAIL")
        self.t1gr_arm = arm
        self.t1gr_seed = int(seed)
        self.t1gr_split = split
        self.t1gr_epoch = 0
        self.ir_condition = ir_condition
        self.depth_condition = depth_condition
        self.ir_by_sid = {str(key): str(Path(value)) for key, value in ir_by_sid.items()}
        self.depth_by_sid = {str(key): str(Path(value)) for key, value in depth_by_sid.items()}
        self.depth_kind_by_sid = {str(key): str(value) for key, value in depth_kind_by_sid.items()}
        self._last_depth_record: dict[int, dict] = {}
        data = dict(kwargs.get("data") or {})
        data["channels"] = 6
        kwargs["data"] = data
        super().__init__(*args, **kwargs)
        self.ids = tuple(Path(path).stem for path in self.im_files)
        if len(self.ids) != len(set(self.ids)):
            raise RuntimeError("T1GR_U6_DUPLICATE_VISIBLE_IDS")
        required = set(self.ids)
        if not required <= set(self.ir_by_sid) or not required <= set(self.depth_by_sid) or not required <= set(self.depth_kind_by_sid):
            raise RuntimeError("T1GR_U6_MODALITY_MAPPING_INCOMPLETE")
        self._ordered, self._position = source_schedule_index(self.ids, seed=self.t1gr_seed)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("T1GR_U6_NEGATIVE_EPOCH")
        self.t1gr_epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        draw = recipient_draw_seed(self.t1gr_seed, self.t1gr_epoch, self.ids[index])
        set_albumentations_seed(self.transforms, draw)
        with scoped_rng(draw):
            return super().__getitem__(index)

    def source_sid(self, recipient: str) -> str:
        if self.ir_condition == "ZERO":
            return ZERO_IR
        if self.ir_condition == "PAIRED":
            return recipient
        if self.ir_condition == "WRONG":
            return fast_source_for_recipient(
                self._ordered, self._position, epoch=self.t1gr_epoch, recipient=recipient
            )
        policy = arm_policy(self.t1gr_arm)
        if self.t1gr_split == "dev":
            return ZERO_IR if policy["dev_native_ir"] == "ZERO_IR" else recipient
        treatment = policy["train_ir"]
        if treatment == "ZERO_IR":
            return ZERO_IR
        if treatment == "CORRECT_PAIRED_IR":
            return recipient
        if treatment == "BALANCED_FULLY_WRONG_IR":
            return fast_source_for_recipient(
                self._ordered, self._position, epoch=self.t1gr_epoch, recipient=recipient
            )
        raise RuntimeError("T1GR_U6_IR_POLICY_INTERNAL_FAIL")

    def _read_ir(self, donor: str, recipient_hw: tuple[int, int], resized_hw: tuple[int, int]) -> np.ndarray:
        if donor == ZERO_IR:
            return np.zeros(resized_hw, dtype=np.uint8)
        path = self.ir_by_sid.get(donor)
        if path is None:
            raise RuntimeError("T1GR_U6_IR_DONOR_PATH_MISSING")
        ir = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if ir is None:
            raise FileNotFoundError("T1GR_U6_IR_READ_FAIL")
        ir = _two_dimensional(ir, "T1GR_U6_IR_DIMENSION_FAIL")
        h0, w0 = recipient_hw
        if ir.shape != (h0, w0):
            ir = cv2.resize(ir, (w0, h0), interpolation=cv2.INTER_LINEAR)
        hr, wr = resized_hw
        if ir.shape != (hr, wr):
            ir = cv2.resize(ir, (wr, hr), interpolation=cv2.INTER_LINEAR)
        return ir

    def _read_depth(
        self, sid: str, recipient_hw: tuple[int, int], resized_hw: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        path = Path(self.depth_by_sid[sid])
        expected_kind = self.depth_kind_by_sid[sid]
        hr, wr = resized_hw
        enabled = self.t1gr_arm == "G3-D" and self.depth_condition == "NATIVE"
        should_decode = enabled and expected_kind == "METRIC_UINT16_PNG"
        native_valid: int | None = None
        native_depth: int | None = None
        if should_decode:
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError("T1GR_U6_DEPTH_READ_FAIL")
            encoded, mask, observed_kind = encode_depth_array(raw, path.suffix)
            if observed_kind != expected_kind:
                raise RuntimeError("T1GR_U6_DEPTH_KIND_DRIFT")
            native_valid = int(np.count_nonzero(mask))
            native_depth = int(np.count_nonzero(encoded))
            h0, w0 = recipient_hw
            if encoded.shape != (h0, w0):
                encoded, mask = resize_depth_valid(encoded, mask, (w0, h0))
            if encoded.shape != (hr, wr):
                encoded, mask = resize_depth_valid(encoded, mask, (wr, hr))
        else:
            if expected_kind not in {"METRIC_UINT16_PNG", "UNKNOWN_SCALE_JPG_QUARANTINED"}:
                raise RuntimeError("T1GR_U6_DEPTH_KIND_DRIFT")
            encoded = np.zeros((hr, wr), dtype=np.uint8)
            mask = np.zeros((hr, wr), dtype=np.uint8)
        record = {
            "sample_id": sid,
            "arm": self.t1gr_arm,
            "split": self.t1gr_split,
            "epoch": int(self.t1gr_epoch),
            "condition": self.depth_condition,
            "depth_kind": expected_kind,
            "depth_enabled": bool(enabled),
            "depth_source_decoded": bool(should_decode),
            "native_valid_pixels": native_valid,
            "native_nonzero_depth_pixels": native_depth,
            "emitted_valid_pixels": int(np.count_nonzero(mask)),
            "emitted_nonzero_depth_pixels": int(np.count_nonzero(encoded)),
            "mask_binary": bool(np.all((mask == 0) | (mask == 255))),
            "mask_zero_implies_depth_zero": bool(np.count_nonzero(encoded[mask == 0]) == 0),
        }
        return encoded, mask, record

    def load_image(self, i: int, rect_mode: bool = True, resize_short: bool = False):
        visible, original_hw, resized_hw = super().load_image(i, rect_mode=rect_mode, resize_short=resize_short)
        sid = Path(self.im_files[i]).stem
        donor = self.source_sid(sid)
        ir = self._read_ir(donor, original_hw, resized_hw)
        depth, mask, record = self._read_depth(sid, original_hw, resized_hw)
        if visible.ndim != 3 or visible.shape[2] != 3 or visible.shape[:2] != ir.shape or ir.shape != depth.shape:
            raise RuntimeError("T1GR_U6_MODALITY_SHAPE_FAIL")
        self._last_depth_record[i] = record
        merged = np.concatenate((visible, ir[:, :, None], depth[:, :, None], mask[:, :, None]), axis=2)
        return merged, original_hw, resized_hw

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        label = super().get_image_and_label(index)
        sid = Path(self.im_files[index]).stem
        donor = self.source_sid(sid)
        record = self._last_depth_record.pop(index, None)
        if record is None:
            raise RuntimeError("T1GR_U6_DEPTH_TRACE_MISSING")
        record = dict(record)
        record["role"] = "anchor"
        label["source_pairs"] = [{
            "recipient": sid,
            "donor": donor,
            "epoch": int(self.t1gr_epoch),
            "role": "anchor",
            "condition": self.ir_condition,
        }]
        label["depth_records"] = [record]
        return label

    def build_transforms(self, hyp=None) -> Compose:
        if self.augment:
            if getattr(hyp, "augmentations", None) is not None:
                raise RuntimeError("T1GR_U6_CUSTOM_ALBUMENTATIONS_FORBIDDEN")
            if float(hyp.mixup) != 0.0 or float(hyp.cutmix) != 0.0 or float(hyp.copy_paste) != 0.0:
                raise RuntimeError("T1GR_U6_MIXING_AUGMENTATION_DRIFT")
            hyp.mosaic = hyp.mosaic if not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
            _replace_transforms(transforms)
        else:
            transforms = Compose([U6LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(U6Format(
            bbox_format="xywh",
            normalize=True,
            return_mask=self.use_segments,
            return_keypoint=self.use_keypoints,
            return_obb=self.use_obb,
            batch_idx=True,
            mask_ratio=hyp.mask_ratio,
            mask_overlap=hyp.overlap_mask,
            bgr=hyp.bgr if self.augment else 0.0,
        ))
        return transforms

    def raw_probe(self, index: int) -> dict:
        visible = cv2.imread(self.im_files[index], cv2.IMREAD_COLOR)
        if visible is None:
            raise FileNotFoundError("T1GR_U6_VISIBLE_READ_FAIL")
        sid = Path(self.im_files[index]).stem
        donor = self.source_sid(sid)
        ir = self._read_ir(donor, visible.shape[:2], visible.shape[:2])
        depth, mask, record = self._read_depth(sid, visible.shape[:2], visible.shape[:2])
        return {
            "sample_id": sid,
            "donor_id": donor,
            "visible_sha256": raw_array_sha256(visible),
            "ir_sha256": raw_array_sha256(ir),
            "depth_sha256": raw_array_sha256(depth),
            "mask_sha256": raw_array_sha256(mask),
            "ir_nonzero": int(np.count_nonzero(ir)),
            "depth_nonzero": int(np.count_nonzero(depth)),
            "mask_nonzero": int(np.count_nonzero(mask)),
            "record": record,
        }
