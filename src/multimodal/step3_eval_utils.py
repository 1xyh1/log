"""Authoritative Step-3 evaluation helpers using Ultralytics validator primitives.

Why this exists:
- Step-3 inputs are already float32 [0,1], so stock validator.preprocess() must NOT
  divide by 255 again.
- Everything after input transfer should follow stock DetectionValidator semantics
  instead of being reimplemented manually (box conversion, NMS, matching).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def _as_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    s = str(device)
    if s in {"0", "cuda"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def extract_detection_tensor(output: Any) -> torch.Tensor:
    """Extract the inference tensor consumed by DetectionValidator.postprocess()."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("one2many", "one2one", "preds"):
            if key in output:
                return extract_detection_tensor(output[key])
        for value in output.values():
            try:
                return extract_detection_tensor(value)
            except (TypeError, ValueError):
                pass
        raise TypeError(f"no detection tensor in output dict keys={list(output)}")
    if isinstance(output, (tuple, list)):
        # Eval-mode Ultralytics Detect commonly returns (decoded_predictions, raw_features).
        for value in output:
            if torch.is_tensor(value) and value.ndim >= 3:
                return value
        for value in output:
            try:
                return extract_detection_tensor(value)
            except (TypeError, ValueError):
                pass
    raise TypeError(f"unsupported model output type: {type(output)!r}")


def make_detection_validator(model, device, names, *, conf=0.001, iou=0.7, max_det=100):
    """Create a DetectionValidator object for its authoritative postprocess/match helpers.

    We deliberately do not call validator.preprocess(); Step-3 tensors are already
    normalized.  The caller moves the real batch tensors to device directly.
    """
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace

    cfg = dict(vars(DEFAULT_CFG))
    cfg.update(
        task="detect",
        conf=conf,
        iou=iou,
        max_det=max_det,
        plots=False,
        save_json=False,
        save_txt=False,
        single_cls=False,
        agnostic_nms=False,
        half=False,
        verbose=False,
    )
    validator = DetectionValidator(
        dataloader=None,
        save_dir=Path("."),
        args=IterableSimpleNamespace(**cfg),
    )
    validator.device = _as_device(device)
    validator.names = {int(k): v for k, v in names.items()}
    validator.nc = len(validator.names)
    validator.end2end = bool(getattr(model, "end2end", False))
    return validator


def move_step3_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move tensors without stock `/255` preprocessing."""
    out = dict(batch)
    for key, value in out.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=device.type == "cuda")
    out["img"] = out["img"].float()
    return out


def evaluate_dataset_stock_semantics(model, dataset, device, names) -> dict:
    """Evaluate a TriModalDataset with stock DetectionValidator semantics.

    Dataset iteration is one image at a time so intervention datasets (NORMAL/ZERO/
    SHUFFLE) stay simple and deterministic.  This is slower than a batched validator
    but the sample probe is tiny and correctness is the priority.
    """
    from ultralytics.utils.metrics import DetMetrics, box_iou

    device = _as_device(device)
    validator = make_detection_validator(model, device, names)
    metrics = DetMetrics(names={int(k): v for k, v in names.items()})

    n_gt_boxes = 0
    n_predictions = 0
    max_confidence = 0.0
    best_iou_per_gt: list[float] = []

    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            batch = dataset.collate_fn([sample])
            batch = move_step3_batch_to_device(batch, device)

            raw = extract_detection_tensor(model._predict_once(batch["img"]))
            preds = validator.postprocess(raw)
            if len(preds) != 1:
                raise RuntimeError(f"expected one prediction dict, got {len(preds)}")

            pbatch = validator._prepare_batch(0, batch)
            pred = validator._prepare_pred(preds[0])
            cls_np = pbatch["cls"].detach().cpu().numpy()
            no_pred = pred["cls"].numel() == 0

            stat = validator._process_batch(pred, pbatch)
            stat.update(
                target_cls=cls_np,
                target_img=np.unique(cls_np),
                conf=np.zeros(0, dtype=np.float32) if no_pred else pred["conf"].detach().cpu().numpy(),
                pred_cls=np.zeros(0, dtype=np.float32) if no_pred else pred["cls"].detach().cpu().numpy(),
                im_name=str(sample["sample_id"]),
            )
            metrics.update_stats(stat)

            n_gt_boxes += int(pbatch["cls"].numel())
            n_predictions += int(pred["cls"].numel())
            if not no_pred:
                max_confidence = max(max_confidence, float(pred["conf"].max()))
            if pbatch["bboxes"].numel() and pred["bboxes"].numel():
                iou_mat = box_iou(pbatch["bboxes"], pred["bboxes"])
                best_iou_per_gt.extend(iou_mat.max(dim=1).values.detach().cpu().tolist())
            elif pbatch["bboxes"].numel():
                best_iou_per_gt.extend([0.0] * int(pbatch["bboxes"].shape[0]))

    metrics.process()
    results = metrics.results_dict
    out = {
        "map50": float(results["metrics/mAP50(B)"]),
        "map50_95": float(results["metrics/mAP50-95(B)"]),
        "n_images": len(dataset),
        "n_gt_boxes": n_gt_boxes,
        "n_predictions": n_predictions,
        "max_confidence": max_confidence,
        "mean_best_iou_per_gt": float(np.mean(best_iou_per_gt)) if best_iou_per_gt else None,
    }

    per_class = {}
    ap_index = getattr(metrics.box, "ap_class_index", np.array([], dtype=int))
    for i, cls_id in enumerate(ap_index.tolist()):
        per_class[str(int(cls_id))] = {
            "ap50": round(float(metrics.box.ap50[i]), 6),
            "ap50_95": round(float(metrics.box.ap[i]), 6),
        }
    out["per_class"] = per_class
    return out
