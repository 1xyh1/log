"""Shared runtime helpers for T-series training and evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from multimodal import modality_preprocess as mp
from multimodal.early_fusion_yolo26 import MODEL_INIT_SEED, build_reference_3ch
from multimodal.raw_sample_index import CLASS_NAMES
from multimodal.step4_f1_c_readiness import (
    EXPECTED_BASE_CHECKPOINT_SHA256,
    verify_base_checkpoint,
    verify_data_yaml,
    verify_raw_data_freshness,
)
from multimodal.trimodal_dataset import TriModalDataset
from multimodal.tseries_core import (
    FORMAL_BATCH, FORMAL_EPOCHS, FORMAL_SEED,
    parameter_sha256, sha256_file, sha256_json, state_sha256, tensor_sha256,
)
from multimodal.tseries_p5_model import TSeriesP5Model

R3_KW = dict(
    epochs=80, batch=4, nbs=4, warmup_epochs=0, workers=0, cache=False,
    imgsz=640, max_det=100, patience=100, close_mosaic=0,
    mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
    scale=0.0, translate=0.0, degrees=0.0, shear=0.0, perspective=0.0,
    multi_scale=0.0, amp=False, fliplr=0.30393, flipud=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0,
    auto_augment=None, erasing=0.0,
    seed=FORMAL_SEED, deterministic=True, end2end=False,
    plots=False, cls_pw=0.0,
    optimizer="MuSGD", lr0=0.00038, lrf=0.88219, momentum=0.94751,
    weight_decay=0.00027, box=9.83241, cls=0.64896, dfl=0.95824,
)

def build_tseries_model(base_checkpoint: Path, treatment_id: str) -> TSeriesP5Model:
    check = verify_base_checkpoint(Path(base_checkpoint), EXPECTED_BASE_CHECKPOINT_SHA256)
    if not check["passed"]:
        raise RuntimeError(f"T_SERIES_BASE_CHECKPOINT_FAIL:{check['errors']}")
    rng = torch.random.get_rng_state()
    torch.manual_seed(MODEL_INIT_SEED)
    try:
        # Critical provenance fix: consume the exact checkpoint that was verified.
        reference = build_reference_3ch(weights=str(base_checkpoint))
        model = TSeriesP5Model(reference, treatment_id=treatment_id)
    finally:
        torch.random.set_rng_state(rng)
    model.nc = 12
    return model

def verify_external_inputs(contract: Mapping, data_yaml: Path, base_checkpoint: Path) -> dict:
    dy = verify_data_yaml(Path(data_yaml), CLASS_NAMES)
    if not dy["passed"]:
        raise RuntimeError(f"T_SERIES_DATA_YAML_FAIL:{dy['errors']}")
    raw = verify_raw_data_freshness(dict(contract))
    if not raw["passed"]:
        raise RuntimeError(f"T_SERIES_RAW_DATA_FAIL:{raw['errors']}")
    base = verify_base_checkpoint(Path(base_checkpoint), EXPECTED_BASE_CHECKPOINT_SHA256)
    if not base["passed"]:
        raise RuntimeError(f"T_SERIES_BASE_CHECKPOINT_FAIL:{base['errors']}")
    return {"data_yaml": dy, "raw_data": raw, "base_checkpoint": base}

def initial_identity(model: TSeriesP5Model) -> dict:
    return {
        "rgb_backbone_state_sha256": state_sha256(model.rgb_backbone),
        "aux_encoder_state_sha256": state_sha256(model.aux_encoder),
        "aux_encoder_param_sha256": parameter_sha256(model.aux_encoder),
        "p5_fusion_state_sha256": state_sha256(model.p5_fusion),
        "p5_fusion_param_sha256": parameter_sha256(model.p5_fusion),
        "p5_bias_sha256": tensor_sha256(model.p5_fusion.proj.bias),
        "tail_state_sha256": state_sha256(model.tail),
        "complete_model_state_sha256": state_sha256(model),
        "state_dict_keys": list(model.state_dict().keys()),
        "requires_grad": {n: bool(p.requires_grad) for n, p in model.named_parameters()},
    }

def metric_from_stats(stats_by_id: dict, ids: list[str], names: dict) -> dict:
    from ultralytics.utils.metrics import DetMetrics
    metrics = DetMetrics(names={int(k): v for k, v in names.items()})
    for sid in ids:
        metrics.update_stats(stats_by_id[sid])
    metrics.process()
    res = metrics.results_dict
    return {
        "map50": float(res["metrics/mAP50(B)"]),
        "map50_95": float(res["metrics/mAP50-95(B)"]),
        "n_images": len(ids),
    }

def collect_detection_stats(model, dataset, device, forward_fn=None) -> dict:
    from multimodal import step3_eval_utils as evu
    names = {int(k): v for k, v in CLASS_NAMES.items()}
    validator = evu.make_detection_validator(model, device, names)
    stats, traces = {}, {}
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            sid = str(sample["sample_id"])
            batch = dataset.collate_fn([sample])
            batch = evu.move_step3_batch_to_device(batch, device)
            if forward_fn is None:
                output = model._predict_once(batch["img"])
                trace = {"mode": "native"}
            else:
                output, trace = forward_fn(sid, sample, batch)
            raw = evu.extract_detection_tensor(output).detach()
            trace = dict(trace)
            trace["detection_sha256"] = tensor_sha256(raw)
            preds = validator.postprocess(raw)
            if len(preds) != 1:
                raise RuntimeError("T_SERIES_EXPECTED_ONE_PREDICTION")
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
                im_name=sid,
            )
            stats[sid] = stat
            traces[sid] = trace
    ids = [str(x) for x in dataset.ids]
    return {
        "full": metric_from_stats(stats, ids, names),
        "loo": {
            held: metric_from_stats(stats, [sid for sid in ids if sid != held], names)
            for held in ids
        },
        "trace": traces,
        "_stats": stats,
    }

def combine_stats_results(results: list[dict], ids: list[str]) -> dict:
    names = {int(k): v for k, v in CLASS_NAMES.items()}
    merged = {}
    for r in results:
        merged.update(r["_stats"])
    return metric_from_stats(merged, ids, names)

def load_checkpoint_model(path: Path, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = (ck.get("ema") or ck.get("model")).float().eval().to(device)
    if not isinstance(model, TSeriesP5Model):
        raise RuntimeError(f"T_SERIES_CHECKPOINT_MODEL_CLASS:{type(model).__name__}")
    return model, ck

def results_csv_metrics(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"T_SERIES_RESULTS_CSV_MISSING:{path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("T_SERIES_RESULTS_CSV_EMPTY")
    def find_key(row, needle):
        matches = [k for k in row if needle in k.replace(" ", "")]
        if not matches:
            raise RuntimeError(f"T_SERIES_RESULTS_KEY_MISSING:{needle}")
        return matches[0]
    key = find_key(rows[-1], "metrics/mAP50-95(B)")
    vals = [float(r[key]) for r in rows]
    return {
        "epochs_recorded": len(rows),
        "last_val_map50_95": vals[-1],
        "late10_median_val_map50_95": float(np.median(vals[-10:])),
        "best_val_map50_95_descriptive": max(vals),
        "best_epoch_descriptive": int(np.argmax(vals)) + 1,
        "metric_key": key,
    }

def protocol_hash(overrides: Mapping) -> str:
    frozen = {
        k: overrides[k]
        for k in sorted(overrides)
        if k not in {"name", "project", "device"}
    }
    return sha256_json(frozen)
