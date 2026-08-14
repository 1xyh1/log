#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmod_qaf.data import DataConfig, TriModalDataset, collate_detection_batch
from mmod_qaf.local_stub import LocalTriModalSmokeModel, load_local_yolo26


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=256)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--ir-width", type=float, default=0.25)
    p.add_argument("--depth-width", type=float, default=0.25)
    p.add_argument("--backward", action="store_true")
    args = p.parse_args()
    ds = TriModalDataset(DataConfig(root=args.data, imgsz=args.imgsz, invalid_depth_policy="skip"))
    batch = next(iter(DataLoader(ds, batch_size=args.batch, collate_fn=collate_detection_batch)))
    base = load_local_yolo26(args.weights)
    model = LocalTriModalSmokeModel(base, args.ir_width, args.depth_width)
    if args.backward:
        model.train()
        out = model(batch["img"])
        surrogate_loss = sum(v["boxes"].abs().mean() + v["scores"].abs().mean() for v in out.values())
        surrogate_loss.backward()
        gradient_abs_sum = {
            "rgb_base": sum(float(p.grad.abs().sum()) for p in model.base.model[:11].parameters() if p.grad is not None),
            "ir": sum(float(p.grad.abs().sum()) for p in model.ir_encoder.parameters() if p.grad is not None),
            "depth": sum(float(p.grad.abs().sum()) for p in model.depth_encoder.parameters() if p.grad is not None),
            "fusion": sum(float(p.grad.abs().sum()) for module in (model.q3, model.q4, model.q5) if module is not None for p in module.parameters() if p.grad is not None),
        }
    else:
        model.eval()
        with torch.no_grad():
            out = model(batch["img"])
        surrogate_loss = None
        gradient_abs_sum = None
    report = {
        "weights": args.weights,
        "input": list(batch["img"].shape),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "surrogate_loss": None if surrogate_loss is None else float(surrogate_loss.detach()),
        "gradient_abs_sum": gradient_abs_sum,
        "output": {
            branch: {k: list(v.shape) if isinstance(v, torch.Tensor) else [list(t.shape) for t in v] for k, v in values.items()}
            for branch, values in out.items()
        },
        "qaf_p4_mean": model.q4.last_diagnostics.mean_modality_weights.tolist(),
        "qaf_p5_mean": model.q5.last_diagnostics.mean_modality_weights.tolist(),
    }
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
