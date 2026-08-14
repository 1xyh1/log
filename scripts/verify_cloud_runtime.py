#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmod_qaf.data import DataConfig, TriModalDataset, collate_detection_batch
from mmod_qaf.model import TriModalModelConfig, build_from_pretrained
from mmod_qaf.train_loop import configure_reproducibility


def _gradient_report(model) -> dict[str, float]:
    report = {
        "rgb": sum(float(p.grad.abs().sum()) for p in model.model[:11].parameters() if p.grad is not None),
    }
    if model.ir_encoder is not None and model.depth_encoder is not None:
        report.update({
            "ir": sum(float(p.grad.abs().sum()) for p in model.ir_encoder.parameters() if p.grad is not None),
            "depth": sum(float(p.grad.abs().sum()) for p in model.depth_encoder.parameters() if p.grad is not None),
            "fusion": sum(
            float(p.grad.abs().sum())
            for name in ("qaf_p3", "qaf_p4", "qaf_p5")
            for module in [getattr(model, name)]
            if module is not None
            for p in module.parameters()
            if p.grad is not None
            ),
        })
    return report


def main():
    p = argparse.ArgumentParser(description="Run an official-Ultralytics multi-step forward/backward smoke test")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fusion", choices=("concat", "qaf"), default="qaf")
    p.add_argument("--rgb-only", action="store_true", help="Validate the same-loop RGB-only baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", help="Optional JSON report path, written only after the gate passes")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    configure_reproducibility(args.seed, deterministic=True)
    device = torch.device(args.device)
    ds = TriModalDataset(DataConfig(root=args.data, imgsz=args.imgsz, invalid_depth_policy="skip"))
    batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=collate_detection_batch)))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    cfg = TriModalModelConfig(
        fusion=args.fusion,
        fuse_p4=not args.rgb_only,
        fuse_p5=not args.rgb_only,
        rgb_only=args.rgb_only,
    )
    model = build_from_pretrained(args.weights, cfg, verbose=args.verbose).to(device)
    model.args = SimpleNamespace(epochs=3, box=9.83241, cls=0.64896, dfl=0.95824)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    optimization_losses = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss_vec, _ = model(batch)
        step_loss = loss_vec.sum()
        if not torch.isfinite(step_loss):
            raise RuntimeError(f"Non-finite optimization loss at step {step}: {step_loss.detach()}")
        step_loss.backward()
        optimizer.step()
        optimization_losses.append(float(step_loss.detach()))

    # Residual fusion starts as an exact RGB identity. Check branch connectivity only after two updates have had a
    # chance to open the residual path and, for Concat, its initially-zero auxiliary mixing weights.
    optimizer.zero_grad(set_to_none=True)
    loss_vec, loss_items = model(batch)
    loss = loss_vec.sum()
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite gradient-check loss: {loss.detach()}")
    loss.backward()
    grad_by_group = _gradient_report(model)
    model.eval()
    with torch.no_grad():
        decoded = model.predict_with_quality(batch["img"], batch["quality"])
    decoded_tensor = decoded[0] if isinstance(decoded, tuple) else decoded
    report = {
        "passed": True,
        "ultralytics_version": __import__("ultralytics").__version__,
        "torch_version": torch.__version__,
        "device": str(device),
        "imgsz": args.imgsz,
        "seed": args.seed,
        "weights": str(Path(args.weights).resolve()),
        "weights_sha256": hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
        "sample_ids": list(batch["sample_id"]),
        "fusion": "rgb_only" if args.rgb_only else args.fusion,
        "loss": float(loss.detach()),
        "optimization_losses": optimization_losses,
        "optimization_steps_before_gradient_check": 2,
        "loss_items": loss_items.detach().float().cpu().tolist(),
        "grad_abs_sum": grad_by_group,
        "decoded_shape": list(decoded_tensor.shape),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "extra_multimodal_parameters": model.extra_parameter_count(),
        "rgb_only": args.rgb_only,
        "fusion_weights": {k: v.float().cpu().tolist() for k, v in model.fusion_diagnostics().items()},
    }
    if not all(v > 0 for v in grad_by_group.values()):
        raise RuntimeError(f"A model branch has zero gradient: {grad_by_group}")
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
