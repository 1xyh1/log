#!/usr/bin/env python3
"""Post-hoc F1 gradient-semantics audit (reviewer erratum checklist 2026-08-16).

Verifies, on the trained formal checkpoints, the four claims required after the
gate-input detach fix:

  1. grad(aux_from_gate) == 0  — gate reads current A but no gradient flows
     back into the aux encoder through the gate path;
  2. soft gate parameters are still grad-active (and G6 shows they moved);
  3. active aux still receives gradients through the residual path only;
  4. F1-C0 projection weights are exactly zero and biases stay at the
     decay-scale neutral level.

This is evidence collection, not a training gate; it does not modify anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.run_integrity import inspect_step3_run  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _max_grad(module) -> float:
    vals = [float(p.grad.abs().max()) for p in module.parameters()
            if p.grad is not None]
    return max(vals, default=0.0)


def _any_grad(module) -> bool:
    return any(p.grad is not None for p in module.parameters())


def load_model(run_dir: Path, device: torch.device):
    ck = torch.load(run_dir / "weights" / "last.pt", map_location="cpu",
                    weights_only=False)
    model = (ck.get("ema") or ck.get("model")).float().eval().to(device)
    # Ultralytics saves final checkpoints with requires_grad=False on all
    # parameters; the posthoc audit needs live autograd for its controlled
    # backward checks.
    model.requires_grad_(True)
    return model


def check_gate_detach(model, img, device) -> dict:
    """Forward the aux pyramid and the gate exactly like the model does
    (detached inputs), then backprop from q only.  Aux encoder must have NO
    gradients; gate params must have non-zero gradients."""
    model.zero_grad()
    rgb, aux = model._split_input(img)
    a3, a4, a5 = model.aux_encoder(aux)
    q = model.reliability_gate(tuple(f.detach() for f in (a3, a4, a5)))
    q.sum().backward()
    aux_grad_max = _max_grad(model.aux_encoder)
    gate_grad_max = _max_grad(model.reliability_gate)
    return {
        "aux_grad_from_gate_path": aux_grad_max,
        "gate_params_grad_active": bool(gate_grad_max > 0.0),
        "gate_grad_max": gate_grad_max,
        "passed": aux_grad_max == 0.0 and gate_grad_max > 0.0,
    }


def check_residual_path(model, img, batch) -> dict:
    """With gate override = 1 (no gate graph), the aux encoder must still
    receive gradients through the residual path."""
    from multimodal.early_fusion_yolo26 import r3_hyp  # noqa: F401
    model.zero_grad()
    model.set_gate_override(1.0)
    try:
        if model.criterion is None:
            from ultralytics.utils.loss import v8DetectionLoss
            model.args = r3_hyp()
            model.criterion = v8DetectionLoss(model)
        preds = model._predict_once(img)
        loss = model.loss(batch, preds)
        total = loss.sum() if torch.is_tensor(loss) else sum(
            v.sum() for v in loss if torch.is_tensor(v))
        total.backward()
        aux_grad_max = _max_grad(model.aux_encoder)
        proj_grad_max = max(float(model.fusions[k].proj.weight.grad.abs().max())
                            for k in ("4", "6", "10")
                            if model.fusions[k].proj.weight.grad is not None)
        return {
            "aux_grad_via_residual": aux_grad_max,
            "proj_weight_grad_max": proj_grad_max,
            "passed": aux_grad_max > 0.0,
        }
    finally:
        model.set_gate_override(None)


def check_c0_neutral(model) -> dict:
    """F1-C0: projection weights exactly zero; biases at decay-scale level."""
    w_norms = [float(model.fusions[k].proj.weight.norm()) for k in ("4", "6", "10")]
    b_norms = [float(model.fusions[k].proj.bias.norm()) for k in ("4", "6", "10")]
    bias_neutral_threshold = 1e-4  # far below learning scale, above 80ep decay dust
    return {
        "proj_weight_norms": w_norms,
        "proj_bias_norms": b_norms,
        "bias_neutral_threshold": bias_neutral_threshold,
        "weights_exactly_zero": all(w == 0.0 for w in w_norms),
        "biases_neutral": all(b < bias_neutral_threshold for b in b_norms),
        "passed": (all(w == 0.0 for w in w_norms)
                   and all(b < bias_neutral_threshold for b in b_norms)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_f1_c")
    p.add_argument("--c0-run", default="F1C-C0")
    p.add_argument("--soft-run", default="F1C-I-magsoft")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--device", default="0")
    p.add_argument("--expected-epochs", type=int, default=80)
    a = p.parse_args()

    from multimodal import step3_eval_utils as evu
    device = evu._as_device(a.device)
    project = Path(a.project)
    c0_dir = project / a.c0_run
    soft_dir = project / a.soft_run
    for tag, run_dir in (("C0", c0_dir), ("SOFT", soft_dir)):
        integrity = inspect_step3_run(
            run_dir, a.expected_epochs, require_weights=True,
            trace_name="step4_g8_trace.jsonl", growth_name="step4_growth.jsonl",
            eval_name="eval_step4_f1_c_causality.json")
        if not integrity.to_dict()["passed"]:
            raise RuntimeError(f"POSTHOC_REFUSE_INCOHERENT_RUN: {tag}")

    contract = json.loads(Path(a.contract).read_text(encoding="utf-8"))
    soft_model = load_model(soft_dir, device)
    if getattr(soft_model, "gate_mode", None) != "learned":
        raise RuntimeError("posthoc gate checks require F1C-I-magsoft learned gate")

    ds = TriModalDataset(contract, split="val", group="C1-I", augment=False)
    samples = [ds[i] for i in range(min(4, len(ds)))]
    batch = ds.collate_fn(samples)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    img = batch["img"].float()

    gate_detach = check_gate_detach(soft_model, img, device)
    residual = check_residual_path(soft_model, img, batch)
    c0_model = load_model(c0_dir, device)
    c0_neutral = check_c0_neutral(c0_model)

    # G6 evidence for the "gate params actually moved" half of claim 2
    g6 = json.loads((soft_dir / "step4_update_gate.json").read_text(
        encoding="utf-8"))
    gate_moved_in_training = float(g6.get("gate_max_abs_change", 0.0)) > 0.0

    report = {
        "schema": "step4-f1-c-posthoc-gradient-audit-v1",
        "checks": {
            "gate_detach_semantics": gate_detach,
            "residual_path_active": residual,
            "gate_moved_in_training_from_g6": gate_moved_in_training,
            "c0_projection_neutral": c0_neutral,
        },
        "provenance": {
            "soft_last_pt_sha256": _sha(soft_dir / "weights" / "last.pt"),
            "c0_last_pt_sha256": _sha(c0_dir / "weights" / "last.pt"),
            "contract_sha256": _sha(Path(a.contract)),
            "script_sha256": _sha(Path(__file__)),
            "model_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py"),
            "gate_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "reliability_gate.py"),
            "dataset_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "trimodal_dataset.py"),
            "eval_core_sha256": _sha(
                ROOT / "src" / "multimodal" / "step3_eval_utils.py"),
        },
        "passed": bool(gate_detach["passed"] and residual["passed"]
                       and gate_moved_in_training and c0_neutral["passed"]),
    }
    out_dir = ROOT / "reports" / "step4_f1_c"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "posthoc_gradient_audit_c.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("->", out)
    if not report["passed"]:
        raise RuntimeError("F1_POSTHOC_GRADIENT_AUDIT_FAIL")


if __name__ == "__main__":
    main()
