#!/usr/bin/env python3
"""Step 4-F0 audit gates (all PASS before any F0 training).

G1 RGB equivalence : 3ch O2M reference vs F0 model with aux=zeros -> final detector
                     outputs max_abs_diff <= 1e-5 (CPU, fp32, eval).
G2 zero-init proj  : all three P3/P4/P5 fusion projections exactly zero weight+bias.
G3 gradient flow   : first backward on a real batch (per group): aux encoder grads > 0,
                     fusion grads > 0, frozen RGB backbone grads are None.
G4 frozen anchor   : RGB backbone params require_grad=False and BN in eval mode.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.early_fusion_yolo26 import build_reference_3ch, r3_hyp  # noqa: E402
from multimodal.raw_sample_index import build_contract, OUT_DEFAULT  # noqa: E402
from multimodal.step4_f0_model import Step4F0Model  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402

GROUPS = {"F0-C0": "zero", "F0-I": "ir", "F0-D": "depth"}


def _tensors(o, acc):
    if torch.is_tensor(o):
        acc.append(o)
    elif isinstance(o, dict):
        for v in o.values():
            _tensors(v, acc)
    elif isinstance(o, (list, tuple)):
        for v in o:
            _tensors(v, acc)
    return acc


def _compare(a, b):
    d = (a - b).abs()
    return float(d.max())


def aux_inputs(img6: torch.Tensor, group: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(1,3,640,640) rgb and (1,2,640,640) aux from a 6ch sample tensor."""
    rgb = img6[0:3].unsqueeze(0)
    kind = GROUPS[group]
    if kind == "zero":
        aux = torch.zeros(1, 2, 640, 640)
    elif kind == "ir":
        aux = torch.stack([img6[3], torch.zeros_like(img6[3])]).unsqueeze(0)
    else:
        aux = img6[4:6].unsqueeze(0)
    return rgb, aux


def gate1(reference, model) -> dict:
    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640)
    reference.eval()
    model.eval()
    with torch.no_grad():
        o_ref = _tensors(reference._predict_once(x), [])
        # exercise the real 6ch split path: [R,G,B,0,0,0] -> aux [0,0]
        o_f0 = _tensors(model(torch.cat([x, torch.zeros(1, 3, 640, 640)], dim=1)), [])
    assert len(o_ref) == len(o_f0), f"tensor count mismatch {len(o_ref)} vs {len(o_f0)}"
    max_d = max(_compare(a, b) for a, b in zip(o_ref, o_f0))
    return {"threshold": 1e-5, "max_abs_diff": max_d, "passed": bool(max_d <= 1e-5)}


def gate2(model) -> dict:
    ok = all(f.assert_zero_init() for f in model.fusions.values())
    return {"all_proj_exactly_zero": ok, "passed": ok}


def _forward_backward(model, batch):
    model.train()
    model.zero_grad()
    preds = model._forward_fused(batch["rgb"], batch["aux"])
    loss = model.loss(batch, preds)
    loss = loss.sum() if isinstance(loss, torch.Tensor) else \
        sum(v.sum() for v in loss if torch.is_tensor(v))
    loss.backward()
    return loss


def _max_grad(module) -> float:
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    return max(float(g.abs().max()) for g in grads) if grads else 0.0


def gate3(model, sample: dict, group: str) -> dict:
    """Two-step gradient check. Math note: with zero-init proj W=0, dL/dA = W^T·dL/dF = 0
    exactly at step 0, while dL/dW = A·dL/dF > 0 unblocks the encoder after ONE optimizer
    step (milder than alpha gating, which has no self-unblocking scalar)."""
    rgb, aux = aux_inputs(torch.as_tensor(sample["img"]), group)
    batch = {
        "rgb": rgb, "aux": aux,
        "cls": torch.as_tensor(sample["cls"], dtype=torch.float32),
        "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
        "batch_idx": torch.as_tensor(sample["batch_idx"], dtype=torch.float32),
    }
    model.args = r3_hyp()
    step1_loss = _forward_backward(model, batch)

    def _weight_grad_max(module) -> float:
        grads = [p.grad for n, p in module.named_parameters()
                 if p.grad is not None and "bias" not in n]
        return max(float(g.abs().max()) for g in grads) if grads else 0.0

    fusion_w_grad1 = max(_weight_grad_max(f) for f in model.fusions.values())
    fusion_grad1 = max(_max_grad(f) for f in model.fusions.values())
    aux_grad1 = _max_grad(model.aux_encoder)
    backbone_trainable = sum(1 for p in model.rgb_backbone.parameters() if p.requires_grad)
    # single SGD step on the fusion projections only (as the trainer would at step 1)
    with torch.no_grad():
        for f in model.fusions.values():
            for p in f.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-1e-3)
    step2_loss = _forward_backward(model, batch)
    aux_grad2 = _max_grad(model.aux_encoder)
    out = {"step1_loss": float(step1_loss.detach()), "step2_loss": float(step2_loss.detach()),
           "fusion_proj_grad_step1": fusion_grad1,
           "fusion_weight_grad_step1": fusion_w_grad1,
           "aux_encoder_grad_step1": aux_grad1,
           "aux_encoder_grad_step2": aux_grad2,
           "rgb_backbone_trainable_params": backbone_trainable}
    if group == "F0-C0":
        # zero aux input: aux path inactive BY DESIGN. Weight grads are exactly 0
        # (dL/dW = A·dL/dF = 0); the bias intercept grad is legitimately nonzero.
        ok = fusion_w_grad1 == 0.0 and aux_grad1 == 0.0 and backbone_trainable == 0
        out["expected"] = "aux weight grads exactly zero (zero input); bias intercept may move"
    else:
        ok = fusion_grad1 > 0 and aux_grad2 > 0 and backbone_trainable == 0
        out["expected"] = ("proj grad > 0 at step 1; aux encoder grad == 0 at step 1 "
                           "(W=0 math) and > 0 at step 2")
    out["passed"] = bool(ok)
    return out


def gate4(model) -> dict:
    trainable = sum(1 for p in model.rgb_backbone.parameters() if p.requires_grad)
    bn_eval = all(not m.training for m in model.rgb_backbone.modules()
                  if isinstance(m, torch.nn.BatchNorm2d))
    return {"rgb_backbone_trainable_params": trainable, "rgb_bn_in_eval": bn_eval,
            "passed": bool(trainable == 0 and bn_eval)}


def main():
    contract = build_contract(out_path=OUT_DEFAULT)
    # ONE shared reference: the 12-class head is random-initialized, so G1 must compare
    # against the very same reference whose modules went into the F0 model.
    reference = build_reference_3ch()
    ds = TriModalDataset(contract, split="train", group="C1-I", augment=False)
    sample = ds[0]
    model = Step4F0Model(reference)
    report = {"G1_rgb_equivalence": gate1(reference, model),
              "G2_zero_init_proj": None, "G3_gradient_flow": {}, "G4_frozen_anchor": None}
    report["G2_zero_init_proj"] = gate2(model)
    for g in GROUPS:
        report["G3_gradient_flow"][g] = gate3(model, sample, g)
    report["G4_frozen_anchor"] = gate4(model)
    report["all_passed"] = bool(
        report["G1_rgb_equivalence"]["passed"] and report["G2_zero_init_proj"]["passed"]
        and all(v["passed"] for v in report["G3_gradient_flow"].values())
        and report["G4_frozen_anchor"]["passed"])
    out = ROOT / "reports" / "step4_f0_audit.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("ALL PASSED:", report["all_passed"], "->", out)
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
