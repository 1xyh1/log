#!/usr/bin/env python3
"""Pre-training hard gates for Step 4-F1 IR scalar reliability fusion."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.early_fusion_yolo26 import (  # noqa: E402
    MODEL_INIT_SEED,
    build_reference_3ch,
    r3_hyp,
)
from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.step4_f1_ir_gate_model import Step4F1IRGateModel  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate0_f0_closeout(f0_project: Path) -> dict:
    """Require the frozen F0 verdict to pin the exact current LOO bytes/code."""
    summary_path = f0_project / "_summary_step4.json"
    loo_path = f0_project / "step4_loo.json"
    if not summary_path.exists() or not loo_path.exists():
        return {
            "summary_exists": summary_path.exists(),
            "loo_exists": loo_path.exists(),
            "passed": False,
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    eval_checks = summary.get("provenance_all_seven") or {}
    loo_checks = summary.get("loo_provenance") or {}
    checks = {
        "schema_v2": summary.get("schema") == "step4-f0-summary-v2",
        "verdict_frozen": summary.get("verdict_frozen") is True,
        "loo_file_pin": summary.get("loo_file_sha256") == _sha(loo_path),
        "summary_source_pin": summary.get("summarize_source_sha256") == _sha(
            ROOT / "scripts" / "summarize_step4.py"
        ),
        "g8_passed": (summary.get("g8_actual") or {}).get("passed") is True,
        "g6_all_passed": bool(summary.get("g6_unified_rejudge")) and all(
            row.get("unified_threshold_rejudge_passed") is True
            for row in summary.get("g6_unified_rejudge", {}).values()
        ),
        "eval_provenance_all_match": bool(eval_checks) and all(
            row.get("match") is True
            for group in eval_checks.values() for row in group.values()
        ),
        "loo_provenance_all_match": bool(loo_checks) and all(
            row.get("match") is True for row in loo_checks.values()
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _tensors(obj, acc):
    if torch.is_tensor(obj):
        acc.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            _tensors(value, acc)
    elif isinstance(obj, (tuple, list)):
        for value in obj:
            _tensors(value, acc)
    return acc


def _max_grad(module) -> float:
    values = [
        float(p.grad.detach().abs().max()) for p in module.parameters()
        if p.grad is not None
    ]
    return max(values, default=0.0)


def _loss_backward(model, batch):
    model.train()
    model.zero_grad()
    preds = model._predict_once(batch["img"])
    loss = model.loss(batch, preds)
    loss = loss.sum() if torch.is_tensor(loss) else sum(
        value.sum() for value in loss if torch.is_tensor(value)
    )
    loss.backward()
    return float(loss.detach())


def _batch(sample):
    return {
        "img": torch.as_tensor(sample["img"])[None],
        "cls": torch.as_tensor(sample["cls"], dtype=torch.float32),
        "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
        "batch_idx": torch.as_tensor(sample["batch_idx"], dtype=torch.float32),
    }


def gate1_initial_rgb_equivalence() -> dict:
    reference = build_reference_3ch()
    model = Step4F1IRGateModel(reference, aux_mode="ir", gate_mode="learned")
    reference.eval()
    model.eval()
    torch.manual_seed(0)
    rgb = torch.rand(1, 3, 640, 640)
    img6 = torch.cat([rgb, torch.rand(1, 1, 640, 640),
                      torch.zeros(1, 2, 640, 640)], dim=1)
    with torch.no_grad():
        expected = _tensors(reference._predict_once(rgb), [])
        actual = _tensors(model._predict_once(img6), [])
    if len(expected) != len(actual) or not expected:
        return {
            "reference_tensor_count": len(expected),
            "f1_tensor_count": len(actual),
            "max_abs_diff": None,
            "threshold": 1e-5,
            "passed": False,
        }
    if any(x.shape != y.shape for x, y in zip(expected, actual)):
        return {
            "reference_tensor_count": len(expected),
            "f1_tensor_count": len(actual),
            "shape_match": False,
            "max_abs_diff": None,
            "threshold": 1e-5,
            "passed": False,
        }
    max_abs = max(float((x - y).abs().max()) for x, y in zip(expected, actual))
    return {"max_abs_diff": max_abs, "threshold": 1e-5,
            "reference_tensor_count": len(expected),
            "f1_tensor_count": len(actual),
            "shape_match": True,
            "passed": bool(max_abs <= 1e-5)}


def gate2_gate_contract() -> dict:
    torch.manual_seed(MODEL_INIT_SEED)
    model = Step4F1IRGateModel(build_reference_3ch())
    features = (
        torch.randn(3, 256, 8, 8),
        torch.randn(3, 256, 4, 4),
        torch.randn(3, 512, 2, 2),
    )
    q = model.reliability_gate(features)
    learned_ok = bool(q.shape == (3, 1) and torch.isfinite(q).all()
                      and (q > 0).all() and (q < 1).all())
    model.set_gate_override(0.0)
    q0 = model._effective_gate(features)
    model.set_gate_override(1.0)
    q1 = model._effective_gate(features)
    model.set_gate_override(None)
    override_ok = bool(torch.equal(q0, torch.zeros_like(q0))
                       and torch.equal(q1, torch.ones_like(q1)))
    zero_proj = all(f.assert_zero_init() for f in model.fusions.values())
    return {
        "learned_q_min": float(q.min()),
        "learned_q_max": float(q.max()),
        "override_exact": override_ok,
        "projections_exact_zero": zero_proj,
        "passed": bool(learned_ok and override_ok and zero_proj),
    }


def gate3_gradient_unlock(contract) -> dict:
    torch.manual_seed(MODEL_INIT_SEED)
    model = Step4F1IRGateModel(build_reference_3ch(), aux_mode="ir")
    model.args = r3_hyp()
    sample = TriModalDataset(
        contract, split="train", group="C1-I", augment=False
    )[0]
    batch = _batch(sample)
    loss1 = _loss_backward(model, batch)
    proj_grad1 = max(_max_grad(fusion) for fusion in model.fusions.values())
    gate_grad1 = _max_grad(model.reliability_gate)
    aux_grad1 = _max_grad(model.aux_encoder)
    with torch.no_grad():
        for fusion in model.fusions.values():
            for param in fusion.parameters():
                if param.grad is not None:
                    param.add_(param.grad, alpha=-1e-3)
    loss2 = _loss_backward(model, batch)
    gate_grad2 = _max_grad(model.reliability_gate)
    aux_grad2 = _max_grad(model.aux_encoder)
    rgb_grad = _max_grad(model.rgb_backbone)
    passed = bool(
        proj_grad1 > 0.0 and gate_grad1 == 0.0 and aux_grad1 == 0.0
        and gate_grad2 > 0.0 and aux_grad2 > 0.0 and rgb_grad == 0.0
        and all(not p.requires_grad for p in model.rgb_backbone.parameters())
    )
    return {
        "step1_loss": loss1,
        "step2_loss": loss2,
        "step1_projection_grad": proj_grad1,
        "step1_gate_grad": gate_grad1,
        "step1_aux_grad": aux_grad1,
        "step2_gate_grad": gate_grad2,
        "step2_aux_grad": aux_grad2,
        "rgb_grad": rgb_grad,
        "passed": passed,
    }


def gate4_p5_route() -> dict:
    model = Step4F1IRGateModel(build_reference_3ch(), aux_mode="ir")
    model.eval()
    rgb = torch.rand(1, 3, 640, 640)
    aux = torch.rand(1, 2, 640, 640)
    captured = {}

    def hook(_module, inputs, _output):
        captured["input"] = inputs[0].detach().clone()

    handle = model.tail[0].register_forward_hook(hook)
    with torch.no_grad():
        model._forward_fused(rgb, aux)
        before = captured["input"]
        model.fusions["10"].proj.weight.fill_(0.25)
        captured.clear()
        model._forward_fused(rgb, aux)
        after = captured["input"]
    handle.remove()
    changed = not torch.allclose(before, after)
    return {"neck11_input_changed_after_p5_residual": changed, "passed": changed}


def gate5_matched_initial_state() -> dict:
    hashes = {}
    for group, aux_mode, gate_mode in (
        ("F1-C0", "zero", "learned"),
        ("F1-I-fixed", "ir", "fixed_one"),
        ("F1-I-soft", "ir", "learned"),
    ):
        torch.manual_seed(MODEL_INIT_SEED)
        model = Step4F1IRGateModel(
            build_reference_3ch(), aux_mode=aux_mode, gate_mode=gate_mode
        )
        h = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        hashes[group] = h.hexdigest()
    passed = len(set(hashes.values())) == 1
    return {"state_sha256": hashes, "passed": passed}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--f0-project", default="runs/step4_f0")
    a = p.parse_args()
    # Read the already-frozen contract; an F1 model audit must not rebuild or mutate it.
    contract = json.loads(Path(a.contract).read_text(encoding="utf-8"))
    report = {
        "schema": "step4-f1-ir-gate-audit-v1",
        "provenance": {
            "contract_sha256": _sha(Path(a.contract)),
            "audit_source_sha256": _sha(Path(__file__)),
            "model_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py"
            ),
            "gate_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "reliability_gate.py"
            ),
            "f0_model_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_f0_model.py"
            ),
            "aux_encoder_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "aux_encoder.py"
            ),
            "feature_fusion_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "feature_fusion.py"
            ),
            "dataset_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "trimodal_dataset.py"
            ),
            "trainability_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "trainability.py"
            ),
            "f0_summary_sha256": _sha(
                Path(a.f0_project) / "_summary_step4.json"
            ),
            "f0_loo_sha256": _sha(Path(a.f0_project) / "step4_loo.json"),
            "f0_summarizer_source_sha256": _sha(
                ROOT / "scripts" / "summarize_step4.py"
            ),
            "f0_closeout_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_closeout.py"
            ),
            "torch_version": torch.__version__,
            "ultralytics_version": __import__("ultralytics").__version__,
        },
        "G0_f0_closeout": gate0_f0_closeout(Path(a.f0_project)),
        "G1_initial_rgb_equivalence": gate1_initial_rgb_equivalence(),
        "G2_gate_contract": gate2_gate_contract(),
        "G3_gradient_unlock": gate3_gradient_unlock(contract),
        "G4_p5_route": gate4_p5_route(),
        "G5_matched_initial_state": gate5_matched_initial_state(),
    }
    report["all_passed"] = all(
        value["passed"] for key, value in report.items() if key.startswith("G")
    )
    out = ROOT / "reports" / "step4_f1_ir_gate" / "pretrain_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
