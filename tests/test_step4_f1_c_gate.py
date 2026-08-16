"""F1-C magnitude gate unit tests (G10 items 1-6, torch CPU only).

G10 (reviewer-frozen):
  1. per-sample log-RMS of one sample == its value inside a batch
  2. batch permutation does not change per-sample descriptors
  3. with the zero-init magnitude branch, new gate q == old gate q initially
  4. magnitude_fc.weight grad is finite/nonzero; a controlled update leaves 0
  5. gate->aux grad stays 0; residual->aux grad stays nonzero (detach semantics)
  6. q is always finite Bx1; RGB and other modality inputs are not modified
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.reliability_gate import (  # noqa: E402
    MagnitudeReliabilityGate, PyramidScalarReliabilityGate, per_sample_log_rms)


def _features(batch=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.rand(batch, 256, 20, 20, generator=g),
        torch.rand(batch, 256, 10, 10, generator=g),
        torch.rand(batch, 512, 5, 5, generator=g),
    )


class TestLogRms:
    def test_batch_matches_single_sample(self):
        a3, a4, a5 = _features()
        batch_rms = per_sample_log_rms(a3)
        single_rms = per_sample_log_rms(a3[1:2])
        assert torch.allclose(batch_rms[1], single_rms[0], atol=0.0, rtol=0.0)

    def test_permutation_invariant(self):
        a3, a4, a5 = _features()
        perm = [2, 0, 1]
        rms_orig = per_sample_log_rms(a3)
        rms_perm = per_sample_log_rms(a3[perm])
        assert torch.equal(rms_orig[perm], rms_perm)

    def test_formula_matches_audit(self):
        a3, _, _ = _features()
        manual = torch.log((a3.pow(2).mean(dim=(1, 2, 3))).sqrt() + 1e-9)
        assert torch.equal(manual, per_sample_log_rms(a3))


class TestInitEquivalence:
    def test_zero_magnitude_branch_matches_old_gate(self):
        feats = _features()
        torch.manual_seed(2026081200)
        old = PyramidScalarReliabilityGate()
        torch.manual_seed(2026081200)
        new = MagnitudeReliabilityGate()
        with torch.no_grad():
            q_old = old(tuple(f.detach() for f in feats))
            q_new = new(tuple(f.detach() for f in feats))
        assert torch.equal(q_old, q_new)  # bitwise equivalence at init

    def test_magnitude_fc_zero_init(self):
        torch.manual_seed(0)
        gate = MagnitudeReliabilityGate()
        assert float(gate.magnitude_fc.weight.abs().max()) == 0.0
        assert gate.magnitude_fc.bias is None


class TestGradients:
    def test_magnitude_fc_grad_finite_nonzero(self):
        torch.manual_seed(0)
        gate = MagnitudeReliabilityGate()
        feats = _features()
        q = gate(feats)
        q.sum().backward()
        grad = gate.magnitude_fc.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert float(grad.abs().max()) > 0.0

    def test_controlled_update_leaves_zero(self):
        torch.manual_seed(0)
        gate = MagnitudeReliabilityGate()
        q = gate(_features())
        q.sum().backward()
        with torch.no_grad():
            gate.magnitude_fc.weight -= 0.1 * gate.magnitude_fc.weight.grad
        assert float(gate.magnitude_fc.weight.abs().max()) > 0.0

    def test_gate_detach_blocks_aux_gradient(self):
        """The F1-C model feeds the gate detached A; reproduce the semantics:
        gradient through the gate must not reach the aux features, even when
        the features THEMSELVES require grad (strong test, reviewer P0)."""
        feats = [f.clone().requires_grad_(True) for f in _features()]
        detached = tuple(f.detach() for f in feats)
        torch.manual_seed(0)
        gate = MagnitudeReliabilityGate()
        q = gate(detached)
        q.sum().backward()
        for f in feats:
            assert f.grad is None

    def test_residual_path_aux_gradient_nonzero(self):
        """Residual path must use the ACTUAL _gated_residual semantics:
        F = R + q * P(A) with the gate detached from A."""
        from multimodal.step4_f1_ir_gate_model import Step4F1IRGateModel

        feats = [f.clone().requires_grad_(True) for f in _features()]
        a3 = feats[0]
        proj = torch.nn.Conv2d(256, 256, 1, bias=True)
        torch.nn.init.normal_(proj.weight, std=0.01)
        q = torch.ones(1, 1)
        r = torch.randn(1, 256, 20, 20)
        fused = Step4F1IRGateModel._gated_residual(proj, r, a3[0:1], q)
        fused.sum().backward()
        assert a3.grad is not None
        assert float(a3.grad.abs().max()) > 0.0


class TestOutputs:
    def test_q_finite_b1(self):
        torch.manual_seed(0)
        gate = MagnitudeReliabilityGate()
        with torch.no_grad():
            q = gate(_features(batch=3))
        assert q.shape == (3, 1)
        assert torch.isfinite(q).all()
        assert float(q.min()) >= 0.0 and float(q.max()) <= 1.0

    def test_inputs_not_modified(self):
        feats = _features()
        copies = [f.clone() for f in feats]
        torch.manual_seed(0)
        gate = MagnitudeReliabilityGate()
        with torch.no_grad():
            gate(feats)
        for original, after in zip(copies, feats):
            assert torch.equal(original, after)

    def test_gate_does_not_touch_rgb_or_depth_channels(self):
        """G10.6 (strong): the gate consumes ONLY the aux pyramid; a 6ch input
        keeps RGB (channels 0-2) and Depth (channels 4-5) byte-identical."""
        from multimodal.step4_f1_ir_gate_model import Step4F1IRGateModel
        from multimodal.early_fusion_yolo26 import build_reference_3ch

        torch.manual_seed(MODEL_SEED := 2026081200)
        model = Step4F1IRGateModel(build_reference_3ch(), aux_mode="ir",
                                   gate_mode="learned", gate_module="magnitude")
        img = torch.rand(2, 6, 80, 80)
        rgb_before = img[:, :3].clone()
        dep_before = img[:, 4:6].clone()
        with torch.no_grad():
            model._predict_once(img)
        assert torch.equal(img[:, :3], rgb_before)
        assert torch.equal(img[:, 4:6], dep_before)
