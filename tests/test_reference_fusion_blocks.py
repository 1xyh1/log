from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.reference_fusion_blocks import (  # noqa: E402
    IdentityConcatFusion,
    RDTTrackStyleDecorrelation,
    ResidualPromptFusion,
    SoftModalityGate,
    StrictOrthogonalDecorrelation,
    inspect_yolo26_backbone_taps,
)
from multimodal.trainability import enforce_frozen_module_eval, freeze_module  # noqa: E402


def test_identity_concat_is_exact_rgb_identity():
    torch.manual_seed(1)
    f = IdentityConcatFusion(8, 3)
    rgb, ir, dep = [torch.randn(2, 8, 10, 12) for _ in range(3)]
    y = f([rgb, ir, dep])
    assert torch.equal(y, rgb)


def test_identity_concat_aux_kernel_gets_gradient():
    torch.manual_seed(2)
    f = IdentityConcatFusion(4, 3)
    rgb = torch.randn(2, 4, 5, 5)
    ir = torch.randn(2, 4, 5, 5)
    dep = torch.randn(2, 4, 5, 5)
    target = torch.randn_like(rgb)
    (f([rgb, ir, dep]) - target).square().mean().backward()
    g = f.proj.weight.grad
    assert g is not None
    assert float(g[:, 4:8].norm()) > 0
    assert float(g[:, 8:12].norm()) > 0


def test_strict_projection_identical_and_orthogonal_cases():
    x = torch.tensor([[[[1.0]], [[0.0]]]])
    same = x.clone()
    proj_same = StrictOrthogonalDecorrelation._project(x, same, 1e-9)
    assert torch.allclose(proj_same, x, atol=1e-6)

    y = torch.tensor([[[[0.0]], [[1.0]]]])
    proj_orth = StrictOrthogonalDecorrelation._project(x, y, 1e-9)
    assert torch.allclose(proj_orth, torch.zeros_like(x), atol=1e-6)


def test_decorrelation_zero_input_finite():
    for cls in (StrictOrthogonalDecorrelation, RDTTrackStyleDecorrelation):
        m = cls(8, 4)
        z = torch.zeros(2, 8, 6, 6)
        y = m(z, z)
        assert y.shape == z.shape
        assert torch.isfinite(y).all()


def test_residual_prompt_identity_initialization():
    m = ResidualPromptFusion(8, 4)
    rgb = torch.randn(2, 8, 7, 7)
    aux = torch.randn_like(rgb)
    assert torch.equal(m(rgb, aux), rgb)


def test_soft_gate_weights_sum_to_one_and_quality_prior_monotonic():
    torch.manual_seed(3)
    m = SoftModalityGate(6, 3)
    xs = [torch.randn(2, 6, 4, 4) for _ in range(3)]
    base = m.weights(xs)
    prior = torch.zeros(2, 3)
    prior[:, 1] = 8.0
    boosted = m.weights(xs, prior)
    assert torch.allclose(base.sum(1), torch.ones(2), atol=1e-6)
    assert torch.allclose(boosted.sum(1), torch.ones(2), atol=1e-6)
    assert torch.all(boosted[:, 1] > base[:, 1])


def test_missing_modality_zero_is_finite():
    m = SoftModalityGate(4, 3)
    rgb = torch.randn(1, 4, 8, 8)
    zero = torch.zeros_like(rgb)
    y, w = m([rgb, zero, zero], return_weights=True)
    assert torch.isfinite(y).all() and torch.isfinite(w).all()
    # identity_start=True protects RGB at initialization.
    assert torch.equal(y, rgb)


def test_frozen_rgb_bn_stats_unchanged_after_outer_train():
    rgb = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4), nn.SiLU())
    freeze_module(rgb, freeze_bn_stats=True)
    before_mean = rgb[1].running_mean.clone()
    before_var = rgb[1].running_var.clone()

    # Simulate an outer model.train() call, then the Step4 freeze callback.
    rgb.train()
    enforce_frozen_module_eval(rgb)
    with torch.no_grad():
        rgb(torch.randn(4, 3, 16, 16))
    assert torch.equal(rgb[1].running_mean, before_mean)
    assert torch.equal(rgb[1].running_var, before_var)
    assert all(not p.requires_grad for p in rgb.parameters())


def test_yolo26_backbone_taps_are_p3_p4_p5_if_snapshot_available():
    pytest.importorskip("ultralytics")
    snapshot = ROOT / "step3_6ch_rgb_equiv_init.pt"
    if not snapshot.exists():
        pytest.skip("project snapshot not present")
    ck = torch.load(snapshot, map_location="cpu", weights_only=False)
    m6 = ck["model"]
    # A Step4 RGB anchor should be 3ch, while current Step3 snapshot is 6ch; the helper
    # still validates feature strides if asked to use the physical input channel count.
    report = inspect_yolo26_backbone_taps(m6, imgsz=640, in_channels=6)
    assert report["P3"]["stride"] == 8
    assert report["P4"]["stride"] == 16
    assert report["P5"]["stride"] == 32
