from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.tseries_core import (  # noqa: E402
    FORMAL_BATCH, FORMAL_EPOCHS, FORMAL_SEED, RUN_NAMES, TREATMENTS,
    apply_treatment, center_full_map, effect_from_results, full_map_dc,
    optimizer_group_snapshot, optimizer_snapshots_equivalent,
    p5_mechanism_metrics, sha256_json, signed_summary,
    single_seed_paired_label,
)


def _metric(v: float):
    ids = [f"s{i}" for i in range(6)]
    return {
        "full": {"map50_95": float(v)},
        "loo": {sid: {"map50_95": float(v)} for sid in ids},
    }


@pytest.mark.parametrize("treatment", ["T0-N", "T1-F", "T2-A"])
def test_exact_treatment_set(treatment):
    assert treatment in TREATMENTS


@pytest.mark.parametrize("treatment", ["T0-N", "T1-F", "T2-A"])
def test_formal_run_names_are_seeded(treatment):
    assert "seed20260812" in RUN_NAMES[treatment]


def test_formal_protocol_constants():
    assert FORMAL_SEED == 20260812
    assert FORMAL_EPOCHS == 80
    assert FORMAL_BATCH == 4


@pytest.mark.parametrize("shape", [(1, 2, 3, 4), (1, 512, 20, 20), (2, 8, 5, 7), (3, 1, 4, 4)])
def test_center_full_map_zero_channel_mean(shape):
    torch.manual_seed(sum(shape))
    x = torch.randn(*shape)
    y = center_full_map(x)
    assert torch.allclose(y.mean((-2, -1)), torch.zeros_like(y.mean((-2, -1))), atol=1e-6, rtol=0)


@pytest.mark.parametrize("shape", [(1, 4, 3, 3), (2, 7, 5, 4), (1, 512, 20, 20)])
def test_full_map_dc_is_spatially_constant(shape):
    x = torch.randn(*shape)
    dc = full_map_dc(x)
    assert dc.shape == x.shape
    assert torch.equal(dc[..., :1, :1].expand_as(dc), dc)


@pytest.mark.parametrize("constant", [-4.0, -0.25, 0.0, 1.0, 7.5])
def test_center_invariant_to_spatial_channel_constant(constant):
    x = torch.randn(2, 3, 7, 9)
    b = torch.full((2, 3, 1, 1), float(constant))
    assert torch.allclose(center_full_map(x + b), center_full_map(x), atol=2e-6, rtol=0)


def test_t0_apply_returns_rgb_exact_object():
    r5 = torch.randn(1, 4, 3, 3, requires_grad=True)
    d = torch.randn_like(r5, requires_grad=True)
    fused, used = apply_treatment(r5, d, "T0-N")
    assert fused is r5
    assert torch.count_nonzero(used) == 0


def test_t0_delta_is_not_in_loss_graph():
    r5 = torch.randn(1, 4, 3, 3, requires_grad=True)
    d = torch.randn(1, 4, 3, 3, requires_grad=True)
    fused, _ = apply_treatment(r5, d, "T0-N")
    fused.sum().backward()
    assert r5.grad is not None
    assert d.grad is None


def test_t1_delta_is_in_loss_graph():
    r5 = torch.randn(1, 4, 3, 3, requires_grad=True)
    d = torch.randn(1, 4, 3, 3, requires_grad=True)
    fused, _ = apply_treatment(r5, d, "T1-F")
    fused.square().mean().backward()
    assert d.grad is not None
    assert float(d.grad.abs().max()) > 0


def test_t2_delta_is_in_loss_graph():
    r5 = torch.randn(1, 4, 3, 3, requires_grad=True)
    d = torch.randn(1, 4, 3, 3, requires_grad=True)
    fused, _ = apply_treatment(r5, d, "T2-A")
    fused.square().mean().backward()
    assert d.grad is not None


def test_unknown_treatment_rejected():
    with pytest.raises(ValueError):
        apply_treatment(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1), "BAD")


def test_t2_bias_forward_cancels_numerically():
    torch.manual_seed(5)
    conv = torch.nn.Conv2d(8, 16, 1, bias=True)
    x = torch.randn(2, 8, 11, 13)
    with torch.no_grad():
        y1 = center_full_map(conv(x))
        bias = conv.bias.detach().clone()
        conv.bias.add_(torch.linspace(-3, 3, steps=16))
        y2 = center_full_map(conv(x))
        conv.bias.copy_(bias)
    assert torch.allclose(y1, y2, atol=3e-6, rtol=0)


def test_fp32_raw_bias_gradient_need_not_be_bitwise_zero():
    # Regression for the implementation adjudication: mathematical cancellation
    # does not imply bitwise-zero FP32 reduction gradients.
    torch.manual_seed(6)
    conv = torch.nn.Conv2d(32, 32, 1, bias=True)
    x = torch.randn(1, 32, 20, 20)
    y = center_full_map(conv(x))
    g = torch.randn_like(y)
    (y * g).sum().backward()
    assert conv.bias.grad is not None
    assert torch.isfinite(conv.bias.grad).all()


@pytest.mark.parametrize("kind", ["T0-N", "T1-F", "T2-A"])
def test_mechanism_metrics_fields(kind):
    d = torch.randn(1, 4, 5, 5)
    r = torch.randn_like(d)
    _, used = apply_treatment(r, d, kind)
    m = p5_mechanism_metrics(d, used)
    for key in (
        "full_rms", "dc_rms", "ac_rms", "dc_over_full",
        "ac_over_full", "used_rms", "post_center_channel_mean_abs_max",
    ):
        assert key in m


def test_t2_mechanism_post_center_mean_small():
    d = torch.randn(2, 17, 9, 9)
    _, used = apply_treatment(torch.zeros_like(d), d, "T2-A")
    assert p5_mechanism_metrics(d, used)["post_center_channel_mean_abs_max"] <= 1e-6


@pytest.mark.parametrize(
    "full,vals,label",
    [
        (0.1, [0.1] * 6, "SEED20260812_POSITIVE_PAIRED_EVIDENCE"),
        (-0.1, [-0.1] * 6, "SEED20260812_NEGATIVE_PAIRED_EVIDENCE"),
        (0.1, [1, 1, 1, -1, -1, -1], "SEED20260812_INCONCLUSIVE_PAIRED_EVIDENCE"),
        (-0.1, [-1, -1, -1, 1, 1, 1], "SEED20260812_INCONCLUSIVE_PAIRED_EVIDENCE"),
    ],
)
def test_single_seed_paired_labels(full, vals, label):
    effect = signed_summary(full, {f"s{i}": v for i, v in enumerate(vals)})
    assert single_seed_paired_label(effect) == label


def test_effect_native_minus_donor_sign():
    native = _metric(0.3)
    donor = _metric(0.2)
    effect = effect_from_results(native, donor)
    assert effect["full"] == pytest.approx(0.1)
    assert effect["positive_folds"] == 6


def test_effect_loo_id_mismatch_rejected():
    native = _metric(0.3)
    donor = _metric(0.2)
    donor["loo"].pop("s5")
    with pytest.raises(RuntimeError, match="LOO_ID_MISMATCH"):
        effect_from_results(native, donor)


def test_sha256_json_order_independent():
    assert sha256_json({"b": 1, "a": 2}) == sha256_json({"a": 2, "b": 1})


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0]))
        self.b = torch.nn.Parameter(torch.tensor([2.0]))


def _opt(model, wd=0.0):
    return torch.optim.SGD([
        {"params": [model.a], "lr": 0.1, "weight_decay": wd},
        {"params": [model.b], "lr": 0.2, "weight_decay": 0.0},
    ])


def test_optimizer_snapshot_records_names_groups_wd():
    m = Tiny()
    snap = optimizer_group_snapshot(m, _opt(m, 0.3))
    assert snap["assignment"]["a"] == 0
    assert snap["assignment"]["b"] == 1
    assert snap["groups"][0]["weight_decay"] == pytest.approx(0.3)


def test_optimizer_equivalence_true():
    a, b = Tiny(), Tiny()
    assert optimizer_snapshots_equivalent(
        optimizer_group_snapshot(a, _opt(a)),
        optimizer_group_snapshot(b, _opt(b)),
    )


def test_optimizer_equivalence_detects_wd_difference():
    a, b = Tiny(), Tiny()
    assert not optimizer_snapshots_equivalent(
        optimizer_group_snapshot(a, _opt(a, 0.0)),
        optimizer_group_snapshot(b, _opt(b, 0.01)),
    )


@pytest.mark.parametrize(
    "path,needles",
    [
        ("src/multimodal/tseries_p5_model.py", [
            "self.p5_fusion = ZeroInitResidualFusion",
            "with torch.no_grad()",
            "y[P5_TAP] = fused5",
            "x = y[P5_TAP]",
            '"p3_direct_injection_count": 0',
            '"p4_direct_injection_count": 0',
            '"p5_direct_injection_count": 1',
        ]),
        ("scripts/run_tseries.py", [
            "build_tseries_model(Path(a.base_checkpoint)",
            "bias.grad.zero_()",
            "T_SERIES_T0_SILENT_OPTIMIZER_UPDATE",
            "T_SERIES_T2_BIAS_OPTIMIZER_UPDATE",
            "optimizer_manifest.json",
            "tseries_mechanism.jsonl",
            "tseries_data_order.jsonl",
        ]),
        ("scripts/smoke_tseries_suite.py", [
            "T_SERIES_STATIC_AUDIT_NOT_PASSING",
            "G12_t0_no_silent_optimizer_update",
            "G14_t2_bias_optimizer_safety",
            "G17_rng_data_order_closure",
        ]),
        ("scripts/eval_tseries_paired.py", [
            "T_SERIES_NATIVE_OVERRIDE_EQUIVALENCE_FAIL",
            "P5_AC_ALL_OWN_MEAN",
            "EXPECTED_DONOR_MAP_SHA256",
            "single_seed_paired_label",
        ]),
        ("scripts/eval_tseries_posttrain.py", [
            "last_val6",
            "train11",
            "all17",
            "late10",
        ]),
        ("scripts/summarize_tseries.py", [
            "STABLE_POSITIVE",
            "no_arbitrary_ap_margin",
            "single_seed_is_not_replication",
            '"depth_go": False',
        ]),
    ],
)
def test_critical_source_contracts(path, needles):
    text = (ROOT / path).read_text(encoding="utf-8")
    for needle in needles:
        assert needle in text


@pytest.mark.parametrize(
    "needle",
    [
        "P5-only direct IR injection",
        "T0-N",
        "T1-F",
        "T2-A",
        "NO P3 direct IR injection",
        "NO P4 direct IR injection",
        "NO reliability gate",
        "NO AC_CONTENT",
        "NO Depth",
        "G18",
        "epoch-0",
        "optimizer",
        "paired-causality",
    ],
)
def test_design_freeze_contains_required_contract(needle):
    text = (ROOT / "docs/step4_tseries/TRAINING_DESIGN_FREEZE.md").read_text(encoding="utf-8")
    assert needle in text


@pytest.mark.parametrize(
    "needle",
    [
        "FP32",
        "numerical-exactness guard",
        "proj.bias.grad",
        "weight decay",
        "bitwise unchanged",
        "new learned gate",
        "post-result patch",
    ],
)
def test_implementation_adjudication_contains_bias_hardening(needle):
    text = (ROOT / "docs/step4_tseries/IMPLEMENTATION_ADJUDICATION.md").read_text(encoding="utf-8")
    assert needle.lower() in text.lower()


def test_erratum_does_not_modify_science():
    text = (ROOT / "docs/step4_a4/feedback/2026-08-19_erratum.md").read_text(encoding="utf-8")
    assert "1/6 positive" in text
    assert "No experimental result changes" in text
    assert "No mechanism label changes" in text


@pytest.mark.parametrize(
    "file",
    [
        "scripts/run_tseries.py",
        "scripts/smoke_tseries_suite.py",
        "scripts/audit_tseries.py",
        "scripts/run_tseries_formal_suite.py",
        "scripts/eval_tseries_posttrain.py",
        "scripts/eval_tseries_paired.py",
        "scripts/summarize_tseries.py",
        "src/multimodal/tseries_core.py",
        "src/multimodal/tseries_p5_model.py",
        "src/multimodal/tseries_runtime.py",
    ],
)
def test_no_depth_treatment_code(file):
    text = (ROOT / file).read_text(encoding="utf-8")
    assert "C2-D" not in text
    assert 'aux_mode = "depth"' not in text


@pytest.mark.parametrize(
    "file",
    [
        "src/multimodal/tseries_p5_model.py",
        "scripts/run_tseries.py",
        "scripts/smoke_tseries_suite.py",
        "scripts/eval_tseries_paired.py",
    ],
)
def test_no_reliability_gate_execution(file):
    text = (ROOT / file).read_text(encoding="utf-8")
    assert "_effective_gate(" not in text
    assert "q_native" not in text


def test_static_audit_runs_package_only(tmp_path):
    out = tmp_path / "audit.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_tseries.py"),
         "--phase", "static", "--out", str(out)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    obj = json.loads(out.read_text(encoding="utf-8"))
    assert obj["schema"] == "step4-tseries-pretraining-audit-v1"
    assert obj["phase"] == "static"
    assert obj["all_passed"] is True
    assert obj["failed"] == []


def test_posttrain_summary_never_directly_authorizes_depth():
    text = (ROOT / "scripts/summarize_tseries.py").read_text(encoding="utf-8")
    assert '"depth_go": False' in text
    assert '"production_go": False' in text


def test_formal_suite_order_is_t0_t1_t2():
    text = (ROOT / "scripts/run_tseries_formal_suite.py").read_text(encoding="utf-8")
    assert 'for treatment in ("T0-N", "T1-F", "T2-A")' in text


def test_formal_suite_is_80_epochs():
    text = (ROOT / "scripts/run_tseries_formal_suite.py").read_text(encoding="utf-8")
    assert '"--epochs", "80"' in text
    assert '"--seed", "20260812"' in text
