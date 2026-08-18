from __future__ import annotations

import hashlib
import sys
import weakref
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.step4_a2_residual_interventions import (  # noqa: E402
    MASK_CONDITIONS, ResidualCondition, SCALES, build_residual_cache,
    classify_paired_effect, forward_with_residual_intervention, gain_conditions,
    mask_conditions, shuffle_conditions,
)


class IdentityIndexed(nn.Module):
    def __init__(self, i: int):
        super().__init__()
        self.i = i
        self.f = -1

    def forward(self, x):
        return x


class FakeAux(nn.Module):
    def forward(self, x):
        # Distinct scale values from untouched recipient aux.
        return x[:, :1] + 1.0, x[:, :1] + 2.0, x[:, :1] + 3.0


class ScaleProj(nn.Module):
    def __init__(self, gain: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(gain))

    def forward(self, x):
        return x * self.weight


class Fusion(nn.Module):
    def __init__(self, gain: float):
        super().__init__()
        self.proj = ScaleProj(gain)


class FakeGate(nn.Module):
    def __init__(self, parent):
        super().__init__()
        object.__setattr__(self, "_parent_ref", weakref.ref(parent))

    def forward(self, features):
        parent = self._parent_ref()
        parent.gate_calls += 1
        parent.gate_inputs.append(tuple(f.detach().clone() for f in features))
        # q depends on all recipient features so changing gate input would be obvious.
        v = sum(f.mean(dim=(1, 2, 3), keepdim=False) for f in features)
        return torch.sigmoid(v).view(-1, 1)


class FakeModel(nn.Module):
    def __init__(self, fixed=False):
        super().__init__()
        self.aux_mode = "ir"
        self.gate_mode = "fixed_one" if fixed else "learned"
        self._gate_override = None
        self.gate_calls = 0
        self.gate_inputs = []
        # Need audited tap indices 4/6/10 to exist in y.
        self.rgb_backbone = nn.ModuleList([IdentityIndexed(i) for i in range(11)])
        self.tail = nn.ModuleList([])
        self.save = set()
        self.aux_encoder = FakeAux()
        self.fusions = nn.ModuleDict({"4": Fusion(1.0), "6": Fusion(2.0), "10": Fusion(3.0)})
        self.reliability_gate = FakeGate(self)
        self.fixed = fixed

    def _split_input(self, x):
        return x[:, :1], x[:, 1:2]

    def _effective_gate(self, features):
        raw = self.reliability_gate(features)
        return torch.ones_like(raw) if self.fixed else raw

    def native(self, x):
        rgb, aux = self._split_input(x)
        y = [None] * 11
        z = rgb
        for m in self.rgb_backbone:
            z = m(z)
            y[m.i] = z
        a = self.aux_encoder(aux)
        q = self._effective_gate(tuple(f.detach() for f in a))
        for scale, tap, feat in zip(SCALES, (4, 6, 10), a):
            delta = self.fusions[str(tap)].proj(feat)
            y[tap] = y[tap] + q.view(-1, 1, 1, 1) * delta
        return y[10]


def model_sha(model):
    h = hashlib.sha256()
    for n, p in sorted(model.state_dict().items()):
        h.update(n.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def recipient():
    # channel0 RGB=10, channel1 aux=2
    return torch.tensor([[[[10.0]], [[2.0]]]])


def test_factorial_is_exact_and_complete():
    assert tuple(c.name for c in mask_conditions()) == MASK_CONDITIONS
    assert set(MASK_CONDITIONS) == {
        "M000", "M100", "M010", "M001", "M110", "M101", "M011", "M111"
    }


def test_shuffle_condition_set_is_two_per_scale():
    names = [c.name for c in shuffle_conditions()]
    assert len(names) == 6
    for scale in SCALES:
        assert f"SHUFFLE_{scale}_COND" in names
        assert f"SHUFFLE_{scale}_ONLY" in names


def test_gain_condition_counts_and_native_only_for_soft():
    assert len(gain_conditions(include_native=False)) == 15
    soft = gain_conditions(include_native=True)
    assert len(soft) == 18
    assert sum(c.gain is None for c in soft) == 3


def test_m111_matches_native_for_learned_gate_bitwise():
    model = FakeModel(fixed=False).eval()
    x = recipient()
    native = model.native(x)
    model.gate_calls = 0
    model.gate_inputs.clear()
    out, trace = forward_with_residual_intervention(
        model, x, ResidualCondition("mask", (1, 1, 1)), recipient_id="r"
    )
    assert torch.equal(out, native)
    assert model.gate_calls == 1
    assert trace["condition"] == "M111"


def test_m111_matches_native_for_fixed_gate_bitwise_and_q_one():
    model = FakeModel(fixed=True).eval()
    x = recipient()
    native = model.native(x)
    model.gate_calls = 0
    out, trace = forward_with_residual_intervention(
        model, x, ResidualCondition("mask", (1, 1, 1)), recipient_id="r"
    )
    assert torch.equal(out, native)
    assert trace["q_native"] == [1.0]


def test_shuffle_happens_after_gate_and_does_not_change_gate_input():
    model = FakeModel(fixed=False).eval()
    x = recipient()
    paired_out, paired_trace = forward_with_residual_intervention(
        model, x, ResidualCondition("mask", (1, 1, 1)), recipient_id="r"
    )
    native_gate_input = tuple(t.clone() for t in model.gate_inputs[-1])
    donor = {
        "P3": torch.tensor([[[[99.0]]]]),
        "P4": torch.tensor([[[[88.0]]]]),
        "P5": torch.tensor([[[[77.0]]]]),
    }
    out, trace = forward_with_residual_intervention(
        model, x, ResidualCondition("shuffle_cond", target_scale="P5"),
        recipient_id="r", donor_id="d", donor_residuals=donor,
    )
    shuffled_gate_input = model.gate_inputs[-1]
    assert all(torch.equal(a, b) for a, b in zip(native_gate_input, shuffled_gate_input))
    assert trace["q_native"] == paired_trace["q_native"]
    assert trace["residual_source_ids"] == {"P3": "r", "P4": "r", "P5": "d"}
    assert not torch.equal(out, paired_out)


def test_shuffle_only_zeros_non_target_coefficients():
    model = FakeModel(fixed=False).eval()
    donor = {s: torch.tensor([[[[20.0 + i]]]]) for i, s in enumerate(SCALES)}
    _, trace = forward_with_residual_intervention(
        model, recipient(), ResidualCondition("shuffle_only", target_scale="P4"),
        recipient_id="r", donor_id="d", donor_residuals=donor,
    )
    assert trace["alpha"]["P3"] == [0.0]
    assert trace["alpha"]["P5"] == [0.0]
    assert trace["alpha"]["P4"] == trace["q_native"]
    assert trace["residual_source_ids"] == {"P3": "r", "P4": "d", "P5": "r"}


def test_gain_changes_only_target_coefficient_and_native_is_recipient_q():
    model = FakeModel(fixed=False).eval()
    _, const = forward_with_residual_intervention(
        model, recipient(), ResidualCondition("gain", target_scale="P3", gain=0.25),
        recipient_id="r",
    )
    assert const["alpha"]["P3"] == [0.25]
    assert const["alpha"]["P4"] == const["q_native"]
    assert const["alpha"]["P5"] == const["q_native"]
    _, native = forward_with_residual_intervention(
        model, recipient(), ResidualCondition("gain", target_scale="P3", gain=None),
        recipient_id="r",
    )
    assert all(native["alpha"][s] == native["q_native"] for s in SCALES)


def test_mask_m000_zeros_all_coefficients():
    model = FakeModel(fixed=False).eval()
    _, trace = forward_with_residual_intervention(
        model, recipient(), ResidualCondition("mask", (0, 0, 0)), recipient_id="r"
    )
    assert all(trace["alpha"][s] == [0.0] for s in SCALES)


def test_self_donor_is_rejected():
    model = FakeModel().eval()
    donor = {s: torch.ones(1, 1, 1, 1) for s in SCALES}
    try:
        forward_with_residual_intervention(
            model, recipient(), ResidualCondition("shuffle_cond", target_scale="P3"),
            recipient_id="r", donor_id="r", donor_residuals=donor,
        )
    except RuntimeError as exc:
        assert "A2_SHUFFLE_SELF_DONOR" in str(exc)
    else:
        raise AssertionError("self donor must be rejected")


def test_eval_forward_does_not_mutate_parameters():
    model = FakeModel().eval()
    before = model_sha(model)
    forward_with_residual_intervention(
        model, recipient(), ResidualCondition("gain", target_scale="P5", gain=0.75),
        recipient_id="r",
    )
    assert model_sha(model) == before


def test_training_mode_and_gate_override_are_rejected():
    model = FakeModel()
    try:
        forward_with_residual_intervention(
            model, recipient(), ResidualCondition("mask", (1, 1, 1)), recipient_id="r"
        )
    except RuntimeError as exc:
        assert "A2_EVAL_ONLY" in str(exc)
    else:
        raise AssertionError("training mode must be rejected")
    model.eval()
    model._gate_override = 0.5
    try:
        forward_with_residual_intervention(
            model, recipient(), ResidualCondition("mask", (1, 1, 1)), recipient_id="r"
        )
    except RuntimeError as exc:
        assert "A2_REFUSE_GATE_OVERRIDE" in str(exc)
    else:
        raise AssertionError("gate override must be rejected")


class TinyDataset:
    def __init__(self):
        self.samples = [
            {"sample_id": "a", "img": torch.tensor([[[10.0]], [[2.0]]])},
            {"sample_id": "b", "img": torch.tensor([[[11.0]], [[3.0]]])},
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def collate_fn(rows):
        return {"img": torch.stack([r["img"] for r in rows], dim=0)}


def test_donor_residual_cache_never_calls_gate():
    model = FakeModel().eval()
    before = model.gate_calls
    cache = build_residual_cache(model, TinyDataset(), torch.device("cpu"))
    assert model.gate_calls == before
    assert set(cache) == {"a", "b"}
    assert all(set(cache[sid]) == set(SCALES) for sid in cache)


def test_paired_effect_classification_boundary_is_frozen_4_of_6():
    primary = {
        "full": 0.1, "loo_median": 0.05, "positive_folds": 4, "negative_folds": 2
    }
    replication = {"full": 0.01}
    assert classify_paired_effect(primary, replication) == "STRONG_POSITIVE"
    primary["positive_folds"] = 3
    assert classify_paired_effect(primary, replication) == "INCONCLUSIVE"


def test_paired_effect_negative_and_replication_sign_required():
    primary = {
        "full": -0.1, "loo_median": -0.05, "positive_folds": 1, "negative_folds": 5
    }
    assert classify_paired_effect(primary, {"full": -0.01}) == "STRONG_NEGATIVE"
    assert classify_paired_effect(primary, {"full": 0.01}) == "INCONCLUSIVE"


def test_evaluator_freezes_f1c_dependency_semantics_and_audit_freshness():
    root = Path(__file__).resolve().parents[1]
    src = (root / "scripts/eval_step4_a2_scale_causality.py").read_text(encoding="utf-8")
    assert "EXPECTED_F1C_SUMMARY_SHA256" in src
    assert "verify_frozen_dependency_closure" in src
    assert "CONTRACT_STALE" in src
    assert "SOURCE_STALE" in src
    assert "VERSION_STALE" in src
    assert "A2_VAL_SET_DRIFT" in src
    assert "verify_preexecution_audit" in src


def test_evaluator_requires_exact_deterministic_donor_map():
    root = Path(__file__).resolve().parents[1]
    src = (root / "scripts/eval_step4_a2_scale_causality.py").read_text(encoding="utf-8")
    assert "expected_donor_map = bijective_derangement(val_ids)" in src
    assert "donor_map != expected_donor_map" in src
    assert "A2_DONOR_MAP_NOT_FROZEN_DETERMINISTIC" in src


def test_runtime_g8_g9_are_not_unconditional_self_assertions():
    root = Path(__file__).resolve().parents[1]
    src = (root / "scripts/eval_step4_a2_scale_causality.py").read_text(encoding="utf-8")
    assert '"G8_stock_validator_semantics": True' not in src
    assert '"G9_provenance_complete": True' not in src
