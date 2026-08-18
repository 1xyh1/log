from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Isolated bundle harness fallback. In the real repository the audited implementation
# from multimodal.reliability_gate is imported instead.
if importlib.util.find_spec("multimodal.reliability_gate") is None:
    rg = types.ModuleType("multimodal.reliability_gate")
    def broadcast_gate(q, ref):
        z = q
        while z.ndim < ref.ndim:
            z = z.unsqueeze(-1)
        return z
    rg.broadcast_gate = broadcast_gate
    sys.modules["multimodal.reliability_gate"] = rg

from multimodal.step4_a3_common import (  # noqa: E402
    SCALES, ap_effect, build_residual_cache_no_gate, classify_ap_effect,
    classify_sample_metric, energy_map, forward_with_custom_residuals,
    git_blob_sha1, infer_feature_stride, pearson_corr, state_sha256, zero_fill_translate,
)
from multimodal.step4_a3_generic_bias import (  # noqa: E402
    build_generic_components, classify_generic_labels, loo_mean_dc,
    loo_mean_residual, native_ac, native_dc,
)
from multimodal.step4_a3_registration import (  # noqa: E402
    RawShift, _phase_peak, cross_fitted_median_shifts, estimate_registration_shift,
    raw_shift_to_feature_cells, shift_surface, valid_content_slices,
)
from multimodal.step4_a3_semantic import (  # noqa: E402
    binary_auroc, boxes_xywhn_to_mask, enrichment, semantic_row,
    summarize_semantic_rows,
)
from multimodal.step4_a3_spatial import spatial_row  # noqa: E402


def test_zero_fill_translate_right_down_no_wrap():
    x = torch.zeros(1, 1, 3, 4)
    x[0, 0, 0, 0] = 5
    y = zero_fill_translate(x, 2, 1)
    assert y[0, 0, 1, 2] == 5
    assert y.sum() == 5
    assert y[0, 0, 0, 0] == 0


def test_zero_fill_translate_left_up():
    x = torch.zeros(1, 1, 3, 4)
    x[0, 0, 2, 3] = 7
    y = zero_fill_translate(x, -2, -1)
    assert y[0, 0, 1, 1] == 7
    assert y.sum() == 7


def test_phase_peak_returns_shift_to_apply_to_moving():
    ref = np.zeros((32, 32), dtype=float)
    ref[8:12, 10:15] = 1
    ref[20, 4] = 2
    moving = np.zeros_like(ref)
    moving[2:, 3:] = ref[:-2, :-3]  # moving is right 3, down 2
    dx, dy, response = _phase_peak(ref, moving)
    assert (dx, dy) == (-3, -2)
    assert response > 1


def test_valid_content_slices_exclude_letterbox_pad():
    sample = {"ratio_pad": ((0.5, 0.5), (10, 20)), "ori_shape": (100, 200)}
    ys, xs = valid_content_slices(sample)
    assert (ys.start, ys.stop) == (20, 70)
    assert (xs.start, xs.stop) == (10, 110)




def test_git_blob_sha1_canonicalizes_crlf(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_bytes(b"one\ntwo\n")
    b.write_bytes(b"one\r\ntwo\r\n")
    assert git_blob_sha1(a) == git_blob_sha1(b)


def test_estimate_registration_shift_uses_common_content():
    h = w = 48
    ref = np.zeros((h, w), dtype=np.float32)
    ref[8:15, 9:18] = 1
    ref[30, 7] = 0.7
    moving = np.zeros_like(ref)
    moving[1:, 2:] = ref[:-1, :-2]
    img = np.stack([ref, ref, ref, moving, np.zeros_like(ref), np.zeros_like(ref)])
    sample = {"img": img, "ratio_pad": ((1.0, 1.0), (0, 0)), "ori_shape": (h, w)}
    s = estimate_registration_shift(sample)
    assert (s.dx, s.dy) == (-2.0, -1.0)


def test_crossfit_median_excludes_held_out():
    raw = {f"s{i}": RawShift(i, -i, 1.0) for i in range(6)}
    out = cross_fitted_median_shifts(raw)
    for sid, row in out.items():
        assert sid not in row["train_ids_for_shift"]
        assert len(row["train_ids_for_shift"]) == 5


def test_raw_shift_to_feature_cells():
    assert raw_shift_to_feature_cells(-17, 18, 8) == (-2, 2)
    assert raw_shift_to_feature_cells(15, -15, 16) == (1, -1)


def test_shift_surface_detects_known_offset():
    ref = torch.zeros(12, 12)
    ref[3:6, 4:8] = 1
    mov = zero_fill_translate(ref[None, None], 1, -2)[0, 0]
    surf = shift_surface(ref, mov, radius=2)
    # Surface uses the shift of reference sampling window relative to moving;
    # the optimum should undo moving's translation.
    assert surf["best_feature_shift"] == {"dx": -1, "dy": 2}
    assert surf["corr_at_best"] >= surf["corr_at_zero"]


def test_energy_map_rms_channels():
    x = torch.tensor([[[3.0]], [[4.0]]])
    e = energy_map(x)
    assert torch.allclose(e, torch.tensor([[math.sqrt(12.5)]]), atol=1e-6)


def test_pearson_corr_identity_and_reverse():
    x = torch.arange(10).float().reshape(2, 5)
    assert pearson_corr(x, x) == pytest.approx(1.0)
    assert pearson_corr(x, -x) == pytest.approx(-1.0)


def test_classify_sample_metric():
    p = {"median": 0.2, "positive": 4, "negative": 2}
    r = {"median": 0.01, "positive": 3, "negative": 3}
    assert classify_sample_metric(p, r) == "STRONG_RECIPIENT_SPECIFIC"
    r["median"] = -0.01
    assert classify_sample_metric(p, r) == "INCONCLUSIVE"


def test_classify_ap_effect_cross_system_sign():
    p = {"full": 0.1, "loo_median": 0.05, "positive_folds": 5, "negative_folds": 1}
    r = {"full": 0.001}
    assert classify_ap_effect(p, r) == "STRONG_POSITIVE"
    r["full"] = -0.001
    assert classify_ap_effect(p, r) == "INCONCLUSIVE"


def test_ap_effect_uses_same_loo_ids():
    base = {"full": {"map50_95": .2}, "loo": {"a": {"map50_95": .1}, "b": {"map50_95": .2}}}
    new = {"full": {"map50_95": .3}, "loo": {"a": {"map50_95": .15}, "b": {"map50_95": .18}}}
    e = ap_effect(new, base)
    assert e["full"] == pytest.approx(.1)
    assert e["positive_folds"] == 1
    assert e["negative_folds"] == 1


def test_boxes_xywhn_to_mask_projects_union():
    boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    mask = boxes_xywhn_to_mask(boxes, (8, 8))
    assert mask.sum() == 16
    assert mask[2:6, 2:6].all()


def test_binary_auroc_perfect_reverse_and_ties():
    y = torch.tensor([0, 0, 1, 1], dtype=torch.bool)
    assert binary_auroc(torch.tensor([0., 1., 2., 3.]), y) == pytest.approx(1.0)
    assert binary_auroc(torch.tensor([3., 2., 1., 0.]), y) == pytest.approx(0.0)
    assert binary_auroc(torch.ones(4), y) == pytest.approx(0.5)


def test_binary_auroc_degenerate_fails():
    with pytest.raises(RuntimeError, match="SEMANTIC_MASK_DEGENERATE"):
        binary_auroc(torch.arange(4.), torch.ones(4, dtype=torch.bool))


def test_enrichment_object_vs_background():
    e = torch.tensor([[2., 2.], [1., 1.]])
    m = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)
    assert enrichment(e, m) == pytest.approx(2.0)


def test_semantic_row_native_better_than_donor():
    # Box covers upper-left quarter. Native puts energy there, donor outside.
    boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5]])
    n = torch.zeros(1, 1, 4, 4)
    d = torch.zeros(1, 1, 4, 4)
    n[..., :2, :2] = 3
    d[..., 2:, 2:] = 3
    row = semantic_row(boxes, n, d)
    assert row["auroc_native"] > row["auroc_donor"]
    assert row["delta_native_minus_donor"] > 0


def test_semantic_coverage_requires_five_of_six():
    rows = {f"s{i}": {"valid": i < 4, "delta_native_minus_donor": 0.1} for i in range(6)}
    with pytest.raises(RuntimeError, match="A3_SEMANTIC_COVERAGE_FAIL"):
        summarize_semantic_rows(rows, expected_n=6)


def test_native_dc_is_spatial_broadcast():
    x = torch.arange(24).float().reshape(1, 2, 3, 4)
    dc = native_dc(x)
    assert dc.shape == x.shape
    for c in range(2):
        assert torch.unique(dc[0, c]).numel() == 1
        assert dc[0, c, 0, 0] == pytest.approx(float(x[0, c].mean()))


def test_native_ac_has_zero_spatial_mean_per_channel():
    x = torch.randn(1, 3, 5, 7)
    ac = native_ac(x)
    assert torch.allclose(ac.mean(dim=(-2, -1)), torch.zeros(1, 3), atol=1e-6)


def test_loo_mean_excludes_recipient():
    rs = {f"s{i}": torch.full((1, 1, 1, 1), float(i)) for i in range(6)}
    mean, ids = loo_mean_residual(rs, "s0")
    assert "s0" not in ids and len(ids) == 5
    assert mean.item() == pytest.approx(3.0)


def test_loo_mean_dc_same_donor_set():
    rs = {f"s{i}": torch.full((1, 2, 2, 2), float(i)) for i in range(6)}
    dc, ids = loo_mean_dc(rs, "s2")
    assert "s2" not in ids
    assert torch.unique(dc[0, 0]).numel() == 1


def test_build_generic_components_no_self():
    rs = {f"s{i}": torch.full((1, 1, 2, 2), float(i)) for i in range(6)}
    comps = build_generic_components(rs, "s3")
    for name in ("LOO_MEAN", "LOO_MEAN_DC"):
        _, donors = comps[name]
        assert "s3" not in donors
    assert comps["NATIVE_DC"][1] is None


def test_generic_label_requires_mean_positive_and_native_not_added():
    effects = {
        "U_mean": {"label": "STRONG_POSITIVE"},
        "native_minus_mean": {"label": "INCONCLUSIVE"},
        "U_dc": {"label": "INCONCLUSIVE"},
        "U_meanDC": {"label": "INCONCLUSIVE"},
        "U_ac": {"label": "STRONG_POSITIVE"},
    }
    lab = classify_generic_labels(effects)
    assert lab["generic_component"] == "GENERIC_COMPONENT_SUPPORTED"
    assert lab["spatial_ac"] == "SPATIAL_AC_SUPPORTED"


class IndexedIdentity(nn.Module):
    def __init__(self, i, f=-1):
        super().__init__()
        self.i = i
        self.f = f
    def forward(self, x):
        return x


class FakeAux(nn.Module):
    def forward(self, x):
        z = x[:, :1].repeat(1, 2, 1, 1)
        return z + 1, z + 2, z + 3


class FakeFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Identity()


class FakeGate(nn.Module):
    def forward(self, feats):
        # Real reliability gate outputs a Bx1 scalar per sample (broadcast_gate contract).
        return feats[0].mean(dim=(1, 2, 3)).unsqueeze(1)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.aux_mode = "ir"
        self.gate_mode = "learned"
        self._gate_override = None
        self.rgb_backbone = nn.ModuleList([IndexedIdentity(i) for i in range(11)])
        self.aux_encoder = FakeAux()
        self.fusions = nn.ModuleDict({"4": FakeFusion(), "6": FakeFusion(), "10": FakeFusion()})
        self.reliability_gate = FakeGate()
        self.tail = nn.ModuleList([IndexedIdentity(11, -1)])
        self.save = set()
    def _split_input(self, x):
        return x[:, :2], x[:, 2:4]
    def _effective_gate(self, feats):
        return self.reliability_gate(feats)


def test_forward_custom_q_is_computed_from_untouched_recipient():
    m = FakeModel().eval()
    x = torch.zeros(1, 4, 4, 4)
    x[:, 2] = 2
    replacement = torch.full((1, 2, 4, 4), 999.)
    before = state_sha256(m)
    _, tr = forward_with_custom_residuals(
        m, x, recipient_id="r", active_scales=["P3"],
        replacements={"P3": replacement}, source_ids={"P3": "donor"},
        condition_name="TEST",
    )
    after = state_sha256(m)
    # A3 intervention cannot change q: q comes from native aux P3 (=3 here).
    assert tr["q_native"] == [3.0]
    assert tr["residual_source_ids"]["P3"] == "donor"
    assert tr["active_scales"] == ["P3"]
    assert before == after


class TinyDataset:
    def __init__(self):
        self.ids = ["a", "b"]
    def __len__(self):
        return 2
    def __getitem__(self, i):
        img = torch.zeros(4, 4, 4)
        img[2] = i + 1
        return {"img": img, "sample_id": self.ids[i]}
    @staticmethod
    def collate_fn(batch):
        return {"img": torch.stack([b["img"] for b in batch])}


def test_no_gate_residual_cache_really_skips_gate():
    m = FakeModel().eval()
    counter = {"n": 0}
    h = m.reliability_gate.register_forward_hook(
        lambda mod, inp, out: counter.__setitem__("n", counter["n"] + 1)
    )
    cache = build_residual_cache_no_gate(m, TinyDataset(), torch.device("cpu"))
    h.remove()
    assert counter["n"] == 0
    assert set(cache) == {"a", "b"}


def test_infer_feature_stride_exact():
    feat = torch.zeros(1, 2, 20, 20)
    assert infer_feature_stride((640, 640), feat) == 32
    with pytest.raises(RuntimeError, match="NONINTEGER"):
        infer_feature_stride((641, 640), feat)


def test_spatial_row_native_identity_beats_flat_donor():
    rgb = torch.zeros(1, 1, 6, 6)
    rgb[..., 1:3, 2:5] = 4
    native = rgb.clone()
    donor = torch.ones_like(rgb)
    row = spatial_row(rgb, native, donor)
    assert row["corr_native"] > row["corr_donor"]
    assert row["delta_native_minus_donor"] > 0
