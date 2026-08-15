"""Reviewer-mandated regression tests for the Step 4-F0 P0 fixes (2026-08-15)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.early_fusion_yolo26 import (  # noqa: E402
    MODEL_INIT_SEED, build_reference_3ch, r3_hyp)
from multimodal.raw_sample_index import build_contract, OUT_DEFAULT  # noqa: E402
from multimodal.step4_f0_model import Step4F0Model  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402
from multimodal.causality_interventions import (  # noqa: E402
    assert_valid_shuffle_map, bijective_derangement)


def _forward_backward(model, batch):
    model.train()
    model.zero_grad()
    preds = model._predict_once(batch["img"])
    loss = model.loss(batch, preds)
    loss = loss.sum() if isinstance(loss, torch.Tensor) else \
        sum(v.sum() for v in loss if torch.is_tensor(v))
    loss.backward()
    return loss


def _batch(sample):
    return {"img": torch.as_tensor(sample["img"])[None],
            "cls": torch.as_tensor(sample["cls"], dtype=torch.float32),
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "batch_idx": torch.as_tensor(sample["batch_idx"], dtype=torch.float32)}


def test_p5_fused_tensor_enters_neck_layer11():
    """Nonzero fusion must change the CURRENT x entering neck layer 11 (f=-1)."""
    ref = build_reference_3ch()
    model = Step4F0Model(ref, aux_mode="depth")
    x = torch.rand(1, 3, 640, 640)
    a = torch.rand(1, 2, 640, 640)
    # zero-proj: fused P5 == RGB P5 -> x entering layer 11 equals reference path
    model.eval()
    with torch.no_grad():
        # hook the input of tail layer 0 (neck layer 11) to verify it equals y[10]
        # NOTE: forward hooks must return None, otherwise the return value REPLACES
        # the module output (setdefault returns the stored tensor -> shape corruption).
        captured = {}

        def _h(m, i, o):
            captured.setdefault("in", i[0].clone())

        model.tail[0].register_forward_hook(_h)
        model._forward_fused(x, torch.zeros_like(a))
        x_entering = captured["in"]
        # now set a nonzero proj weight AND a NONZERO aux input: the fused P5 must
        # change what enters neck layer 11 (untrained BN(0)=0 would not move it).
        with torch.no_grad():
            model.fusions["10"].proj.weight.fill_(0.5)
        captured.clear()
        model._forward_fused(x, a)
        x_entering2 = captured["in"]
    assert not torch.allclose(x_entering, x_entering2), \
        "neck layer 11 still receives the pre-fusion P5 (stale x)"


def test_gate3_each_group_starts_from_zero_init():
    contract = build_contract(out_path=OUT_DEFAULT)
    for g, dsg in (("F0-C0", "C0-N"), ("F0-I", "C1-I"), ("F0-D", "C2-D")):
        torch.manual_seed(MODEL_INIT_SEED)
        m = Step4F0Model(build_reference_3ch(), aux_mode={"F0-C0": "zero", "F0-I": "ir",
                                                          "F0-D": "depth"}[g])
        assert all(f.assert_zero_init() for f in m.fusions.values()), g
        sample = TriModalDataset(contract, split="train", group=dsg, augment=False)[0]
        m.args = r3_hyp()
        _forward_backward(m, _batch(sample))


def test_gate3_depth_uses_nonzero_D_M():
    contract = build_contract(out_path=OUT_DEFAULT)
    sample = TriModalDataset(contract, split="train", group="C2-D", augment=False)[0]
    img6 = torch.as_tensor(sample["img"])
    assert float(img6[4].abs().max()) > 0 and float(img6[5].abs().max()) > 0, \
        "C2-D sample must carry nonzero D and M planes"
    m = Step4F0Model(build_reference_3ch(), aux_mode="depth")
    m.args = r3_hyp()
    _forward_backward(m, _batch(sample))
    w = m.fusions["4"].proj.weight.grad
    assert w is not None and float(w.abs().max()) > 0


def test_f0_c0_weight_grad_zero_but_bias_allowed():
    contract = build_contract(out_path=OUT_DEFAULT)
    sample = TriModalDataset(contract, split="train", group="C0-N", augment=False)[0]
    m = Step4F0Model(build_reference_3ch(), aux_mode="zero")
    m.args = r3_hyp()
    _forward_backward(m, _batch(sample))
    for f in m.fusions.values():
        assert f.proj.weight.grad is None or float(f.proj.weight.grad.abs().max()) == 0.0
        assert f.proj.bias.grad is not None and float(f.proj.bias.grad.abs().max()) > 0


def test_actual_yield_g8_matches_expected():
    """The runner records expected/actual order+flip hashes and actual_matches_expected."""
    # exercised structurally: run the smoke trace through the fixed contract
    import glob
    traces = sorted(Path("runs/step4_f0").glob("smoke-*/step4_g8_trace.jsonl"))
    if not traces:
        import pytest
        pytest.skip("no smoke traces present")
    for tp in traces:
        for line in tp.read_text().strip().splitlines():
            t = json.loads(line)
            assert t.get("actual_order_sha256"), "actual order hash missing"
            assert t.get("actual_flip_sha256"), "actual flip hash missing"
            assert t.get("actual_matches_expected") is True


def test_shuffle_is_bijective_no_self_cross_group():
    contract = build_contract(out_path=OUT_DEFAULT)
    from multimodal.raw_sample_index import group_of
    for split in ("train", "val", "all17"):
        ids = contract[f"{split}_ids"]
        m = bijective_derangement(ids)
        assert assert_valid_shuffle_map(m, ids)
        assert all(group_of(m[s]) != group_of(s) for s in m)


def test_stale_or_short_formal_run_rejected():
    from multimodal.run_integrity import inspect_step3_run
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "args.yaml").write_text("epochs: 1\nname: x\nexist_ok: false\n")
        (p / "results.csv").write_text("epoch,train/Loss\n1,1.0\n")
        (p / "step4_g8_trace.jsonl").write_text("{}" + "\n")
        (p / "step4_growth.jsonl").write_text("{}" + "\n")
        rep = inspect_step3_run(p, 80, require_weights=False,
                                trace_name="step4_g8_trace.jsonl",
                                growth_name="step4_growth.jsonl")
        assert not rep.to_dict()["passed"]


def test_three_group_initial_state_sha_equal():
    """F0-C0/F0-I/F0-D must start from identical state except the aux_mode flag."""
    shas = {}
    for g in ("zero", "ir", "depth"):
        torch.manual_seed(MODEL_INIT_SEED)
        m = Step4F0Model(build_reference_3ch(), aux_mode=g)
        import hashlib
        h = hashlib.sha256()
        for n, p in sorted(m.state_dict().items()):
            h.update(n.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        shas[g] = h.hexdigest()
    assert len(set(shas.values())) == 1
