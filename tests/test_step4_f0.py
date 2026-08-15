"""Step 4-F0 contract tests: RGB equivalence / zero-init / gradient flow / shuffle."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.early_fusion_yolo26 import build_reference_3ch, r3_hyp  # noqa: E402
from multimodal.raw_sample_index import build_contract, OUT_DEFAULT  # noqa: E402
from multimodal.step4_f0_model import Step4F0Model  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402


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


def test_f0_rgb_equivalence():
    ref = build_reference_3ch()
    model = Step4F0Model(ref, aux_mode="zero")
    ref.eval()
    model.eval()
    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640)
    x6 = torch.cat([x, torch.zeros(1, 3, 640, 640)], dim=1)
    with torch.no_grad():
        o_ref = _tensors(ref._predict_once(x), [])
        o_f0 = _tensors(model(x6), [])
    assert len(o_ref) == len(o_f0)
    for a, b in zip(o_ref, o_f0):
        assert float((a - b).abs().max()) <= 1e-5


def test_zero_init_projection():
    model = Step4F0Model(build_reference_3ch())
    for f in model.fusions.values():
        assert f.proj.weight.abs().max().item() == 0.0
        assert f.proj.bias.abs().max().item() == 0.0


def test_gradient_flow():
    contract = build_contract(out_path=OUT_DEFAULT)
    sample = TriModalDataset(contract, split="train", group="C1-I", augment=False)[0]
    model = Step4F0Model(build_reference_3ch(), aux_mode="ir")
    model.args = r3_hyp()
    img = torch.as_tensor(sample["img"])[None]
    batch = {"img": img,
             "cls": torch.as_tensor(sample["cls"], dtype=torch.float32),
             "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
             "batch_idx": torch.as_tensor(sample["batch_idx"], dtype=torch.float32)}
    model.train()
    model.zero_grad()
    preds = model._predict_once(batch["img"])
    loss = model.loss(batch, preds)
    loss = loss.sum() if isinstance(loss, torch.Tensor) else \
        sum(v.sum() for v in loss if torch.is_tensor(v))
    loss.backward()
    proj_grad = max(p.grad.abs().max() for f in model.fusions.values()
                    for p in f.parameters() if p.grad is not None)
    assert proj_grad > 0
    # W=0 math: encoder grad is exactly 0 at step 1; after one proj step it unblocks
    with torch.no_grad():
        for f in model.fusions.values():
            for p in f.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-1e-3)
    model.zero_grad()
    preds = model._predict_once(batch["img"])
    loss2 = model.loss(batch, preds)
    loss2 = loss2.sum() if isinstance(loss2, torch.Tensor) else \
        sum(v.sum() for v in loss2 if torch.is_tensor(v))
    loss2.backward()
    enc_grad = max(p.grad.abs().max() for p in model.aux_encoder.parameters()
                   if p.grad is not None)
    assert enc_grad > 0
    assert sum(1 for p in model.rgb_backbone.parameters() if p.requires_grad) == 0


def _derangement(ids):
    from multimodal.raw_sample_index import group_of
    groups = {}
    for sid in ids:
        groups.setdefault(group_of(sid), []).append(sid)
    donors = {sid: [d for g, ds in groups.items() if g != group_of(sid) for d in ds]
              for sid in ids}
    result = {}
    remaining = set(ids)
    while remaining:
        # most-constrained first: fewest available cross-group donors
        best = min(remaining,
                   key=lambda s: len([d for d in donors[s] if d in remaining and d != s]))
        pool = [d for d in donors[best] if d in remaining and d != best] or \
               [d for d in ids if d != best]  # donors may be reused (protocol: only no self-match)
        result[best] = pool[0]
        remaining.remove(best)
    return result


def test_shuffle_consistency():
    contract = build_contract(out_path=OUT_DEFAULT)
    ids = contract["val_ids"]
    m1 = _derangement(ids)
    m2 = _derangement(ids)
    assert m1 == m2  # deterministic across calls
    assert all(m1[s] != s for s in m1)  # no self-match
    # group-aware where feasible: donor group differs from rgb group
    from multimodal.raw_sample_index import group_of
    assert all(group_of(m1[s]) != group_of(s) for s in m1)
