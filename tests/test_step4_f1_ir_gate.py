from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")

from multimodal.reliability_gate import (  # noqa: E402
    PyramidScalarReliabilityGate,
    broadcast_gate,
)
from multimodal.step4_f1_interventions import (  # noqa: E402
    IRCorruptionDatasetView,
    corrupt_ir_plane,
)


def test_scalar_gate_shape_range_and_backward():
    torch.manual_seed(1)
    gate = PyramidScalarReliabilityGate(hidden=16)
    features = [
        torch.randn(2, 256, 8, 8, requires_grad=True),
        torch.randn(2, 256, 4, 4, requires_grad=True),
        torch.randn(2, 512, 2, 2, requires_grad=True),
    ]
    q = gate(features)
    assert q.shape == (2, 1)
    assert torch.isfinite(q).all() and (q > 0).all() and (q < 1).all()
    q.sum().backward()
    assert all(feature.grad is not None for feature in features)
    assert max(float(feature.grad.abs().max()) for feature in features) > 0


def test_broadcast_gate_contract():
    x = torch.randn(3, 8, 5, 5)
    q = torch.tensor([[0.0], [0.5], [1.0]])
    b = broadcast_gate(q, x)
    assert b.shape == (3, 1, 1, 1)
    with pytest.raises(ValueError):
        broadcast_gate(torch.ones(3), x)


def test_ir_corruption_is_deterministic_and_bounded():
    plane = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    mask = np.ones((8, 8), dtype=bool)
    a = corrupt_ir_plane(
        plane, kind="noise", severity=0.7, sample_id="sample", content_mask=mask
    )
    b = corrupt_ir_plane(
        plane, kind="noise", severity=0.7, sample_id="sample", content_mask=mask
    )
    assert np.array_equal(a, b)
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0


def test_ir_corruption_view_changes_only_ir():
    class Dummy:
        ids = ["x"]

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {
                "img": np.ones((6, 8, 8), dtype=np.float32) * 0.5,
                "ori_shape": (8, 8),
                "ratio_pad": ((1.0, 1.0), (0, 0)),
                "sample_id": "x",
                "aux_sample_id": "x",
            }

        @staticmethod
        def collate_fn(batch):
            return batch

    base = Dummy()[0]
    changed = IRCorruptionDatasetView(Dummy(), kind="zero", severity=1.0)[0]
    assert np.array_equal(changed["img"][:3], base["img"][:3])
    assert np.array_equal(changed["img"][4:], base["img"][4:])
    assert np.count_nonzero(changed["img"][3]) == 0


def test_f1_initial_detector_identity_if_local_yolo_snapshot_available():
    pytest.importorskip("ultralytics")
    from multimodal.early_fusion_yolo26 import WEIGHTS_DEFAULT, build_reference_3ch
    from multimodal.step4_f1_ir_gate_model import Step4F1IRGateModel

    if not Path(WEIGHTS_DEFAULT).exists():
        pytest.skip("local YOLO26 pretrained checkpoint is unavailable")
    reference = build_reference_3ch()
    model = Step4F1IRGateModel(reference, aux_mode="ir")
    reference.eval()
    model.eval()
    rgb = torch.rand(1, 3, 320, 320)
    img6 = torch.cat([rgb, torch.rand(1, 1, 320, 320),
                      torch.zeros(1, 2, 320, 320)], dim=1)

    def tensors(obj, out):
        if torch.is_tensor(obj):
            out.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                tensors(value, out)
        elif isinstance(obj, (tuple, list)):
            for value in obj:
                tensors(value, out)
        return out

    with torch.no_grad():
        expected = tensors(reference._predict_once(rgb), [])
        actual = tensors(model._predict_once(img6), [])
    assert len(expected) == len(actual)
    assert max(float((a - b).abs().max()) for a, b in zip(expected, actual)) <= 1e-5
