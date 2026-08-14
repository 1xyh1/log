from dataclasses import asdict

import pytest
import torch

import mmod_qaf.model as model_module
from mmod_qaf.model import TriModalModelConfig, load_training_checkpoint, save_training_checkpoint


def test_checkpoint_roundtrip_preserves_model_and_preprocess_config(monkeypatch, tmp_path):
    source = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
    model_cfg = TriModalModelConfig(
        fuse_p3=False,
        fuse_p4=False,
        fuse_p5=False,
        rgb_only=True,
    )
    train_cfg = {
        "seed": 17,
        "deterministic": True,
        "data": {
            "train": {
                "root": "data/train",
                "imgsz": 768,
                "ir_mode": "gray",
                "depth_min_mm": 300,
                "depth_max_mm": 19999,
                "depth_resize": "valid_bilinear",
            }
        },
    }
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(path, source, optimizer, 4, model_cfg, train_cfg, {"mAP50-95": 0.25})

    restored = torch.nn.Linear(3, 2)
    monkeypatch.setattr(model_module, "build_from_pretrained", lambda *_args, **_kwargs: restored)
    loaded, checkpoint = load_training_checkpoint(path, weights="unused.pt", device="cpu")

    assert checkpoint["format"] == "trimodal-yolo26-qaf-v0.3"
    assert checkpoint["epoch"] == 4
    assert checkpoint["model_cfg"] == asdict(model_cfg)
    assert checkpoint["train_cfg"] == train_cfg
    assert checkpoint["metrics"] == {"mAP50-95": 0.25}
    for expected, actual in zip(source.parameters(), loaded.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_checkpoint_load_is_strict(monkeypatch, tmp_path):
    path = tmp_path / "broken.pt"
    model_cfg = TriModalModelConfig(fuse_p3=False, fuse_p4=False, fuse_p5=False, rgb_only=True)
    torch.save(
        {
            "format": "trimodal-yolo26-qaf-v0.3",
            "model_cfg": asdict(model_cfg),
            "model_state": {"unexpected": torch.ones(1)},
            "train_cfg": {},
        },
        path,
    )
    monkeypatch.setattr(model_module, "build_from_pretrained", lambda *_args, **_kwargs: torch.nn.Linear(3, 2))
    with pytest.raises(RuntimeError):
        load_training_checkpoint(path, weights="unused.pt", device="cpu")


def test_checkpoint_format_is_rejected_before_build(monkeypatch, tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"format": "trimodal-yolo26-qaf-v0.2"}, path)
    monkeypatch.setattr(
        model_module,
        "build_from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        load_training_checkpoint(path, weights="unused.pt", device="cpu")
