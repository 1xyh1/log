from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.run_integrity import inspect_step3_run, sha256_file  # noqa: E402


def _write_fake_run(root: Path, epochs: int = 80):
    yaml = pytest.importorskip("yaml")
    root.mkdir(parents=True, exist_ok=True)
    (root / "args.yaml").write_text(yaml.safe_dump({"epochs": epochs, "name": root.name}), encoding="utf-8")
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "metrics/mAP50-95(B)"])
        w.writeheader()
        for e in range(1, epochs + 1):
            w.writerow({"epoch": e, "metrics/mAP50-95(B)": 0.1})
    (root / "step3_g8_trace.jsonl").write_text(
        "".join(json.dumps({"epoch": e}) + "\n" for e in range(epochs)), encoding="utf-8"
    )
    (root / "step3_kernel_growth.jsonl").write_text(
        "".join(json.dumps({"epoch": e + 1}) + "\n" for e in range(epochs)), encoding="utf-8"
    )
    (root / "manifest.json").write_text(
        json.dumps({"run_kind": "formal", "expected_epochs": epochs}), encoding="utf-8"
    )


def test_run_integrity_rejects_overwritten_formal(tmp_path):
    run = tmp_path / "C0-N"
    _write_fake_run(run, epochs=1)
    report = inspect_step3_run(run, expected_epochs=80, require_weights=False)
    assert not report.passed
    assert any("EPOCH" in e or "ROW_COUNT" in e for e in report.errors)


def test_run_integrity_rejects_stale_eval_hash(tmp_path):
    run = tmp_path / "C1-I"
    _write_fake_run(run, epochs=3)
    before = sha256_file(run / "results.csv")
    (run / "eval_step3_causality.json").write_text(
        json.dumps({"provenance": {"results_sha256": before}}), encoding="utf-8"
    )
    with (run / "results.csv").open("a", encoding="utf-8") as f:
        f.write("4,0.2\n")
    report = inspect_step3_run(run, expected_epochs=3, require_weights=False)
    assert not report.passed
    assert "STALE_EVAL_PROVENANCE:results_sha256" in report.errors


def test_eval_gt_conversion_matches_stock_prepare_batch():
    pytest.importorskip("ultralytics")
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace, ops

    args = IterableSimpleNamespace(**vars(DEFAULT_CFG))
    v = DetectionValidator(dataloader=None, save_dir=ROOT / "runs", args=args)
    v.device = torch.device("cpu")
    xywh = torch.tensor([[0.50, 0.50, 0.20, 0.40], [0.25, 0.30, 0.10, 0.10]])
    batch = {
        "img": torch.zeros(1, 6, 640, 640),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": xywh.clone(),
        "batch_idx": torch.tensor([0.0, 0.0]),
        "ori_shape": ((640, 640),),
        "ratio_pad": (((1.0, 1.0), (0, 0)),),
        "im_file": ("dummy.jpg",),
    }
    pbatch = v._prepare_batch(0, batch)
    expected = ops.xywh2xyxy(xywh) * 640.0
    assert torch.allclose(pbatch["bboxes"], expected)


def test_eval_perfect_prediction_scores_perfect_iou():
    pytest.importorskip("ultralytics")
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace

    v = DetectionValidator(
        dataloader=None,
        save_dir=ROOT / "runs",
        args=IterableSimpleNamespace(**vars(DEFAULT_CFG)),
    )
    v.device = torch.device("cpu")
    batch = {"cls": torch.tensor([0.0]), "bboxes": torch.tensor([[100.0, 100.0, 200.0, 220.0]])}
    pred = {
        "cls": torch.tensor([0.0]),
        "bboxes": batch["bboxes"].clone(),
        "conf": torch.tensor([0.9]),
        "extra": torch.empty(1, 0),
    }
    out = v._process_batch(pred, batch)["tp"]
    assert out.shape == (1, 10)
    assert out.all()
