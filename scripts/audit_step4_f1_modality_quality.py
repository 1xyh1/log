#!/usr/bin/env python3
"""Describe RGB/IR/Depth input quality without changing the frozen data path."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.modality_quality import (  # noqa: E402
    describe_trimodal_sample,
    diagnostic_flags,
)
from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate(samples: list[dict], path: tuple[str, ...]) -> dict:
    values = []
    for sample in samples:
        obj = sample
        for key in path:
            obj = obj[key]
        if obj is not None:
            values.append(float(obj))
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--split", choices=["train", "val", "all17"], default="val")
    p.add_argument("--out", default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    contract_path = Path(a.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    # Reuse the authoritative dataset twice; no combined group or alternate loader.
    ir_ds = TriModalDataset(contract, split=a.split, group="C1-I", augment=False)
    depth_ds = TriModalDataset(contract, split=a.split, group="C2-D", augment=False)
    if ir_ds.ids != depth_ds.ids:
        raise RuntimeError("IR/Depth quality views have different sample order")

    rows = []
    for idx, sid in enumerate(ir_ds.ids):
        ir_sample = ir_ds[idx]
        depth_sample = depth_ds[idx]
        if not np.array_equal(ir_sample["img"][:3], depth_sample["img"][:3]):
            raise RuntimeError(f"RGB view differs between C1-I and C2-D for {sid}")
        if (ir_sample["ori_shape"] != depth_sample["ori_shape"]
                or ir_sample["ratio_pad"] != depth_sample["ratio_pad"]):
            raise RuntimeError(f"LetterBox metadata differs between views for {sid}")
        ir_stats = describe_trimodal_sample(ir_sample)
        depth_stats = describe_trimodal_sample(depth_sample)
        row = {
            "sample_id": sid,
            "rgb_luma": ir_stats["rgb_luma"],
            "ir": ir_stats["ir"],
            "depth": depth_stats["depth"],
            "depth_valid_ratio": depth_stats["depth_valid_ratio"],
        }
        row["diagnostic_flags"] = diagnostic_flags(row)
        rows.append(row)

    report = {
        "schema": "step4-f1-modality-quality-diagnostic-v1",
        "role": ("descriptive input audit only; not a perceptual score, not a gate "
                 "target, and not part of NORMAL/ZERO/SHUFFLE"),
        "split": a.split,
        "n_samples": len(rows),
        "provenance": {
            "contract_sha256": _sha(contract_path),
            "script_sha256": _sha(Path(__file__)),
            "quality_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "modality_quality.py"
            ),
            "dataset_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "trimodal_dataset.py"
            ),
            "preprocess_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "modality_preprocess.py"
            ),
            "raw_sample_index_sha256": _sha(
                ROOT / "src" / "multimodal" / "raw_sample_index.py"
            ),
        },
        "aggregate": {
            "rgb_dynamic_range": _aggregate(rows, ("rgb_luma", "dynamic_range_p99_p01")),
            "ir_dynamic_range": _aggregate(rows, ("ir", "dynamic_range_p99_p01")),
            "ir_gradient_abs_mean": _aggregate(rows, ("ir", "gradient_abs_mean")),
            "depth_valid_ratio": _aggregate(rows, ("depth_valid_ratio",)),
        },
        "samples": rows,
    }
    out = Path(a.out) if a.out else (
        ROOT / "reports" / "step4_f1_ir_gate" / f"modality_quality_{a.split}.json"
    )
    if out.exists() and not a.overwrite:
        raise RuntimeError(f"REFUSE_OVERWRITE_QUALITY_REPORT: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2, ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
