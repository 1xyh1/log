#!/usr/bin/env python3
"""B1 primary causality evaluation plus forced-gate diagnostics.

B1 = F1 architecture + training-time corruption; the evaluation protocol is
identical to F1 (NORMAL/ZERO/SHUFFLE last.pt primary).

Primary protocol remains NORMAL / ZERO-AUX / SHUFFLE on last.pt.  FORCE-Q0 and
FORCE-Q1 are additional val-only mechanism probes and never replace the primary
three-way causal contract.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import step3_eval_utils as evu  # noqa: E402
from multimodal.causality_interventions import (  # noqa: E402
    assert_valid_shuffle_map,
    bijective_derangement,
)
from multimodal.raw_sample_index import CLASS_NAMES, OUT_DEFAULT  # noqa: E402
from multimodal.run_integrity import inspect_step3_run  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402

GROUP_SPECS = {
    "F1C-C0": {"dataset": "C0-N", "aux_mode": "zero", "gate_mode": "learned",
               "gate_module": "magnitude"},
    "F1C-I-fixed": {"dataset": "C1-I", "aux_mode": "ir", "gate_mode": "fixed_one",
                    "gate_module": "magnitude"},
    "F1C-I-magsoft": {"dataset": "C1-I", "aux_mode": "ir", "gate_mode": "learned",
                      "gate_module": "magnitude"},
    "F1C-I-soft": {"dataset": "C1-I", "aux_mode": "ir", "gate_mode": "learned",
                   "gate_module": "original"},
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def late10(run_dir: Path) -> dict:
    rows = list(csv.DictReader((run_dir / "results.csv").open(encoding="utf-8")))
    values = [
        float(row["metrics/mAP50-95(B)"])
        for row in rows if row.get("metrics/mAP50-95(B)")
    ][-10:]
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else None,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _collect_gate_values(model, dataset, device) -> dict:
    values = {}
    device = evu._as_device(device)
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            batch = dataset.collate_fn([sample])
            batch = evu.move_step3_batch_to_device(batch, device)
            model._predict_once(batch["img"])
            raw = model.last_raw_gate
            effective = model.last_effective_gate
            if raw is None or effective is None or raw.numel() != 1 or effective.numel() != 1:
                raise RuntimeError("F1 checkpoint did not expose a scalar gate")
            values[str(sample["sample_id"])] = {
                "raw_q": float(raw.item()),
                "effective_q": float(effective.item()),
            }
    return values


def _dataset(contract, split, group, *, aux_zero=False, aux_map=None):
    return TriModalDataset(
        contract, split=split, group=group, augment=False,
        aux_zero=aux_zero, aux_id_map=aux_map,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=sorted(GROUP_SPECS), required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--project", default="runs/step4_f1_c")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--device", default="0")
    p.add_argument("--expected-epochs", type=int, default=80)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    device = evu._as_device(a.device)
    spec = GROUP_SPECS[a.group]

    contract_path = Path(a.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    names = dict(CLASS_NAMES)
    run_dir = Path(a.project) / a.run_name
    out = run_dir / "eval_step4_f1_c_causality.json"
    if out.exists() and not a.overwrite:
        raise RuntimeError(f"REFUSE_OVERWRITE_F1C_CAUSALITY_EVAL: {out}")
    integrity = inspect_step3_run(
        run_dir, a.expected_epochs, require_weights=True,
        trace_name="step4_g8_trace.jsonl", growth_name="step4_growth.jsonl",
        eval_name="eval_step4_f1_c_causality.json",
    )
    if not integrity.to_dict()["passed"]:
        print(json.dumps(integrity.to_dict(), indent=2, ensure_ascii=False))
        raise RuntimeError("REFUSE_EVAL_INCOHERENT_RUN")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_manifest = {
        "schema": "step4-f1-c-manifest-v1",
        "group": a.group,
        "physical_run_name": a.run_name,
        "aux_mode": spec["aux_mode"],
        "gate_mode": spec["gate_mode"],
        "dataset_group": spec["dataset"],
        "expected_epochs": a.expected_epochs,
    }
    mismatched = {
        key: {"recorded": manifest.get(key), "expected": expected}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatched:
        raise RuntimeError(f"REFUSE_RUN_IDENTITY_MISMATCH: {mismatched}")

    maps = {}
    for split, filename in (
        ("train", "shuffle_map_train.json"),
        ("val", "shuffle_map_val.json"),
        ("all17", "shuffle_map_all17.json"),
    ):
        path = run_dir / filename
        ids = contract[f"{split}_ids"]
        if path.exists():
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if not assert_valid_shuffle_map(candidate, ids):
                raise RuntimeError(f"REFUSE_INVALID_EXISTING_SHUFFLE_MAP: {path}")
            maps[split] = candidate
        else:
            maps[split] = bijective_derangement(ids)
            if not assert_valid_shuffle_map(maps[split], ids):
                raise RuntimeError(f"FAILED_TO_BUILD_VALID_SHUFFLE_MAP: {split}")
            path.write_text(json.dumps(maps[split], indent=2), encoding="utf-8")

    result = {
        "schema": "step4-f1-c-stock-validator-semantics-v1",
        "group": a.group,
        "aux_mode": spec["aux_mode"],
        "gate_mode": spec["gate_mode"],
        "late10": late10(run_dir),
        "protocol": {
            "primary": ["NORMAL", "ZERO-AUX", "SHUFFLE"],
            "primary_checkpoint": "last.pt",
            "diagnostic_only": ["FORCE-Q0", "FORCE-Q1"],
        },
        "provenance": {
            "results_sha256": _sha(run_dir / "results.csv"),
            "args_sha256": _sha(run_dir / "args.yaml"),
            "last_pt_sha256": _sha(run_dir / "weights" / "last.pt"),
            "best_pt_sha256": _sha(run_dir / "weights" / "best.pt"),
            "manifest_sha256": _sha(run_dir / "manifest.json"),
            "contract_sha256": _sha(contract_path),
            "evaluator_source_sha256": _sha(Path(__file__)),
            "model_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py"
            ),
            "gate_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "reliability_gate.py"
            ),
            "step3_eval_utils_sha256": _sha(
                ROOT / "src" / "multimodal" / "step3_eval_utils.py"
            ),
            "trimodal_dataset_sha256": _sha(
                ROOT / "src" / "multimodal" / "trimodal_dataset.py"
            ),
            "f0_model_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_f0_model.py"
            ),
            "aux_encoder_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "aux_encoder.py"
            ),
            "feature_fusion_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "feature_fusion.py"
            ),
            "trainability_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "trainability.py"
            ),
            "causality_interventions_sha256": _sha(
                ROOT / "src" / "multimodal" / "causality_interventions.py"
            ),
            "raw_sample_index_sha256": _sha(
                ROOT / "src" / "multimodal" / "raw_sample_index.py"
            ),
            "torch_version": torch.__version__,
            "ultralytics_version": __import__("ultralytics").__version__,
            "shuffle_map_sha256": {
                split: _sha(run_dir / filename) for split, filename in (
                    ("train", "shuffle_map_train.json"),
                    ("val", "shuffle_map_val.json"),
                    ("all17", "shuffle_map_all17.json"),
                )
            },
        },
    }

    for checkpoint in ("last.pt", "best.pt"):
        ck = torch.load(
            run_dir / "weights" / checkpoint, map_location="cpu", weights_only=False
        )
        model = (ck.get("ema") or ck.get("model")).float().eval().to(device)
        if not hasattr(model, "set_gate_override"):
            raise RuntimeError("checkpoint is not a Step4F1IRGateModel")
        if (getattr(model, "aux_mode", None) != spec["aux_mode"]
                or getattr(model, "gate_mode", None) != spec["gate_mode"]
                or getattr(model, "gate_module_kind", None) != spec["gate_module"]):
            raise RuntimeError(
                "REFUSE_CHECKPOINT_IDENTITY_MISMATCH: "
                f"aux={getattr(model, 'aux_mode', None)} "
                f"gate={getattr(model, 'gate_mode', None)} expected={spec}"
            )
        model.set_gate_override(None)
        result[checkpoint] = {}
        result[checkpoint]["gate_values"] = {}
        for variant in ("NORMAL", "ZERO-AUX", "SHUFFLE"):
            result[checkpoint][variant] = {}
            result[checkpoint]["gate_values"][variant] = {}
            for split in ("train", "val", "all17"):
                dataset = _dataset(
                    contract, split, spec["dataset"],
                    aux_zero=(variant == "ZERO-AUX"),
                    aux_map=(maps[split] if variant == "SHUFFLE" else None),
                )
                result[checkpoint][variant][split] = (
                    evu.evaluate_dataset_stock_semantics(model, dataset, device, names)
                )
                result[checkpoint]["gate_values"][variant][split] = (
                    _collect_gate_values(model, dataset, device)
                )

        normal_val = _dataset(contract, "val", spec["dataset"])
        try:
            for variant, q in (("FORCE-Q0", 0.0), ("FORCE-Q1", 1.0)):
                model.set_gate_override(q)
                result[checkpoint][variant] = {
                    "val": evu.evaluate_dataset_stock_semantics(
                        model, normal_val, device, names
                    )
                }
        finally:
            model.set_gate_override(None)

        print(
            f"[{a.group}/{checkpoint}] val AP="
            f"{result[checkpoint]['NORMAL']['val']['map50_95']:.4f} "
            f"ZERO={result[checkpoint]['ZERO-AUX']['val']['map50_95']:.4f} "
            f"SHUFFLE={result[checkpoint]['SHUFFLE']['val']['map50_95']:.4f} "
            f"Q0={result[checkpoint]['FORCE-Q0']['val']['map50_95']:.4f} "
            f"Q1={result[checkpoint]['FORCE-Q1']['val']['map50_95']:.4f}"
        )

    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
