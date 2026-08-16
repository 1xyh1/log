#!/usr/bin/env python3
"""Evaluation-only IR degradation probe for a trained F1-I-soft checkpoint.

This report tests whether the learned task gate responds to auxiliary corruption and
whether that response protects detection.  It is diagnostic evidence only; the
formal result remains NORMAL/ZERO/SHUFFLE from eval_step4_f1_causality.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import step3_eval_utils as evu  # noqa: E402
from multimodal.raw_sample_index import CLASS_NAMES, OUT_DEFAULT  # noqa: E402
from multimodal.run_integrity import inspect_step3_run  # noqa: E402
from multimodal.step4_f1_interventions import IRCorruptionDatasetView  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate_values(model, dataset, device) -> dict:
    device = evu._as_device(device)
    out = {}
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            batch = dataset.collate_fn([sample])
            batch = evu.move_step3_batch_to_device(batch, device)
            model._predict_once(batch["img"])
            raw = model.last_raw_gate
            effective = model.last_effective_gate
            if raw is None or effective is None:
                raise RuntimeError("F1 model did not expose gate values")
            out[str(sample["sample_id"])] = {
                "raw_q": float(raw.item()),
                "effective_q": float(effective.item()),
            }
    return out


def _summary(values: dict) -> dict:
    q = sorted(float(row["raw_q"]) for row in values.values())
    n = len(q)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                "p10": None, "p50": None, "p90": None}

    def quantile(p: float) -> float:
        # nearest-rank quantile (n small, avoid interpolation surprises)
        return q[min(n - 1, max(0, round(p * (n - 1))))]

    return {
        "n": n,
        "mean": statistics.mean(q),
        "std": statistics.stdev(q) if n > 1 else 0.0,
        "min": q[0],
        "max": q[-1],
        "p10": quantile(0.10),
        "p50": quantile(0.50),
        "p90": quantile(0.90),
    }


def _evaluate_with_override(model, dataset, device, names, override):
    model.set_gate_override(override)
    return evu.evaluate_dataset_stock_semantics(model, dataset, device, names)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="F1-I-soft")
    p.add_argument("--project", default="runs/step4_f1_ir_gate")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--device", default="0")
    p.add_argument("--expected-epochs", type=int, default=80)
    p.add_argument("--checkpoint", choices=["last.pt", "best.pt"], default="last.pt")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    device = evu._as_device(a.device)
    run_dir = Path(a.project) / a.run_name
    out = run_dir / f"eval_step4_f1_quality_{a.checkpoint.removesuffix('.pt')}.json"
    if out.exists() and not a.overwrite:
        raise RuntimeError(f"REFUSE_OVERWRITE_QUALITY_EVAL: {out}")
    integrity = inspect_step3_run(
        run_dir, a.expected_epochs, require_weights=True,
        trace_name="step4_g8_trace.jsonl", growth_name="step4_growth.jsonl",
        eval_name="eval_step4_f1_causality.json",
    )
    if not integrity.to_dict()["passed"]:
        print(json.dumps(integrity.to_dict(), indent=2, ensure_ascii=False))
        raise RuntimeError("REFUSE_QUALITY_EVAL_INCOHERENT_RUN")

    contract_path = Path(a.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    base = TriModalDataset(contract, split="val", group="C1-I", augment=False)
    ck = torch.load(
        run_dir / "weights" / a.checkpoint, map_location="cpu", weights_only=False
    )
    model = (ck.get("ema") or ck.get("model")).float().eval().to(device)
    if getattr(model, "gate_mode", None) != "learned":
        raise RuntimeError("quality probe requires the learned F1-I-soft checkpoint")
    if getattr(model, "aux_mode", None) != "ir":
        raise RuntimeError("quality probe requires an IR auxiliary checkpoint")
    names = dict(CLASS_NAMES)

    conditions = [("identity", 0.0), ("zero", 1.0)]
    for kind in ("blur", "contrast", "noise", "shift"):
        conditions.extend((kind, value) for value in (0.25, 0.50, 0.75, 1.0))

    report = {
        "schema": "step4-f1-ir-quality-probe-v1",
        "role": "diagnostic only; does not replace NORMAL/ZERO/SHUFFLE",
        "checkpoint": a.checkpoint,
        "provenance": {
            "checkpoint_sha256": _sha(run_dir / "weights" / a.checkpoint),
            "contract_sha256": _sha(contract_path),
            "script_sha256": _sha(Path(__file__)),
            "interventions_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_f1_interventions.py"
            ),
            "evaluator_core_sha256": _sha(
                ROOT / "src" / "multimodal" / "step3_eval_utils.py"
            ),
            "model_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py"
            ),
            "gate_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "reliability_gate.py"
            ),
            "dataset_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "trimodal_dataset.py"
            ),
            "torch_version": torch.__version__,
            "ultralytics_version": __import__("ultralytics").__version__,
        },
        "conditions": {},
    }
    try:
        for kind, severity in conditions:
            dataset = IRCorruptionDatasetView(base, kind=kind, severity=severity)
            model.set_gate_override(None)
            values = _gate_values(model, dataset, device)
            learned = _evaluate_with_override(model, dataset, device, names, None)
            forced_zero = _evaluate_with_override(model, dataset, device, names, 0.0)
            forced_one = _evaluate_with_override(model, dataset, device, names, 1.0)
            key = f"{kind}:{severity:.2f}"
            report["conditions"][key] = {
                "kind": kind,
                "severity": severity,
                "raw_q": _summary(values),
                "per_sample_gate": values,
                "learned_gate": learned,
                "force_q0": forced_zero,
                "force_q1": forced_one,
                "learned_minus_force_q1_map50_95": (
                    learned["map50_95"] - forced_one["map50_95"]
                ),
            }
            print(
                f"[{key}] q={report['conditions'][key]['raw_q']['mean']:.4f} "
                f"AP={learned['map50_95']:.4f} q1={forced_one['map50_95']:.4f}"
            )
    finally:
        model.set_gate_override(None)

    identity_q = report["conditions"]["identity:0.00"]["raw_q"]["mean"]
    report["interpretation_inputs"] = {
        "identity_q_mean": identity_q,
        "corruptions_with_lower_q_than_identity": [
            key for key, row in report["conditions"].items()
            if key != "identity:0.00" and row["raw_q"]["mean"] < identity_q
        ],
        "corruptions_where_gate_beats_force_q1": [
            key for key, row in report["conditions"].items()
            if row["learned_minus_force_q1_map50_95"] > 0
        ],
        "warning": ("q movement alone is not success; detection retention versus "
                    "FORCE-Q1 is required"),
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
