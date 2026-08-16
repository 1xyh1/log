#!/usr/bin/env python3
"""No-retraining val6 leave-one-out closeout for F1 IR gating."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import step3_eval_utils as evu  # noqa: E402
from multimodal.causality_interventions import assert_valid_shuffle_map  # noqa: E402
from multimodal.raw_sample_index import CLASS_NAMES, OUT_DEFAULT  # noqa: E402
from multimodal.run_integrity import inspect_step3_run  # noqa: E402
from multimodal.step4_f1_closeout import (  # noqa: E402
    LOO_SCHEMA,
    compute_f1_deltas,
    validate_f1_loo_payload,
)
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


MODEL_SPECS = {
    "C0": {"aux_mode": "zero", "gate_mode": "learned", "group": "B1-C0"},
    "FIXED": {"aux_mode": "ir", "gate_mode": "fixed_one", "group": "B1-I-fixed"},
    "SOFT": {"aux_mode": "ir", "gate_mode": "learned", "group": "B1-I-soft"},
}


def _load(run_dir: Path, device, expected: dict, checkpoint: str = "last.pt"):
    ck = torch.load(run_dir / "weights" / checkpoint, map_location="cpu",
                    weights_only=False)
    model = (ck.get("ema") or ck.get("model")).float().eval().to(device)
    if (getattr(model, "aux_mode", None) != expected["aux_mode"]
            or getattr(model, "gate_mode", None) != expected["gate_mode"]):
        raise RuntimeError(
            f"F1 LOO checkpoint identity mismatch in {run_dir}: "
            f"aux={getattr(model, 'aux_mode', None)} "
            f"gate={getattr(model, 'gate_mode', None)} expected={expected}"
        )
    return model


def _score(model, contract, dataset_group, excluded, device, names,
           *, aux_zero=False, aux_map=None) -> float:
    ds = TriModalDataset(
        contract, split="val", group=dataset_group, augment=False,
        aux_zero=aux_zero, aux_id_map=aux_map,
        exclude_ids={excluded} if excluded else None,
    )
    return float(evu.evaluate_dataset_stock_semantics(model, ds, device, names)["map50_95"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_f1_b_corruption")
    p.add_argument("--c0-run", default="B1-C0")
    p.add_argument("--fixed-run", default="B1-I-fixed")
    p.add_argument("--soft-run", default="B1-I-soft")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--device", default="0")
    p.add_argument("--expected-epochs", type=int, default=80)
    p.add_argument("--checkpoint", choices=["last.pt", "best.pt"],
                   default="last.pt")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    device = evu._as_device(a.device)
    names = dict(CLASS_NAMES)
    contract_path = Path(a.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    project = Path(a.project)
    runs = {
        "C0": project / a.c0_run,
        "FIXED": project / a.fixed_run,
        "SOFT": project / a.soft_run,
    }
    for tag, run_dir in runs.items():
        integrity = inspect_step3_run(
            run_dir, a.expected_epochs, require_weights=True,
            trace_name="step4_g8_trace.jsonl", growth_name="step4_growth.jsonl",
            eval_name="eval_step4_f1_b_causality.json",
        )
        if not integrity.to_dict()["passed"]:
            raise RuntimeError(f"REFUSE_F1_LOO_INCOHERENT_RUN: {tag}")
        eval_obj = json.loads(
            (run_dir / "eval_step4_f1_b_causality.json").read_text(encoding="utf-8")
        )
        if eval_obj.get("group") != MODEL_SPECS[tag]["group"]:
            raise RuntimeError(
                f"REFUSE_F1_LOO_EVAL_IDENTITY_MISMATCH: {tag} "
                f"recorded={eval_obj.get('group')}"
            )

    val_ids = list(contract["val_ids"])
    if len(val_ids) != 6:
        raise RuntimeError(f"F1 LOO protocol expects val6, got {len(val_ids)}")
    shuffle_maps = {}
    for tag in ("FIXED", "SOFT"):
        path = runs[tag] / "shuffle_map_val.json"
        mapping = json.loads(path.read_text(encoding="utf-8"))
        if not assert_valid_shuffle_map(mapping, val_ids):
            raise RuntimeError(f"invalid {tag} val shuffle map")
        shuffle_maps[tag] = mapping
    if shuffle_maps["FIXED"] != shuffle_maps["SOFT"]:
        raise RuntimeError("F1 fixed/soft shuffle maps differ")

    models = {
        tag: _load(run_dir, device, MODEL_SPECS[tag], a.checkpoint)
        for tag, run_dir in runs.items()
    }
    result = {
        "schema": LOO_SCHEMA,
        "checkpoint": a.checkpoint,
        "method": ("val6 leave-one-out; excluded image leaves the anchor set only; "
                   "the full deterministic donor mapping is retained"),
        "val_ids": val_ids,
        "runs": {tag: str(path) for tag, path in runs.items()},
        "provenance": {
            **{f"{tag}_ckpt_sha256": _sha(path / "weights" / a.checkpoint)
               for tag, path in runs.items()},
            "contract_sha256": _sha(contract_path),
            "loo_source_sha256": _sha(Path(__file__)),
            "eval_core_sha256": _sha(
                ROOT / "src" / "multimodal" / "step3_eval_utils.py"
            ),
            "dataset_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "trimodal_dataset.py"
            ),
            "model_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py"
            ),
            "gate_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "reliability_gate.py"
            ),
            "f1_closeout_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_f1_closeout.py"
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
            "fixed_shuffle_sha256": _sha(runs["FIXED"] / "shuffle_map_val.json"),
            "soft_shuffle_sha256": _sha(runs["SOFT"] / "shuffle_map_val.json"),
            "torch_version": torch.__version__,
            "ultralytics_version": __import__("ultralytics").__version__,
        },
        "folds": {},
    }
    for excluded in [None] + val_ids:
        key = "full" if excluded is None else excluded
        rows = {}
        c0 = _score(models["C0"], contract, "C0-N", excluded, device, names)
        rows["C0"] = {"NORMAL": round(c0, 6), "copy_of_normal": True}
        for tag in ("FIXED", "SOFT"):
            rows[tag] = {
                "NORMAL": round(_score(
                    models[tag], contract, "C1-I", excluded, device, names
                ), 6),
                "ZERO-AUX": round(_score(
                    models[tag], contract, "C1-I", excluded, device, names,
                    aux_zero=True,
                ), 6),
                "SHUFFLE": round(_score(
                    models[tag], contract, "C1-I", excluded, device, names,
                    aux_map=shuffle_maps[tag],
                ), 6),
                "copy_of_normal": False,
            }
        result["folds"][key] = rows
        print(f"[fold={key}] C0={c0:.4f} FIXED={rows['FIXED']['NORMAL']:.4f} "
              f"SOFT={rows['SOFT']['NORMAL']:.4f}", flush=True)

    result["deltas"] = compute_f1_deltas(result["folds"], val_ids)
    if a.checkpoint == "last.pt":
        proof = validate_f1_loo_payload(result)
    else:
        # best.pt is diagnostic-only; the shared F1 payload validator enforces
        # the frozen last.pt primary protocol, so re-check the payload math
        # locally without the checkpoint constraint.
        recomputed = compute_f1_deltas(result["folds"], val_ids)
        proof = {"passed": recomputed == result["deltas"],
                 "errors": [] if recomputed == result["deltas"]
                 else ["DELTA_MISMATCH"]}
    if not proof["passed"]:
        raise RuntimeError(f"F1_LOO_PAYLOAD_SELF_CHECK_FAILED: {proof['errors']}")
    out = project / f"step4_f1_b_loo_{a.checkpoint.removesuffix('.pt')}.json"
    if out.exists() and not a.overwrite:
        raise RuntimeError(f"REFUSE_OVERWRITE_F1_LOO: {out}")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
