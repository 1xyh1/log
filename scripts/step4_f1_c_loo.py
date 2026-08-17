#!/usr/bin/env python3
"""No-retraining val6 leave-one-out closeout for F1-C (four groups).

Frozen LOO promotion conditions (reviewer 2026-08-17):
    magsoft - original-soft:  full > 0, LOO median > 0, positive folds >= 4/6
    magsoft - C0:             same class of conditions
    magsoft - fixed:          same class of conditions
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
from multimodal.causality_interventions import assert_valid_shuffle_map  # noqa: E402
from multimodal.raw_sample_index import CLASS_NAMES, OUT_DEFAULT  # noqa: E402
from multimodal.run_integrity import inspect_step3_run  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402

MODEL_SPECS = {
    "C0": {"aux_mode": "zero", "gate_mode": "learned",
           "gate_module": "magnitude", "group": "F1C-C0"},
    "FIXED": {"aux_mode": "ir", "gate_mode": "fixed_one",
              "gate_module": "magnitude", "group": "F1C-I-fixed"},
    "MAGSOFT": {"aux_mode": "ir", "gate_mode": "learned",
                "gate_module": "magnitude", "group": "F1C-I-magsoft"},
    "ORIGSOFT": {"aux_mode": "ir", "gate_mode": "learned",
                 "gate_module": "original", "group": "F1C-I-soft"},
}

LOO_SCHEMA = "step4-f1-c-loo-v1"
DELTA_SPECS = {
    "MAGSOFT_minus_C0": ("MAGSOFT", "NORMAL", "C0", "NORMAL"),
    "MAGSOFT_minus_FIXED": ("MAGSOFT", "NORMAL", "FIXED", "NORMAL"),
    "MAGSOFT_minus_ORIGSOFT": ("MAGSOFT", "NORMAL", "ORIGSOFT", "NORMAL"),
    "MAGSOFT_N_minus_Z": ("MAGSOFT", "NORMAL", "MAGSOFT", "ZERO-AUX"),
    "MAGSOFT_N_minus_S": ("MAGSOFT", "NORMAL", "MAGSOFT", "SHUFFLE"),
    "ORIGSOFT_N_minus_Z": ("ORIGSOFT", "NORMAL", "ORIGSOFT", "ZERO-AUX"),
    "ORIGSOFT_N_minus_S": ("ORIGSOFT", "NORMAL", "ORIGSOFT", "SHUFFLE"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _delta_series(folds: dict, val_ids: list, tag: str, variant: str,
                  base_tag: str, base_variant: str) -> dict:
    full = round(folds["full"][tag][variant]
                 - folds["full"][base_tag][base_variant], 6)
    per_fold = {f: round(folds[f][tag][variant]
                         - folds[f][base_tag][base_variant], 6) for f in val_ids}
    vals = list(per_fold.values())
    return {"full": full, "per_fold": per_fold,
            "positive_folds": sum(1 for x in vals if x > 0),
            "n_folds": len(vals),
            "median": round(statistics.median(vals), 6) if vals else None,
            "min": round(min(vals), 6), "max": round(max(vals), 6)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_f1_c")
    p.add_argument("--c0-run", default="F1C-C0")
    p.add_argument("--fixed-run", default="F1C-I-fixed")
    p.add_argument("--magsoft-run", default="F1C-I-magsoft")
    p.add_argument("--origsoft-run", default="F1C-I-soft")
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
        "MAGSOFT": project / a.magsoft_run,
        "ORIGSOFT": project / a.origsoft_run,
    }
    for tag, run_dir in runs.items():
        integrity = inspect_step3_run(
            run_dir, a.expected_epochs, require_weights=True,
            trace_name="step4_g8_trace.jsonl", growth_name="step4_growth.jsonl",
            eval_name="eval_step4_f1_c_causality.json",
        )
        if not integrity.to_dict()["passed"]:
            raise RuntimeError(f"REFUSE_F1C_LOO_INCOHERENT_RUN: {tag}")
        eval_obj = json.loads(
            (run_dir / "eval_step4_f1_c_causality.json").read_text(encoding="utf-8"))
        if eval_obj.get("group") != MODEL_SPECS[tag]["group"]:
            raise RuntimeError(
                f"REFUSE_F1C_LOO_EVAL_IDENTITY_MISMATCH: {tag}")
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        expected = MODEL_SPECS[tag]
        if not (
            manifest.get("schema") == "step4-f1-c-manifest-v1"
            and manifest.get("run_kind") == "formal"
            and manifest.get("group") == expected["group"]
            and manifest.get("aux_mode") == expected["aux_mode"]
            and manifest.get("gate_mode") == expected["gate_mode"]
            and manifest.get("gate_module") == expected["gate_module"]
        ):
            raise RuntimeError(f"REFUSE_F1C_LOO_MANIFEST_IDENTITY_MISMATCH:{tag}")

    val_ids = list(contract["val_ids"])
    if len(val_ids) != 6:
        raise RuntimeError(f"F1C LOO protocol expects val6, got {len(val_ids)}")
    shuffle_maps = {}
    for tag in ("FIXED", "MAGSOFT", "ORIGSOFT"):
        path = runs[tag] / "shuffle_map_val.json"
        mapping = json.loads(path.read_text(encoding="utf-8"))
        if not assert_valid_shuffle_map(mapping, val_ids):
            raise RuntimeError(f"invalid {tag} val shuffle map")
        shuffle_maps[tag] = mapping
    if not (shuffle_maps["FIXED"] == shuffle_maps["MAGSOFT"]
            == shuffle_maps["ORIGSOFT"]):
        raise RuntimeError("F1C shuffle maps differ across groups")

    def load(run_dir: Path, expected: dict):
        ck = torch.load(run_dir / "weights" / a.checkpoint, map_location="cpu",
                        weights_only=False)
        model = (ck.get("ema") or ck.get("model")).float().eval().to(device)
        for key in ("aux_mode", "gate_mode"):
            if getattr(model, key, None) != expected[key]:
                raise RuntimeError(f"identity mismatch {key} in {run_dir}")
        if getattr(model, "gate_module_kind", None) != expected["gate_module"]:
            raise RuntimeError(f"gate_module mismatch in {run_dir}")
        return model

    models = {tag: load(rd, MODEL_SPECS[tag]) for tag, rd in runs.items()}

    def score(tag, dataset_group, excluded, *, aux_zero=False, aux_map=None):
        ds = TriModalDataset(
            contract, split="val", group=dataset_group, augment=False,
            aux_zero=aux_zero, aux_id_map=aux_map,
            exclude_ids={excluded} if excluded else None)
        return float(evu.evaluate_dataset_stock_semantics(
            models[tag], ds, device, names)["map50_95"])

    result = {
        "schema": LOO_SCHEMA,
        "checkpoint": a.checkpoint,
        "val_ids": val_ids,
        "runs": {tag: str(path) for tag, path in runs.items()},
        "provenance": {
            **{f"{tag}_ckpt_sha256": _sha(path / "weights" / a.checkpoint)
               for tag, path in runs.items()},
            **{f"{tag}_manifest_sha256": _sha(path / "manifest.json")
               for tag, path in runs.items()},
            **{f"{tag}_eval_sha256": _sha(path / "eval_step4_f1_c_causality.json")
               for tag, path in runs.items()},
            **{f"{tag}_shuffle_map_val_sha256": _sha(runs[tag] / "shuffle_map_val.json")
               for tag in ("FIXED", "MAGSOFT", "ORIGSOFT")},
            "contract_sha256": _sha(contract_path),
            "loo_source_sha256": _sha(Path(__file__)),
            "eval_core_sha256": _sha(ROOT / "src" / "multimodal" / "step3_eval_utils.py"),
            "dataset_source_sha256": _sha(ROOT / "src" / "multimodal" / "trimodal_dataset.py"),
            "model_source_sha256": _sha(ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py"),
            "gate_source_sha256": _sha(ROOT / "src" / "multimodal" / "reliability_gate.py"),
            "causality_interventions_sha256": _sha(
                ROOT / "src" / "multimodal" / "causality_interventions.py"),
            "torch_version": torch.__version__,
            "ultralytics_version": __import__("ultralytics").__version__,
        },
        "folds": {},
    }
    for excluded in [None] + val_ids:
        key = "full" if excluded is None else excluded
        rows = {}
        c0 = score("C0", "C0-N", excluded)
        rows["C0"] = {"NORMAL": round(c0, 6), "copy_of_normal": True}
        for tag in ("FIXED", "MAGSOFT", "ORIGSOFT"):
            rows[tag] = {
                "NORMAL": round(score(tag, "C1-I", excluded), 6),
                "ZERO-AUX": round(score(tag, "C1-I", excluded, aux_zero=True), 6),
                "SHUFFLE": round(score(tag, "C1-I", excluded,
                                       aux_map=shuffle_maps[tag]), 6),
                "copy_of_normal": False,
            }
        result["folds"][key] = rows
        print(f"[fold={key}] C0={c0:.4f} FIXED={rows['FIXED']['NORMAL']:.4f} "
              f"MAGSOFT={rows['MAGSOFT']['NORMAL']:.4f} "
              f"ORIGSOFT={rows['ORIGSOFT']['NORMAL']:.4f}", flush=True)

    deltas = {}
    for key, (tag, variant, base_tag, base_variant) in DELTA_SPECS.items():
        deltas[key] = _delta_series(result["folds"], val_ids, tag, variant,
                                    base_tag, base_variant)
    result["deltas"] = deltas
    # self-proof: recompute and compare
    recomputed = {key: _delta_series(result["folds"], val_ids, tag, variant,
                                     base_tag, base_variant)
                  for key, (tag, variant, base_tag, base_variant)
                  in DELTA_SPECS.items()}
    if recomputed != deltas:
        raise RuntimeError("F1C_LOO_PAYLOAD_SELF_CHECK_FAILED")

    out = project / f"step4_f1_c_loo_{a.checkpoint.removesuffix('.pt')}.json"
    if out.exists() and not a.overwrite:
        raise RuntimeError(f"REFUSE_OVERWRITE_F1C_LOO: {out}")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
