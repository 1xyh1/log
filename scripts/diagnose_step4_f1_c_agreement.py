#!/usr/bin/env python3
"""F1-C-A0: RGB-IR agreement diagnosis on trained B1-soft checkpoints.

NO training, NO re-evaluation of AP.  Extracts per-image per-scale features
from the trained model and asks whether an RGB-IR agreement descriptor can
stably distinguish CORRECT pairing (NORMAL) from WRONG pairing (SHUFFLE) or
MISSING aux (ZERO) — on val6 and across the 17 degraded conditions.

Descriptors (preregistered):
    agreement_i = cos( GAP(R_i), GAP(P_i(A_i)) )      # 1xC cosine
    aux_rel_energy_i = ||P_i(A_i)||_2 / ||R_i||_2
    q = learned gate value
The projected-aux feature P_i(A_i) is exactly the residual BEFORE gate
scaling, so agreement measures what the network would add, not what q lets
through.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import step3_eval_utils as evu  # noqa: E402
from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.run_integrity import inspect_step3_run  # noqa: E402
from multimodal.step4_f1_interventions import IRCorruptionDatasetView  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402

SCALES = {4: "P3", 6: "P4", 10: "P5"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((a @ b) / denom)


def _descriptors(model, img, device) -> dict:
    """Replicate the model's fused forward and capture per-scale descriptors."""
    rgb, aux = model._split_input(img)
    y = [None] * (len(model.rgb_backbone) + len(model.tail))
    x = rgb
    for m in model.rgb_backbone:
        x = m(x)
        y[m.i] = x
    a3, a4, a5 = model.aux_encoder(aux)
    out = {}
    for layer, a in ((4, a3), (6, a4), (10, a5)):
        r = y[layer]
        p_a = model.fusions[str(layer)].proj(a)
        gap_r = torch.nn.functional.adaptive_avg_pool2d(r, 1).flatten(1)
        gap_pa = torch.nn.functional.adaptive_avg_pool2d(p_a, 1).flatten(1)
        out[SCALES[layer]] = {
            "agreement": _cos(gap_r[0], gap_pa[0]),
            "aux_rel_energy": float(p_a.norm() / (r.norm() + 1e-12)),
            "aux_abs_norm": float(p_a.norm()),
            "rgb_abs_norm": float(r.norm()),
        }
    q = model.reliability_gate(tuple(f.detach() for f in (a3, a4, a5)))
    out["q"] = float(q.item())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_f1_b_corruption")
    p.add_argument("--soft-run", default="B1-I-soft")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--device", default="0")
    p.add_argument("--expected-epochs", type=int, default=80)
    a = p.parse_args()
    device = evu._as_device(a.device)
    run_dir = Path(a.project) / a.soft_run
    integrity = inspect_step3_run(
        run_dir, a.expected_epochs, require_weights=True,
        trace_name="step4_g8_trace.jsonl", growth_name="step4_growth.jsonl",
        eval_name="eval_step4_f1_b_causality.json")
    if not integrity.to_dict()["passed"]:
        raise RuntimeError("REFUSE_DIAGNOSIS_INCOHERENT_RUN")

    contract = json.loads(Path(a.contract).read_text(encoding="utf-8"))
    base = TriModalDataset(contract, split="val", group="C1-I", augment=False)

    report = {
        "schema": "step4-f1-c-agreement-diagnosis-v1",
        "role": "diagnostic only; no AP re-evaluation",
        "provenance": {
            "soft_last_pt_sha256": _sha(run_dir / "weights" / "last.pt"),
            "soft_best_pt_sha256": _sha(run_dir / "weights" / "best.pt"),
            "contract_sha256": _sha(Path(a.contract)),
            "script_sha256": _sha(Path(__file__)),
            "model_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py"),
            "gate_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "reliability_gate.py"),
            "dataset_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "trimodal_dataset.py"),
        },
        "checkpoints": {},
    }

    for ck_name in ("last.pt", "best.pt"):
        ck = torch.load(run_dir / "weights" / ck_name, map_location="cpu",
                        weights_only=False)
        model = (ck.get("ema") or ck.get("model")).float().eval().to(device)
        model.requires_grad_(False)
        ck_report = {"val6_pairing": {}, "degraded": {}}
        with torch.no_grad():
            # val6 pairing: NORMAL / ZERO-AUX / SHUFFLE per image
            from multimodal.causality_interventions import (
                assert_valid_shuffle_map, bijective_derangement)
            val_ids = list(contract["val_ids"])
            shuffle_map = bijective_derangement(val_ids)
            assert assert_valid_shuffle_map(shuffle_map, val_ids)
            for variant in ("NORMAL", "ZERO-AUX", "SHUFFLE"):
                ds = TriModalDataset(
                    contract, split="val", group="C1-I", augment=False,
                    aux_zero=(variant == "ZERO-AUX"),
                    aux_id_map=(shuffle_map if variant == "SHUFFLE" else None))
                per_image = {}
                for idx in range(len(ds)):
                    sample = ds[idx]
                    batch = ds.collate_fn([sample])
                    batch = evu.move_step3_batch_to_device(batch, device)
                    per_image[str(sample["sample_id"])] = _descriptors(
                        model, batch["img"], device)
                ck_report["val6_pairing"][variant] = per_image
            # pairing contrast: NORMAL - SHUFFLE per image (correct vs wrong)
            contrast = {}
            for sid in val_ids:
                n = ck_report["val6_pairing"]["NORMAL"][sid]
                s = ck_report["val6_pairing"]["SHUFFLE"][sid]
                contrast[sid] = {
                    scale: n[scale]["agreement"] - s[scale]["agreement"]
                    for scale in SCALES.values()
                }
            ck_report["normal_minus_shuffle_agreement"] = contrast
            ck_report["normal_minus_shuffle_agreement_summary"] = {
                scale: {
                    "per_image": {sid: contrast[sid][scale] for sid in val_ids},
                    "positive_images": sum(
                        1 for sid in val_ids if contrast[sid][scale] > 0),
                    "median": statistics.median(
                        contrast[sid][scale] for sid in val_ids),
                    "min": min(contrast[sid][scale] for sid in val_ids),
                    "max": max(contrast[sid][scale] for sid in val_ids),
                }
                for scale in SCALES.values()
            }
            # degraded conditions: identity + 17 corruptions on val6
            conditions = [("identity", 0.0), ("zero", 1.0)]
            for kind in ("blur", "contrast", "noise", "shift"):
                conditions.extend((kind, v) for v in (0.25, 0.50, 0.75, 1.0))
            for kind, severity in conditions:
                ds = IRCorruptionDatasetView(base, kind=kind, severity=severity)
                per_image = {}
                for idx in range(len(ds)):
                    sample = ds[idx]
                    batch = ds.collate_fn([sample])
                    batch = evu.move_step3_batch_to_device(batch, device)
                    per_image[str(sample["sample_id"])] = _descriptors(
                        model, batch["img"], device)
                ck_report["degraded"][f"{kind}:{severity:.2f}"] = {
                    "kind": kind, "severity": severity, "per_image": per_image,
                    "agreement_mean": {
                        scale: statistics.mean(
                            row[scale]["agreement"]
                            for row in per_image.values())
                        for scale in SCALES.values()
                    },
                    "q_mean": statistics.mean(
                        row["q"] for row in per_image.values()),
                }
        report["checkpoints"][ck_name] = ck_report
        print(f"[{ck_name}] N-S agreement summary done")

    out_dir = ROOT / "reports" / "step4_f1_c_agreement"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "agreement_diagnosis.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
