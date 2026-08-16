#!/usr/bin/env python3
"""F1-C-A0 supplement: all17 agreement axis + growth-drift analysis.

The val6 agreement diagnosis showed GAP-cosine cannot separate NORMAL from
SHUFFLE on val6.  This script adds the preregistered all17 axis and quantifies
the epoch-39 -> 80 paired-signal decay from the growth trace (projection norm
drift, aux encoder norm drift, gate norm drift) and the train/val mAP curves.
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
from multimodal.causality_interventions import (  # noqa: E402
    assert_valid_shuffle_map, bijective_derangement)
from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.run_integrity import inspect_step3_run  # noqa: E402
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


def _agreement(model, img) -> dict:
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
        }
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
    all17_ids = list(contract["all17_ids"])
    shuffle_map = bijective_derangement(all17_ids)
    assert assert_valid_shuffle_map(shuffle_map, all17_ids)
    map_payload = json.dumps(shuffle_map, sort_keys=True, ensure_ascii=False)
    map_sha = hashlib.sha256(map_payload.encode("utf-8")).hexdigest()

    report = {
        "schema": "step4-f1-c-agreement-all17-v1.1",
        "provenance": {
            "soft_last_pt_sha256": _sha(run_dir / "weights" / "last.pt"),
            "soft_best_pt_sha256": _sha(run_dir / "weights" / "best.pt"),
            "results_csv_sha256": _sha(run_dir / "results.csv"),
            "growth_trace_sha256": _sha(run_dir / "step4_growth.jsonl"),
            "contract_sha256": _sha(Path(a.contract)),
            "script_sha256": _sha(Path(__file__)),
            "model_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py"),
            "gate_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "reliability_gate.py"),
            "dataset_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "trimodal_dataset.py"),
            "eval_core_sha256": _sha(
                ROOT / "src" / "multimodal" / "step3_eval_utils.py"),
            "causality_interventions_sha256": _sha(
                ROOT / "src" / "multimodal" / "causality_interventions.py"),
            "run_integrity_sha256": _sha(
                ROOT / "src" / "multimodal" / "run_integrity.py"),
            "all17_shuffle_map_sha256": map_sha,
            "torch_version": torch.__version__,
            "ultralytics_version": __import__("ultralytics").__version__,
        },
        "all17_shuffle_map": shuffle_map,
        "all17": {},
        "growth_drift": {},
    }

    for ck_name in ("last.pt", "best.pt"):
        ck = torch.load(run_dir / "weights" / ck_name, map_location="cpu",
                        weights_only=False)
        model = (ck.get("ema") or ck.get("model")).float().eval().to(device)
        model.requires_grad_(False)
        per_variant = {}
        with torch.no_grad():
            for variant in ("NORMAL", "SHUFFLE"):
                ds = TriModalDataset(
                    contract, split="all17", group="C1-I", augment=False,
                    aux_id_map=(shuffle_map if variant == "SHUFFLE" else None))
                per_image = {}
                for idx in range(len(ds)):
                    sample = ds[idx]
                    batch = ds.collate_fn([sample])
                    batch = evu.move_step3_batch_to_device(batch, device)
                    per_image[str(sample["sample_id"])] = _agreement(
                        model, batch["img"])
                per_variant[variant] = per_image
        contrast = {}
        for sid in all17_ids:
            n = per_variant["NORMAL"][sid]
            s = per_variant["SHUFFLE"][sid]
            contrast[sid] = {scale: n[scale]["agreement"] - s[scale]["agreement"]
                             for scale in SCALES.values()}
        report["all17"][ck_name] = {
            "normal_minus_shuffle_agreement": {
                scale: {
                    "per_image": {sid: contrast[sid][scale] for sid in all17_ids},
                    "positive_images": sum(1 for sid in all17_ids
                                           if contrast[sid][scale] > 0),
                    "median": statistics.median(contrast[sid][scale]
                                                for sid in all17_ids),
                    "min": min(contrast[sid][scale] for sid in all17_ids),
                    "max": max(contrast[sid][scale] for sid in all17_ids),
                }
                for scale in SCALES.values()
            },
            "n_images": len(all17_ids),
        }
        print(f"[{ck_name}] all17 N-S agreement done")

    # ---- growth drift: projection/aux/gate norm trajectories ----
    growth = [json.loads(x) for x in
              (run_dir / "step4_growth.jsonl").read_text(
                  encoding="utf-8").splitlines() if x.strip()]
    def traj(key):
        return [float(row[key]) for row in growth]
    report["growth_drift"] = {
        "projP3_norm": {"at_epoch39": traj("projP3_norm")[38],
                        "at_epoch80": traj("projP3_norm")[-1],
                        "ratio_80_39": traj("projP3_norm")[-1] / max(traj("projP3_norm")[38], 1e-12)},
        "projP4_norm": {"at_epoch39": traj("projP4_norm")[38],
                        "at_epoch80": traj("projP4_norm")[-1],
                        "ratio_80_39": traj("projP4_norm")[-1] / max(traj("projP4_norm")[38], 1e-12)},
        "projP5_norm": {"at_epoch39": traj("projP5_norm")[38],
                        "at_epoch80": traj("projP5_norm")[-1],
                        "ratio_80_39": traj("projP5_norm")[-1] / max(traj("projP5_norm")[38], 1e-12)},
        "aux_encoder_norm": {"at_epoch39": traj("aux_encoder_norm")[38],
                             "at_epoch80": traj("aux_encoder_norm")[-1]},
        "gate_param_norm": {"at_epoch39": traj("gate_param_norm")[38],
                            "at_epoch80": traj("gate_param_norm")[-1]},
        "q_mean": {"at_epoch39": growth[38]["effective_q"]["mean"],
                   "at_epoch80": growth[-1]["effective_q"]["mean"]},
    }
    # mAP curve from results.csv
    import csv
    with open(run_dir / "results.csv", encoding="utf-8", newline="") as f:
        rows = [float(r["metrics/mAP50-95(B)"])
                for r in csv.DictReader(f) if r.get("metrics/mAP50-95(B)")]
    report["growth_drift"]["val_mAP"] = {
        "at_epoch39": rows[38], "at_epoch80": rows[-1],
        "peak": max(rows), "peak_epoch": rows.index(max(rows)) + 1,
    }

    out_dir = ROOT / "reports" / "step4_f1_c_agreement"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "agreement_all17_growth_v1_1.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
