#!/usr/bin/env python3
"""F1-C-A1: magnitude-aware gate-input descriptor audit (NO training).

The reviewer's exploratory finding: aux_rel_energy (P5) tracks corruption
severity (identity 0.057 -> noise 0.247) while learned q barely moves
(0.494-0.505), Spearman(energy, sweep-optimal q) ~ -0.85.  This script audits
candidate gate-input descriptors against the q-sweep evidence:

    * slices of the actual joint LayerNorm(concat(GAP(A3..A5))) gate input
    * raw GAP(A) L2 magnitude (no LayerNorm)
    * per-scale log-RMS(A_i)
    * RGB-relative energy ||P_i(A_i)|| / ||R_i||
    * projected residual energy ||P_i(A_i)||
    * per-scale spatial cosine (secondary)

The hard sweep-optimal q is retained as exploratory evidence, with tie-aware
ranks and an identifiability flag.  The primary diagnostic target is the
continuous AP(q=0)-AP(q=1) contrast, supplemented by leave-one-corruption-
family-out correlations.  None of these statistics is a promotion criterion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import step3_eval_utils as evu  # noqa: E402
from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.run_integrity import inspect_step3_run  # noqa: E402
from multimodal.step4_f1_c_descriptor_audit import (  # noqa: E402
    Q_GRID,
    SCAN_IDENTIFIABLE_RANGE,
    correlation_report,
    scan_targets,
)
from multimodal.step4_f1_interventions import IRCorruptionDatasetView  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402

SCALES = {4: "P3", 6: "P4", 10: "P5"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptors(model, img) -> dict:
    rgb, aux = model._split_input(img)
    y = [None] * (len(model.rgb_backbone) + len(model.tail))
    x = rgb
    for m in model.rgb_backbone:
        x = m(x)
        y[m.i] = x
    a3, a4, a5 = model.aux_encoder(aux)
    out = {}
    gaps = []
    features = []
    for layer, a in ((4, a3), (6, a4), (10, a5)):
        r = y[layer]
        p_a = model.fusions[str(layer)].proj(a)
        gap_a = torch.nn.functional.adaptive_avg_pool2d(a, 1).flatten(1)
        gaps.append(gap_a)
        features.append((layer, a, r, p_a, gap_a))

    joint_gap = torch.cat(gaps, dim=1)
    gate_input = model.reliability_gate.norm(joint_gap)
    offset = 0
    for layer, a, r, p_a, gap_a in features:
        width = gap_a.shape[1]
        gate_slice = gate_input[:, offset:offset + width]
        offset += width
        rms = float(a.pow(2).mean().sqrt())
        out[SCALES[layer]] = {
            "gate_ln_slice_rms": float(gate_slice.pow(2).mean().sqrt()),
            "raw_gap_magnitude": float(gap_a.norm()),
            "log_rms": math.log(rms + 1e-9),
            "rgb_relative_energy": float(p_a.norm() / (r.norm() + 1e-12)),
            "proj_energy": float(p_a.norm()),
            "spatial_cos_mean": float(torch.nn.functional.cosine_similarity(
                r, p_a, dim=1).mean()),
        }
    out["gate_input_summary"] = {
        "joint_gap_norm": float(joint_gap.norm()),
        "gate_ln_full_norm": float(gate_input.norm()),
        "gate_ln_full_rms": float(gate_input.pow(2).mean().sqrt()),
    }
    return out


def _condition_specs() -> list[tuple[str, float]]:
    conditions = [("identity", 0.0), ("zero", 1.0)]
    for kind in ("blur", "contrast", "noise", "shift"):
        conditions.extend((kind, value) for value in (0.25, 0.50, 0.75, 1.0))
    return conditions


def _validate_quality(quality: dict, quality_path: Path, checkpoint: str,
                      checkpoint_path: Path, contract_path: Path) -> None:
    if quality.get("schema") != "step4-f1-b-ir-quality-probe-v1":
        raise RuntimeError(f"QUALITY_SCHEMA_MISMATCH: {quality_path}")
    if quality.get("checkpoint") != checkpoint:
        raise RuntimeError(
            f"QUALITY_CHECKPOINT_MISMATCH: recorded={quality.get('checkpoint')} "
            f"expected={checkpoint}"
        )
    provenance = quality.get("provenance") or {}
    expected_sha = {
        "checkpoint_sha256": _sha(checkpoint_path),
        "contract_sha256": _sha(contract_path),
    }
    for key, value in expected_sha.items():
        if provenance.get(key) != value:
            raise RuntimeError(
                f"QUALITY_STALE_PROVENANCE:{key}: "
                f"recorded={provenance.get(key)} current={value}"
            )
    expected_conditions = {
        f"{kind}:{severity:.2f}" for kind, severity in _condition_specs()
    }
    conditions = quality.get("conditions") or {}
    if set(conditions) != expected_conditions:
        raise RuntimeError("QUALITY_CONDITION_SET_MISMATCH")
    for key, row in conditions.items():
        try:
            scan_targets(row.get("scan_overrides") or {})
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"QUALITY_SCAN_INVALID:{key}:{exc}") from exc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_f1_b_corruption")
    p.add_argument("--soft-run", default="B1-I-soft")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--device", default="0")
    p.add_argument("--expected-epochs", type=int, default=80)
    p.add_argument("--checkpoint", choices=["last.pt", "best.pt"],
                   default="last.pt")
    a = p.parse_args()
    device = evu._as_device(a.device)
    run_dir = Path(a.project) / a.soft_run
    integrity = inspect_step3_run(
        run_dir, a.expected_epochs, require_weights=True,
        trace_name="step4_g8_trace.jsonl", growth_name="step4_growth.jsonl",
        eval_name="eval_step4_f1_b_causality.json")
    if not integrity.to_dict()["passed"]:
        raise RuntimeError("REFUSE_DESCRIPTOR_AUDIT_INCOHERENT_RUN")

    quality_path = run_dir / f"eval_step4_f1_b_quality_{a.checkpoint.removesuffix('.pt')}.json"
    if not quality_path.exists():
        raise RuntimeError(f"QUALITY_EVIDENCE_MISSING: {quality_path}")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))

    contract_path = Path(a.contract)
    checkpoint_path = run_dir / "weights" / a.checkpoint
    _validate_quality(quality, quality_path, a.checkpoint, checkpoint_path,
                      contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    base = TriModalDataset(contract, split="val", group="C1-I", augment=False)
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = (ck.get("ema") or ck.get("model")).float().eval().to(device)
    model.requires_grad_(False)

    conditions = _condition_specs()

    per_condition = {}
    with torch.no_grad():
        for kind, severity in conditions:
            ds = IRCorruptionDatasetView(base, kind=kind, severity=severity)
            means = {scale: {key: [] for key in (
                "gate_ln_slice_rms", "raw_gap_magnitude", "log_rms",
                "rgb_relative_energy", "proj_energy", "spatial_cos_mean")}
                for scale in SCALES.values()}
            gate_means = {key: [] for key in (
                "joint_gap_norm", "gate_ln_full_norm", "gate_ln_full_rms")}
            for idx in range(len(ds)):
                sample = ds[idx]
                batch = ds.collate_fn([sample])
                batch = evu.move_step3_batch_to_device(batch, device)
                desc = _descriptors(model, batch["img"])
                for scale in SCALES.values():
                    for key in means[scale]:
                        means[scale][key].append(desc[scale][key])
                for key in gate_means:
                    gate_means[key].append(desc["gate_input_summary"][key])
            per_condition[f"{kind}:{severity:.2f}"] = {
                **{
                    scale: {key: sum(vals) / len(vals) for key, vals in d.items()}
                    for scale, d in means.items()
                },
                "gate_input_summary": {
                    key: sum(vals) / len(vals) for key, vals in gate_means.items()
                },
            }

    targets = {
        key: scan_targets(row["scan_overrides"])
        for key, row in quality["conditions"].items()
    }
    cond_keys = [f"{k}:{s:.2f}" for k, s in conditions]
    families = [kind for kind, _ in conditions]
    correlations = {}
    for scale in SCALES.values():
        for desc_key in ("gate_ln_slice_rms", "raw_gap_magnitude", "log_rms",
                         "rgb_relative_energy", "proj_energy",
                         "spatial_cos_mean"):
            xs = [per_condition[k][scale][desc_key] for k in cond_keys]
            correlations[f"{scale}_{desc_key}"] = correlation_report(
                xs, [targets[k] for k in cond_keys], families
            )

    report = {
        "schema": "step4-f1-c-descriptor-audit-v2",
        "checkpoint": a.checkpoint,
        "provenance": {
            "checkpoint_sha256": _sha(checkpoint_path),
            "quality_sha256": _sha(quality_path),
            "quality_recorded_checkpoint_sha256":
                quality["provenance"]["checkpoint_sha256"],
            "quality_recorded_contract_sha256":
                quality["provenance"]["contract_sha256"],
            "contract_sha256": _sha(contract_path),
            "script_sha256": _sha(Path(__file__)),
            "statistics_source_sha256": _sha(
                ROOT / "src" / "multimodal" /
                "step4_f1_c_descriptor_audit.py"),
            "model_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py"),
            "gate_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "reliability_gate.py"),
            "dataset_source_sha256": _sha(
                ROOT / "src" / "multimodal" / "trimodal_dataset.py"),
            "corruption_view_sha256": _sha(
                ROOT / "src" / "multimodal" / "step4_f1_interventions.py"),
            "step3_eval_utils_sha256": _sha(
                ROOT / "src" / "multimodal" / "step3_eval_utils.py"),
            "run_integrity_sha256": _sha(
                ROOT / "src" / "multimodal" / "run_integrity.py"),
            "torch_version": torch.__version__,
            "ultralytics_version": __import__("ultralytics").__version__,
        },
        "target_definition": {
            "q_grid": list(Q_GRID),
            "best_q_tie_break": "lowest q among equal AP values",
            "best_q_identifiable_min_scan_range": SCAN_IDENTIFIABLE_RANGE,
            "primary_continuous_target": "AP(q=0)-AP(q=1)",
            "hard_best_q_role": "exploratory only",
        },
        "sweep_targets": targets,
        "per_condition_descriptors": per_condition,
        "correlations": correlations,
        "descriptor_semantics": {
            "gate_ln_slice_rms": (
                "per-scale slice RMS after the gate's actual joint "
                "LayerNorm(concat(GAP(A3),GAP(A4),GAP(A5)))"),
            "spatial_cos_mean": (
                "mean per-location channel cosine cos(R[:, :, h, w], "
                "P(A)[:, :, h, w])"),
        },
        "caveats": (
            "exploratory evidence only: same val6 and non-independent "
            "conditions. A scalar norm or slice RMS cannot establish whether "
            "the full LayerNorm vector direction contains predictive "
            "information. Hard best-q correlations are secondary because the "
            "five-point sweep is discrete and frequently weakly identifiable."
        ),
    }
    out_dir = ROOT / "reports" / "step4_f1_c_agreement"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"descriptor_audit_v2_{a.checkpoint.removesuffix('.pt')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                              allow_nan=False),
                   encoding="utf-8")
    def rounded(value):
        return round(value, 4) if value is not None else None

    concise = {
        key: {
            "rho_q0_minus_q1": rounded(value["spearman_vs_q0_minus_q1"]),
            "rho_best_q_identifiable": rounded(
                value["spearman_vs_best_q_identifiable_only"]),
            "n_identifiable": value["n_identifiable_for_best_q"],
        }
        for key, value in correlations.items()
    }
    print(json.dumps(concise, indent=2, ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
