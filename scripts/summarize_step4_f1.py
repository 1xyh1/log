#!/usr/bin/env python3
"""Fail-fast F1 IR-gate closeout and next-branch decision."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.run_integrity import inspect_step3_run  # noqa: E402
from multimodal.step4_closeout import g8_check  # noqa: E402
from multimodal.step4_f1_closeout import (  # noqa: E402
    LOO_SCHEMA,
    validate_f1_loo_payload,
)


GROUP_SPECS = {
    "C0": {"group": "F1-C0", "aux_mode": "zero", "gate_mode": "learned"},
    "FIXED": {"group": "F1-I-fixed", "aux_mode": "ir", "gate_mode": "fixed_one"},
    "SOFT": {"group": "F1-I-soft", "aux_mode": "ir", "gate_mode": "learned"},
}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha(recorded: str | None, path: Path, label: str) -> dict:
    current = _sha(path) if path.exists() else None
    match = bool(recorded and current and recorded == current)
    if not match:
        raise RuntimeError(
            f"STALE_PROVENANCE {label}: recorded={recorded} current={current}"
        )
    return {"recorded": recorded, "current": current, "match": True}


def _verify_eval(eval_obj, run_dir: Path, contract_path: Path,
                 expected: dict) -> dict:
    if eval_obj.get("schema") != "step4-f1-stock-validator-semantics-v1":
        raise RuntimeError(f"bad F1 eval schema in {run_dir}")
    for key in ("group", "aux_mode", "gate_mode"):
        if eval_obj.get(key) != expected[key]:
            raise RuntimeError(
                f"F1_EVAL_IDENTITY_MISMATCH {run_dir.name}:{key} "
                f"recorded={eval_obj.get(key)} expected={expected[key]}"
            )
    p = eval_obj.get("provenance") or {}
    targets = {
        "results_sha256": run_dir / "results.csv",
        "args_sha256": run_dir / "args.yaml",
        "last_pt_sha256": run_dir / "weights" / "last.pt",
        "best_pt_sha256": run_dir / "weights" / "best.pt",
        "manifest_sha256": run_dir / "manifest.json",
        "contract_sha256": contract_path,
        "evaluator_source_sha256": ROOT / "scripts" / "eval_step4_f1_causality.py",
        "model_source_sha256": ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py",
        "gate_source_sha256": ROOT / "src" / "multimodal" / "reliability_gate.py",
        "step3_eval_utils_sha256": ROOT / "src" / "multimodal" / "step3_eval_utils.py",
        "trimodal_dataset_sha256": ROOT / "src" / "multimodal" / "trimodal_dataset.py",
        "f0_model_source_sha256": ROOT / "src" / "multimodal" / "step4_f0_model.py",
        "aux_encoder_source_sha256": ROOT / "src" / "multimodal" / "aux_encoder.py",
        "feature_fusion_source_sha256": ROOT / "src" / "multimodal" / "feature_fusion.py",
        "trainability_source_sha256": ROOT / "src" / "multimodal" / "trainability.py",
        "causality_interventions_sha256": ROOT / "src" / "multimodal" / "causality_interventions.py",
        "raw_sample_index_sha256": ROOT / "src" / "multimodal" / "raw_sample_index.py",
    }
    checks = {
        key: _require_sha(p.get(key), path, f"{run_dir.name}:{key}")
        for key, path in targets.items()
    }
    for split, filename in (
        ("train", "shuffle_map_train.json"),
        ("val", "shuffle_map_val.json"),
        ("all17", "shuffle_map_all17.json"),
    ):
        checks[f"shuffle:{split}"] = _require_sha(
            (p.get("shuffle_map_sha256") or {}).get(split),
            run_dir / filename,
            f"{run_dir.name}:shuffle:{split}",
        )
    import torch
    import ultralytics
    for key, current in (("torch_version", torch.__version__),
                         ("ultralytics_version", ultralytics.__version__)):
        recorded = p.get(key)
        if recorded != current:
            raise RuntimeError(
                f"STALE_RUNTIME {run_dir.name}:{key} recorded={recorded} current={current}"
            )
        checks[key] = {"recorded": recorded, "current": current, "match": True}
    return checks


def _verify_g6(tag: str, gate: dict) -> dict:
    """Re-judge the recorded update evidence instead of trusting passed=true."""
    rgb_ok = gate.get("rgb_backbone_unchanged") is True
    q_ok = gate.get("q_finite_and_bounded") is True
    aux_delta = float(gate.get("aux_encoder_global_rel_l2", float("nan")))
    gate_delta = float(gate.get("gate_max_abs_change", float("nan")))
    proj = [float(value) for value in gate.get("proj_weight_norms", [])]
    proj_bias = [float(value) for value in gate.get("proj_bias_norms", [])]
    if (len(proj) != 3 or len(proj_bias) != 3
            or not all(math.isfinite(value) for value in proj + proj_bias)):
        passed = False
    elif tag == "C0":
        passed = rgb_ok and q_ok and aux_delta < 1e-3 and max(proj) == 0.0
    elif tag == "FIXED":
        q = gate.get("last_epoch_effective_q") or {}
        passed = (
            rgb_ok and q_ok and aux_delta > 1e-3 and min(proj) > 0.0
            and q.get("min") == 1.0 and q.get("max") == 1.0
        )
    else:
        passed = (
            rgb_ok and q_ok and aux_delta > 1e-3 and min(proj) > 0.0
            and gate_delta > 0.0
        )
    if not passed:
        raise RuntimeError(f"F1_G6_REJUDGE_FAIL {tag}: {gate}")
    return {
        "rgb_backbone_unchanged": rgb_ok,
        "aux_encoder_global_rel_l2": aux_delta,
        "gate_max_abs_change": gate_delta,
        "proj_weight_norms": proj,
        "proj_bias_norms": proj_bias,
        "q_finite_and_bounded": q_ok,
        "passed": True,
    }


def _verify_manifest(manifest: dict, tag: str, run_dir: Path,
                     contract_path: Path, audit_path: Path,
                     expected_epochs: int) -> dict:
    expected = GROUP_SPECS[tag]
    identity = {
        "schema": "step4-f1-ir-gate-manifest-v1",
        "group": expected["group"],
        "physical_run_name": run_dir.name,
        "run_kind": "formal",
        "aux_mode": expected["aux_mode"],
        "gate_mode": expected["gate_mode"],
        "expected_epochs": expected_epochs,
    }
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"F1_MANIFEST_IDENTITY_MISMATCH {tag}:{key} "
                f"recorded={manifest.get(key)} expected={value}"
            )
    targets = {
        "contract_sha256": contract_path,
        "pretrain_audit_sha256": audit_path,
        "runner_source_sha256": ROOT / "scripts" / "run_step4_f1_ir_gate.py",
        "model_source_sha256": ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py",
        "gate_source_sha256": ROOT / "src" / "multimodal" / "reliability_gate.py",
        "f0_model_source_sha256": ROOT / "src" / "multimodal" / "step4_f0_model.py",
        "aux_encoder_source_sha256": ROOT / "src" / "multimodal" / "aux_encoder.py",
        "feature_fusion_source_sha256": ROOT / "src" / "multimodal" / "feature_fusion.py",
        "trainability_source_sha256": ROOT / "src" / "multimodal" / "trainability.py",
        "dataset_source_sha256": ROOT / "src" / "multimodal" / "trimodal_dataset.py",
        "preprocess_source_sha256": ROOT / "src" / "multimodal" / "modality_preprocess.py",
    }
    checks = {
        key: _require_sha(manifest.get(key), path, f"MANIFEST:{tag}:{key}")
        for key, path in targets.items()
    }
    import torch
    import ultralytics
    for key, current in (("torch_version", torch.__version__),
                         ("ultralytics_version", ultralytics.__version__)):
        recorded = manifest.get(key)
        if recorded != current:
            raise RuntimeError(
                f"STALE_RUNTIME MANIFEST:{tag}:{key} "
                f"recorded={recorded} current={current}"
            )
        checks[key] = {"recorded": recorded, "current": current, "match": True}
    return checks


def _verify_quality(report: dict, run_dir: Path, contract_path: Path) -> dict:
    if report.get("schema") != "step4-f1-ir-quality-probe-v2":
        raise RuntimeError("bad F1 quality schema (A0 probe v2 required)")
    if report.get("checkpoint") != "last.pt":
        raise RuntimeError("F1 closeout requires last.pt quality evidence")
    p = report.get("provenance") or {}
    targets = {
        "checkpoint_sha256": run_dir / "weights" / "last.pt",
        "contract_sha256": contract_path,
        "script_sha256": ROOT / "scripts" / "eval_step4_f1_quality.py",
        "interventions_source_sha256": ROOT / "src" / "multimodal" / "step4_f1_interventions.py",
        "evaluator_core_sha256": ROOT / "src" / "multimodal" / "step3_eval_utils.py",
        "model_source_sha256": ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py",
        "gate_source_sha256": ROOT / "src" / "multimodal" / "reliability_gate.py",
        "dataset_source_sha256": ROOT / "src" / "multimodal" / "trimodal_dataset.py",
    }
    checks = {
        key: _require_sha(p.get(key), path, f"QUALITY:{key}")
        for key, path in targets.items()
    }
    import torch
    import ultralytics
    for key, current in (("torch_version", torch.__version__),
                         ("ultralytics_version", ultralytics.__version__)):
        recorded = p.get(key)
        if recorded != current:
            raise RuntimeError(
                f"STALE_RUNTIME QUALITY:{key} recorded={recorded} current={current}"
            )
        checks[key] = {"recorded": recorded, "current": current, "match": True}
    return checks


def _verify_loo(loo, runs: dict[str, Path], contract_path: Path) -> dict:
    if loo.get("schema") != LOO_SCHEMA or loo.get("checkpoint") != "last.pt":
        raise RuntimeError("bad F1 LOO schema/checkpoint")
    p = loo.get("provenance") or {}
    targets = {
        **{f"{tag}_last_pt_sha256": path / "weights" / "last.pt"
           for tag, path in runs.items()},
        "contract_sha256": contract_path,
        "loo_source_sha256": ROOT / "scripts" / "step4_f1_loo.py",
        "eval_core_sha256": ROOT / "src" / "multimodal" / "step3_eval_utils.py",
        "dataset_source_sha256": ROOT / "src" / "multimodal" / "trimodal_dataset.py",
        "model_source_sha256": ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py",
        "gate_source_sha256": ROOT / "src" / "multimodal" / "reliability_gate.py",
        "f1_closeout_source_sha256": ROOT / "src" / "multimodal" / "step4_f1_closeout.py",
        "f0_model_source_sha256": ROOT / "src" / "multimodal" / "step4_f0_model.py",
        "aux_encoder_source_sha256": ROOT / "src" / "multimodal" / "aux_encoder.py",
        "feature_fusion_source_sha256": ROOT / "src" / "multimodal" / "feature_fusion.py",
        "trainability_source_sha256": ROOT / "src" / "multimodal" / "trainability.py",
        "causality_interventions_sha256": ROOT / "src" / "multimodal" / "causality_interventions.py",
        "raw_sample_index_sha256": ROOT / "src" / "multimodal" / "raw_sample_index.py",
        "fixed_shuffle_sha256": runs["FIXED"] / "shuffle_map_val.json",
        "soft_shuffle_sha256": runs["SOFT"] / "shuffle_map_val.json",
    }
    checks = {
        key: _require_sha(p.get(key), path, f"LOO:{key}")
        for key, path in targets.items()
    }
    import torch
    import ultralytics
    for key, current in (("torch_version", torch.__version__),
                         ("ultralytics_version", ultralytics.__version__)):
        recorded = p.get(key)
        if recorded != current:
            raise RuntimeError(
                f"STALE_RUNTIME LOO:{key} recorded={recorded} current={current}"
            )
        checks[key] = {"recorded": recorded, "current": current, "match": True}
    return checks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_f1_ir_gate")
    p.add_argument("--c0-run", default="F1-C0")
    p.add_argument("--fixed-run", default="F1-I-fixed")
    p.add_argument("--soft-run", default="F1-I-soft")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument(
        "--audit-report",
        default=str(ROOT / "reports" / "step4_f1_ir_gate" / "pretrain_audit.json"),
    )
    p.add_argument("--expected-epochs", type=int, default=80)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    project = Path(a.project)
    contract_path = Path(a.contract)
    audit_path = Path(a.audit_report)
    runs = {
        "C0": project / a.c0_run,
        "FIXED": project / a.fixed_run,
        "SOFT": project / a.soft_run,
    }
    out = project / "_summary_step4_f1.json"
    if out.exists() and not a.overwrite:
        raise RuntimeError(f"REFUSE_OVERWRITE_F1_SUMMARY: {out}")
    integrity = {}
    g6 = {}
    for tag, run_dir in runs.items():
        rep = inspect_step3_run(
            run_dir, a.expected_epochs, require_weights=True,
            trace_name="step4_g8_trace.jsonl", growth_name="step4_growth.jsonl",
            eval_name="eval_step4_f1_causality.json",
        ).to_dict()
        if not rep["passed"]:
            raise RuntimeError(f"F1_INTEGRITY_FAIL: {tag}")
        gate = _read(run_dir / "step4_update_gate.json")
        if gate.get("passed") is not True:
            raise RuntimeError(f"F1_G6_FAIL: {tag}")
        g6[tag] = _verify_g6(tag, gate)
        integrity[tag] = rep

    evals = {
        tag: _read(run_dir / "eval_step4_f1_causality.json")
        for tag, run_dir in runs.items()
    }
    provenance = {
        tag: _verify_eval(evals[tag], runs[tag], contract_path, GROUP_SPECS[tag])
        for tag in runs
    }
    manifests = {tag: _read(run_dir / "manifest.json") for tag, run_dir in runs.items()}
    manifest_provenance = {
        tag: _verify_manifest(
            manifests[tag], tag, runs[tag], contract_path, audit_path,
            a.expected_epochs,
        )
        for tag in runs
    }
    initial_state_keys = (
        "initial_model_state_sha256",
        "initial_rgb_backbone_sha256",
        "initial_aux_encoder_sha256",
        "initial_fusion_sha256",
        "initial_gate_sha256",
    )
    matched_initial_state = {}
    for key in initial_state_keys:
        values = {tag: manifests[tag].get(key) for tag in runs}
        passed = bool(all(values.values()) and len(set(values.values())) == 1)
        if not passed:
            raise RuntimeError(f"F1_INITIAL_STATE_MISMATCH {key}: {values}")
        matched_initial_state[key] = {"values": values, "passed": True}
    g8 = g8_check(runs, a.expected_epochs)
    if not g8["passed"]:
        raise RuntimeError(f"F1_G8_CLOSEOUT_FAIL: {g8}")
    loo_path = project / "step4_f1_loo.json"
    if not loo_path.exists():
        raise RuntimeError("F1_LOO_MISSING")
    loo = _read(loo_path)
    loo_payload_validation = validate_f1_loo_payload(loo)
    if not loo_payload_validation["passed"]:
        raise RuntimeError(
            f"F1_LOO_PAYLOAD_INVALID: {loo_payload_validation['errors']}"
        )
    loo_provenance = _verify_loo(loo, runs, contract_path)
    for tag in runs:
        loo_sha = (loo.get("provenance") or {}).get(f"{tag}_last_pt_sha256")
        eval_sha = (evals[tag].get("provenance") or {}).get("last_pt_sha256")
        if not loo_sha or loo_sha != eval_sha:
            raise RuntimeError(
                f"F1_LOO_EVAL_CHECKPOINT_MISMATCH {tag}: "
                f"loo={loo_sha} eval={eval_sha}"
            )

    quality_path = runs["SOFT"] / "eval_step4_f1_quality_last.json"
    if not quality_path.exists():
        raise RuntimeError("F1_LAST_PT_QUALITY_EVIDENCE_MISSING")
    quality = _read(quality_path)
    quality_provenance = _verify_quality(quality, runs["SOFT"], contract_path)

    # Post-hoc gradient-semantics audit (detach erratum checklist).
    posthoc_path = ROOT / "reports" / "step4_f1_ir_gate" / "posthoc_gradient_audit.json"
    if not posthoc_path.exists():
        raise RuntimeError("F1_POSTHOC_AUDIT_MISSING")
    posthoc = _read(posthoc_path)
    if posthoc.get("schema") != "step4-f1-posthoc-gradient-audit-v1" \
            or posthoc.get("passed") is not True:
        raise RuntimeError("F1_POSTHOC_AUDIT_NOT_PASSED")
    posthoc_prov = posthoc.get("provenance") or {}
    posthoc_targets = {
        "soft_last_pt_sha256": runs["SOFT"] / "weights" / "last.pt",
        "c0_last_pt_sha256": runs["C0"] / "weights" / "last.pt",
        "contract_sha256": contract_path,
        "script_sha256": ROOT / "scripts" / "audit_step4_f1_posthoc.py",
        "model_source_sha256": ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py",
        "gate_source_sha256": ROOT / "src" / "multimodal" / "reliability_gate.py",
        "dataset_source_sha256": ROOT / "src" / "multimodal" / "trimodal_dataset.py",
        "eval_core_sha256": ROOT / "src" / "multimodal" / "step3_eval_utils.py",
    }
    posthoc_checks = {
        key: _require_sha(posthoc_prov.get(key), path, f"POSTHOC:{key}")
        for key, path in posthoc_targets.items()
    }
    # cross-consistency: posthoc checkpoints must be the same last.pt files
    for tag, key in (("SOFT", "soft_last_pt_sha256"), ("C0", "c0_last_pt_sha256")):
        eval_sha = (evals[tag].get("provenance") or {}).get("last_pt_sha256")
        if not posthoc_prov.get(key) or posthoc_prov[key] != eval_sha:
            raise RuntimeError(
                f"F1_POSTHOC_EVAL_CHECKPOINT_MISMATCH {tag}: "
                f"posthoc={posthoc_prov.get(key)} eval={eval_sha}"
            )

    def score(tag, variant):
        return float(evals[tag]["last.pt"][variant]["val"]["map50_95"])

    c0 = score("C0", "NORMAL")
    fixed = score("FIXED", "NORMAL")
    normal = score("SOFT", "NORMAL")
    zero = score("SOFT", "ZERO-AUX")
    shuffle = score("SOFT", "SHUFFLE")
    soft_loo = loo["deltas"]["SOFT_minus_C0"]
    gate_loo = loo["deltas"]["SOFT_minus_FIXED"]
    q_rows = evals["SOFT"]["last.pt"]["gate_values"]["NORMAL"]["val"]
    q_values = sorted(float(row["raw_q"]) for row in q_rows.values())
    n_q = len(q_values)

    def qnt(p: float) -> float:
        return q_values[min(n_q - 1, max(0, round(p * (n_q - 1))))]

    q_summary = {
        "n": n_q,
        "mean": statistics.mean(q_values),
        "std": statistics.stdev(q_values) if n_q > 1 else 0.0,
        "min": q_values[0],
        "max": q_values[-1],
        "p10": qnt(0.10),
        "p50": qnt(0.50),
        "p90": qnt(0.90),
        "range": q_values[-1] - q_values[0],
    }

    # Threshold context the reviewer asked to be stored explicitly: formula,
    # epoch budget, measured per-group changes and the control noise floor.
    g6_context = {
        "decay_threshold_formula": "1e-3 * epochs / 80",
        "epochs": a.expected_epochs,
        "scaled_threshold_used_by_runner": 1e-3 * (a.expected_epochs / 80.0),
        "formal_uses_original_1e_3": a.expected_epochs == 80,
        "control_noise_floor": g6["C0"]["aux_encoder_global_rel_l2"],
        "measured_changes": {
            tag: {
                "aux_encoder_global_rel_l2": row["aux_encoder_global_rel_l2"],
                "gate_max_abs_change": row["gate_max_abs_change"],
                "proj_weight_norms": row["proj_weight_norms"],
                "proj_bias_norms": row["proj_bias_norms"],
            }
            for tag, row in g6.items()
        },
        "note": ("linear epoch scaling exists ONLY for smoke chain-aliveness; "
                 "formal 80ep runs use the original 1e-3 threshold "
                 "(scale factor 1.0) and quality is judged by the causal and "
                 "LOO protocol, never by these thresholds"),
    }

    conditions = quality.get("conditions") or {}
    identity = conditions.get("identity:0.00")
    if not identity:
        raise RuntimeError("F1_QUALITY_IDENTITY_CONDITION_MISSING")
    degraded = {key: row for key, row in conditions.items() if key != "identity:0.00"}
    if len(degraded) != 17:
        raise RuntimeError(
            f"F1_QUALITY_CONDITION_COUNT_MISMATCH: expected=17 got={len(degraded)}"
        )
    identity_q = float(identity["raw_q"]["mean"])
    lower_q = [
        key for key, row in degraded.items()
        if float(row["raw_q"]["mean"]) < identity_q - 1e-4
    ]
    beats_q1 = [
        key for key, row in degraded.items()
        if float(row["learned_minus_force_q1_map50_95"]) > 0.0
    ]
    reliability_supported = len(lower_q) >= 9 and len(beats_q1) >= 1
    reliability = {
        "identity_q_mean": identity_q,
        "degraded_conditions": len(degraded),
        "lower_q_than_identity_by_1e-4": lower_q,
        "lower_q_count": len(lower_q),
        "gate_beats_force_q1_conditions": beats_q1,
        "gate_beats_force_q1_count": len(beats_q1),
        "supported": reliability_supported,
        "criterion": (
            "at least 9/17 degraded conditions lower mean q by >1e-4 and at "
            "least one degraded condition has learned AP > FORCE-Q1"
        ),
    }

    # ---- A0 (reviewer): constant-attenuation vs adaptive contribution ----
    a0 = quality.get("interpretation_inputs") or {}
    adaptive_gain_identity = a0.get("identity_learned_minus_qclean")
    adaptive_near_zero = (
        adaptive_gain_identity is not None
        and abs(adaptive_gain_identity) < 0.01
    )
    a0_block = {
        "force_qclean_value": a0.get("force_qclean_value"),
        "identity_learned_minus_qclean": adaptive_gain_identity,
        "adaptive_contribution_near_zero": adaptive_near_zero,
        "macro_mean_learned_ap": a0.get("macro_mean_learned_ap"),
        "worst_condition": a0.get("worst_condition"),
        "worst_condition_learned_ap": a0.get("worst_condition_learned_ap"),
        "corruptions_where_gate_beats_force_qclean":
            a0.get("corruptions_where_gate_beats_force_qclean"),
        "family_q_severity_monotonicity":
            a0.get("family_q_severity_monotonicity"),
        "conclusion": ("CONSTANT ATTENUATION HELPS, ADAPTIVITY CONTRIBUTES "
                       "NOTHING" if adaptive_near_zero else
                       "adaptive contribution not yet ruled out; F1-B test "
                       "required"),
    }

    # ---- best/late10 stability block (reviewer: IR gain is checkpoint-sensitive)
    import csv
    stability = {}
    for tag in ("C0", "FIXED", "SOFT"):
        stability[tag] = {
            "last_val": float(evals[tag]["last.pt"]["NORMAL"]["val"]["map50_95"]),
            "best_val": float(evals[tag]["best.pt"]["NORMAL"]["val"]["map50_95"]),
            "late10": evals[tag].get("late10", {}),
        }
    with open(runs["C0"] / "results.csv", encoding="utf-8", newline="") as f:
        c0_tail = [float(r["metrics/mAP50-95(B)"])
                   for r in csv.DictReader(f) if r.get("metrics/mAP50-95(B)")][-10:]
    f0_summary = _read(ROOT / "runs" / "step4_f0" / "_summary_step4.json")
    f0_c0_last = f0_summary["groups"]["F0-C0"]["last_normal_val"]
    stability_block = {
        "per_group": stability,
        # results.csv stores ~3-decimal values; compare both sides at that
        # precision.
        "f1_c0_last_is_late10_minimum": bool(
            round(stability["C0"]["last_val"], 3) == round(min(c0_tail), 3)),
        "f1_c0_late10_tail": c0_tail,
        "frozen_f0_c0_r1_last_val": f0_c0_last,
        "f0_c0_r1_minus_f1_fixed": f0_c0_last - fixed,
        "interpretation": (
            "IR complementary is a candidate claim valid only under THIS F1 "
            "single seed, the preregistered last.pt and the matched "
            "learned-gate C0; best/late10 do not form consistent support and "
            "the frozen F0 control still exceeds F1-I-fixed — the fixed "
            "residual must NOT be frozen as the final competition structure"),
    }

    causal_pass = normal > c0 and normal > zero and normal > shuffle
    loo_pass = soft_loo["median"] > 0 and soft_loo["positive_folds"] >= 4
    gate_beats_fixed = normal > fixed and gate_loo["median"] > 0
    if causal_pass and loo_pass and gate_beats_fixed and reliability_supported:
        decision = "PROMOTE_F1_GATE_CONFIRM_ONE_SEED"
        next_step = "run one confirmation seed; keep Depth out"
    elif causal_pass and loo_pass and gate_beats_fixed:
        decision = "SOFT_SCALAR_HELPED_BUT_RELIABILITY_NOT_PROVEN"
        next_step = "F1-B deterministic IR corruption/dropout training; keep architecture fixed"
    elif causal_pass and loo_pass:
        decision = "IR_COMPLEMENTARY_BUT_GATE_NOT_PROVEN_BETTER_THAN_Q1"
        next_step = "keep the simpler fixed residual unless quality probe shows robustness gain"
    elif normal > zero and normal > shuffle:
        decision = "MODEL_USES_IR_BUT_NO_NET_BENEFIT"
        next_step = "F1-B deterministic IR corruption training; no new fusion block"
    elif q_summary["range"] < 0.02:
        decision = "GATE_NEAR_CONSTANT"
        next_step = "F1-B quality/reliability learning; consider RGB-IR agreement input"
    else:
        decision = "F1_GATE_FAILED_CAUSAL_PROTOCOL"
        next_step = "stop before spatial gate/QAF and inspect intervention signs"

    summary = {
        "schema": "step4-f1-summary-v3",
        "loo_file_sha256": _sha(loo_path),
        "quality_file_sha256": _sha(quality_path),
        "posthoc_file_sha256": _sha(posthoc_path),
        "summarize_source_sha256": _sha(Path(__file__)),
        "f1_closeout_source_sha256": _sha(
            ROOT / "src" / "multimodal" / "step4_f1_closeout.py"
        ),
        "shared_closeout_source_sha256": _sha(
            ROOT / "src" / "multimodal" / "step4_closeout.py"
        ),
        "integrity": integrity,
        "provenance": provenance,
        "manifest_provenance": manifest_provenance,
        "matched_initial_state": matched_initial_state,
        "g6": g6,
        "g6_threshold_context": g6_context,
        "g8": g8,
        "loo_payload_validation": loo_payload_validation,
        "loo_provenance": loo_provenance,
        "quality_provenance": quality_provenance,
        "posthoc": {"checks": posthoc["checks"],
                    "provenance_checks": posthoc_checks,
                    "passed": posthoc["passed"]},
        "a0_constant_vs_adaptive": a0_block,
        "stability_block": stability_block,
        "primary_last_val6": {
            "C0": c0,
            "FIXED_NORMAL": fixed,
            "SOFT_NORMAL": normal,
            "SOFT_ZERO": zero,
            "SOFT_SHUFFLE": shuffle,
            "SOFT_minus_C0": normal - c0,
            "SOFT_minus_FIXED": normal - fixed,
            "SOFT_N_minus_Z": normal - zero,
            "SOFT_N_minus_S": normal - shuffle,
        },
        "soft_loo": soft_loo,
        "soft_minus_fixed_loo": gate_loo,
        "val6_raw_q": q_summary,
        "quality_reliability_evidence": reliability,
        "decision": decision,
        "next_step": next_step,
        "verdict_frozen": True,
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
