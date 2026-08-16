#!/usr/bin/env python3
"""F1-B closeout summarizer v2: full provenance re-verification + G9 per-sample
re-judgement + B1 promotion rules.

v2 closeout (reviewer 2026-08-17, no retraining): the summarizer now executes
its own verification instead of trusting recorded booleans —
  * eval/manifest/LOO/quality/posthoc provenance blocks are re-hashed;
  * G6 is re-judged from the recorded measurements (passed=true is NOT trusted);
  * G9 records are re-judged per sample (schedule, SHAs, semantics, IDs);
  * last/best/late10 stability block records the mid-training paired-signal
    peak and its decay to last.pt.
"""
from __future__ import annotations

import argparse
import csv
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
from multimodal.step4_f1_b_corruption import (  # noqa: E402
    SEVERITIES, TRAIN_KINDS, sample_schedule, schedule_sha256)
from multimodal.step4_f1_closeout import (  # noqa: E402
    LOO_SCHEMA, validate_f1_loo_payload)

GROUP_SPECS = {
    "C0": {"group": "B1-C0", "aux_mode": "zero", "gate_mode": "learned"},
    "FIXED": {"group": "B1-I-fixed", "aux_mode": "ir", "gate_mode": "fixed_one"},
    "SOFT": {"group": "B1-I-soft", "aux_mode": "ir", "gate_mode": "learned"},
}

UNIFIED_ACTIVE_THRESHOLD = 1e-3


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(obj) -> str:
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"),
                         sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_sha(recorded, path: Path, label: str) -> dict:
    current = _sha(path) if path.exists() else None
    match = bool(recorded and current and recorded == current)
    if not match:
        raise RuntimeError(
            f"STALE_PROVENANCE {label}: recorded={recorded} current={current}"
        )
    return {"recorded": recorded, "current": current, "match": True}


# --------------------------------------------------------------------------
# Provenance verification blocks (v2 closeout)
# --------------------------------------------------------------------------

def _verify_eval(eval_obj: dict, run_dir: Path, contract_path: Path,
                 expected: dict) -> dict:
    if eval_obj.get("schema") != "step4-f1-b-stock-validator-semantics-v1":
        raise RuntimeError(f"bad B1 eval schema in {run_dir}")
    for key in ("group", "aux_mode", "gate_mode"):
        if eval_obj.get(key) != expected[key]:
            raise RuntimeError(
                f"B1_EVAL_IDENTITY_MISMATCH {run_dir.name}:{key} "
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
        "evaluator_source_sha256": ROOT / "scripts" / "eval_step4_f1_b_causality.py",
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
    return checks


def _verify_manifest(manifest: dict, tag: str, run_dir: Path,
                     contract_path: Path, audit_path: Path,
                     expected_epochs: int) -> dict:
    expected = GROUP_SPECS[tag]
    identity = {
        "schema": "step4-f1-b-manifest-v1",
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
                f"B1_MANIFEST_IDENTITY_MISMATCH {tag}:{key} "
                f"recorded={manifest.get(key)} expected={value}"
            )
    targets = {
        "contract_sha256": contract_path,
        "pretrain_audit_sha256": audit_path,
        "runner_source_sha256": ROOT / "scripts" / "run_step4_f1_b.py",
        "corruption_source_sha256": ROOT / "src" / "multimodal" / "step4_f1_b_corruption.py",
        "f1_v4_summary_sha256": ROOT / "runs" / "step4_f1_ir_gate" / "_summary_step4_f1.json",
        "design_freeze_sha256": ROOT / "docs" / "step4_f1_b_corruption" / "DESIGN_FREEZE.md",
        "model_source_sha256": ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py",
        "gate_source_sha256": ROOT / "src" / "multimodal" / "reliability_gate.py",
        "f0_model_source_sha256": ROOT / "src" / "multimodal" / "step4_f0_model.py",
        "aux_encoder_source_sha256": ROOT / "src" / "multimodal" / "aux_encoder.py",
        "feature_fusion_source_sha256": ROOT / "src" / "multimodal" / "feature_fusion.py",
        "trainability_source_sha256": ROOT / "src" / "multimodal" / "trainability.py",
        "dataset_source_sha256": ROOT / "src" / "multimodal" / "trimodal_dataset.py",
        "preprocess_source_sha256": ROOT / "src" / "multimodal" / "modality_preprocess.py",
        "quality_mask_source_sha256": ROOT / "src" / "multimodal" / "modality_quality.py",
    }
    return {
        key: _require_sha(manifest.get(key), path, f"MANIFEST:{tag}:{key}")
        for key, path in targets.items()
    }


def _verify_g6(tag: str, gate: dict) -> dict:
    """Re-judge the recorded update evidence; passed=true is NOT trusted."""
    rgb_ok = gate.get("rgb_backbone_unchanged") is True
    q_ok = gate.get("q_finite_and_bounded") is True
    aux_delta = float(gate.get("aux_encoder_global_rel_l2", float("nan")))
    gate_delta = float(gate.get("gate_max_abs_change", float("nan")))
    proj = [float(v) for v in gate.get("proj_weight_norms", [])]
    proj_bias = [float(v) for v in gate.get("proj_bias_norms", [])]
    if (len(proj) != 3 or len(proj_bias) != 3
            or not all(math.isfinite(v) for v in proj + proj_bias)):
        raise RuntimeError(f"B1_G6_REJUDGE_FAIL {tag}: {gate}")
    if tag == "C0":
        passed = rgb_ok and q_ok and aux_delta < UNIFIED_ACTIVE_THRESHOLD \
            and max(proj) == 0.0
    elif tag == "FIXED":
        q = gate.get("last_epoch_effective_q") or {}
        passed = (rgb_ok and q_ok and aux_delta > UNIFIED_ACTIVE_THRESHOLD
                  and min(proj) > 0.0
                  and q.get("min") == 1.0 and q.get("max") == 1.0)
    else:
        passed = (rgb_ok and q_ok and aux_delta > UNIFIED_ACTIVE_THRESHOLD
                  and min(proj) > 0.0 and gate_delta > 0.0)
    if not passed:
        raise RuntimeError(f"B1_G6_REJUDGE_FAIL {tag}: {gate}")
    return {
        "rgb_backbone_unchanged": rgb_ok,
        "aux_encoder_global_rel_l2": aux_delta,
        "gate_max_abs_change": gate_delta,
        "proj_weight_norms": proj,
        "proj_bias_norms": proj_bias,
        "q_finite_and_bounded": q_ok,
        "passed": True,
    }


def _verify_quality(report: dict, run_dir: Path, fixed_dir: Path,
                    contract_path: Path) -> dict:
    if report.get("schema") != "step4-f1-b-ir-quality-probe-v1":
        raise RuntimeError("bad B1 quality schema")
    if report.get("checkpoint") != "last.pt":
        raise RuntimeError("B1 closeout requires last.pt quality evidence")
    p = report.get("provenance") or {}
    targets = {
        "checkpoint_sha256": run_dir / "weights" / "last.pt",
        "fixed_checkpoint_sha256": fixed_dir / "weights" / "last.pt",
        "contract_sha256": contract_path,
        "script_sha256": ROOT / "scripts" / "eval_step4_f1_b_quality.py",
        "interventions_source_sha256": ROOT / "src" / "multimodal" / "step4_f1_interventions.py",
        "evaluator_core_sha256": ROOT / "src" / "multimodal" / "step3_eval_utils.py",
        "model_source_sha256": ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py",
        "gate_source_sha256": ROOT / "src" / "multimodal" / "reliability_gate.py",
        "dataset_source_sha256": ROOT / "src" / "multimodal" / "trimodal_dataset.py",
    }
    return {
        key: _require_sha(p.get(key), path, f"QUALITY:{key}")
        for key, path in targets.items()
    }


def _verify_posthoc(posthoc: dict, runs: dict[str, Path],
                    contract_path: Path) -> dict:
    if posthoc.get("schema") != "step4-f1-b-posthoc-gradient-audit-v1" \
            or posthoc.get("passed") is not True:
        raise RuntimeError("B1_POSTHOC_AUDIT_NOT_PASSED")
    p = posthoc.get("provenance") or {}
    targets = {
        "soft_last_pt_sha256": runs["SOFT"] / "weights" / "last.pt",
        "c0_last_pt_sha256": runs["C0"] / "weights" / "last.pt",
        "contract_sha256": contract_path,
        "script_sha256": ROOT / "scripts" / "audit_step4_f1_b_posthoc.py",
        "model_source_sha256": ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py",
        "gate_source_sha256": ROOT / "src" / "multimodal" / "reliability_gate.py",
        "dataset_source_sha256": ROOT / "src" / "multimodal" / "trimodal_dataset.py",
        "eval_core_sha256": ROOT / "src" / "multimodal" / "step3_eval_utils.py",
    }
    checks = {
        key: _require_sha(p.get(key), path, f"POSTHOC:{key}")
        for key, path in targets.items()
    }
    return checks


def _verify_loo(loo: dict, runs: dict[str, Path], contract_path: Path) -> dict:
    if loo.get("schema") != LOO_SCHEMA or loo.get("checkpoint") != "last.pt":
        raise RuntimeError("bad B1 LOO schema/checkpoint")
    p = loo.get("provenance") or {}
    targets = {
        **{f"{tag}_last_pt_sha256": path / "weights" / "last.pt"
           for tag, path in runs.items()},
        "contract_sha256": contract_path,
        "loo_source_sha256": ROOT / "scripts" / "step4_f1_b_loo.py",
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
    }
    return {
        key: _require_sha(p.get(key), path, f"LOO:{key}")
        for key, path in targets.items()
    }


# --------------------------------------------------------------------------
# G9 per-sample re-judgement
# --------------------------------------------------------------------------

def rejudge_g9(run_dirs: dict[str, Path], expected_epochs: int,
               seed: int, contract: dict) -> dict:
    train_ids = list(contract["train_ids"])
    errors: list[str] = []
    per_group: dict[str, dict] = {}
    cross_epoch_expected: dict[int, set] = {}

    for tag, run_dir in run_dirs.items():
        trace = _read_jsonl(run_dir / "step4_b1_g9_trace.jsonl")
        records = _read_jsonl(run_dir / "step4_b1_g9_records.jsonl")
        aux_mode = GROUP_SPECS[tag]["aux_mode"]
        if len(trace) != expected_epochs:
            errors.append(f"G9_TRACE_ROW_COUNT:{tag}:{len(trace)}")
        by_epoch: dict[int, list] = {}
        for r in records:
            by_epoch.setdefault(int(r["epoch"]), []).append(r)
        if set(by_epoch.keys()) != set(range(expected_epochs)):
            errors.append(f"G9_RECORDS_EPOCH_SET:{tag}")

        for epoch in range(min(expected_epochs, len(trace))):
            row = trace[epoch]
            recs = by_epoch.get(epoch, [])
            if len(recs) != len(train_ids):
                errors.append(f"G9_RECORDS_COUNT:{tag}:e{epoch}")
            # ID set must be complete and duplicate-free (checked even when the
            # count is off, so a duplicated row is named explicitly)
            rec_ids = [str(r["sample_id"]) for r in recs]
            if sorted(rec_ids) != sorted(train_ids):
                errors.append(f"G9_ID_SET_INCOMPLETE:{tag}:e{epoch}")
            if len(set(rec_ids)) != len(rec_ids):
                errors.append(f"G9_ID_DUPLICATES:{tag}:e{epoch}")
            # canonical records SHA
            canonical_rows = [
                {k: r[k] for k in (
                    "epoch", "sample_id", "kind", "severity", "ir_sha_before",
                    "ir_sha_after", "rgb_unchanged", "depth_unchanged",
                    "labels_bboxes_same_object")}
                for r in recs
            ]
            canonical_sha = _sha_json(sorted(canonical_rows,
                                             key=lambda r: r["sample_id"]))
            if row.get("records_sha256") != canonical_sha:
                errors.append(f"G9_RECORDS_SHA_MISMATCH:{tag}:e{epoch}")
            # actual schedule SHA recomputed from the records themselves
            actual_rows = [{"sample_id": str(r["sample_id"]), "kind": r["kind"],
                            "severity": r["severity"]} for r in recs]
            recomputed_actual_sha = _sha_json(
                sorted(actual_rows, key=lambda r: r["sample_id"]))
            if row.get("actual_schedule_sha256") != recomputed_actual_sha:
                errors.append(f"G9_ACTUAL_SCHEDULE_SHA:{tag}:e{epoch}")
            # per-sample re-judgement
            for r in recs:
                sid = str(r["sample_id"])
                expected_sched = sample_schedule(seed, epoch, sid)
                if (r["kind"] != expected_sched["kind"]
                        or r["severity"] != expected_sched["severity"]):
                    errors.append(f"G9_SCHEDULE_MISMATCH:{tag}:e{epoch}:{sid}")
                    continue
                if r["kind"] not in TRAIN_KINDS:
                    errors.append(f"G9_KIND_INVALID:{tag}:e{epoch}:{sid}")
                    continue
                if r["kind"] in ("noise", "blur", "contrast"):
                    if r["severity"] not in SEVERITIES:
                        errors.append(f"G9_SEVERITY_INVALID:{tag}:e{epoch}:{sid}")
                elif r["kind"] == "zero" and r["severity"] != 1.0:
                    errors.append(f"G9_ZERO_SEVERITY:{tag}:e{epoch}:{sid}")
                elif r["kind"] == "clean" and r["severity"] != 0.0:
                    errors.append(f"G9_CLEAN_SEVERITY:{tag}:e{epoch}:{sid}")
                if not (r["rgb_unchanged"] and r["depth_unchanged"]
                        and r["labels_bboxes_same_object"]):
                    errors.append(f"G9_CHANNELS_CHANGED:{tag}:e{epoch}:{sid}")
                ir_same = r["ir_sha_before"] == r["ir_sha_after"]
                if aux_mode == "zero":
                    if not ir_same:
                        errors.append(f"G9_C0_IR_CHANGED:{tag}:e{epoch}:{sid}")
                else:
                    if ir_same != (r["kind"] == "clean"):
                        errors.append(f"G9_IR_SEMANTICS:{tag}:e{epoch}:{sid}")
            # schedule anchor
            expected = schedule_sha256(seed, epoch, train_ids)
            if row.get("expected_schedule_sha256") != expected:
                errors.append(f"G9_EXPECTED_SCHEDULE:{tag}:e{epoch}")
            if row.get("expected_matches_actual") is not True:
                errors.append(f"G9_EXPECTED_ACTUAL_FLAG:{tag}:e{epoch}")
            # kind counts
            counts = {}
            for r in recs:
                counts[r["kind"]] = counts.get(r["kind"], 0) + 1
            if row.get("kind_counts") != counts:
                errors.append(f"G9_KIND_COUNTS:{tag}:e{epoch}")
            cross_epoch_expected.setdefault(epoch, set()).add(expected)
        per_group[tag] = {"trace_rows": len(trace), "record_rows": len(records)}

    for epoch, shas in cross_epoch_expected.items():
        if len(shas) != 1:
            errors.append(f"G9_CROSS_GROUP_SCHEDULE:e{epoch}")

    return {"errors": errors, "passed": not errors, "per_group": per_group}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_f1_b_corruption")
    p.add_argument("--c0-run", default="B1-C0")
    p.add_argument("--fixed-run", default="B1-I-fixed")
    p.add_argument("--soft-run", default="B1-I-soft")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--expected-epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    project = Path(a.project)
    contract_path = Path(a.contract)
    audit_path = ROOT / "reports" / "step4_f1_b_corruption" / "pretrain_audit.json"
    contract = _read(contract_path)
    runs = {
        "C0": project / a.c0_run,
        "FIXED": project / a.fixed_run,
        "SOFT": project / a.soft_run,
    }
    out = project / "_summary_step4_f1_b.json"
    if out.exists() and not a.overwrite:
        raise RuntimeError(f"REFUSE_OVERWRITE_B1_SUMMARY: {out}")

    # ---- integrity + G6 re-judgement ----
    integrity = {}
    g6 = {}
    for tag, run_dir in runs.items():
        rep = inspect_step3_run(
            run_dir, a.expected_epochs, require_weights=True,
            trace_name="step4_g8_trace.jsonl", growth_name="step4_growth.jsonl",
            eval_name="eval_step4_f1_b_causality.json",
        ).to_dict()
        if not rep["passed"]:
            raise RuntimeError(f"B1_INTEGRITY_FAIL: {tag}")
        gate = _read(run_dir / "step4_update_gate.json")
        g6[tag] = _verify_g6(tag, gate)
        integrity[tag] = rep

    # ---- provenance blocks (v2 closeout) ----
    evals = {
        tag: _read(run_dir / "eval_step4_f1_b_causality.json")
        for tag, run_dir in runs.items()
    }
    provenance = {
        tag: _verify_eval(evals[tag], runs[tag], contract_path, GROUP_SPECS[tag])
        for tag in runs
    }
    manifests = {tag: _read(run_dir / "manifest.json") for tag, run_dir in runs.items()}
    manifest_provenance = {
        tag: _verify_manifest(manifests[tag], tag, runs[tag], contract_path,
                              audit_path, a.expected_epochs)
        for tag in runs
    }
    matched_initial_state = {}
    for key in ("initial_model_state_sha256", "initial_rgb_backbone_sha256",
                "initial_aux_encoder_sha256", "initial_fusion_sha256",
                "initial_gate_sha256"):
        values = {tag: manifests[tag].get(key) for tag in runs}
        if not (all(values.values()) and len(set(values.values())) == 1):
            raise RuntimeError(f"B1_INITIAL_STATE_MISMATCH {key}: {values}")
        matched_initial_state[key] = {"values": values, "passed": True}

    g8 = g8_check(runs, a.expected_epochs)
    if not g8["passed"]:
        raise RuntimeError(f"B1_G8_CLOSEOUT_FAIL: {g8}")

    g9 = rejudge_g9(runs, a.expected_epochs, a.seed, contract)
    if not g9["passed"]:
        raise RuntimeError(f"B1_G9_REJUDGE_FAIL: {g9['errors'][:20]}")

    loo_path = project / "step4_f1_b_loo.json"
    if not loo_path.exists():
        raise RuntimeError("B1_LOO_MISSING")
    loo = _read(loo_path)
    loo_validation = validate_f1_loo_payload(loo)
    if not loo_validation["passed"]:
        raise RuntimeError(f"B1_LOO_PAYLOAD_INVALID: {loo_validation['errors']}")
    loo_provenance = _verify_loo(loo, runs, contract_path)

    quality_path = runs["SOFT"] / "eval_step4_f1_b_quality_last.json"
    if not quality_path.exists():
        raise RuntimeError("B1_LAST_PT_QUALITY_EVIDENCE_MISSING")
    quality = _read(quality_path)
    quality_provenance = _verify_quality(quality, runs["SOFT"], runs["FIXED"],
                                         contract_path)

    posthoc_path = ROOT / "reports" / "step4_f1_ir_gate" / "posthoc_gradient_audit_b.json"
    if not posthoc_path.exists():
        raise RuntimeError("B1_POSTHOC_AUDIT_MISSING")
    posthoc = _read(posthoc_path)
    posthoc_checks = _verify_posthoc(posthoc, runs, contract_path)

    def score(tag, variant):
        return float(evals[tag]["last.pt"][variant]["val"]["map50_95"])

    def best_score(tag, variant):
        return float(evals[tag]["best.pt"][variant]["val"]["map50_95"])

    c0 = score("C0", "NORMAL")
    fixed = score("FIXED", "NORMAL")
    normal = score("SOFT", "NORMAL")
    zero = score("SOFT", "ZERO-AUX")
    shuffle = score("SOFT", "SHUFFLE")
    soft_loo = loo["deltas"]["SOFT_minus_C0"]
    gate_loo = loo["deltas"]["SOFT_minus_FIXED"]

    causal_pass = normal > c0 and normal > zero and normal > shuffle
    loo_pass = soft_loo["median"] > 0 and soft_loo["positive_folds"] >= 4
    beats_fixed = normal > fixed and gate_loo["median"] > 0

    # ---- stability block (reviewer: mid-training paired signal decays) ----
    stability = {}
    for tag in ("C0", "FIXED", "SOFT"):
        stability[tag] = {
            "last_val": score(tag, "NORMAL"),
            "best_val": best_score(tag, "NORMAL"),
            "best_N_minus_Z": best_score(tag, "NORMAL") - best_score(tag, "ZERO-AUX"),
            "best_N_minus_S": best_score(tag, "NORMAL") - best_score(tag, "SHUFFLE"),
            "late10": evals[tag].get("late10", {}),
        }
    with open(runs["SOFT"] / "results.csv", encoding="utf-8", newline="") as f:
        soft_rows = [float(r["metrics/mAP50-95(B)"])
                     for r in csv.DictReader(f)
                     if r.get("metrics/mAP50-95(B)")]
    best_epoch = soft_rows.index(max(soft_rows)) + 1
    stability_block = {
        "per_group": stability,
        "soft_best_epoch_1based": best_epoch,
        "soft_best_epoch_val": max(soft_rows),
        "soft_last_val": soft_rows[-1],
        "note": ("paired IR signal appears at mid-training checkpoints but does "
                 "not persist to the preregistered last.pt; the adaptive gate "
                 "was never proven better than constant QCLEAN — best.pt is "
                 "auxiliary evidence and does not overturn the last.pt verdict"),
    }

    # ---- B1 promotion rules (frozen DESIGN_FREEZE section 6) ----
    conditions = quality.get("conditions") or {}
    identity = conditions.get("identity:0.00")
    if not identity:
        raise RuntimeError("B1_QUALITY_IDENTITY_CONDITION_MISSING")
    own_qclean = float(identity["raw_q"]["mean"])
    degraded = {k: v for k, v in conditions.items() if k != "identity:0.00"}
    if len(degraded) != 17:
        raise RuntimeError(f"B1_QUALITY_CONDITION_COUNT: {len(degraded)}")

    macro_soft = statistics.mean(
        row["learned_gate"]["map50_95"] for row in degraded.values())
    # "FIXED" = the separately-trained B1-I-fixed model evaluated on the same
    # corrupted datasets (frozen protocol); NOT force_q1 on the soft ckpt.
    macro_fixed = statistics.mean(
        row["separately_trained_fixed"]["map50_95"] for row in degraded.values())
    macro_qclean = statistics.mean(
        row["force_qclean"]["map50_95"] for row in degraded.values())
    worst4_keys = sorted(degraded, key=lambda k:
                         degraded[k]["learned_gate"]["map50_95"])[:4]
    worst4_soft = statistics.mean(
        degraded[k]["learned_gate"]["map50_95"] for k in worst4_keys)
    worst4_fixed = statistics.mean(
        degraded[k]["separately_trained_fixed"]["map50_95"] for k in worst4_keys)
    worst4_qclean = statistics.mean(
        degraded[k]["force_qclean"]["map50_95"] for k in worst4_keys)
    learned_minus_qclean_pos = sum(
        1 for row in degraded.values()
        if row["learned_minus_force_qclean_map50_95"] > 0)
    n_monotone_families = sum(
        1 for fam, m in (quality.get("interpretation_inputs") or {})
        .get("family_q_severity_monotonicity", {}).items()
        if m.get("monotone_direction") == "down")

    macro_pass = macro_soft > macro_fixed and macro_soft > macro_qclean
    worst4_pass = worst4_soft > worst4_fixed and worst4_soft > worst4_qclean
    adaptive_pass = learned_minus_qclean_pos >= 9
    reliability_pass = macro_pass and worst4_pass and adaptive_pass

    if causal_pass and loo_pass and beats_fixed and reliability_pass:
        decision = "PROMOTE_B1_ADAPTIVE_RELIABILITY_CONFIRM_ONE_SEED"
        next_step = "one confirmation seed; keep Depth out"
    elif causal_pass and loo_pass and beats_fixed:
        decision = "B1_CORRUPTION_HELPED_BUT_ADAPTIVITY_NOT_PROVEN"
        next_step = ("F1-B further iterations or RGB-IR agreement input; "
                     "keep architecture fixed")
    elif causal_pass and loo_pass:
        decision = "B1_IR_COMPLEMENTARY_GATE_NOT_BETTER_THAN_FIXED"
        next_step = "keep the fixed residual unless robustness gain appears"
    elif normal > zero and normal > shuffle:
        decision = "B1_MODEL_USES_IR_BUT_NO_NET_BENEFIT"
        next_step = "inspect intervention signs; no new fusion block"
    else:
        decision = "B1_GATE_FAILED_CAUSAL_PROTOCOL"
        next_step = "stop before spatial gate/QAF and inspect intervention signs"

    summary = {
        "schema": "step4-f1-b-summary-v2",
        "loo_file_sha256": _sha(loo_path),
        "quality_file_sha256": _sha(quality_path),
        "posthoc_file_sha256": _sha(posthoc_path),
        "summarize_source_sha256": _sha(Path(__file__)),
        "integrity": integrity,
        "provenance": provenance,
        "manifest_provenance": manifest_provenance,
        "matched_initial_state": matched_initial_state,
        "g6_rejudged": g6,
        "g8": g8,
        "g9_rejudged": g9,
        "loo_payload_validation": loo_validation,
        "loo_provenance": loo_provenance,
        "quality_provenance": quality_provenance,
        "posthoc": {"checks": posthoc["checks"],
                    "provenance_checks": posthoc_checks,
                    "passed": posthoc["passed"]},
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
        "b1_promotion_evidence": {
            "own_force_qclean": own_qclean,
            "macro_soft_fixed_qclean": [macro_soft, macro_fixed, macro_qclean],
            "worst4_keys": worst4_keys,
            "worst4_soft_fixed_qclean": [worst4_soft, worst4_fixed, worst4_qclean],
            "learned_minus_qclean_positive_count": learned_minus_qclean_pos,
            "monotone_families_down": n_monotone_families,
            "macro_pass": macro_pass,
            "worst4_pass": worst4_pass,
            "adaptive_pass": adaptive_pass,
            "reliability_pass": reliability_pass,
        },
        "decision": decision,
        "next_step": next_step,
        "verdict_frozen": True,
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
