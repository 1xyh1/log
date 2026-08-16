#!/usr/bin/env python3
"""F1-B closeout summarizer: G9 per-sample re-judgement + B1 promotion rules.

Unlike the F1 summarizer this one does NOT trust the G9 trace booleans: it
re-reads step4_b1_g9_records.jsonl, recomputes the canonical records SHA,
re-derives every per-sample assertion (IR before/after semantics, RGB/Depth/
label/bbox untouched, kind/severity validity), recomputes kind counts and the
expected schedule, and only then applies the frozen B1 promotion rules
(clean SOFT > C0 and > separately-trained FIXED with LOO, macro and worst-4
degraded AP above BOTH fixed and the B1-soft's own FORCE-QCLEAN, learned-
QCLEAN positive on >= 9/17 degraded conditions).
"""
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
from multimodal.step4_f1_b_corruption import (  # noqa: E402
    SEVERITIES, TRAIN_KINDS, sample_schedule, schedule_sha256)
from multimodal.step4_f1_closeout import (  # noqa: E402
    LOO_SCHEMA, validate_f1_loo_payload)

GROUP_SPECS = {
    "C0": {"group": "B1-C0", "aux_mode": "zero", "gate_mode": "learned"},
    "FIXED": {"group": "B1-I-fixed", "aux_mode": "ir", "gate_mode": "fixed_one"},
    "SOFT": {"group": "B1-I-soft", "aux_mode": "ir", "gate_mode": "learned"},
}


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


def rejudge_g9(run_dirs: dict[str, Path], expected_epochs: int,
               seed: int, contract: dict) -> dict:
    """Per-sample G9 re-judgement: trust nothing, recompute everything."""
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
        # group records by epoch
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
                sid = str(r["sample_id"])
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
        per_group[tag] = {
            "trace_rows": len(trace),
            "record_rows": len(records),
        }

    # cross-group expected schedule identical per epoch
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
    contract = _read(contract_path)
    runs = {
        "C0": project / a.c0_run,
        "FIXED": project / a.fixed_run,
        "SOFT": project / a.soft_run,
    }
    out = project / "_summary_step4_f1_b.json"
    if out.exists() and not a.overwrite:
        raise RuntimeError(f"REFUSE_OVERWRITE_B1_SUMMARY: {out}")

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
        if gate.get("passed") is not True:
            raise RuntimeError(f"B1_G6_FAIL: {tag}")
        g6[tag] = gate
        integrity[tag] = rep

    evals = {
        tag: _read(run_dir / "eval_step4_f1_b_causality.json")
        for tag, run_dir in runs.items()
    }
    for tag, ev in evals.items():
        if ev.get("schema") != "step4-f1-b-stock-validator-semantics-v1":
            raise RuntimeError(f"bad B1 eval schema in {tag}")
        expected = GROUP_SPECS[tag]
        for key in ("group", "aux_mode", "gate_mode"):
            if ev.get(key) != expected[key]:
                raise RuntimeError(
                    f"B1_EVAL_IDENTITY_MISMATCH {tag}:{key} "
                    f"recorded={ev.get(key)} expected={expected[key]}"
                )

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

    quality_path = runs["SOFT"] / "eval_step4_f1_b_quality_last.json"
    if not quality_path.exists():
        raise RuntimeError("B1_LAST_PT_QUALITY_EVIDENCE_MISSING")
    quality = _read(quality_path)
    if quality.get("schema") != "step4-f1-b-ir-quality-probe-v1":
        raise RuntimeError("bad B1 quality schema")

    def score(tag, variant):
        return float(evals[tag]["last.pt"][variant]["val"]["map50_95"])

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
        "schema": "step4-f1-b-summary-v1",
        "loo_file_sha256": _sha(loo_path),
        "quality_file_sha256": _sha(quality_path),
        "summarize_source_sha256": _sha(Path(__file__)),
        "integrity": integrity,
        "g6": g6,
        "g8": g8,
        "g9_rejudged": g9,
        "loo_payload_validation": loo_validation,
        "primary_last_val6": {
            "C0": c0,
            "FIXED_NORMAL": fixed,
            "SOFT_NORMAL": normal,
            "SOFT_ZERO": zero,
            "SOFT_SHUFFLE": shuffle,
            "SOFT_minus_C0": normal - c0,
            "SOFT_minus_FIXED": normal - fixed,
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
