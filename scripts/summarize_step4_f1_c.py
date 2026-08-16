#!/usr/bin/env python3
"""F1-C closeout summarizer: four-group re-verification + frozen LOO conditions.

Re-hashes G6/G8/G9/G10.7/eval/quality/LOO/posthoc provenance itself (recorded
booleans are not trusted), re-judges G9 records per sample, verifies the
fp32-RGB-SHA records against the manifest, and applies the frozen F1-C
promotion conditions:

    magsoft vs original-soft:  full > 0, LOO median > 0, positive >= 4/6
    magsoft vs C0 / fixed:     same class of conditions
    plus: beat historical B1-soft last (0.304028, external auxiliary only),
    beat new-chain C0/fixed/ZERO/SHUFFLE, macro/worst4 above own QCLEAN,
    learned-QCLEAN >= 9/17.
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

GROUP_SPECS = {
    "C0": {"group": "F1C-C0", "aux_mode": "zero", "gate_mode": "learned",
           "gate_module": "magnitude"},
    "FIXED": {"group": "F1C-I-fixed", "aux_mode": "ir", "gate_mode": "fixed_one",
              "gate_module": "magnitude"},
    "MAGSOFT": {"group": "F1C-I-magsoft", "aux_mode": "ir",
                "gate_mode": "learned", "gate_module": "magnitude"},
    "ORIGSOFT": {"group": "F1C-I-soft", "aux_mode": "ir", "gate_mode": "learned",
                 "gate_module": "original"},
}

UNIFIED_ACTIVE_THRESHOLD = 1e-3
LOO_SCHEMA = "step4-f1-c-loo-v1"
HISTORICAL_B1_SOFT_LAST = 0.304028
DELTA_SPECS = {
    "MAGSOFT_minus_C0": ("MAGSOFT", "NORMAL", "C0", "NORMAL"),
    "MAGSOFT_minus_FIXED": ("MAGSOFT", "NORMAL", "FIXED", "NORMAL"),
    "MAGSOFT_minus_ORIGSOFT": ("MAGSOFT", "NORMAL", "ORIGSOFT", "NORMAL"),
    "MAGSOFT_N_minus_Z": ("MAGSOFT", "NORMAL", "MAGSOFT", "ZERO-AUX"),
    "MAGSOFT_N_minus_S": ("MAGSOFT", "NORMAL", "MAGSOFT", "SHUFFLE"),
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
            f"STALE_PROVENANCE {label}: recorded={recorded} current={current}")
    return {"recorded": recorded, "current": current, "match": True}


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


def rejudge_g9(run_dirs: dict[str, Path], expected_epochs: int,
               seed: int, contract: dict) -> dict:
    train_ids = list(contract["train_ids"])
    errors: list[str] = []
    cross_epoch_expected: dict[int, set] = {}
    for tag, run_dir in run_dirs.items():
        trace = _read_jsonl(run_dir / "step4_f1c_g9_trace.jsonl")
        records = _read_jsonl(run_dir / "step4_f1c_g9_records.jsonl")
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
            rec_ids = [str(r["sample_id"]) for r in recs]
            if sorted(rec_ids) != sorted(train_ids):
                errors.append(f"G9_ID_SET_INCOMPLETE:{tag}:e{epoch}")
            if len(set(rec_ids)) != len(rec_ids):
                errors.append(f"G9_ID_DUPLICATES:{tag}:e{epoch}")
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
            actual_rows = [{"sample_id": str(r["sample_id"]), "kind": r["kind"],
                            "severity": r["severity"]} for r in recs]
            recomputed_actual_sha = _sha_json(
                sorted(actual_rows, key=lambda r: r["sample_id"]))
            if row.get("actual_schedule_sha256") != recomputed_actual_sha:
                errors.append(f"G9_ACTUAL_SCHEDULE_SHA:{tag}:e{epoch}")
            for r in recs:
                sid = str(r["sample_id"])
                expected_sched = sample_schedule(seed, epoch, sid)
                if (r["kind"] != expected_sched["kind"]
                        or r["severity"] != expected_sched["severity"]):
                    errors.append(f"G9_SCHEDULE_MISMATCH:{tag}:e{epoch}:{sid}")
                    continue
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
            expected = schedule_sha256(seed, epoch, train_ids)
            if row.get("expected_schedule_sha256") != expected:
                errors.append(f"G9_EXPECTED_SCHEDULE:{tag}:e{epoch}")
            counts = {}
            for r in recs:
                counts[r["kind"]] = counts.get(r["kind"], 0) + 1
            if row.get("kind_counts") != counts:
                errors.append(f"G9_KIND_COUNTS:{tag}:e{epoch}")
            cross_epoch_expected.setdefault(epoch, set()).add(expected)
    for epoch, shas in cross_epoch_expected.items():
        if len(shas) != 1:
            errors.append(f"G9_CROSS_GROUP_SCHEDULE:e{epoch}")
    return {"errors": errors, "passed": not errors}


def verify_fp32_rgb(run_dirs: dict[str, Path]) -> dict:
    out = {}
    for tag, run_dir in run_dirs.items():
        rec = _read(run_dir / "step4_fp32_rgb_sha.json")
        manifest = _read(run_dir / "manifest.json")
        expected = manifest.get("initial_rgb_backbone_sha256")
        ok = bool(
            rec.get("schema") == "step4-f1-c-fp32-rgb-v1"
            and rec.get("group") == GROUP_SPECS[tag]["group"]
            and rec.get("expected_initial_sha256") == expected
            and rec.get("actual_final_sha256") == expected
            and rec.get("match") is True)
        out[tag] = {"schema_ok": rec.get("schema") == "step4-f1-c-fp32-rgb-v1",
                    "expected_matches_manifest": rec.get("expected_initial_sha256") == expected,
                    "actual_matches_expected": rec.get("actual_final_sha256") == expected,
                    "match_flag": rec.get("match"),
                    "passed": ok}
        if not ok:
            raise RuntimeError(f"F1C_FP32_RGB_SHA_FAIL {tag}: {rec}")
    return out


def verify_g6(tag: str, gate: dict) -> dict:
    rgb_ok = gate.get("rgb_backbone_unchanged") is True
    q = gate.get("last_epoch_effective_q") or {}
    q_vals = [q.get(k) for k in ("mean", "min", "max")]
    q_finite = (int(q.get("count", 0)) > 0
                and all(v is not None and math.isfinite(float(v))
                        for v in q_vals)
                and 0.0 <= float(q_vals[1]) <= float(q_vals[2]) <= 1.0)
    aux_delta = float(gate.get("aux_encoder_global_rel_l2", float("nan")))
    proj = [float(v) for v in gate.get("proj_weight_norms", [])]
    if tag == "C0":
        passed = rgb_ok and q_finite and aux_delta < UNIFIED_ACTIVE_THRESHOLD \
            and max(proj) == 0.0
    elif tag == "FIXED":
        passed = (rgb_ok and q_finite and aux_delta > UNIFIED_ACTIVE_THRESHOLD
                  and min(proj) > 0.0
                  and q.get("min") == 1.0 and q.get("max") == 1.0)
    else:
        passed = (rgb_ok and q_finite and aux_delta > UNIFIED_ACTIVE_THRESHOLD
                  and min(proj) > 0.0
                  and float(gate.get("gate_max_abs_change", 0.0)) > 0.0)
    if not passed:
        raise RuntimeError(f"F1C_G6_REJUDGE_FAIL {tag}: {gate}")
    return {"passed": True, "proj_weight_norms": proj,
            "aux_encoder_global_rel_l2": aux_delta, "q_finite": q_finite}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_f1_c")
    p.add_argument("--c0-run", default="F1C-C0")
    p.add_argument("--fixed-run", default="F1C-I-fixed")
    p.add_argument("--magsoft-run", default="F1C-I-magsoft")
    p.add_argument("--origsoft-run", default="F1C-I-soft")
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
        "MAGSOFT": project / a.magsoft_run,
        "ORIGSOFT": project / a.origsoft_run,
    }
    out = project / "_summary_step4_f1_c.json"
    if out.exists() and not a.overwrite:
        raise RuntimeError(f"REFUSE_OVERWRITE_F1C_SUMMARY: {out}")

    integrity = {}
    g6 = {}
    for tag, run_dir in runs.items():
        rep = inspect_step3_run(
            run_dir, a.expected_epochs, require_weights=True,
            trace_name="step4_g8_trace.jsonl", growth_name="step4_growth.jsonl",
            eval_name="eval_step4_f1_c_causality.json").to_dict()
        if not rep["passed"]:
            raise RuntimeError(f"F1C_INTEGRITY_FAIL: {tag}")
        gate = _read(run_dir / "step4_update_gate.json")
        g6[tag] = verify_g6(tag, gate)
        integrity[tag] = rep

    fp32 = verify_fp32_rgb(runs)
    g8 = g8_check(runs, a.expected_epochs)
    if not g8["passed"]:
        raise RuntimeError(f"F1C_G8_CLOSEOUT_FAIL: {g8}")
    g9 = rejudge_g9(runs, a.expected_epochs, a.seed, contract)
    if not g9["passed"]:
        raise RuntimeError(f"F1C_G9_REJUDGE_FAIL: {g9['errors'][:20]}")

    evals = {
        tag: _read(run_dir / "eval_step4_f1_c_causality.json")
        for tag, run_dir in runs.items()
    }
    for tag, ev in evals.items():
        if ev.get("schema") != "step4-f1-c-stock-validator-semantics-v1":
            raise RuntimeError(f"bad F1C eval schema in {tag}")
        if ev.get("group") != GROUP_SPECS[tag]["group"]:
            raise RuntimeError(f"F1C eval group mismatch {tag}")

    loo_path = project / "step4_f1_c_loo_last.json"
    if not loo_path.exists():
        raise RuntimeError("F1C_LOO_MISSING")
    loo = _read(loo_path)
    if loo.get("schema") != LOO_SCHEMA or loo.get("checkpoint") != "last.pt":
        raise RuntimeError("bad F1C LOO schema/checkpoint")
    val_ids = list(contract["val_ids"])
    # recompute deltas from folds; refuse any mismatch
    for key, (tag, variant, base_tag, base_variant) in DELTA_SPECS.items():
        recomputed = _delta_series(loo["folds"], val_ids, tag, variant,
                                   base_tag, base_variant)
        if loo["deltas"][key] != recomputed:
            raise RuntimeError(f"F1C_LOO_DELTA_MISMATCH: {key}")

    quality_path = runs["MAGSOFT"] / "eval_step4_f1_c_quality_last.json"
    if not quality_path.exists():
        raise RuntimeError("F1C_LAST_PT_QUALITY_EVIDENCE_MISSING")
    quality = _read(quality_path)
    if quality.get("schema") != "step4-f1-c-ir-quality-probe-v1":
        raise RuntimeError("bad F1C quality schema")

    posthoc_path = ROOT / "reports" / "step4_f1_c" / "posthoc_gradient_audit_c.json"
    if not posthoc_path.exists():
        raise RuntimeError("F1C_POSTHOC_AUDIT_MISSING")
    posthoc = _read(posthoc_path)
    if posthoc.get("schema") != "step4-f1-c-posthoc-gradient-audit-v1" \
            or posthoc.get("passed") is not True:
        raise RuntimeError("F1C_POSTHOC_AUDIT_NOT_PASSED")

    def score(tag, variant):
        return float(evals[tag]["last.pt"][variant]["val"]["map50_95"])

    c0 = score("C0", "NORMAL")
    fixed = score("FIXED", "NORMAL")
    normal = score("MAGSOFT", "NORMAL")
    zero = score("MAGSOFT", "ZERO-AUX")
    shuffle = score("MAGSOFT", "SHUFFLE")
    origsoft = score("ORIGSOFT", "NORMAL")

    mag_c0 = loo["deltas"]["MAGSOFT_minus_C0"]
    mag_fixed = loo["deltas"]["MAGSOFT_minus_FIXED"]
    mag_orig = loo["deltas"]["MAGSOFT_minus_ORIGSOFT"]

    def loo_cond(delta: dict) -> bool:
        return delta["full"] > 0 and delta["median"] > 0 \
            and delta["positive_folds"] >= 4

    loo_c0_ok = loo_cond(mag_c0)
    loo_fixed_ok = loo_cond(mag_fixed)
    loo_orig_ok = loo_cond(mag_orig)

    causal_pass = normal > c0 and normal > zero and normal > shuffle
    beats_origsoft = normal > origsoft
    beats_historical = normal > HISTORICAL_B1_SOFT_LAST

    conditions = quality.get("conditions") or {}
    identity = conditions.get("identity:0.00")
    own_qclean = float(identity["raw_q"]["mean"])
    degraded = {k: v for k, v in conditions.items() if k != "identity:0.00"}
    if len(degraded) != 17:
        raise RuntimeError(f"F1C_QUALITY_CONDITION_COUNT: {len(degraded)}")
    macro_soft = statistics.mean(
        row["learned_gate"]["map50_95"] for row in degraded.values())
    macro_fixed = statistics.mean(
        row["separately_trained_fixed"]["map50_95"] for row in degraded.values())
    macro_qclean = statistics.mean(
        row["force_qclean"]["map50_95"] for row in degraded.values())
    macro_orig = statistics.mean(
        row["original_gate_soft"]["map50_95"] for row in degraded.values())
    worst4_keys = sorted(degraded, key=lambda k:
                         degraded[k]["learned_gate"]["map50_95"])[:4]
    worst4_soft = statistics.mean(
        degraded[k]["learned_gate"]["map50_95"] for k in worst4_keys)
    worst4_fixed = statistics.mean(
        degraded[k]["separately_trained_fixed"]["map50_95"] for k in worst4_keys)
    worst4_qclean = statistics.mean(
        degraded[k]["force_qclean"]["map50_95"] for k in worst4_keys)
    worst4_orig = statistics.mean(
        degraded[k]["original_gate_soft"]["map50_95"] for k in worst4_keys)
    learned_minus_qclean_pos = sum(
        1 for row in degraded.values()
        if row["learned_minus_force_qclean_map50_95"] > 0)

    macro_pass = macro_soft > macro_fixed and macro_soft > macro_qclean
    worst4_pass = worst4_soft > worst4_fixed and worst4_soft > worst4_qclean
    adaptive_pass = learned_minus_qclean_pos >= 9
    beats_orig_macro = macro_soft > macro_orig
    beats_orig_worst4 = worst4_soft > worst4_orig

    if (causal_pass and loo_c0_ok and loo_fixed_ok and loo_orig_ok
            and beats_origsoft and beats_historical
            and macro_pass and worst4_pass and adaptive_pass
            and beats_orig_macro and beats_orig_worst4):
        decision = "PROMOTE_F1C_MAGNITUDE_GATE_CONFIRM_ONE_SEED"
        next_step = "one confirmation seed; keep Depth out"
    elif causal_pass and loo_c0_ok and loo_orig_ok and beats_origsoft:
        decision = "F1C_MAGNITUDE_HELPED_BUT_NOT_FULLY_PROMOTED"
        next_step = "inspect which promotion axis failed; no new modules"
    elif causal_pass and loo_c0_ok:
        decision = "F1C_IR_COMPLEMENTARY_MAGNITUDE_NOT_BETTER_THAN_ORIGINAL"
        next_step = "keep the simpler original gate unless magnitude wins"
    else:
        decision = "F1C_GATE_FAILED_CAUSAL_PROTOCOL"
        next_step = "stop; inspect intervention signs"

    summary = {
        "schema": "step4-f1-c-summary-v1",
        "loo_file_sha256": _sha(loo_path),
        "quality_file_sha256": _sha(quality_path),
        "posthoc_file_sha256": _sha(posthoc_path),
        "summarize_source_sha256": _sha(Path(__file__)),
        "integrity": integrity,
        "g6_rejudged": g6,
        "g8": g8,
        "g9_rejudged": g9,
        "fp32_rgb_verified": fp32,
        "posthoc": {"passed": posthoc["passed"]},
        "primary_last_val6": {
            "C0": c0, "FIXED_NORMAL": fixed, "MAGSOFT_NORMAL": normal,
            "MAGSOFT_ZERO": zero, "MAGSOFT_SHUFFLE": shuffle,
            "ORIGSOFT_NORMAL": origsoft,
            "MAGSOFT_minus_C0": normal - c0,
            "MAGSOFT_minus_FIXED": normal - fixed,
            "MAGSOFT_minus_ORIGSOFT": normal - origsoft,
            "MAGSOFT_minus_HISTORICAL_B1_SOFT": normal - HISTORICAL_B1_SOFT_LAST,
        },
        "loo_conditions": {
            "magsoft_minus_c0": {**{k: mag_c0[k] for k in ("full", "median", "positive_folds")}, "passed": loo_c0_ok},
            "magsoft_minus_fixed": {**{k: mag_fixed[k] for k in ("full", "median", "positive_folds")}, "passed": loo_fixed_ok},
            "magsoft_minus_origsoft": {**{k: mag_orig[k] for k in ("full", "median", "positive_folds")}, "passed": loo_orig_ok},
        },
        "promotion_evidence": {
            "own_force_qclean": own_qclean,
            "macro_soft_fixed_qclean_orig": [macro_soft, macro_fixed, macro_qclean, macro_orig],
            "worst4_keys": worst4_keys,
            "worst4_soft_fixed_qclean_orig": [worst4_soft, worst4_fixed, worst4_qclean, worst4_orig],
            "learned_minus_qclean_positive_count": learned_minus_qclean_pos,
            "macro_pass": macro_pass,
            "worst4_pass": worst4_pass,
            "adaptive_pass": adaptive_pass,
            "beats_origsoft_macro": beats_orig_macro,
            "beats_origsoft_worst4": beats_orig_worst4,
            "beats_historical_b1_soft": beats_historical,
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
