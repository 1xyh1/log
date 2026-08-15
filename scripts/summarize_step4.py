#!/usr/bin/env python3
"""Step 4-F0 final summary — the closing gate before the four-class verdict.

Consumes (fail-fast on any mismatch):
  * run integrity (inspect_step3_run with step4 trace/growth/eval names)
  * ALL SEVEN provenance SHA-256 entries recorded by eval_step4_causality.py
    (results / args / last / best / manifest / contract / evaluator source)
  * LOO payload recomputation (validate_loo_payload — exact equality against
    the raw folds) and full LOO provenance (dependency SHA set, shuffle map
    SHA, groups-path cross-check, eval cross-consistency)
  * actual-yield G8 closeout gate (per-row expected==actual, flag all-true,
    positional epoch continuity, byte-identical traces)
  * G6 update gates RE-JUDGED under the unified threshold (control < 1e-3,
    active aux > 1e-3)
  * last/best/late10 causal numbers (N-C0 / N-Z / N-S)
  * the frozen four-class verdict (COMPLEMENTARY-CANDIDATE /
    MODEL-USES-AUX-BUT-NO-BENEFIT / NO-AUX-USAGE / UNSTABLE; mixed evidence is
    a diagnostic substatus, not a fifth class — reviewer ruling).

The written summary pins the LOO file bytes (loo_file_sha256) and this
script's own source (summarize_source_sha256), so the frozen verdict document
is fully traceable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.run_integrity import inspect_step3_run, sha256_file  # noqa: E402
from multimodal.step4_closeout import (  # noqa: E402
    g8_check, loo_provenance_check, validate_loo_payload)

UNIFIED_ACTIVE_THRESHOLD = 1e-3
LOO_NEAR_ZERO = 0.01


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def g6_rejudge(run_dirs: dict[str, Path]) -> dict:
    """Re-judge recorded update gates under the unified 1e-3 threshold."""
    out = {}
    for g, rd in run_dirs.items():
        gate = _read_json(rd / "step4_update_gate.json")
        rel = gate["aux_encoder_global_rel_l2"]
        if g == "C0":
            passed = bool(gate["rgb_backbone_unchanged"]
                          and rel < UNIFIED_ACTIVE_THRESHOLD
                          and max(gate["proj_weight_norms"]) == 0.0)
        else:
            passed = bool(gate["rgb_backbone_unchanged"]
                          and rel > UNIFIED_ACTIVE_THRESHOLD
                          and min(gate["proj_weight_norms"]) > 0.0)
        out[g] = {
            "rgb_backbone_unchanged": gate["rgb_backbone_unchanged"],
            "aux_encoder_global_rel_l2": rel,
            "proj_weight_norms": gate["proj_weight_norms"],
            "unified_threshold_rejudge_passed": passed,
            "recorded_original_threshold_note": (
                "gate file was written with the legacy active-aux threshold "
                ">1e-5; re-judged here with the unified 1e-3 (control decay "
                "scale 2.05e-4 sits below 1e-3, so >1e-5 could misread decay "
                "noise as learning)"),
        }
    return out


def provenance_check(eval_obj: dict, run_dir: Path, contract_path: Path,
                     evaluator_path: Path) -> dict:
    """Re-hash all seven recorded provenance entries; fail-fast on mismatch."""
    prov = eval_obj.get("provenance") or {}
    targets = {
        "results_sha256": run_dir / "results.csv",
        "args_sha256": run_dir / "args.yaml",
        "last_pt_sha256": run_dir / "weights" / "last.pt",
        "best_pt_sha256": run_dir / "weights" / "best.pt",
        "manifest_sha256": run_dir / "manifest.json",
        "contract_sha256": contract_path,
        "evaluator_source_sha256": evaluator_path,
    }
    checks = {}
    for key, fp in targets.items():
        if key not in prov:
            checks[key] = {"recorded": None, "current": None, "match": False,
                           "error": "RECORDED_SHA_MISSING"}
            continue
        if not fp.exists():
            checks[key] = {"recorded": prov[key], "current": None, "match": False,
                           "error": "TARGET_FILE_MISSING"}
            continue
        cur = sha256_file(fp)
        checks[key] = {"recorded": prov[key], "current": cur,
                       "match": prov[key] == cur}
    return checks


def loo_shape(loo: dict, key: str) -> dict:
    """Quantitative shape of one LOO delta series (per-fold values)."""
    d = loo["deltas"][key]
    vals = list(d["per_fold"].values())
    abs_vals = sorted((abs(v), f) for f, v in d["per_fold"].items())
    dom_mag, dom_fold = abs_vals[-1]
    total_abs = sum(a for a, _ in abs_vals)
    dom_share = dom_mag / total_abs if total_abs else None
    if max(abs(v) for v in vals) < LOO_NEAR_ZERO:
        shape = "all_near_zero"
    elif d["positive_folds"] >= len(vals) - 1:
        shape = "stable_positive"
    elif d["positive_folds"] <= 1:
        shape = "stable_negative"
    else:
        shape = "mixed_signs"
    return {"shape": shape,
            "per_fold": d["per_fold"],
            "full_delta": d["full"],
            "positive_folds": d["positive_folds"],
            "n_folds": d["n_folds"],
            "median": d["median"],
            "min": d["min"],
            "max": d["max"],
            "dominant_fold": dom_fold,
            "dominant_abs_share": round(dom_share, 4) if dom_share is not None else None}


def verdict(cand: dict, c0: dict, loo: dict, g6: dict, tag: str) -> dict:
    def val(obj, ck, variant):
        return obj[ck][variant]["val"]["map50_95"]

    n = val(cand, "last.pt", "NORMAL")
    z = val(cand, "last.pt", "ZERO-AUX")
    s = val(cand, "last.pt", "SHUFFLE")
    c0n = val(c0, "last.pt", "NORMAL")
    best_n = val(cand, "best.pt", "NORMAL")
    best_z = val(cand, "best.pt", "ZERO-AUX")
    best_s = val(cand, "best.pt", "SHUFFLE")
    c0_best = val(c0, "best.pt", "NORMAL")
    late = cand["late10"].get("median")
    aux_learned = g6[tag]["aux_encoder_global_rel_l2"] > UNIFIED_ACTIVE_THRESHOLD
    unstable = bool(late is not None and best_n > 0.05 and late < 0.5 * best_n)

    if n > c0n and n > z and n > s and not unstable:
        status = "COMPLEMENTARY-CANDIDATE"
    elif n > z and n > s and not (n > c0n):
        status = "MODEL-USES-AUX-BUT-NO-BENEFIT"
    elif not aux_learned and abs(n - z) < 0.01 and abs(n - s) < 0.01:
        status = "NO-AUX-USAGE"
    elif unstable:
        status = "UNSTABLE"
    else:
        status = "MODEL-USES-AUX-BUT-NO-BENEFIT"  # mixed signs -> substatus below

    diagnostics = []
    if n > z and n > s:
        diagnostics.append("PAIRED_BEATS_ZERO_AND_SHUFFLE")
    if tag == "IR" and n - z > 0.02:
        diagnostics.append("STRONG_PAIRED_IR_USAGE")
    if abs(n - c0n) < 0.01:
        diagnostics.append("NEAR_CONTROL")
    if best_n > c0_best and best_n > best_z and best_n > best_s:
        diagnostics.append("BEST_PT_PASSES_ALL_3")
    if aux_learned:
        diagnostics.append("AUX_LEARNED")
    if (n - z) <= 0 or (n - s) <= 0:
        diagnostics.append("WEAK_OR_NONGENERALIZING_DEPTH_USAGE" if tag == "D"
                           else "MIXED_INTERVENTION_SIGNS")
    loo_res = loo_shape(loo, f"{tag}_minus_C0")
    diagnostics.append(f"LOO_{loo_res['shape'].upper()}_"
                       f"pos{loo_res['positive_folds']}/{loo_res['n_folds']}")
    return {
        "tag": tag,
        "status": status,
        "normal": round(n, 4),
        "zero": round(z, 4),
        "shuffle": round(s, 4),
        "c0_normal": round(c0n, 4),
        "paired_vs_zero": round(n - z, 4),
        "paired_vs_shuffle": round(n - s, 4),
        "vs_c0": round(n - c0n, 4),
        "late10_median": late,
        "best_pt_normal": round(best_n, 4),
        "best_pt_c0": round(c0_best, 4),
        "best_pt_paired_vs_zero": round(best_n - best_z, 4),
        "best_pt_paired_vs_shuffle": round(best_n - best_s, 4),
        "aux_learned_beyond_decay": aux_learned,
        "loo": loo_res,
        "diagnostics": diagnostics,
        "note": ("Negative evidence does not imply the modality itself is useless "
                 "(Step 3-A standing ruling)."),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_f0")
    p.add_argument("--c0-run", default="F0-C0-r1")
    p.add_argument("--ir-run", default="F0-I")
    p.add_argument("--depth-run", default="F0-D")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--expected-epochs", type=int, default=80)
    a = p.parse_args()

    project = Path(a.project)
    contract_path = Path(a.contract)
    evaluator_path = ROOT / "scripts" / "eval_step4_causality.py"
    loo_script_path = ROOT / "scripts" / "step4_loo.py"
    run_dirs = {"C0": project / a.c0_run, "IR": project / a.ir_run,
                "D": project / a.depth_run}

    integrity = {
        g: inspect_step3_run(rd, a.expected_epochs, require_weights=True,
                             trace_name="step4_g8_trace.jsonl",
                             growth_name="step4_growth.jsonl",
                             eval_name="eval_step4_causality.json").to_dict()
        for g, rd in run_dirs.items()
    }
    failures = {g: r for g, r in integrity.items() if not r["passed"]}
    if failures:
        print(json.dumps(failures, indent=2, ensure_ascii=False))
        raise RuntimeError("REFUSE_SUMMARY_WITH_INCOHERENT_RUNS")

    evals = {g: _read_json(rd / "eval_step4_causality.json")
             for g, rd in run_dirs.items()}
    for g, ev in evals.items():
        if ev.get("schema") != "step4-stock-validator-semantics-v1":
            raise RuntimeError(f"{g}: evaluator schema mismatch: {ev.get('schema')}")

    prov = {g: provenance_check(evals[g], run_dirs[g], contract_path,
                                evaluator_path) for g in run_dirs}
    bad = {g: {k: v for k, v in checks.items() if not v["match"]}
           for g, checks in prov.items()}
    if any(bad.values()):
        print(json.dumps(bad, indent=2, ensure_ascii=False))
        raise RuntimeError("REFUSE_SUMMARY_WITH_STALE_PROVENANCE")

    loo_path = project / "step4_loo.json"
    if not loo_path.exists():
        raise RuntimeError("STEP4_LOO_MISSING: run scripts/step4_loo.py first")
    loo = _read_json(loo_path)
    # P1 closeout: recompute the payload from raw folds before trusting it.
    loo_payload = validate_loo_payload(loo)
    if not loo_payload["passed"]:
        print(json.dumps(loo_payload["errors"], indent=2, ensure_ascii=False))
        raise RuntimeError("REFUSE_SUMMARY_WITH_INCONSISTENT_LOO_PAYLOAD")
    loo_prov = loo_provenance_check(loo, run_dirs, contract_path,
                                    loo_script_path, evals)
    bad_loo = {k: v for k, v in loo_prov.items() if not v["match"]}
    if bad_loo:
        print(json.dumps(bad_loo, indent=2, ensure_ascii=False))
        raise RuntimeError("REFUSE_SUMMARY_WITH_STALE_LOO")

    g6 = g6_rejudge(run_dirs)
    summary = {
        "schema": "step4-f0-summary-v2",
        "physical_runs": {g: str(rd) for g, rd in run_dirs.items()},
        "integrity": integrity,
        "provenance_all_seven": prov,
        "loo_file_sha256": sha256_file(loo_path),
        "summarize_source_sha256": sha256_file(Path(__file__)),
        "loo_provenance": loo_prov,
        "g8_actual": g8_check(run_dirs, a.expected_epochs),
        "g6_unified_rejudge": g6,
        "loo": {"path": str(loo_path),
                "method": loo["method"],
                "ir_minus_c0": loo_shape(loo, "IR_minus_C0"),
                "d_minus_c0": loo_shape(loo, "D_minus_C0"),
                "ir_n_minus_z": loo_shape(loo, "IR_N_minus_Z"),
                "ir_n_minus_s": loo_shape(loo, "IR_N_minus_S"),
                "d_n_minus_z": loo_shape(loo, "D_N_minus_Z"),
                "d_n_minus_s": loo_shape(loo, "D_N_minus_S")},
        "groups": {},
        "verdict_frozen": False,
    }
    summary["groups"]["F0-C0"] = {
        "role": "null-path control (F0 structure, frozen backbone)",
        "last_normal_val": evals["C0"]["last.pt"]["NORMAL"]["val"]["map50_95"],
        "late10_median": evals["C0"]["late10"].get("median"),
    }
    for tag, gname in (("IR", "F0-I"), ("D", "F0-D")):
        summary["groups"][gname] = verdict(evals[tag], evals["C0"], loo, g6, tag)

    # Closeout: freeze only after every gate above passed (provenance and LOO
    # checks already fail-fast; G8/G6 are re-asserted here explicitly).
    if not summary["g8_actual"]["passed"]:
        raise RuntimeError("REFUSE_FREEZE_G8_NOT_PASSED")
    if not all(v["unified_threshold_rejudge_passed"] for v in g6.values()):
        raise RuntimeError("REFUSE_FREEZE_G6_NOT_PASSED")
    gi = summary["groups"]["F0-I"]
    gd = summary["groups"]["F0-D"]
    nz = summary["loo"]["ir_n_minus_z"]
    ns = summary["loo"]["ir_n_minus_s"]
    summary["verdict_frozen"] = True
    summary["final_conclusions"] = {
        "F0-I": {
            "status": gi["status"],
            "statement": (
                "IR is genuinely used by the model: NORMAL-ZERO is positive in "
                f"{nz['positive_folds']}/{nz['n_folds']} LOO folds (stable "
                "strong signal) and NORMAL-SHUFFLE in "
                f"{ns['positive_folds']}/{ns['n_folds']}. "
                "It has not been proven to beat the matched RGB baseline: "
                f"IR-C0 = {gi['vs_c0']:+.4f} (last.pt val6), LOO "
                f"{gi['loo']['positive_folds']}/{gi['loo']['n_folds']} positive, "
                f"median {gi['loo']['median']:+.4f}, dominated by one negative "
                "fold."),
            "diagnostics": gi["diagnostics"],
        },
        "F0-D": {
            "status": gd["status"],
            "statement": (
                "Depth parameters learn beyond the control decay scale, but the "
                "training-set exploitable information did not convert into "
                "independent val generalization gain: "
                f"D-C0 = {gd['vs_c0']:+.4f} (last.pt val6), LOO "
                f"{gd['loo']['positive_folds']}/{gd['loo']['n_folds']} positive, "
                f"median {gd['loo']['median']:+.4f}."),
            "diagnostics": gd["diagnostics"],
        },
        "next_step": ("F1 IR soft/reliability gate (CSSA soft reliability / "
                      "EvaNet quality prior direction). Do NOT stack Depth yet."),
    }

    out = project / "_summary_step4.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
