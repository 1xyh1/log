#!/usr/bin/env python3
"""Step-3 summary that refuses mixed/stale formal artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.run_integrity import inspect_step3_run  # noqa: E402


def _read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def g8_check(run_dirs: dict[str, Path]) -> dict:
    traces = {g: _read_jsonl(p / "step3_g8_trace.jsonl") for g, p in run_dirs.items()}
    n = min(len(x) for x in traces.values())
    all_actual = all(
        "actual_order_sha256" in traces[g][e] and "actual_flip_sha256" in traces[g][e]
        for g in traces for e in range(n)
    )
    mismatches = []
    per_group_actual = {}
    for e in range(n):
        if all_actual:
            orders = {traces[g][e]["actual_order_sha256"] for g in traces}
            flips = {traces[g][e]["actual_flip_sha256"] for g in traces}
        else:
            # Preserved C1/C2 traces are legacy planned evidence.  New runner keeps
            # these compatible fields in addition to stronger actual-yield fields.
            orders = {traces[g][e].get("sample_order_sha256") for g in traces}
            flips = {traces[g][e].get("flip_schedule_sha256") for g in traces}
        if len(orders) != 1 or None in orders or len(flips) != 1 or None in flips:
            mismatches.append(e)
    for g, t in traces.items():
        per_group_actual[g] = all("actual_order_sha256" in x and "actual_flip_sha256" in x
                                  for x in t)
    # Honest evidence wording (reviewer): planned schedule matched across all 80 epochs;
    # actual DataLoader yield only proven for groups with actual_* fields (C0-N-r1).
    evidence_level = "actual_yield_all" if all(per_group_actual.values()) else \
        "legacy_planned_match_actual_yield_unavailable_for_some_groups"
    return {
        "epochs_compared": n,
        "order_and_flip_hashes_match": not mismatches,
        "mismatched_epochs": mismatches,
        "evidence_level": evidence_level,
        "actual_yield_per_group": per_group_actual,
        "claim": "80-epoch planned sampler/flip schedule matched across groups. "
                 "Actual DataLoader yield proven for groups with actual_* fields only.",
        "passed": not mismatches,
    }


def verdict(cand: dict, c0: dict, growth: list[dict]) -> dict:
    def val(obj, variant):
        return obj["last.pt"][variant]["val"]["map50_95"]

    n, z, s = val(cand, "NORMAL"), val(cand, "ZERO"), val(cand, "SHUFFLE")
    c0n = val(c0, "NORMAL")
    q = max((growth[-1].get(k, 0.0) for k in ("qI", "qD", "qM")), default=0.0) if growth else 0.0
    loo = cand.get("val6_loo", {})
    late = cand["late10"].get("median")
    best = cand["best.pt"]["NORMAL"]["val"]["map50_95"]
    unstable = bool(late is not None and best > 0.05 and late < 0.5 * best)
    if loo.get("status") == "DIAGNOSTIC_ONLY" and loo.get("positive_folds") is not None:
        unstable |= loo["positive_folds"] <= 1

    if n > c0n and n > z and n > s and not unstable:
        status = "COMPLEMENTARY-CANDIDATE"
    elif n > z and n > s and not (n > c0n):
        status = "MODEL-USES-AUX-BUT-NO-BENEFIT"
    elif q <= 1e-4 and abs(n - z) < 0.01 and abs(n - s) < 0.01:
        status = "NO-AUX-USAGE"
    elif unstable:
        status = "UNSTABLE"
    else:
        # Reviewer ruling: MIXED-EVIDENCE is a DIAGNOSTIC SUBSTATUS, not a fifth class.
        # Mixed intervention signs with learned aux kernels map to
        # MODEL-USES-AUX-BUT-NO-BENEFIT and the sign pattern is recorded explicitly.
        status = "MODEL-USES-AUX-BUT-NO-BENEFIT"
    paired_vs_zero = round(n - z, 4)
    paired_vs_shuffle = round(n - s, 4)
    diagnostics = []
    if q > 1e-4:
        diagnostics.append("aux kernels learned (q>1e-4)")
    if paired_vs_zero > 0:
        diagnostics.append("paired vs zero: positive")
    elif paired_vs_zero < 0:
        diagnostics.append("paired vs zero: negative")
    if paired_vs_shuffle > 0:
        diagnostics.append("paired vs shuffle: positive")
    elif paired_vs_shuffle < 0:
        diagnostics.append("paired vs shuffle: negative")
    if loo.get("status") == "DIAGNOSTIC_ONLY" and loo.get("positive_folds") is not None:
        diagnostics.append(f"LOO positive_folds={loo['positive_folds']}/{len(loo.get('deltas', []))} "
                           f"median_delta={loo.get('median_delta')}")
    return {
        "normal": n,
        "zero": z,
        "shuffle": s,
        "c0_normal": c0n,
        "late10_median": late,
        "best_pt_normal": best,
        "final_aux_q": q,
        "paired_vs_zero": paired_vs_zero,
        "paired_vs_shuffle": paired_vs_shuffle,
        "val6_loo": {k: loo.get(k) for k in
                     ("positive_folds", "median_delta", "min_delta", "max_delta", "status")},
        "diagnostics": diagnostics,
        "status": status,
        "note": "Negative Step3 early-fusion evidence does not imply the modality itself is useless.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step3_earlyfusion")
    p.add_argument("--c0-run", required=True)
    p.add_argument("--c1-run", default="C1-I")
    p.add_argument("--c2-run", default="C2-D")
    p.add_argument("--expected-epochs", type=int, default=80)
    a = p.parse_args()

    project = Path(a.project)
    physical = {"C0-N": a.c0_run, "C1-I": a.c1_run, "C2-D": a.c2_run}
    run_dirs = {g: project / name for g, name in physical.items()}

    integrity = {
        g: inspect_step3_run(path, a.expected_epochs, require_weights=True).to_dict()
        for g, path in run_dirs.items()
    }
    failures = {g: r for g, r in integrity.items() if not r["passed"]}
    if failures:
        print(json.dumps(failures, indent=2, ensure_ascii=False))
        raise RuntimeError("REFUSE_SUMMARY_WITH_INCOHERENT_RUNS")

    evals = {}
    for g, path in run_dirs.items():
        eval_path = path / "eval_step3_causality.json"
        evals[g] = json.loads(eval_path.read_text(encoding="utf-8"))
        if evals[g].get("schema") != "step3-stock-validator-semantics-v2":
            raise RuntimeError(f"{g}: evaluator schema is not the repaired authoritative v2")

    summary = {
        "schema": "step3-summary-v2",
        "physical_runs": physical,
        "integrity": integrity,
        "g8": g8_check(run_dirs),
        "groups": {},
    }
    c0 = evals["C0-N"]
    summary["groups"]["C0-N"] = {
        "role": "null-path control",
        "last_normal_val": c0["last.pt"]["NORMAL"]["val"]["map50_95"],
        "late10_median": c0["late10"].get("median"),
    }
    for g in ("C1-I", "C2-D"):
        growth = _read_jsonl(run_dirs[g] / "step3_kernel_growth.jsonl")
        summary["groups"][g] = verdict(evals[g], c0, growth)

    out = project / "_summary_step3_v2.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
