#!/usr/bin/env python3
"""Step 3-A summary: G8 cross-group check + Δ framework + four-status verdict.

Primary axis = last.pt NORMAL/ZERO/SHUFFLE (protocol); best.pt diagnostic only.
No +0.02 thresholds; interpretation boundary: Step 3-A negative != modality useless.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GROUPS = ("C0-N", "C1-I", "C2-D")


def g8_check(project: Path) -> dict:
    traces = {}
    for g in GROUPS:
        lines = (project / g / "step3_g8_trace.jsonl").read_text().strip().splitlines()
        traces[g] = [json.loads(l) for l in lines]
    n = min(len(t) for t in traces.values())
    order_ok = flip_ok = True
    mism = []
    for e in range(n):
        orders = {traces[g][e]["sample_order_sha256"] for g in GROUPS}
        flips = {traces[g][e]["flip_schedule_sha256"] for g in GROUPS}
        if len(orders) > 1 or len(flips) > 1:
            order_ok, flip_ok = False, False
            mism.append(e)
    return {"epochs_compared": n, "sample_order_all_epochs_match": order_ok,
            "flip_schedule_all_epochs_match": flip_ok,
            "mismatched_epochs": mism, "passed": bool(order_ok and flip_ok)}


def verdict(group: str, cand: dict, c0: dict, growth: list) -> dict:
    def v(ck, variant, split="val"):
        return ck["last.pt"][variant][split]["map50_95"]

    n = v(cand, "NORMAL")
    z = v(cand, "ZERO")
    s = v(cand, "SHUFFLE")
    c0n = v(c0, "NORMAL")
    late10 = cand["late10"]["median"]
    best = cand["best.pt"]["NORMAL"]["val"]["map50_95"]
    last_q = growth[-1] if growth else {}
    used_q = max(last_q.get("qI", 0), last_q.get("qD", 0), last_q.get("qM", 0))
    loo = cand.get("val6_loo", {})
    deltas = loo.get("deltas", [])
    d = {"normal": n, "zero": z, "shuffle": s, "c0_normal": c0n,
         "late10_median": late10, "best_pt_normal": best, "final_aux_q": used_q}
    over_c0 = n > c0n
    over_ablation = n > z and n > s
    aux_used = used_q > 1e-4
    unstable = (late10 is not None and late10 < 0.5 * best and best > 0.05) or \
               (loo.get("positive_folds") is not None and loo["positive_folds"] <= 1)
    if over_c0 and over_ablation and not unstable:
        d["status"] = "COMPLEMENTARY-CANDIDATE"
    elif over_ablation and not over_c0:
        d["status"] = "MODEL-USES-AUX-BUT-NO-BENEFIT"
    elif not aux_used and abs(n - z) < 0.01 and abs(n - s) < 0.01:
        d["status"] = "NO-AUX-USAGE"
    elif unstable:
        d["status"] = "UNSTABLE"
    else:
        d["status"] = "MIXED-EVIDENCE"
    d["note"] = ("interpretation boundary: NEGATIVE here only means shared-backbone early "
                 "fusion showed no complementarity evidence; IR/Depth learnability was "
                 "already proven in Step 2. No significance claims on 6-val.")
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step3_earlyfusion")
    a = p.parse_args()
    project = Path(a.project)
    evals = {}
    for g in GROUPS:
        evals[g] = json.loads((project / g / "eval_step3_causality.json").read_text(encoding="utf-8"))
        growth = [json.loads(l) for l in
                  (project / g / "step3_kernel_growth.jsonl").read_text().strip().splitlines()]
        evals[g]["_growth"] = growth
    summary = {"g8": g8_check(project), "groups": {}}
    for g in ("C1-I", "C2-D"):
        summary["groups"][g] = verdict(g, evals[g], evals["C0-N"], evals[g]["_growth"])
    summary["groups"]["C0-N"] = {
        "role": "null-path control (aux channels all-zero)",
        "last_normal_val": evals["C0-N"]["last.pt"]["NORMAL"]["val"]["map50_95"],
        "late10_median": evals["C0-N"]["late10"]["median"],
        "final_aux_q": max(summary["groups"].get("C1-I", {}).get("final_aux_q", 0), 0) or None,
    }
    # C0-N kernel growth must stay exactly 0
    c0_growth = evals["C0-N"]["_growth"]
    c0_max = max((max(g["wI_norm"], g["wD_norm"], g["wM_norm"]) for g in c0_growth), default=0)
    summary["C0N_aux_weights_strictly_zero"] = bool(c0_max == 0.0)
    out = project / "_summary_step3.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"G8: passed={summary['g8']['passed']} epochs={summary['g8']['epochs_compared']} "
          f"mismatches={summary['g8']['mismatched_epochs']}")
    print(f"C0-N aux weights strictly zero: {summary['C0N_aux_weights_strictly_zero']} "
          f"(max={c0_max})")
    for g in ("C1-I", "C2-D"):
        d = summary["groups"][g]
        print(f"{g}: N={d['normal']:.4f} Z={d['zero']:.4f} S={d['shuffle']:.4f} "
              f"C0={d['c0_normal']:.4f} late10={d['late10_median']} best={d['best_pt_normal']:.4f} "
              f"q={d['final_aux_q']:.4f} -> {d['status']}")
    print("->", out)


if __name__ == "__main__":
    main()
