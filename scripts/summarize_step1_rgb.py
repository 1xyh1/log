#!/usr/bin/env python3
"""Step 1/2 Sample Probe ranking summary (three-set metrics, multi-seed mean+/-std).

Reads runs/<project>/<group>/experiment_manifest.json (eval_sets embedded,
legacy eval_dual fallback), computes per-group stats, writes _summary.json.

Two checkpoint axes per group:
    FIXED = last.pt (epoch-80 fixed budget)  -> PRIMARY ranking axis
    BEST  = best.pt (6-val selected)         -> auxiliary (unstable on 6-val)
Three evaluation sets per axis:
    train11 = "did the model learn the training samples" (NOT a gap metric)
    val6    = held-out transfer on the fixed split
    all17   = pipeline sanity only (mixed set, NOT a generalization metric)
Single-seed runs: std / direction_consistent / wins_all_paired_seeds = null
(no significance claims with n=1; paired direction reported from 2+ seeds).
PROVISIONAL: 17-image sample probe, not a competition baseline.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _pick(src: dict) -> dict:
    def g(sub, key):
        return src.get(sub, {}).get(key)
    return {"train_map50_95": g("train", "map50_95"),
            "val_map50": g("val", "map50"),
            "val_map50_95": g("val", "map50_95"),
            "all17_map50_95": g("all17", "map50_95")}


def load_run(project: Path, group: str):
    m = json.loads((project / group / "experiment_manifest.json").read_text(encoding="utf-8"))
    ev = m.get("eval_sets") or m.get("eval_dual") or {}
    evl = m.get("eval_sets_last") or m.get("eval_dual_last") or {}
    return {
        "group": m["group"],
        "head": m["head"],
        "recipe": m["recipe"],
        "recipe_source": m["recipe_source"],
        "seed": m["seed"],
        "best": _pick(ev),
        "fixed": _pick(evl),
        "manifest": m,
    }


def stats(pairs):
    """pairs: [(seed, value)]; None values dropped; std None for single seed."""
    vals = [v for _, v in pairs if v is not None]
    if not vals:
        return None
    return {
        "mean": round(statistics.mean(vals), 4),
        "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
        "per_seed": {str(s): v for s, v in pairs if v is not None},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step1_rgb")
    p.add_argument("--out", default="runs/step1_rgb/_summary.json")
    a = p.parse_args()
    project = Path(a.project)
    rows = []
    for d in sorted(project.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        man = d / "experiment_manifest.json"
        if man.is_file():
            rows.append(load_run(project, d.name))

    # group base name (strip -sYYYYMMDD seed suffix)
    def base(g: str) -> str:
        return g.rsplit("-s", 1)[0]

    grouped = {}
    for r in rows:
        grouped.setdefault(base(r["group"]), []).append(r)

    summary = {"provisional": True, "note": "17-image sample probe (11 train/6 val, 9 classes present). "
               "FIXED (last.pt epoch-80) is the primary axis: 6-val best.pt selection is unstable. "
               "train11 = fit check; all17 = pipeline sanity only. Single-seed stats are null. "
               "Not a competition baseline; rerun on full 2000-image set.", "groups": {}}
    print(f"{'group':<10}{'head':<5}{'recipe':<16}{'n':<3}"
          f"{'FIXED train':<14}{'FIXED val':<14}{'FIXED all17':<16}{'BEST val':<14}")
    print("-" * 96)
    for g in sorted(grouped):
        runs = grouped[g]
        hdr = runs[0]

        def s(axis, key):
            return stats([(r["seed"], r[axis][key]) for r in runs])

        ft, f, fa = s("fixed", "train_map50_95"), s("fixed", "val_map50_95"), s("fixed", "all17_map50_95")
        bt, b, ba = s("best", "train_map50_95"), s("best", "val_map50_95"), s("best", "all17_map50_95")

        def fmt(x):
            return f"{x['mean']:.4f}" if x else "n/a"

        print(f"{g:<10}{hdr['head']:<5}{hdr['recipe']:<16}{len(runs):<3}"
              f"{fmt(ft):<14}{fmt(f):<14}{fmt(fa):<16}{fmt(b):<14}")
        summary["groups"][g] = {
            "head": hdr["head"], "recipe": hdr["recipe"], "recipe_source": hdr["recipe_source"],
            "n_seeds": len(runs),
            "train11_mAP50-95_fixed": ft,
            "val_mAP50-95_fixed": f,
            "all17_mAP50-95_fixed": fa,
            "train11_mAP50-95_bestpt": bt,
            "val_mAP50-95_bestpt": b,
            "all17_mAP50-95_bestpt": ba,
        }

    # provisional winner: PRIMARY = FIXED (last.pt) val mean; BEST reported as reference
    def winner_on(key, label):
        ranked = sorted(
            [(g, d) for g, d in summary["groups"].items() if d[key] is not None],
            key=lambda kv: -kv[1][key]["mean"])
        if not ranked:
            return None
        w = ranked[0]
        out = {
            "group": w[0], "head": w[1]["head"], "recipe": w[1]["recipe"],
            "val_mAP50-95_mean": w[1][key]["mean"],
            "note": f"PROVISIONAL Step-1 winner on 17-image probe ({label} axis). Rerun on full data.",
        }
        if len(ranked) > 1:
            wd, rd = w[1][key], ranked[1][1][key]
            gap = wd["mean"] - rd["mean"]
            common_seeds = sorted(set(wd["per_seed"]) & set(rd["per_seed"]))
            paired = [(wd["per_seed"][s], rd["per_seed"][s]) for s in common_seeds]
            wins = sum(1 for x, y in paired if x > y)
            out["runner_up"] = ranked[1][0]
            out["margin_vs_runnerup"] = round(gap, 4)
            # n=1: no direction claim; from 2+ paired seeds only
            out["direction_consistent"] = bool(paired and wins == len(paired)) if len(paired) >= 2 else None
            out["wins_all_paired_seeds"] = f"{wins}/{len(paired)}" if len(paired) >= 2 else None
            if wd["std"] is not None and rd["std"] is not None:
                out["mean_gap_exceeds_combined_std"] = bool(gap > wd["std"] + rd["std"])
            else:
                out["mean_gap_exceeds_combined_std"] = None
        return out

    summary["winner_fixed"] = winner_on("val_mAP50-95_fixed", "FIXED last.pt")
    summary["winner_bestpt"] = winner_on("val_mAP50-95_bestpt", "BEST best.pt")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    w = summary["winner_fixed"]
    if w:
        print(f"\nwinner (FIXED last.pt): {w['group']} val mAP50-95 {w['val_mAP50-95_mean']:.4f}")
        if "runner_up" in w:
            print(f"  vs {w['runner_up']}: margin {w['margin_vs_runnerup']:.4f}, "
                  f"direction_consistent={w['direction_consistent']}, "
                  f"paired_seed_wins={w['wins_all_paired_seeds']}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
