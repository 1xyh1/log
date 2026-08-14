#!/usr/bin/env python3
"""Step 3-A causal evaluation using authoritative DetectionValidator primitives.

NORMAL / ZERO / SHUFFLE x last.pt + best.pt x train/val/all17.
Primary checkpoint remains last.pt.  Evaluation refuses an incoherent 80-epoch run.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.raw_sample_index import CLASS_NAMES, OUT_DEFAULT, build_contract, group_of  # noqa: E402
from multimodal.run_integrity import inspect_step3_run, sha256_file  # noqa: E402
from multimodal.step3_eval_utils import evaluate_dataset_stock_semantics  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402

EVALUATOR_SCHEMA = "step3-stock-validator-semantics-v2"


def _device(s: str) -> str:
    if s in {"0", "cuda"}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return s


def _load_checkpoint_model(path: Path, device: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ckpt.get("ema")
    if model is None:
        model = ckpt.get("model")
    if model is None:
        raise RuntimeError(f"checkpoint contains neither ema nor model: {path}")
    return model.float().eval().to(device)


def _late10(run_dir: Path) -> dict:
    rows = list(csv.DictReader((run_dir / "results.csv").open(encoding="utf-8")))
    vals = [float(r["metrics/mAP50-95(B)"]) for r in rows if r.get("metrics/mAP50-95(B)")]
    last10 = vals[-10:]
    if not last10:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(last10), 6),
        "median": round(statistics.median(last10), 6),
        "std": round(statistics.stdev(last10), 6) if len(last10) > 1 else None,
        "min": round(min(last10), 6),
        "max": round(max(last10), 6),
    }


def _perfect_cross_group_matching(ids: list[str]) -> dict[str, str] | None:
    """Deterministic bipartite matching: every sample gets a unique donor from another group."""
    candidates = {
        sid: sorted(d for d in ids if d != sid and group_of(d) != group_of(sid))
        for sid in ids
    }
    # Most constrained left nodes first; deterministic tie break by sample id.
    left = sorted(ids, key=lambda sid: (len(candidates[sid]), sid))
    donor_to_sid: dict[str, str] = {}

    def dfs(sid: str, seen: set[str]) -> bool:
        for donor in candidates[sid]:
            if donor in seen:
                continue
            seen.add(donor)
            old = donor_to_sid.get(donor)
            if old is None or dfs(old, seen):
                donor_to_sid[donor] = sid
                return True
        return False

    if not all(dfs(sid, set()) for sid in left):
        return None
    out = {sid: donor for donor, sid in donor_to_sid.items()}
    return out if len(out) == len(ids) else None


def build_group_aware_derangement(ids: list[str]) -> dict[str, str]:
    """Bijective, no-self donor map; cross-group everywhere when mathematically possible."""
    ids = list(ids)
    cross = _perfect_cross_group_matching(ids)
    if cross is not None:
        result = cross
    else:
        # Deterministic cyclic fallback, search rotations until no self match.
        ordered = sorted(ids)
        result = None
        for shift in range(1, len(ordered)):
            cand = {sid: ordered[(i + shift) % len(ordered)] for i, sid in enumerate(ordered)}
            if all(s != d for s, d in cand.items()):
                result = cand
                break
        if result is None:
            raise RuntimeError("cannot build derangement")

    if set(result) != set(ids) or set(result.values()) != set(ids):
        raise RuntimeError("shuffle map is not a bijection")
    if any(s == d for s, d in result.items()):
        raise RuntimeError("shuffle map contains self donor")
    return result


def _load_or_create_shuffle_map(run_dir: Path, split: str, ids: list[str]) -> dict[str, str]:
    fp = run_dir / f"shuffle_map_{split}.json"
    expected_keys = set(ids)
    if fp.exists():
        obj = json.loads(fp.read_text(encoding="utf-8"))
        if set(obj) == expected_keys and set(obj.values()) == expected_keys and all(k != v for k, v in obj.items()):
            return obj
    obj = build_group_aware_derangement(ids)
    fp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return obj


def _provenance(run_dir: Path) -> dict:
    fields = {
        "results_sha256": run_dir / "results.csv",
        "args_sha256": run_dir / "args.yaml",
        "last_pt_sha256": run_dir / "weights" / "last.pt",
        "best_pt_sha256": run_dir / "weights" / "best.pt",
        "manifest_sha256": run_dir / "manifest.json",
        "g8_sha256": run_dir / "step3_g8_trace.jsonl",
        "kernel_growth_sha256": run_dir / "step3_kernel_growth.jsonl",
    }
    out = {k: sha256_file(v) if v.exists() else None for k, v in fields.items()}
    out.update(
        evaluator_schema=EVALUATOR_SCHEMA,
        evaluated_at=datetime.now().astimezone().isoformat(),
    )
    try:
        import ultralytics
        out["ultralytics_version"] = ultralytics.__version__
    except Exception:
        out["ultralytics_version"] = "unknown"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=["C0-N", "C1-I", "C2-D"], required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--project", default="runs/step3_earlyfusion")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--device", default="0")
    p.add_argument("--expected-epochs", type=int, default=80)
    p.add_argument("--c0-run-name", default=None,
                   help="Optional coherent C0 run for val6 LOO candidate-vs-control comparison.")
    args = p.parse_args()

    args.run_name = args.run_name or args.group
    args.device = _device(args.device)
    project = Path(args.project)
    run_dir = project / args.run_name

    integrity = inspect_step3_run(
        run_dir,
        expected_epochs=args.expected_epochs,
        require_weights=True,
        check_eval_provenance=False,  # we are about to replace legacy/stale eval output
    )
    if not integrity.passed:
        print(json.dumps(integrity.to_dict(), indent=2, ensure_ascii=False))
        raise RuntimeError(
            "REFUSE_EVALUATION_OF_INCOHERENT_RUN: " + "; ".join(integrity.errors)
        )

    contract = build_contract(out_path=args.contract)
    names = {i: CLASS_NAMES[i] for i in range(len(CLASS_NAMES))}

    shuffle_maps = {
        split: _load_or_create_shuffle_map(run_dir, split, contract[f"{split}_ids"])
        for split in ("train", "val", "all17")
    }

    results = {
        "schema": EVALUATOR_SCHEMA,
        "group": args.group,
        "physical_run_name": args.run_name,
        "integrity": integrity.to_dict(),
        "late10": _late10(run_dir),
        "provenance": _provenance(run_dir),
    }

    for ck_name in ("last.pt", "best.pt"):
        ck_path = run_dir / "weights" / ck_name
        model = _load_checkpoint_model(ck_path, args.device)
        results[ck_name] = {}
        for variant in ("NORMAL", "ZERO", "SHUFFLE"):
            results[ck_name][variant] = {}
            for split in ("train", "val", "all17"):
                ds = TriModalDataset(
                    contract,
                    split=split,
                    group=args.group,
                    augment=False,
                    aux_zero=(variant == "ZERO"),
                    aux_id_map=shuffle_maps[split] if variant == "SHUFFLE" else None,
                )
                metrics = evaluate_dataset_stock_semantics(model, ds, args.device, names)
                results[ck_name][variant][split] = metrics

        normal = results[ck_name]["NORMAL"]["val"]
        zero = results[ck_name]["ZERO"]["val"]
        shuffle = results[ck_name]["SHUFFLE"]["val"]
        print(
            f"[{args.group}/{args.run_name}] {ck_name}: "
            f"N={normal['map50_95']:.6f} Z={zero['map50_95']:.6f} S={shuffle['map50_95']:.6f} "
            f"preds={normal['n_predictions']} max_conf={normal['max_confidence']:.4f} "
            f"mean_best_iou={normal['mean_best_iou_per_gt']}"
        )

    # LOO is diagnostic only and must not block candidate evaluation while C0 is being recovered.
    if args.group == "C0-N":
        results["val6_loo"] = {"status": "NOT_APPLICABLE_CONTROL"}
    elif args.c0_run_name:
        c0_dir = project / args.c0_run_name
        c0_integrity = inspect_step3_run(c0_dir, expected_epochs=args.expected_epochs, require_weights=True)
        if not c0_integrity.passed:
            results["val6_loo"] = {
                "status": "SKIPPED_C0_INCOHERENT",
                "errors": c0_integrity.errors,
            }
        else:
            cand = _load_checkpoint_model(run_dir / "weights" / "last.pt", args.device)
            c0 = _load_checkpoint_model(c0_dir / "weights" / "last.pt", args.device)
            deltas, fold_support = [], []
            val_ids = contract["val_ids"]
            for j, removed in enumerate(val_ids):
                subset = [sid for i, sid in enumerate(val_ids) if i != j]
                sub_contract = {**contract, "val_ids": subset}
                cand_ds = TriModalDataset(sub_contract, "val", args.group, augment=False)
                c0_ds = TriModalDataset(sub_contract, "val", "C0-N", augment=False)
                mc = evaluate_dataset_stock_semantics(cand, cand_ds, args.device, names)["map50_95"]
                m0 = evaluate_dataset_stock_semantics(c0, c0_ds, args.device, names)["map50_95"]
                deltas.append(float(mc - m0))
                classes_left = sorted({
                    int(line.split()[0])
                    for sid in subset
                    for line in (Path(contract["_labels_dir"]) / f"{sid}.txt").read_text().splitlines()
                    if line.strip()
                })
                fold_support.append({"removed": removed, "classes_left": classes_left})
            results["val6_loo"] = {
                "status": "DIAGNOSTIC_ONLY",
                "deltas": deltas,
                "positive_folds": sum(d > 0 for d in deltas),
                "median_delta": float(np.median(deltas)),
                "min_delta": min(deltas),
                "max_delta": max(deltas),
                "fold_class_support": fold_support,
            }
    else:
        results["val6_loo"] = {"status": "SKIPPED_NO_C0_RUN_NAME"}

    out = run_dir / "eval_step3_causality.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
