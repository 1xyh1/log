#!/usr/bin/env python3
"""Step 4-F0 LOO (leave-one-out) probe on val6 — NO retraining.

Primary question (reviewer, 2026-08-16): with each val6 image removed in turn,
what is the SHAPE of the IR-minus-C0 / D-minus-C0 mAP50-95 deltas —
all near zero, mixed signs, or dominated by one image?  This decides whether
F0-I is "no net gain at all" vs "aggregate tie on a tiny val set with a stable
direction".

Evaluation: last.pt only (primary protocol axis).  Folds = full val6 + 6
leave-one-out subsets.  Variants per group:
    F0-C0-r1 : NORMAL only — ZERO-AUX/SHUFFLE coincide with NORMAL by
               construction (the C0-N group mask already zeroes aux channels),
               so fold values are copied and marked copy_of_normal.
    F0-I     : NORMAL / ZERO-AUX / SHUFFLE
    F0-D     : NORMAL / ZERO-AUX / SHUFFLE
SHUFFLE fold semantics: the excluded image leaves the ANCHOR set only; the
donor pool stays the full val6 bijective derangement (shuffle_map_val.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import step3_eval_utils as evu  # noqa: E402
from multimodal.causality_interventions import assert_valid_shuffle_map  # noqa: E402
from multimodal.raw_sample_index import CLASS_NAMES, build_contract, OUT_DEFAULT  # noqa: E402
from multimodal.run_integrity import inspect_step3_run  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402

DATASET_GROUP = {"F0-C0": "C0-N", "F0-I": "C1-I", "F0-D": "C2-D"}

# Execution-semantics dependencies whose provenance must be frozen alongside the
# LOO numbers (reviewer P1): evaluator core, dataset, and the model class chain
# that torch pickle needs to load the checkpoints.
DEPENDENCY_SOURCES = {
    "step3_eval_utils": "src/multimodal/step3_eval_utils.py",
    "trimodal_dataset": "src/multimodal/trimodal_dataset.py",
    "step4_f0_model": "src/multimodal/step4_f0_model.py",
    "aux_encoder": "src/multimodal/aux_encoder.py",
    "feature_fusion": "src/multimodal/feature_fusion.py",
    "trainability": "src/multimodal/trainability.py",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_last_model(run_dir: Path, device: torch.device):
    ck = torch.load(run_dir / "weights" / "last.pt", map_location="cpu",
                    weights_only=False)
    return (ck.get("ema") or ck.get("model")).float().eval().to(device)


def evaluate_fold(model, contract, group, excluded, device, names,
                  aux_zero=False, aux_id_map=None) -> float:
    ds = TriModalDataset(contract, split="val", group=group, augment=False,
                         aux_zero=aux_zero, aux_id_map=aux_id_map,
                         exclude_ids={excluded} if excluded else None)
    res = evu.evaluate_dataset_stock_semantics(model, ds, device, names)
    return float(res["map50_95"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_f0")
    p.add_argument("--c0-run", default="F0-C0-r1")
    p.add_argument("--ir-run", default="F0-I")
    p.add_argument("--depth-run", default="F0-D")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--device", default="0")
    p.add_argument("--expected-epochs", type=int, default=80)
    a = p.parse_args()
    device = evu._as_device(a.device)

    contract = build_contract(out_path=a.contract)
    # Class names come from the frozen contract constant (CLASS_NAMES), not from
    # a machine-local dataset.yaml — keeps the LOO rerunnable from the repo alone.
    names = dict(CLASS_NAMES)

    project = Path(a.project)
    run_dirs = {"C0": project / a.c0_run, "IR": project / a.ir_run,
                "D": project / a.depth_run}
    for tag, run_dir in run_dirs.items():
        integrity = inspect_step3_run(
            run_dir, a.expected_epochs, require_weights=True,
            trace_name="step4_g8_trace.jsonl", growth_name="step4_growth.jsonl",
            eval_name="eval_step4_causality.json")
        if not integrity.to_dict()["passed"]:
            print(json.dumps(integrity.to_dict(), indent=2, ensure_ascii=False))
            raise RuntimeError(f"REFUSE_LOO_INCOHERENT_RUN: {tag}")

    models = {tag: load_last_model(rd, device) for tag, rd in run_dirs.items()}
    val_ids = list(contract["val_ids"])
    assert len(val_ids) == 6, val_ids
    # Shuffle maps: each group uses its OWN map; both are re-validated here as
    # bijective no-self derangements and asserted byte-equal (they come from the
    # same deterministic derangement of val6).
    shuffle_maps = {}
    for tag in ("IR", "D"):
        fp = run_dirs[tag] / "shuffle_map_val.json"
        m = json.loads(fp.read_text(encoding="utf-8"))
        assert assert_valid_shuffle_map(m, val_ids), f"{tag} shuffle map invalid"
        shuffle_maps[tag] = m
    assert shuffle_maps["IR"] == shuffle_maps["D"], "IR/D shuffle maps differ"

    # folds: None = full val6; otherwise one excluded id
    folds = [None] + val_ids
    results = {"schema": "step4-loo-v1",
               "method": "val6 leave-one-out on last.pt; SHUFFLE donor pool stays "
                        "the full val6 derangement (excluded image leaves anchor "
                        "set only); C0 ZERO/SHUFFLE = NORMAL by group-mask "
                        "construction (copied, marked).",
               "checkpoint": "last.pt",
               "val_ids": val_ids,
               "groups": {tag: str(rd) for tag, rd in run_dirs.items()},
               "provenance": {
                   f"{tag}_last_pt_sha256": sha256(rd / "weights" / "last.pt")
                   for tag, rd in run_dirs.items()
               },
               "folds": {}}
    results["provenance"]["contract_sha256"] = sha256(Path(a.contract))
    results["provenance"]["loo_source_sha256"] = sha256(Path(__file__))
    for dep, rel in DEPENDENCY_SOURCES.items():
        results["provenance"][f"dep_{dep}_sha256"] = sha256(ROOT / rel)
    results["provenance"]["ir_shuffle_map_val_sha256"] = sha256(
        run_dirs["IR"] / "shuffle_map_val.json")
    results["provenance"]["d_shuffle_map_val_sha256"] = sha256(
        run_dirs["D"] / "shuffle_map_val.json")

    n_done = 0
    total = 7 * 3  # one printed line per fold x group (IR/D lines run 3 variants)
    for fold in folds:
        fold_key = "full" if fold is None else fold
        fold_res = {}
        for tag in ("C0", "IR", "D"):
            group = DATASET_GROUP[{"C0": "F0-C0", "IR": "F0-I", "D": "F0-D"}[tag]]
            variants = {"NORMAL": evaluate_fold(
                models[tag], contract, group, fold, device, names)}
            if tag == "C0":
                variants["ZERO-AUX"] = variants["NORMAL"]
                variants["SHUFFLE"] = variants["NORMAL"]
                copied = True
            else:
                variants["ZERO-AUX"] = evaluate_fold(
                    models[tag], contract, group, fold, device, names,
                    aux_zero=True)
                variants["SHUFFLE"] = evaluate_fold(
                    models[tag], contract, group, fold, device, names,
                    aux_id_map=shuffle_maps[tag])
                copied = False
            fold_res[tag] = {v: round(x, 6) for v, x in variants.items()}
            fold_res[tag]["copy_of_normal"] = copied
            n_done += 1
            print(f"[{n_done}/{total}] fold={fold_key} {tag} NORMAL="
                  f"{fold_res[tag]['NORMAL']:.4f}", flush=True)
        results["folds"][fold_key] = fold_res

    # ------------------------------------------------------------------ deltas
    def delta(tag: str, variant: str, base_tag: str = "C0",
              base_variant: str = "NORMAL") -> dict:
        full = (results["folds"]["full"][tag][variant]
                - results["folds"]["full"][base_tag][base_variant])
        per_fold = {f: round(results["folds"][f][tag][variant]
                             - results["folds"][f][base_tag][base_variant], 6)
                    for f in val_ids}
        vals = list(per_fold.values())
        pos = sum(1 for x in vals if x > 0)
        return {"full": round(full, 6),
                "per_fold": per_fold,
                "positive_folds": pos,
                "n_folds": len(vals),
                "median": round(statistics.median(vals), 6) if vals else None,
                "min": round(min(vals), 6),
                "max": round(max(vals), 6)}

    results["deltas"] = {
        "IR_minus_C0": delta("IR", "NORMAL"),
        "D_minus_C0": delta("D", "NORMAL"),
        "IR_N_minus_Z": delta("IR", "NORMAL", "IR", "ZERO-AUX"),
        "IR_N_minus_S": delta("IR", "NORMAL", "IR", "SHUFFLE"),
        "D_N_minus_Z": delta("D", "NORMAL", "D", "ZERO-AUX"),
        "D_N_minus_S": delta("D", "NORMAL", "D", "SHUFFLE"),
    }

    out = project / "step4_loo.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print("->", out)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_fold"}
                      for k, v in results["deltas"].items()},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
