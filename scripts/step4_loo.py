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

Deltas are computed by the single shared implementation
multimodal.step4_closeout.compute_deltas, and the finished payload is
self-checked with validate_loo_payload before writing (producer self-proof).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import step3_eval_utils as evu  # noqa: E402
from multimodal.raw_sample_index import CLASS_NAMES, build_contract, OUT_DEFAULT  # noqa: E402
from multimodal.run_integrity import inspect_step3_run  # noqa: E402
from multimodal.step4_closeout import (  # noqa: E402
    LOO_SCHEMA, compute_deltas, load_validated_shuffle_maps,
    resolve_dep_targets, validate_loo_payload)
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402

DATASET_GROUP = {"F0-C0": "C0-N", "F0-I": "C1-I", "F0-D": "C2-D"}


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
    # Each group uses its OWN shuffle map; both re-validated as bijective
    # no-self derangements and asserted equal (shared deterministic source).
    shuffle_maps = load_validated_shuffle_maps(run_dirs, val_ids)

    # folds: None = full val6; otherwise one excluded id
    folds = [None] + val_ids
    results = {"schema": LOO_SCHEMA,
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
    for dep, fp in resolve_dep_targets().items():
        results["provenance"][f"dep_{dep}_sha256"] = sha256(fp)
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

    results["deltas"] = compute_deltas(results["folds"], val_ids)

    # Producer self-proof: the payload must pass the same validator that the
    # summarizer runs before consuming it.
    proof = validate_loo_payload(results)
    if not proof["passed"]:
        raise RuntimeError(f"LOO_PAYLOAD_SELF_CHECK_FAILED: {proof['errors']}")

    out = project / "step4_loo.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print("->", out)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_fold"}
                      for k, v in results["deltas"].items()},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
