#!/usr/bin/env python3
"""Step 4-F0 causality evaluation: NORMAL / ZERO-AUX / SHUFFLE x best/last x
train11/val6/all17 (+ late10, per-class). Reuses the fixed Step-3 evaluator core:
the F0 model consumes the SAME 6ch TriModalDataset contract and splits internally,
so stock-validator-semantics evaluation carries over unchanged.

Primary axis = last.pt NORMAL/ZERO-AUX/SHUFFLE (protocol). ZERO-AUX zeroes the aux
channels in the 6ch batch (group dataset already zeroes inactive channels, so the
F0-C0 control's three variants coincide by construction).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import step3_eval_utils as evu  # noqa: E402
from multimodal.causality_interventions import (  # noqa: E402
    assert_valid_shuffle_map, bijective_derangement)
from multimodal.raw_sample_index import build_contract, OUT_DEFAULT  # noqa: E402
from multimodal.run_integrity import inspect_step3_run  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402

GROUPS = {"F0-C0": "zero", "F0-I": "ir", "F0-D": "depth"}
DATASET_GROUP = {"F0-C0": "C0-N", "F0-I": "C1-I", "F0-D": "C2-D"}


def late10(run_dir: Path) -> dict:
    import csv
    rows = list(csv.DictReader((run_dir / "results.csv").open()))
    vals = [float(r["metrics/mAP50-95(B)"]) for r in rows if r.get("metrics/mAP50-95(B)")]
    last10 = vals[-10:]
    return {"mean": round(statistics.mean(last10), 4),
            "median": round(statistics.median(last10), 4),
            "std": round(statistics.stdev(last10), 4) if len(last10) > 1 else None,
            "min": round(min(last10), 4), "max": round(max(last10), 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=["F0-C0", "F0-I", "F0-D"], required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--project", default="runs/step4_f0")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--device", default="0")
    p.add_argument("--expected-epochs", type=int, default=80)
    a = p.parse_args()
    device = evu._as_device(a.device)

    contract = build_contract(out_path=a.contract)
    import yaml as _yaml
    names = _yaml.safe_load(Path(
        "D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml")
        .read_text(encoding="utf-8"))["names"]
    names = {int(k): v for k, v in names.items()}

    run_dir = Path(a.project) / a.run_name
    # P0-5: integrity gate before any evaluation (stale/mixed/short runs rejected)
    integrity = inspect_step3_run(run_dir, a.expected_epochs, require_weights=True,
                                  trace_name="step4_g8_trace.jsonl",
                                  growth_name="step4_growth.jsonl")
    if not integrity.to_dict()["passed"]:
        print(json.dumps(integrity.to_dict(), indent=2, ensure_ascii=False))
        raise RuntimeError("REFUSE_EVAL_INCOHERENT_RUN")
    ds_group = DATASET_GROUP[a.group]
    # deterministic SHUFFLE maps: bijective no-self cross-group (P0-4); existing maps
    # are RE-VALIDATED and rebuilt if they fail the formal SHUFFLE gate.
    maps = {}
    for split, key in (("train", "shuffle_map_train.json"), ("val", "shuffle_map_val.json"),
                       ("all17", "shuffle_map_all17.json")):
        fp = run_dir / key
        ids = contract[f"{split}_ids"]
        if fp.exists():
            old = json.loads(fp.read_text(encoding="utf-8"))
            if not assert_valid_shuffle_map(old, ids):
                print(f"[warn] {key} failed the bijection gate -> rebuilding")
                fp.unlink()
        if not fp.exists():
            maps[split] = bijective_derangement(ids)
            assert assert_valid_shuffle_map(maps[split], ids), key
            fp.write_text(json.dumps(maps[split], indent=2), encoding="utf-8")
        else:
            maps[split] = json.loads(fp.read_text(encoding="utf-8"))

    results = {"schema": "step4-stock-validator-semantics-v1", "group": a.group,
               "aux_mode": GROUPS[a.group], "late10": late10(run_dir)}
    for ck_name in ("last.pt", "best.pt"):
        ck = torch.load(run_dir / "weights" / ck_name, map_location="cpu", weights_only=False)
        model = (ck.get("ema") or ck.get("model")).float().eval().to(device)
        results[ck_name] = {}
        for variant in ("NORMAL", "ZERO-AUX", "SHUFFLE"):
            results[ck_name][variant] = {}
            for split in ("train", "val", "all17"):
                ds = TriModalDataset(
                    contract, split=split, group=ds_group, augment=False,
                    aux_zero=(variant == "ZERO-AUX"),
                    aux_id_map=(maps[split] if variant == "SHUFFLE" else None))
                results[ck_name][variant][split] = evu.evaluate_dataset_stock_semantics(
                    model, ds, device, names)
        print(f"[{a.group}/{a.run_name}] {ck_name} NORMAL val mAP50-95="
              f"{results[ck_name]['NORMAL']['val']['map50_95']:.4f} "
              f"ZERO-AUX={results[ck_name]['ZERO-AUX']['val']['map50_95']:.4f} "
              f"SHUFFLE={results[ck_name]['SHUFFLE']['val']['map50_95']:.4f}")
    out = run_dir / "eval_step4_causality.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
