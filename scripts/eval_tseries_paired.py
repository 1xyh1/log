#!/usr/bin/env python3
"""Post-training P5 recipient-vs-donor paired causality for T1-F/T2-A."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import step3_eval_utils as evu  # noqa: E402
from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402
from multimodal.tseries_core import (  # noqa: E402
    RUN_NAMES, effect_from_results, sha256_file, single_seed_paired_label, tensor_sha256,
)
from multimodal.tseries_runtime import collect_detection_stats, load_checkpoint_model  # noqa: E402

SCHEMA = "step4-tseries-posttrain-paired-v1"
EXPECTED_DONOR_MAP_SHA256 = "c5cd8e852663eae3243bc5e4c263c6f2c26c7b2faa977ae3b60cb5c1ea122af5"

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def clean_result(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "_stats"}

def build_residual_cache(model, dataset, device) -> dict:
    cache = {}
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            sid = str(sample["sample_id"])
            batch = dataset.collate_fn([sample])
            batch = evu.move_step3_batch_to_device(batch, device)
            residual = model.p5_residual_from_input(batch["img"])
            residual_cpu = residual.detach().cpu().clone()
            cache[sid] = {
                "tensor": residual_cpu,
                "source_id": sid,
                "sha256": tensor_sha256(residual_cpu),
                "post_center_channel_mean_abs_max": float(
                    residual_cpu.float().mean(dim=(-2, -1)).abs().max().item()
                ),
            }
    return cache

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_tseries")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--donor-map", default="reports/step4_a2/val_donor_map.json")
    p.add_argument("--device", default="0")
    p.add_argument("--out", default="reports/step4_tseries/posttrain_paired.json")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    out = ROOT / a.out
    if out.exists() and not a.overwrite:
        raise RuntimeError(f"T_SERIES_REFUSE_OVERWRITE:{out}")

    contract = load_json(Path(a.contract))
    donor_path = ROOT / a.donor_map
    if sha256_file(donor_path) != EXPECTED_DONOR_MAP_SHA256:
        raise RuntimeError("T_SERIES_DONOR_MAP_SHA_DRIFT")
    donor_map = load_json(donor_path)
    expected_val = [str(x) for x in contract["val_ids"]]
    if set(donor_map) != set(expected_val):
        raise RuntimeError("T_SERIES_DONOR_MAP_ID_DRIFT")
    if any(str(k) == str(v) for k, v in donor_map.items()):
        raise RuntimeError("T_SERIES_DONOR_SELF_MATCH")

    devarg = str(a.device)
    if devarg == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    elif devarg.startswith("cuda:"):
        device = torch.device(devarg)
    else:
        device = torch.device(f"cuda:{devarg}")

    dataset = TriModalDataset(contract, split="val", group="C1-I", augment=False)
    if list(dataset.ids) != expected_val:
        raise RuntimeError("T_SERIES_VAL6_DRIFT")

    systems = {}
    for treatment in ("T1-F", "T2-A"):
        run_dir = (ROOT / a.project / RUN_NAMES[treatment]).resolve()
        ckpt_path = run_dir / "weights/last.pt"
        manifest_path = run_dir / "manifest.json"
        model, _ = load_checkpoint_model(ckpt_path, device)
        if model.treatment_id != treatment:
            raise RuntimeError("T_SERIES_PAIRED_TREATMENT_MISMATCH")
        cache = build_residual_cache(model, dataset, device)

        def native_forward(sid, sample, batch):
            entry = cache[sid]
            residual = entry["tensor"].to(device=device)
            normal = model._predict_once(batch["img"])
            override = model.predict_with_p5_residual(batch["img"], residual)
            nraw = evu.extract_detection_tensor(normal).detach()
            oraw = evu.extract_detection_tensor(override).detach()
            if not torch.equal(nraw, oraw):
                raise RuntimeError(f"T_SERIES_NATIVE_OVERRIDE_EQUIVALENCE_FAIL:{treatment}:{sid}")
            return override, {
                "role": "recipient",
                "recipient_id": sid,
                "residual_source_id": sid,
                "residual_definition": "FULL" if treatment == "T1-F" else "AC_ALL",
                "residual_sha256": entry["sha256"],
                "post_center_channel_mean_abs_max": entry["post_center_channel_mean_abs_max"],
                "native_raw_sha256": tensor_sha256(nraw),
                "override_raw_sha256": tensor_sha256(oraw),
                "bitwise_native_override_equal": True,
            }

        def donor_forward(sid, sample, batch):
            donor = str(donor_map[sid])
            entry = cache[donor]
            if entry["source_id"] != donor:
                raise RuntimeError(f"T_SERIES_DONOR_CACHE_SOURCE_FAIL:{treatment}:{sid}")
            residual = entry["tensor"].to(device=device)
            output = model.predict_with_p5_residual(batch["img"], residual)
            return output, {
                "role": "donor",
                "recipient_id": sid,
                "residual_source_id": donor,
                "donor_id": donor,
                "residual_definition": "FULL" if treatment == "T1-F" else "AC_ALL",
                "residual_sha256": entry["sha256"],
                "post_center_channel_mean_abs_max": entry["post_center_channel_mean_abs_max"],
                "donor_self_centered": bool(
                    treatment != "T2-A"
                    or entry["post_center_channel_mean_abs_max"] <= 1e-6
                ),
            }

        native = collect_detection_stats(model, dataset, device, native_forward)
        donor = collect_detection_stats(model, dataset, device, donor_forward)
        effect = effect_from_results(native, donor)
        label = single_seed_paired_label(effect)

        # Runtime provenance: every donor source must be the frozen paired donor.
        for sid, tr in donor["trace"].items():
            if tr["residual_source_id"] != str(donor_map[sid]):
                raise RuntimeError(f"T_SERIES_DONOR_SOURCE_FAIL:{treatment}:{sid}")
            if treatment == "T2-A" and not tr.get("donor_self_centered"):
                raise RuntimeError(f"T_SERIES_DONOR_AC_SELF_CENTER_FAIL:{sid}")
        for sid, tr in native["trace"].items():
            if not tr.get("bitwise_native_override_equal"):
                raise RuntimeError(f"T_SERIES_NATIVE_ANCHOR_FAIL:{treatment}:{sid}")

        systems[treatment] = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "last_pt_sha256": sha256_file(ckpt_path),
            "manifest_sha256": sha256_file(manifest_path),
            "residual_definition": "P5_FULL" if treatment == "T1-F" else "P5_AC_ALL_OWN_MEAN",
            "native": clean_result(native),
            "donor": clean_result(donor),
            "paired_effect_native_minus_donor": effect,
            "single_seed_label": label,
            "native_override_equivalence": "bitwise_per_sample",
        }

    report = {
        "schema": SCHEMA,
        "authority": "single-seed retrained paired causality; not multi-seed replication",
        "donor_map_sha256": EXPECTED_DONOR_MAP_SHA256,
        "val_ids": expected_val,
        "systems": systems,
        "provenance": {
            "evaluator_sha256": sha256_file(ROOT / "scripts/eval_tseries_paired.py"),
            "model_source_sha256": sha256_file(ROOT / "src/multimodal/tseries_p5_model.py"),
            "core_source_sha256": sha256_file(ROOT / "src/multimodal/tseries_core.py"),
        },
        "interpretation_discipline": {
            "performance_gain_is_not_paired_causality": True,
            "single_seed_is_not_replication": True,
            "t2_donor_ac_uses_donor_own_projected_residual_and_own_spatial_mean": True,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"schema": SCHEMA, "out": str(out)}, indent=2))

if __name__ == "__main__":
    main()
