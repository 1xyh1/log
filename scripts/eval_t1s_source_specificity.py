#!/usr/bin/env python3
"""T1-S: exhaustive P5 FULL residual source-specificity audit for retrained T1-F."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import step3_eval_utils as evu  # noqa: E402
from multimodal.raw_sample_index import CLASS_NAMES, OUT_DEFAULT  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402
from multimodal.tseries_core import RUN_NAMES, sha256_file, tensor_sha256  # noqa: E402
from multimodal.tseries_runtime import (  # noqa: E402
    collect_detection_stats,
    load_checkpoint_model,
    metric_from_stats,
)
from multimodal.t1s_source_specificity import (  # noqa: E402
    EXPECTED_DERANGEMENTS,
    VAL6_IDS,
    decide_source_specificity,
    distribution_summary,
    fixed_donor_index,
    generate_derangements,
    is_derangement,
    mapping_dict,
    rank_and_percentile,
    verify_exact_derangement_family,
)

SCHEMA_MATRIX = "step4-t1s-source-matrix-v1"
SCHEMA_DERANGEMENTS = "step4-t1s-derangements-v1"
SCHEMA_SUMMARY = "step4-t1s-summary-v1"
AUDIT_SCHEMA = "step4-t1s-preexecution-audit-v1"

EXPECTED_T1_LAST_SHA256 = "8380e21504fabd0d8c3715398739bbb0bed5aaafd9c822dfc14c9503af2daeee"
EXPECTED_T1_MANIFEST_SHA256 = "081afec392d96ee2d570a3424e5f015f05ee308297daed8900ece5584c707312"
EXPECTED_DONOR_MAP_SHA256 = "c5cd8e852663eae3243bc5e4c263c6f2c26c7b2faa977ae3b60cb5c1ea122af5"
EXPECTED_POSTTRAIN_PERFORMANCE_SHA256 = "4b38f3ef8d7defac4b91b42e8651b667ca8320e990666af225dea4e6886ea93a"
EXPECTED_POSTTRAIN_PAIRED_SHA256 = "249702b5542055d7bf36595b28cb1e5a0b42312d05c55d459a8b0890e3a2b5cd"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def refuse_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        raise RuntimeError(f"T1S_REFUSE_OVERWRITE:{existing}")


def choose_device(arg: str) -> torch.device:
    arg = str(arg)
    if arg == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if arg.startswith("cuda:"):
        return torch.device(arg)
    return torch.device(f"cuda:{arg}")


def verify_formal_audit(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"T1S_PREEXECUTION_AUDIT_MISSING:{path}")
    obj = load_json(path)
    if obj.get("schema") != AUDIT_SCHEMA or obj.get("phase") != "formal":
        raise RuntimeError("T1S_PREEXECUTION_AUDIT_SCHEMA")
    if obj.get("all_passed") is not True:
        raise RuntimeError("T1S_PREEXECUTION_AUDIT_NOT_PASSING")
    gates = obj.get("gates") or {}
    if set(gates) != {f"G{i}" for i in range(1, 16)} or not all(gates.values()):
        raise RuntimeError(f"T1S_GATES_NOT_ALL_PASS:{gates}")
    pins = obj.get("source_hashes") or {}
    current = {
        "design_sha256": sha256_file(ROOT / "docs/step4_t1s/DESIGN_FREEZE.md"),
        "core_sha256": sha256_file(ROOT / "src/multimodal/t1s_source_specificity.py"),
        "evaluator_sha256": sha256_file(ROOT / "scripts/eval_t1s_source_specificity.py"),
        "audit_sha256": sha256_file(ROOT / "scripts/audit_t1s.py"),
        "tests_sha256": sha256_file(ROOT / "tests/test_t1s.py"),
        "readme_sha256": sha256_file(ROOT / "T1S_README.md"),
    }
    stale = {
        k: {"recorded": pins.get(k), "current": v}
        for k, v in current.items()
        if pins.get(k) != v
    }
    if stale:
        raise RuntimeError(f"T1S_PREEXECUTION_AUDIT_STALE:{stale}")
    return obj


def assert_metric_close(actual: dict, expected: dict, label: str, atol: float = 1e-15) -> None:
    for key in ("map50", "map50_95"):
        a = float(actual[key])
        e = float(expected[key])
        if not math.isclose(a, e, rel_tol=0.0, abs_tol=atol):
            raise RuntimeError(f"T1S_{label}_METRIC_MISMATCH:{key}:{a}!={e}")
    if int(actual["n_images"]) != int(expected["n_images"]):
        raise RuntimeError(f"T1S_{label}_N_IMAGES_MISMATCH")


def residual_rms(t: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(t.detach().float() ** 2)).item())


def build_residual_cache(model, dataset, device, paired_t1: dict) -> dict:
    cache = {}
    expected_native_trace = paired_t1["native"]["trace"]
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            sid = str(sample["sample_id"])
            batch = dataset.collate_fn([sample])
            batch = evu.move_step3_batch_to_device(batch, device)
            residual = model.p5_residual_from_input(batch["img"])
            residual_cpu = residual.detach().cpu().clone()
            sha = tensor_sha256(residual_cpu)
            expected_sha = expected_native_trace[sid]["residual_sha256"]
            if sha != expected_sha:
                raise RuntimeError(f"T1S_RESIDUAL_CACHE_SHA_FAIL:{sid}:{sha}!={expected_sha}")
            cache[sid] = {
                "tensor": residual_cpu,
                "source_id": sid,
                "sha256": sha,
                "shape": list(residual_cpu.shape),
                "rms": residual_rms(residual_cpu),
                "spatial_channel_mean_abs_max": float(
                    residual_cpu.float().mean(dim=(-2, -1)).abs().max().item()
                ),
            }
    if set(cache) != set(VAL6_IDS):
        raise RuntimeError("T1S_RESIDUAL_CACHE_ID_FAIL")
    return cache


def clean_cache(cache: dict) -> dict:
    return {
        sid: {k: v for k, v in entry.items() if k != "tensor"}
        for sid, entry in cache.items()
    }


def assemble_metric(matrix_stats: dict, mapping: dict, ids: list[str]) -> dict:
    stats = {rid: matrix_stats[rid][str(mapping[rid])] for rid in ids}
    names = {int(k): v for k, v in CLASS_NAMES.items()}
    return metric_from_stats(stats, ids, names)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="runs/step4_tseries")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--donor-map", default="reports/step4_a2/val_donor_map.json")
    p.add_argument("--performance", default="reports/step4_tseries/posttrain_performance.json")
    p.add_argument("--paired", default="reports/step4_tseries/posttrain_paired.json")
    p.add_argument("--audit", default="reports/step4_t1s/preexecution_audit.json")
    p.add_argument("--device", default="0")
    p.add_argument("--out-dir", default="reports/step4_t1s")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    audit = verify_formal_audit(ROOT / a.audit)
    out_dir = ROOT / a.out_dir
    matrix_path = out_dir / "source_matrix.json"
    der_path = out_dir / "derangements.json"
    summary_path = out_dir / "t1s_summary.json"
    refuse_outputs([matrix_path, der_path, summary_path], a.overwrite)

    performance_path = ROOT / a.performance
    paired_path = ROOT / a.paired
    donor_path = ROOT / a.donor_map
    if sha256_file(performance_path) != EXPECTED_POSTTRAIN_PERFORMANCE_SHA256:
        raise RuntimeError("T1S_POSTTRAIN_PERFORMANCE_SHA_DRIFT")
    if sha256_file(paired_path) != EXPECTED_POSTTRAIN_PAIRED_SHA256:
        raise RuntimeError("T1S_POSTTRAIN_PAIRED_SHA_DRIFT")
    if sha256_file(donor_path) != EXPECTED_DONOR_MAP_SHA256:
        raise RuntimeError("T1S_DONOR_MAP_SHA_DRIFT")

    performance = load_json(performance_path)
    paired = load_json(paired_path)
    donor_map = {str(k): str(v) for k, v in load_json(donor_path).items()}
    contract = load_json(Path(a.contract))

    ids = [str(x) for x in contract["val_ids"]]
    if tuple(ids) != VAL6_IDS:
        raise RuntimeError(f"T1S_VAL6_ID_OR_ORDER_DRIFT:{ids}")
    if not is_derangement(donor_map, ids):
        raise RuntimeError("T1S_FROZEN_DONOR_NOT_DERANGEMENT")

    t1_run = (ROOT / a.project / RUN_NAMES["T1-F"]).resolve()
    ckpt_path = t1_run / "weights/last.pt"
    manifest_path = t1_run / "manifest.json"
    if sha256_file(ckpt_path) != EXPECTED_T1_LAST_SHA256:
        raise RuntimeError("T1S_T1_LAST_SHA_DRIFT")
    if sha256_file(manifest_path) != EXPECTED_T1_MANIFEST_SHA256:
        raise RuntimeError("T1S_T1_MANIFEST_SHA_DRIFT")

    device = choose_device(a.device)
    model, _ = load_checkpoint_model(ckpt_path, device)
    if model.treatment_id != "T1-F":
        raise RuntimeError(f"T1S_MODEL_TREATMENT_MISMATCH:{model.treatment_id}")

    dataset = TriModalDataset(contract, split="val", group="C1-I", augment=False)
    if [str(x) for x in dataset.ids] != ids:
        raise RuntimeError("T1S_DATASET_VAL6_DRIFT")

    paired_t1 = paired["systems"]["T1-F"]
    cache = build_residual_cache(model, dataset, device, paired_t1)

    # Normal native checkpoint forward: direct runtime anchor.
    normal = collect_detection_stats(model, dataset, device)

    matrix_stats = {rid: {} for rid in ids}
    matrix_trace = {rid: {} for rid in ids}

    # Six fixed-source columns => exactly 36 recipient/source cells.
    for source_id in ids:
        entry = cache[source_id]

        def fixed_source_forward(sid, sample, batch, source_id=source_id, entry=entry):
            residual = entry["tensor"].to(device=device)
            output = model.predict_with_p5_residual(batch["img"], residual)
            return output, {
                "recipient_id": sid,
                "source_id": source_id,
                "residual_sha256": entry["sha256"],
                "role": "matrix_cell",
            }

        col = collect_detection_stats(model, dataset, device, fixed_source_forward)
        for rid in ids:
            matrix_stats[rid][source_id] = col["_stats"][rid]
            matrix_trace[rid][source_id] = col["trace"][rid]

    accepted_native_trace = paired_t1["native"]["trace"]
    for rid in ids:
        diag_sha = matrix_trace[rid][rid]["detection_sha256"]
        direct_sha = normal["trace"][rid]["detection_sha256"]
        accepted_sha = accepted_native_trace[rid]["detection_sha256"]
        if not (diag_sha == direct_sha == accepted_sha):
            raise RuntimeError(
                f"T1S_NATIVE_ANCHOR_FAIL:{rid}:{diag_sha}:{direct_sha}:{accepted_sha}"
            )

    identity_map = {rid: rid for rid in ids}
    identity_metric = assemble_metric(matrix_stats, identity_map, ids)
    assert_metric_close(identity_metric, normal["full"], "NATIVE_DIRECT")
    assert_metric_close(identity_metric, paired_t1["native"]["full"], "NATIVE_ACCEPTED")

    accepted_donor_trace = paired_t1["donor"]["trace"]
    for rid in ids:
        source_id = donor_map[rid]
        got_sha = matrix_trace[rid][source_id]["detection_sha256"]
        expected_sha = accepted_donor_trace[rid]["detection_sha256"]
        if got_sha != expected_sha:
            raise RuntimeError(
                f"T1S_FIXED_DONOR_ANCHOR_FAIL:{rid}:{source_id}:{got_sha}!={expected_sha}"
            )
    fixed_donor_metric = assemble_metric(matrix_stats, donor_map, ids)
    assert_metric_close(fixed_donor_metric, paired_t1["donor"]["full"], "FIXED_DONOR_ACCEPTED")

    # ZERO is an inference-time residual ablation inside the T1 checkpoint.
    def zero_forward(sid, sample, batch):
        z = torch.zeros_like(cache[sid]["tensor"]).to(device=device)
        output = model.predict_with_p5_residual(batch["img"], z)
        return output, {
            "recipient_id": sid,
            "source_id": "ZERO",
            "residual_sha256": tensor_sha256(z.detach().cpu()),
            "role": "zero_residual",
        }

    zero = collect_detection_stats(model, dataset, device, zero_forward)

    family = verify_exact_derangement_family(ids)
    if not family["passed"]:
        raise RuntimeError(f"T1S_DERANGEMENT_FAMILY_FAIL:{family}")
    ders = generate_derangements(ids)
    donor_idx = fixed_donor_index(donor_map, ders, ids)

    der_rows = []
    for idx, perm in enumerate(ders):
        mapping = mapping_dict(ids, perm)
        metric = assemble_metric(matrix_stats, mapping, ids)
        der_rows.append({
            "index": idx,
            "mapping": mapping,
            "map50": float(metric["map50"]),
            "map50_95": float(metric["map50_95"]),
            "is_frozen_a2_donor_map": idx == donor_idx,
        })

    if len(der_rows) != EXPECTED_DERANGEMENTS:
        raise RuntimeError("T1S_DERANGEMENT_COUNT_FAIL")
    frozen_row = der_rows[donor_idx]
    if not math.isclose(
        float(frozen_row["map50_95"]), float(fixed_donor_metric["map50_95"]),
        rel_tol=0.0, abs_tol=1e-15,
    ):
        raise RuntimeError("T1S_FROZEN_DONOR_DERANGEMENT_METRIC_FAIL")

    d95 = [r["map50_95"] for r in der_rows]
    d50 = [r["map50"] for r in der_rows]
    i95 = float(identity_metric["map50_95"])
    z95 = float(zero["full"]["map50_95"])
    f95 = float(fixed_donor_metric["map50_95"])

    decision = decide_source_specificity(i95, z95, d95)

    matrix_report = {
        "schema": SCHEMA_MATRIX,
        "authority": "T1-F last.pt; Step3 validator; 36 exact recipient/source cells",
        "val_ids": ids,
        "checkpoint": {
            "path": str(ckpt_path.relative_to(ROOT)),
            "sha256": sha256_file(ckpt_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "residual_definition": "P5_FULL_POST_PROJECTION",
        "residual_cache": clean_cache(cache),
        "cells": matrix_trace,
        "zero_trace": zero["trace"],
        "anchors": {
            "native_diagonal_bitwise_per_sample": True,
            "fixed_a2_donor_bitwise_per_sample": True,
            "native_metric": identity_metric,
            "fixed_donor_metric": fixed_donor_metric,
            "zero_metric": zero["full"],
        },
        "provenance": {
            "preexecution_audit_sha256": sha256_file(ROOT / a.audit),
            "posttrain_performance_sha256": sha256_file(performance_path),
            "posttrain_paired_sha256": sha256_file(paired_path),
            "donor_map_sha256": sha256_file(donor_path),
            "evaluator_sha256": sha256_file(ROOT / "scripts/eval_t1s_source_specificity.py"),
        },
    }

    der_report = {
        "schema": SCHEMA_DERANGEMENTS,
        "val_ids": ids,
        "exact_family": family,
        "fixed_a2_donor_index": donor_idx,
        "rows": der_rows,
        "distribution_map50_95": distribution_summary(d95),
        "distribution_map50": distribution_summary(d50),
    }

    summary = {
        "schema": SCHEMA_SUMMARY,
        "evaluation_only": True,
        "t1_architecture_gain_upstream": performance["systems"]["T1-F"]["last_val6"]["full"],
        "identity_native": identity_metric,
        "zero_residual": zero["full"],
        "frozen_a2_donor": fixed_donor_metric,
        "derangements": {
            "count": len(der_rows),
            "map50_95": distribution_summary(d95),
            "map50": distribution_summary(d50),
        },
        "ranks": {
            "identity_vs_derangements": rank_and_percentile(i95, d95),
            "frozen_a2_donor_vs_derangements": rank_and_percentile(f95, d95),
            "zero_vs_derangements": rank_and_percentile(z95, d95),
        },
        "contrasts": {
            "identity_minus_zero_map50_95": i95 - z95,
            "identity_minus_derangement_median_map50_95": i95 - float(np.median(d95)),
            "derangement_median_minus_zero_map50_95": float(np.median(d95)) - z95,
        },
        "decision": decision,
        "training_go": False,
        "depth_go": False,
        "production_go": False,
        "interpretation_discipline": {
            "fixed_donor_is_one_of_265_not_the_only_causal_test": True,
            "zero_is_t1_checkpoint_inference_ablation_not_t0_checkpoint": True,
            "architecture_gain_is_not_equated_with_paired_multimodal_gain": True,
            "single_seed_source_specificity_is_not_replication": True,
            "no_new_training_performed": True,
        },
        "provenance": {
            "matrix_sha_pending_until_write": True,
            "derangements_sha_pending_until_write": True,
            "preexecution_audit_sha256": sha256_file(ROOT / a.audit),
            "t1_last_pt_sha256": sha256_file(ckpt_path),
            "t1_manifest_sha256": sha256_file(manifest_path),
            "posttrain_paired_sha256": sha256_file(paired_path),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(json.dumps(matrix_report, indent=2, ensure_ascii=False), encoding="utf-8")
    der_path.write_text(json.dumps(der_report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["provenance"]["matrix_sha256"] = sha256_file(matrix_path)
    summary["provenance"]["derangements_sha256"] = sha256_file(der_path)
    summary["provenance"].pop("matrix_sha_pending_until_write", None)
    summary["provenance"].pop("derangements_sha_pending_until_write", None)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "schema": SCHEMA_SUMMARY,
        "identity_map50_95": i95,
        "zero_map50_95": z95,
        "frozen_donor_map50_95": f95,
        "derangements": len(der_rows),
        "decision": decision,
        "outputs": {
            "matrix": str(matrix_path),
            "derangements": str(der_path),
            "summary": str(summary_path),
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
