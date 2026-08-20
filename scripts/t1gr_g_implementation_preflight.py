#!/usr/bin/env python3
"""Real-data and model preflight that alone authorizes the nine smoke runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_e5_core import (  # noqa: E402
    FROZEN_E5_SECURITY_POLICY_SHA256,
    SCHEMA_RECIPE,
    compare_environment,
    environment_probe,
    payload_ok as e5_payload_ok,
)
from multimodal.t1gr_g_core import ARMS, SCHEMA_DESIGN_AUDIT, SCHEMA_RUN_PLAN, SEEDS, payload_ok  # noqa: E402
from multimodal.t1gr_g_dataset import (  # noqa: E402
    T1GRDataset,
    scoped_rng,
    set_albumentations_seed,
    transform_graph,
)
from multimodal.t1gr_g_impl_core import SCHEMA_PREFLIGHT, payload_sha256, validate_impl_spec  # noqa: E402
from multimodal.t1gr_g_model import assert_same_seed_arm_identity, build_t1gr_g_model  # noqa: E402
from multimodal.t1gr_g_runtime import (  # noqa: E402
    implementation_source_hashes,
    read_json,
    validate_frozen_chain,
    verify_multimodal_view,
)
from multimodal.t1gr_secure_io import (  # noqa: E402
    Deadline,
    assert_public_safe,
    atomic_json_write,
    ensure_private_input,
    ensure_public_output,
    ensure_repo_input,
    fail,
    file_lock,
    read_json_bounded,
    safe_error_message,
    sha256_file,
    sha256_json,
)

SCRIPT_VERSION = "t1gr-g-implementation-preflight-v1"


def _dataset_kwargs(cfg, data, img_path, batch: int, *, augment: bool) -> dict:
    return {
        "img_path": img_path,
        "imgsz": cfg.imgsz,
        "batch_size": batch,
        "augment": augment,
        "hyp": cfg,
        "rect": False,
        "cache": None,
        "single_cls": False,
        "stride": 32,
        "pad": 0.0,
        "prefix": "preflight: ",
        "task": "detect",
        "classes": cfg.classes,
        "data": data,
        "fraction": 1.0,
    }


def _equal_value(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_equal_value(a, b) for a, b in zip(left, right))
    return left == right


def _labels_equal(left: dict, right: dict) -> bool:
    ignored = {"img", "source_pairs"}
    keys_left = set(left) - ignored
    keys_right = set(right) - ignored
    if keys_left != keys_right:
        return False
    return all(_equal_value(left[key], right[key]) for key in keys_left)


def _build_probe_datasets(view, recipe, seed: int):
    try:
        from ultralytics.cfg import DEFAULT_CFG, get_cfg
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.data.utils import check_det_dataset
    except Exception as exc:
        raise RuntimeError("T1GR_G_PREFLIGHT_DATASET_IMPORT_FAIL") from exc
    overrides = dict(recipe["train_args"])
    overrides.update({"task": "detect", "mode": "train", "data": str(view["dataset_yaml"]), "model": "yolo26s.yaml"})
    cfg = get_cfg(DEFAULT_CFG, overrides=overrides)
    data4 = check_det_dataset(str(view["dataset_yaml"]))
    data4["channels"] = 4
    data3 = dict(data4)
    data3["channels"] = 3
    common = _dataset_kwargs(cfg, data3, data4["train"], int(recipe["train_args"]["batch"]), augment=True)
    stock = YOLODataset(**common)
    arms = {}
    for arm in ARMS:
        kwargs = _dataset_kwargs(cfg, data4, data4["train"], int(recipe["train_args"]["batch"]), augment=True)
        arms[arm] = T1GRDataset(
            **kwargs,
            ir_by_sid=view["ir_maps"]["train"],
            arm=arm,
            seed=int(seed),
            split="train",
        )
        arms[arm].set_epoch(0)
    dev_kwargs = _dataset_kwargs(cfg, data4, data4["val"], int(recipe["train_args"]["batch"]), augment=False)
    dev = T1GRDataset(
        **dev_kwargs,
        ir_by_sid=view["ir_maps"]["dev"],
        arm="G1-P",
        seed=int(seed),
        split="dev",
    )
    return stock, arms, dev


def _data_probe(view, recipe) -> dict:
    stock, arms, dev = _build_probe_datasets(view, recipe, SEEDS[0])
    n = len(stock)
    indexes = (0, n // 2, n - 1)
    probe_rows = []
    for probe_number, index in enumerate(indexes):
        paired = arms["G1-P"]
        set_albumentations_seed(stock.transforms, paired.draw_seed(index))
        with scoped_rng(paired.draw_seed(index)):
            stock_item = stock[index]
        arm_items = {arm: dataset[index] for arm, dataset in arms.items()}
        stock_parity = bool(
            torch.equal(stock_item["img"], arm_items["G1-P"]["img"][:3])
            and _labels_equal(stock_item, arm_items["G1-P"])
        )
        visible_identity = len({
            sha256_json({
                "bytes": item["img"][:3].contiguous().numpy().tobytes().hex(),
                "shape": list(item["img"][:3].shape),
            })
            for item in arm_items.values()
        }) == 1
        label_identity = all(_labels_equal(arm_items["G0-N"], item) for item in arm_items.values())
        g0_pair = arm_items["G0-N"]["source_pairs"][0]
        g1_pair = arm_items["G1-P"]["source_pairs"][0]
        g2_pair = arm_items["G2-S"]["source_pairs"][0]
        source_only = bool(
            g0_pair["recipient"] == g0_pair["donor"]
            and g1_pair["recipient"] == g1_pair["donor"]
            and g2_pair["recipient"] != g2_pair["donor"]
            and g0_pair["recipient"] == g1_pair["recipient"] == g2_pair["recipient"]
        )
        channels_ok = all(tuple(item["img"].shape[:1]) == (4,) for item in arm_items.values())
        passed = stock_parity and visible_identity and label_identity and source_only and channels_ok
        probe_rows.append({
            "probe_number": probe_number,
            "stock_e5_visible_and_label_bitwise_parity": stock_parity,
            "same_seed_cross_arm_visible_bitwise_identity": visible_identity,
            "same_seed_cross_arm_label_identity": label_identity,
            "g2_changes_only_source_identity_before_geometry": source_only,
            "four_channel_output": channels_ok,
            "passed": passed,
        })
    dev_item = dev[0]
    dev_zero_ir = bool(torch.count_nonzero(dev_item["img"][3]).item() == 0)
    graph = transform_graph(arms["G1-P"].transforms)
    dev_graph = transform_graph(dev.transforms)
    graph_checks = {
        "mosaic_wrapper": "T1GRMosaic" in graph,
        "perspective_wrapper": "T1GRRandomPerspective" in graph,
        "visible_hsv_wrapper": "T1GRVisibleHSV" in graph,
        "visible_albumentations_wrapper": "T1GRVisibleAlbumentations" in graph,
        "format_wrapper": "T1GRFormat" in graph,
        "zero_ir_letterbox_wrapper": "T1GRLetterBox" in dev_graph,
    }
    passed = all(row["passed"] for row in probe_rows) and all(graph_checks.values()) and dev_zero_ir
    return {
        "probe_count": len(probe_rows),
        "probes": probe_rows,
        "transform_graph_commitment": sha256_json(graph),
        "transform_graph_checks": graph_checks,
        "dev_zero_ir_after_letterbox_and_format": dev_zero_ir,
        "passed": passed,
    }


def _model_probe(checkpoint: Path, recipe: dict) -> dict:
    seed_rows = []
    treatment = {}
    for seed in SEEDS:
        identities = []
        for arm in ARMS:
            _, identity = build_t1gr_g_model(checkpoint, recipe, arm=arm, seed=seed)
            identities.append(identity)
            treatment[arm] = identity["model_treatment_id"]
        compact = assert_same_seed_arm_identity(identities)
        compact["identity_field_count"] = 12
        seed_rows.append(compact)
    treatment_ok = treatment == {"G0-N": "T0-N", "G1-P": "T1-F", "G2-S": "T1-F"}
    return {
        "seed_identity_rows": seed_rows,
        "treatment_mapping": treatment,
        "same_seed_all_arm_identity": all(row["all_identity_fields_equal"] for row in seed_rows),
        "g0_null_g1_g2_full": treatment_ok,
        "initialization_claim": "AUDITABLE_INITIALIZATION_ONLY",
        "numerical_repeatability_claimed": False,
        "passed": all(row["all_identity_fields_equal"] for row in seed_rows) and treatment_ok,
    }


def run(args) -> dict:
    repo = ROOT.resolve(strict=True)
    security_path = ensure_repo_input(repo, "config/t1gr_e5_security_policy.json", "config")
    if sha256_file(security_path) != FROZEN_E5_SECURITY_POLICY_SHA256:
        fail("E5_SECURITY_POLICY_SHA_DRIFT")
    security = read_json_bounded(security_path, 1 << 20, "t1gr-e5-security-policy-v2")
    recipe_path = ensure_repo_input(repo, "reports/step4_t1gr/e5_v2_step1_recipe_public.json", "reports/step4_t1gr")
    design_path = ensure_repo_input(repo, "config/t1gr_g_design.frozen.json", "config")
    spec_path = ensure_repo_input(repo, "config/t1gr_g_implementation_spec.frozen.json", "config")
    audit_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_g_design_audit_public.json", "reports/step4_t1gr")
    plan_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_g_run_plan_public.json", "reports/step4_t1gr")
    view_public_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_g_multimodal_view_public.json", "reports/step4_t1gr")
    view_manifest_path = ensure_private_input(Path(args.view_manifest), repo)
    checkpoint = Path(args.base_checkpoint).expanduser().resolve(strict=False)
    if not checkpoint.is_file():
        fail("BASE_CHECKPOINT_NOT_FOUND")
    out = ensure_public_output(
        repo, "reports/step4_t1gr/t1gr_g_implementation_preflight_public.json", security["public_output_prefix"]
    )
    deadline = Deadline(float(args.timeout_seconds or security["view_verify_timeout_seconds"]))
    with file_lock(out.with_suffix(out.suffix + ".lock"), 5.0, 900.0):
        recipe = read_json_bounded(recipe_path, int(security["max_public_json_bytes"]), SCHEMA_RECIPE)
        design = read_json(design_path)
        spec = read_json(spec_path)
        audit = read_json(audit_path)
        plan = read_json(plan_path)
        view_public = read_json(view_public_path)
        validate_impl_spec(spec)
        chain = validate_frozen_chain(design, recipe, spec)
        if sha256_file(design_path, deadline) != spec["upstream"]["design_file_sha256"]:
            fail("T1GR_G_DESIGN_FILE_SHA_DRIFT")
        if not e5_payload_ok(recipe):
            fail("T1GR_G_RECIPE_INTEGRITY_FAIL")
        if (
            audit.get("schema") != SCHEMA_DESIGN_AUDIT
            or not payload_ok(audit)
            or audit.get("design_freeze_passed") is not True
            or audit.get("implementation_entry_authorized") is not True
        ):
            fail("T1GR_G_DESIGN_AUDIT_NOT_PASS")
        if (
            plan.get("schema") != SCHEMA_RUN_PLAN
            or not payload_ok(plan)
            or plan.get("n_runs") != 9
            or plan.get("order_frozen") is not True
            or plan.get("multiseed_training_authorized") is not False
        ):
            fail("T1GR_G_RUN_PLAN_NOT_FROZEN")
        if view_public.get("schema") != "t1gr-g-multimodal-view-public-v1" or not e5_payload_ok(view_public):
            fail("T1GR_G_VIEW_PUBLIC_INTEGRITY_FAIL")
        if view_public.get("view_gate_passed") is not True:
            fail("T1GR_G_VIEW_GATE_NOT_PASS")
        if sha256_file(view_manifest_path, deadline) != view_public.get("view_manifest_private_sha256"):
            fail("T1GR_G_VIEW_PRIVATE_SHA_DRIFT")
        view = verify_multimodal_view(view_manifest_path, recipe, deadline=deadline)
        checkpoint_sha = sha256_file(checkpoint, deadline)
        if checkpoint_sha != recipe["base_checkpoint_sha256"]:
            fail("T1GR_G_CHECKPOINT_SHA_DRIFT")
        environment = environment_probe()
        compare_environment(environment, recipe["environment"])
        data_probe = _data_probe(view, recipe)
        if not data_probe["passed"]:
            fail("T1GR_G_DATA_PROBE_FAIL")
        model_probe = _model_probe(checkpoint, recipe)
        if not model_probe["passed"]:
            fail("T1GR_G_MODEL_PROBE_FAIL")
        source_hashes = implementation_source_hashes(repo)
        checks = {
            "frozen_chain": all(chain["checks"].values()),
            "design_audit": True,
            "nine_run_plan": True,
            "multimodal_view": True,
            "environment": True,
            "checkpoint": True,
            "recipient_keyed_joint_augmentation": data_probe["passed"],
            "same_seed_model_identity": model_probe["passed"],
            "source_set_complete": len(source_hashes) == 14,
            "holdout_unavailable": True,
        }
        passed = all(checks.values())
        request_fingerprint = sha256_json({
            "script": SCRIPT_VERSION,
            "recipe": sha256_file(recipe_path, deadline),
            "design": sha256_file(design_path, deadline),
            "spec": sha256_file(spec_path, deadline),
            "audit": sha256_file(audit_path, deadline),
            "plan": sha256_file(plan_path, deadline),
            "view_public": sha256_file(view_public_path, deadline),
            "view_private": sha256_file(view_manifest_path, deadline),
            "checkpoint": checkpoint_sha,
            "environment": environment,
            "sources": source_hashes,
        })
        report = {
            "schema": SCHEMA_PREFLIGHT,
            "script_version": SCRIPT_VERSION,
            "recipe_public_sha256": sha256_file(recipe_path, deadline),
            "design_file_sha256": sha256_file(design_path, deadline),
            "implementation_spec_sha256": sha256_file(spec_path, deadline),
            "design_audit_sha256": sha256_file(audit_path, deadline),
            "run_plan_sha256": sha256_file(plan_path, deadline),
            "view_public_sha256": sha256_file(view_public_path, deadline),
            "view_manifest_private_sha256": sha256_file(view_manifest_path, deadline),
            "base_checkpoint_sha256": checkpoint_sha,
            "environment": environment,
            "checks": checks,
            "data_probe": data_probe,
            "model_probe": model_probe,
            "implementation_source_hashes": source_hashes,
            "passed_count": sum(bool(value) for value in checks.values()),
            "total_count": len(checks),
            "preflight_gate_passed": passed,
            "smoke_training_authorized": passed,
            "multiseed_training_authorized": False,
            "execution_authorized": False,
            "final_holdout_ids_available_to_preflight": False,
            "final_holdout_open_authorized": False,
            "depth_go": False,
            "production_go": False,
            "next_action": "run the frozen nine one-epoch smoke suite",
        }
        assert_public_safe(report)
        digest, reused = atomic_json_write(out, report, private=False, request_fingerprint=request_fingerprint)
        if not passed:
            fail("T1GR_G_PREFLIGHT_GATE_FAIL")
        return {
            "status": "PASS",
            "idempotent_reuse": reused,
            "public_output_sha256": digest,
            "smoke_training_authorized": True,
            "multiseed_training_authorized": False,
            "final_holdout_open_authorized": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print(json.dumps({"status": "FAIL", "error": "USER_INTERRUPT"}), file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": safe_error_message(exc)}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
