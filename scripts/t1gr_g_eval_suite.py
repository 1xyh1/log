#!/usr/bin/env python3
"""Evaluate all formal last.pt checkpoints on ZERO-IR DEV and five component LOFO subsets."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_e5_core import (  # noqa: E402
    FROZEN_E5_SECURITY_POLICY_SHA256,
    SCHEMA_RECIPE,
    compare_environment,
    environment_probe,
    payload_ok as e5_payload_ok,
    private_umask,
    ultralytics_offline_guard,
    wall_clock_watchdog,
    write_private_failure_report,
)
from multimodal.t1gr_g_core import (  # noqa: E402
    SCHEMA_RESULTS,
    balanced_component_folds,
    payload_sha256,
)
from multimodal.t1gr_g_dataset import T1GRDataset  # noqa: E402
from multimodal.t1gr_g_impl_core import SCHEMA_EVAL, finite_metric, parse_component_map  # noqa: E402
from multimodal.t1gr_g_model import T1GRP5Model  # noqa: E402
from multimodal.t1gr_g_runtime import (  # noqa: E402
    build_epoch_fresh_dataloader,
    frozen_launch_rows,
    implementation_source_hashes,
    read_json,
    run_name,
    run_report_rel,
    verify_multimodal_view,
)
from multimodal.t1gr_secure_io import (  # noqa: E402
    Deadline,
    assert_public_safe,
    atomic_json_write,
    check_existing_output,
    ensure_private_input,
    ensure_public_output,
    ensure_repo_input,
    fail,
    file_lock,
    is_within,
    read_json_bounded,
    safe_error_message,
    sha256_file,
    sha256_json,
)

SCRIPT_VERSION = "t1gr-g-zero-ir-dev-eval-suite-v1"


def _make_dataset(cfg, data: dict, img_path: str, view: dict, arm: str, seed: int) -> T1GRDataset:
    dataset = T1GRDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=cfg.batch,
        augment=False,
        hyp=cfg,
        rect=True,
        cache=None,
        single_cls=False,
        stride=32,
        pad=0.5,
        prefix="T1GR DEV: ",
        task="detect",
        classes=cfg.classes,
        data=data,
        fraction=1.0,
        ir_by_sid=view["ir_maps"]["dev"],
        arm=arm,
        seed=seed,
        split="dev",
    )
    return dataset


def _evaluate(model, *, cfg, data: dict, img_path: str, view: dict, arm: str, seed: int, save_dir: Path, timeout: float) -> dict:
    from ultralytics.models.yolo.detect.val import DetectionValidator

    dataset = _make_dataset(cfg, data, img_path, view, arm, seed)
    if torch.count_nonzero(dataset[0]["img"][3]).item() != 0:
        fail("T1GR_G_EVAL_ZERO_IR_PREPROCESS_FAIL")
    loader = build_epoch_fresh_dataloader(
        dataset, batch=int(cfg.batch), workers=16, shuffle=False, rank=-1, drop_last=False, pin_memory=False
    )
    if int(loader.num_workers) != 16:
        fail("T1GR_G_EVAL_WORKER_DRIFT")
    validator_args = {
        "task": "detect",
        "mode": "val",
        "model": "yolo26s.yaml",
        "data": str(view["dataset_yaml"]),
        "device": cfg.device,
        "batch": int(cfg.batch),
        "imgsz": int(cfg.imgsz),
        "workers": 16,
        "split": "val",
        "conf": float(cfg.conf),
        "iou": float(cfg.iou),
        "max_det": int(cfg.max_det),
        "half": bool(cfg.half),
        "dnn": bool(cfg.dnn),
        "plots": False,
        "save_json": False,
        "verbose": False,
        "end2end": True,
    }
    try:
        validator = DetectionValidator(dataloader=loader, save_dir=save_dir, args=validator_args)
        with wall_clock_watchdog(timeout, "T1GR_G_EVAL_TIMEOUT"):
            validator(model=model)
        box = getattr(validator.metrics, "box", None)
        if box is None:
            fail("T1GR_G_EVAL_BOX_METRICS_MISSING")
        return {
            "map50_95": finite_metric(float(box.map)),
            "map50": finite_metric(float(box.map50)),
            "sample_count": len(dataset),
            "ids_commitment": sha256_json(sorted(dataset.ids)),
            "workers": int(loader.num_workers),
            "ir_input": "ZERO_IR",
        }
    finally:
        loader.shutdown()


def run(args) -> dict:
    repo = ROOT.resolve(strict=True)
    security_path = ensure_repo_input(repo, "config/t1gr_e5_security_policy.json", "config")
    if sha256_file(security_path) != FROZEN_E5_SECURITY_POLICY_SHA256:
        fail("E5_SECURITY_POLICY_SHA_DRIFT")
    security = read_json_bounded(security_path, 1 << 20, "t1gr-e5-security-policy-v2")
    recipe_path = ensure_repo_input(repo, "reports/step4_t1gr/e5_v2_step1_recipe_public.json", "reports/step4_t1gr")
    design_path = ensure_repo_input(repo, "config/t1gr_g_design.frozen.json", "config")
    smoke_audit_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_g_smoke_audit_public.json", "reports/step4_t1gr")
    view_public_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_g_multimodal_view_public.json", "reports/step4_t1gr")
    view_manifest_path = ensure_private_input(Path(args.view_manifest), repo)
    component_map_path = ensure_private_input(Path(args.dev_component_map), repo)
    run_root = Path(args.run_root).expanduser().resolve(strict=False)
    if is_within(run_root, repo) or not run_root.is_dir():
        fail("T1GR_G_EVAL_RUN_ROOT_FAIL")
    output_results = ensure_public_output(repo, "reports/step4_t1gr/per_seed_results.json", security["public_output_prefix"])
    output_audit = ensure_public_output(repo, "reports/step4_t1gr/t1gr_g_eval_public.json", security["public_output_prefix"])
    timeout = float(args.timeout_seconds)
    deadline = Deadline(timeout * 54)
    with file_lock(output_audit.with_suffix(output_audit.suffix + ".lock"), 5.0, 900.0):
        recipe = read_json_bounded(recipe_path, int(security["max_public_json_bytes"]), SCHEMA_RECIPE)
        design = read_json(design_path)
        smoke_audit = read_json(smoke_audit_path)
        view_public = read_json(view_public_path)
        if not all(e5_payload_ok(value) for value in (recipe, smoke_audit, view_public)):
            fail("T1GR_G_EVAL_PUBLIC_INPUT_INTEGRITY_FAIL")
        if smoke_audit.get("multiseed_training_authorized") is not True:
            fail("T1GR_G_EVAL_FORMAL_AUTHORITY_MISSING")
        view_manifest_sha = sha256_file(view_manifest_path, deadline)
        if view_manifest_sha != view_public.get("view_manifest_private_sha256"):
            fail("T1GR_G_EVAL_VIEW_SHA_DRIFT")
        view = verify_multimodal_view(view_manifest_path, recipe, deadline=deadline)
        component_obj = read_json(component_map_path)
        component_by_id = parse_component_map(component_obj, view["ids"]["dev"])
        fold_by_id = balanced_component_folds(component_by_id, 5)
        fold_counts = {fold: sum(value == fold for value in fold_by_id.values()) for fold in range(5)}
        if any(count <= 0 for count in fold_counts.values()):
            fail("T1GR_G_EVAL_EMPTY_FOLD")
        source_hashes = implementation_source_hashes(repo)
        if source_hashes != smoke_audit.get("implementation_source_hashes"):
            fail("T1GR_G_EVAL_IMPLEMENTATION_DRIFT")
        rows = frozen_launch_rows(design)
        formal_reports = []
        formal = {}
        for row in rows:
            path = ensure_repo_input(repo, run_report_rel("formal", row["seed"], row["arm"]), "reports/step4_t1gr")
            report = read_json(path)
            if (
                not e5_payload_ok(report)
                or report.get("mode") != "formal"
                or report.get("run_gate_passed") is not True
                or report.get("dev_suite_eval_authorized") is not True
                or int(report.get("epochs_completed", -1)) != 80
                or report.get("primary_checkpoint") != "last.pt"
                or report.get("validation_ir") != "ZERO_IR"
            ):
                fail("T1GR_G_FORMAL_REPORT_NOT_EVALUABLE")
            key = (int(row["seed"]), str(row["arm"]))
            formal[key] = (report, path)
            formal_reports.append({"seed": key[0], "arm": key[1], "report_sha256": sha256_file(path)})
        request_fingerprint = sha256_json({
            "script": SCRIPT_VERSION,
            "recipe": sha256_file(recipe_path, deadline),
            "design": sha256_file(design_path, deadline),
            "smoke_audit": sha256_file(smoke_audit_path, deadline),
            "view": view_manifest_sha,
            "component_map": sha256_file(component_map_path, deadline),
            "formal_reports": formal_reports,
            "fold_assignment": sha256_json(fold_by_id),
            "run_root_binding": sha256_json(str(run_root).casefold() if os.name == "nt" else str(run_root)),
        })
        existing_results = check_existing_output(output_results, request_fingerprint)
        existing_audit = check_existing_output(output_audit, request_fingerprint)
        if existing_results is not None and existing_audit is not None:
            return {
                "status": "PASS",
                "idempotent_reuse": True,
                "per_seed_results_sha256": existing_results[1],
                "eval_audit_sha256": existing_audit[1],
            }
        if (existing_results is None) != (existing_audit is None):
            fail("T1GR_G_EVAL_PARTIAL_PUBLIC_OUTPUT")
        try:
            import ultralytics
            from ultralytics.cfg import DEFAULT_CFG, get_cfg
            from ultralytics.data.utils import check_det_dataset
        except Exception:
            fail("T1GR_G_EVAL_IMPORT_FAIL")
        if str(ultralytics.__version__) != recipe["environment"]["ultralytics_version"]:
            fail("T1GR_G_EVAL_ULTRALYTICS_DRIFT")
        environment = environment_probe()
        compare_environment(environment, recipe["environment"])
        eval_root = run_root / "T1GR_G_ZERO_IR_DEV_EVAL"
        if eval_root.exists():
            fail("T1GR_G_EVAL_DIRECTORY_ALREADY_EXISTS")
        eval_root.mkdir(parents=True, mode=0o700)
        cfg_overrides = dict(recipe["eval_args"])
        cfg_overrides.update({
            "task": "detect",
            "mode": "val",
            "data": str(view["dataset_yaml"]),
            "model": "yolo26s.yaml",
            "device": recipe["runtime"]["device"],
            "batch": int(recipe["train_args"]["batch"]),
            "imgsz": int(recipe["train_args"]["imgsz"]),
            "workers": 16,
        })
        cfg = get_cfg(DEFAULT_CFG, overrides=cfg_overrides)
        data = check_det_dataset(str(view["dataset_yaml"]))
        data["channels"] = 4
        subset_files = {}
        for fold in range(5):
            keep = [sid for sid in view["ids"]["dev"] if fold_by_id[sid] != fold]
            path = eval_root / f"lofo_{fold}.private.txt"
            path.write_text("".join(view["image_maps"]["dev"][sid] + "\n" for sid in keep), encoding="utf-8")
            subset_files[fold] = (path, keep)
        offline_state = {}
        permission_state = {}
        result_rows = []
        phase = "load_models"
        try:
            with ultralytics_offline_guard(bypass_amp_download_check=False) as offline, private_umask() as permissions:
                offline_state.update(offline)
                permission_state.update(permissions)
                for position, row in enumerate(rows):
                    seed, arm = int(row["seed"]), str(row["arm"])
                    run_report, run_report_path = formal[(seed, arm)]
                    last = run_root / run_name("formal", seed, arm) / "weights" / "last.pt"
                    if not last.is_file() or sha256_file(last, deadline) != run_report.get("last_pt_sha256"):
                        fail("T1GR_G_EVAL_LAST_CHECKPOINT_DRIFT")
                    phase = f"load_{position}"
                    checkpoint_obj = torch.load(last, map_location="cpu", weights_only=False)
                    model = (checkpoint_obj.get("ema") or checkpoint_obj.get("model")).float().eval()
                    if not isinstance(model, T1GRP5Model):
                        fail("T1GR_G_EVAL_MODEL_CLASS_DRIFT")
                    if getattr(model, "treatment_id", None) != run_report.get("model_treatment_id"):
                        fail("T1GR_G_EVAL_MODEL_TREATMENT_DRIFT")
                    phase = f"full_{position}"
                    full = _evaluate(
                        model,
                        cfg=cfg,
                        data=data,
                        img_path=str(data["val"]),
                        view=view,
                        arm=arm,
                        seed=seed,
                        save_dir=eval_root / f"s{seed}_{arm.replace('-', '_')}_full",
                        timeout=timeout,
                    )
                    if full["sample_count"] != 198 or full["ids_commitment"] != recipe["ids_commitments"]["dev"]:
                        fail("T1GR_G_EVAL_FULL_DEV_ID_DRIFT")
                    lofo = {}
                    lofo_evidence = {}
                    for fold in range(5):
                        phase = f"lofo_{position}_{fold}"
                        metric = _evaluate(
                            model,
                            cfg=cfg,
                            data=data,
                            img_path=str(subset_files[fold][0]),
                            view=view,
                            arm=arm,
                            seed=seed,
                            save_dir=eval_root / f"s{seed}_{arm.replace('-', '_')}_lofo_{fold}",
                            timeout=timeout,
                        )
                        expected_count = 198 - fold_counts[fold]
                        if metric["sample_count"] != expected_count:
                            fail("T1GR_G_EVAL_LOFO_COUNT_DRIFT")
                        lofo[f"fold_{fold}"] = metric["map50_95"]
                        lofo_evidence[f"fold_{fold}"] = {
                            "sample_count": metric["sample_count"],
                            "ids_commitment": metric["ids_commitment"],
                        }
                    result_rows.append({
                        "seed": seed,
                        "arm": arm,
                        "dev_map50_95": full["map50_95"],
                        "lofo_map50_95": lofo,
                        "run_manifest_sha256": sha256_file(run_report_path, deadline),
                        "last_checkpoint_sha256": sha256_file(last, deadline),
                        "full_dev_ids_commitment": full["ids_commitment"],
                        "lofo_evidence": lofo_evidence,
                    })
        except BaseException as exc:
            try:
                write_private_failure_report(
                    eval_root, exc, phase, int(recipe["runtime"]["private_traceback_max_bytes"])
                )
            except Exception:
                pass
            raise
        results = {
            "schema": SCHEMA_RESULTS,
            "authority": "DEV_ONLY_ZERO_IR_PRIMARY",
            "metric": "mAP50-95",
            "checkpoint": "last.pt",
            "max_det": 100,
            "inference_ir": "ZERO_IR",
            "fold_count": 5,
            "fold_unit": "E3_LEAKAGE_COMPONENT",
            "rows": result_rows,
            "final_holdout_accessed": False,
            "final_holdout_open_authorized": False,
        }
        audit = {
            "schema": SCHEMA_EVAL,
            "script_version": SCRIPT_VERSION,
            "recipe_public_sha256": sha256_file(recipe_path, deadline),
            "design_file_sha256": sha256_file(design_path, deadline),
            "smoke_audit_sha256": sha256_file(smoke_audit_path, deadline),
            "view_manifest_private_sha256": view_manifest_sha,
            "dev_component_map_private_sha256": sha256_file(component_map_path, deadline),
            "formal_run_evidence": formal_reports,
            "implementation_source_hashes": source_hashes,
            "environment": environment,
            "run_count": len(result_rows),
            "evaluation_count": len(result_rows) * 6,
            "full_dev_count": 198,
            "fold_counts": {f"fold_{fold}": count for fold, count in fold_counts.items()},
            "fold_assignment_commitment": sha256_json(fold_by_id),
            "metric": "mAP50-95",
            "max_det": 100,
            "checkpoint": "last.pt",
            "inference_ir": "ZERO_IR",
            "external_network_integrations": offline_state,
            "private_artifact_permissions": permission_state,
            "eval_gate_passed": len(result_rows) == 9,
            "final_holdout_ids_available_to_evaluator": False,
            "final_holdout_open_authorized": False,
            "depth_go": False,
            "production_go": False,
            "next_action": "run the frozen cross-seed summarizer; keep FINAL HOLDOUT sealed",
        }
        assert_public_safe(results)
        assert_public_safe(audit)
        results_sha, _ = atomic_json_write(
            output_results, results, private=False, request_fingerprint=request_fingerprint
        )
        audit_sha, _ = atomic_json_write(output_audit, audit, private=False, request_fingerprint=request_fingerprint)
        return {
            "status": "PASS",
            "idempotent_reuse": False,
            "per_seed_results_sha256": results_sha,
            "eval_audit_sha256": audit_sha,
            "run_count": len(result_rows),
            "final_holdout_open_authorized": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-manifest", required=True)
    parser.add_argument("--dev-component-map", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=10800.0)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": safe_error_message(exc)}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
