#!/usr/bin/env python3
"""Evaluate all T1-U6 formal last.pt checkpoints on frozen DEV views."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_u6_core import (  # noqa: E402
    ARMS,
    SCHEMA_EVAL_AUDIT,
    SCHEMA_PREFLIGHT,
    SCHEMA_RESULTS,
    SCHEMA_RUN,
    SCHEMA_SMOKE_AUDIT,
    implementation_source_hashes,
    launch_rows,
    payload_ok,
    run_name,
    run_report_rel,
)
from multimodal.t1gr_u6_dataset import T1GRU6Dataset  # noqa: E402
from multimodal.t1gr_u6_model import first_conv  # noqa: E402
from multimodal.t1gr_u6_runtime import assert_same_server_environment, verify_u6_view  # noqa: E402
from multimodal.t1gr_e5_core import (  # noqa: E402
    FROZEN_E5_SECURITY_POLICY_SHA256,
    SCHEMA_RECIPE,
    environment_probe,
    payload_ok as e5_payload_ok,
    private_umask,
    ultralytics_offline_guard,
    wall_clock_watchdog,
    write_private_failure_report,
)
from multimodal.t1gr_g_core import balanced_component_folds  # noqa: E402
from multimodal.t1gr_g_impl_core import finite_metric, parse_component_map  # noqa: E402
from multimodal.t1gr_g_runtime import (  # noqa: E402
    build_epoch_fresh_dataloader,
    implementation_source_hashes as primary_source_hashes,
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

SCRIPT_VERSION = "t1gr-u6-server-multicondition-lofo-eval-v1"
LOFO_ARMS = frozenset({"G0-N", "G1-P", "G3-D"})
DOMAIN_ARMS = frozenset({"G1-P", "G3-D"})
WRONG_DIAGNOSTIC_ARMS = frozenset({"G1-P", "G2-S"})


def _make_dataset(
    cfg,
    data: dict,
    img_path: str,
    view: dict,
    *,
    arm: str,
    seed: int,
    ir_condition: str,
    depth_condition: str,
):
    return T1GRU6Dataset(
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
        prefix="T1-U6 DEV: ",
        task="detect",
        classes=cfg.classes,
        data=data,
        fraction=1.0,
        ir_by_sid=view["ir_maps"]["dev"],
        depth_by_sid=view["depth_maps"]["dev"],
        depth_kind_by_sid=view["depth_kind_maps"]["dev"],
        arm=arm,
        seed=seed,
        split="dev",
        ir_condition=ir_condition,
        depth_condition=depth_condition,
    )


def _evaluate(
    model,
    *,
    cfg,
    data: dict,
    img_path: str,
    view: dict,
    arm: str,
    seed: int,
    ir_condition: str,
    depth_condition: str,
    save_dir: Path,
    timeout: float,
) -> dict:
    from ultralytics.models.yolo.detect.val import DetectionValidator

    dataset = _make_dataset(
        cfg,
        data,
        img_path,
        view,
        arm=arm,
        seed=seed,
        ir_condition=ir_condition,
        depth_condition=depth_condition,
    )
    probe = dataset[0]["img"]
    if int(probe.shape[0]) != 6:
        fail("T1GR_U6_EVAL_CHANNEL_FAIL")
    if ir_condition == "ZERO" and int(torch.count_nonzero(probe[3]).item()) != 0:
        fail("T1GR_U6_EVAL_ZERO_IR_FAIL")
    if depth_condition == "ZERO" and int(torch.count_nonzero(probe[4:]).item()) != 0:
        fail("T1GR_U6_EVAL_ZERO_DEPTH_FAIL")
    mask = probe[5]
    if not bool(torch.all((mask == 0) | (mask == 255)).item()):
        fail("T1GR_U6_EVAL_MASK_BINARY_FAIL")
    loader = build_epoch_fresh_dataloader(
        dataset,
        batch=int(cfg.batch),
        workers=16,
        shuffle=False,
        rank=-1,
        drop_last=False,
        pin_memory=False,
    )
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
        with wall_clock_watchdog(timeout, "T1GR_U6_EVAL_TIMEOUT"):
            validator(model=model)
        box = getattr(validator.metrics, "box", None)
        if box is None:
            fail("T1GR_U6_EVAL_METRICS_MISSING")
        return {
            "map50_95": finite_metric(float(box.map)),
            "map50": finite_metric(float(box.map50)),
            "sample_count": len(dataset),
            "ids_commitment": sha256_json(sorted(dataset.ids)),
            "arm": arm,
            "ir_condition": ir_condition,
            "depth_condition": depth_condition,
        }
    finally:
        loader.shutdown()


def _stem_gate(model, arm: str) -> list[int]:
    conv = first_conv(model)
    if conv.in_channels != 6:
        fail("T1GR_U6_EVAL_MODEL_STEM_FAIL")
    weights = conv.weight.detach().float().cpu()
    counts = [int(torch.count_nonzero(weights[:, index]).item()) for index in range(6)]
    if arm == "G0-N" and counts[3:] != [0, 0, 0]:
        fail("T1GR_U6_EVAL_G0_STEM_FAIL")
    if arm in {"G1-P", "G2-S"} and (counts[3] <= 0 or counts[4:] != [0, 0]):
        fail("T1GR_U6_EVAL_IR_STEM_FAIL")
    if arm == "G3-D" and any(value <= 0 for value in counts[3:]):
        fail("T1GR_U6_EVAL_G3_STEM_FAIL")
    return counts


def run(args) -> dict:
    repo = ROOT.resolve(strict=True)
    security_path = ensure_repo_input(repo, "config/t1gr_e5_security_policy.json", "config")
    if sha256_file(security_path) != FROZEN_E5_SECURITY_POLICY_SHA256:
        fail("E5_SECURITY_POLICY_SHA_DRIFT")
    security = read_json_bounded(security_path, 1 << 20, "t1gr-e5-security-policy-v2")
    recipe_path = ensure_repo_input(repo, "reports/step4_t1gr/e5_v2_step1_recipe_public.json", "reports/step4_t1gr")
    preflight_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_u6_server_preflight_public.json", "reports/step4_t1gr")
    smoke_audit_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_u6_server_smoke_audit_public.json", "reports/step4_t1gr")
    view_manifest_path = ensure_private_input(Path(args.u6_view_manifest), repo)
    component_map_path = ensure_private_input(Path(args.dev_component_map), repo)
    run_root = Path(args.run_root).expanduser().resolve(strict=False)
    if is_within(run_root, repo) or not run_root.is_dir():
        fail("T1GR_U6_EVAL_RUN_ROOT_FAIL")
    output_results = ensure_public_output(repo, "reports/step4_t1gr/t1gr_u6_server_results_public.json", security["public_output_prefix"])
    output_audit = ensure_public_output(repo, "reports/step4_t1gr/t1gr_u6_server_eval_public.json", security["public_output_prefix"])
    timeout = float(args.timeout_seconds)
    deadline = Deadline(timeout * 120)
    with file_lock(output_audit.with_suffix(output_audit.suffix + ".lock"), 5.0, 900.0):
        recipe = read_json_bounded(recipe_path, int(security["max_public_json_bytes"]), SCHEMA_RECIPE)
        preflight = read_json_bounded(preflight_path, int(security["max_public_json_bytes"]), SCHEMA_PREFLIGHT)
        smoke_audit = read_json_bounded(smoke_audit_path, int(security["max_public_json_bytes"]), SCHEMA_SMOKE_AUDIT)
        if not e5_payload_ok(recipe) or not payload_ok(preflight) or not payload_ok(smoke_audit):
            fail("T1GR_U6_EVAL_INPUT_INTEGRITY_FAIL")
        if smoke_audit.get("formal_training_authorized") is not True or smoke_audit.get("final_holdout_open_authorized") is not False:
            fail("T1GR_U6_EVAL_AUTHORITY_FAIL")
        view_sha = sha256_file(view_manifest_path, deadline)
        if view_sha != preflight.get("u6_view_manifest_private_sha256"):
            fail("T1GR_U6_EVAL_VIEW_SHA_DRIFT")
        view = verify_u6_view(view_manifest_path, recipe, deadline=deadline)
        component_obj = json.loads(component_map_path.read_text(encoding="utf-8"))
        component_by_id = parse_component_map(component_obj, view["ids"]["dev"])
        fold_by_id = balanced_component_folds(component_by_id, 5)
        fold_counts = {fold: sum(value == fold for value in fold_by_id.values()) for fold in range(5)}
        if any(count <= 0 for count in fold_counts.values()):
            fail("T1GR_U6_EVAL_EMPTY_FOLD")
        sources, upstream = implementation_source_hashes(repo), primary_source_hashes(repo)
        if (
            sources != smoke_audit.get("implementation_source_hashes")
            or upstream != smoke_audit.get("legacy_primary_suite_source_hashes")
        ):
            fail("T1GR_U6_EVAL_SOURCE_DRIFT")
        formal, evidence = {}, []
        for row in launch_rows():
            path = ensure_repo_input(repo, run_report_rel("formal", row["seed"], row["arm"]), "reports/step4_t1gr")
            report = read_json_bounded(path, int(security["max_public_json_bytes"]), SCHEMA_RUN)
            if (
                not payload_ok(report)
                or report.get("run_gate_passed") is not True
                or report.get("mode") != "formal"
                or report.get("arm") != row["arm"]
                or int(report.get("seed", -1)) != int(row["seed"])
                or int(report.get("suite_position_zero_based", -1)) != int(row["position"])
                or int(report.get("lane_position_zero_based", -1)) != int(row["lane_position"])
                or int(report.get("epochs_completed", -1)) != 80
                or report.get("primary_checkpoint") != "last.pt"
                or report.get("dev_suite_eval_authorized") is not True
                or report.get("implementation_source_hashes") != sources
                or report.get("legacy_primary_suite_source_hashes") != upstream
                or report.get("u6_view_manifest_private_sha256") != view_sha
                or report.get("environment") != preflight.get("server_environment")
            ):
                fail("T1GR_U6_FORMAL_REPORT_NOT_EVALUABLE")
            key = (int(row["seed"]), str(row["arm"]))
            formal[key] = (report, path)
            evidence.append({"seed": key[0], "arm": key[1], "report_sha256": sha256_file(path)})
        request = sha256_json({
            "script": SCRIPT_VERSION,
            "recipe": sha256_file(recipe_path, deadline),
            "preflight": sha256_file(preflight_path, deadline),
            "smoke_audit": sha256_file(smoke_audit_path, deadline),
            "u6_view": view_sha,
            "component_map": sha256_file(component_map_path, deadline),
            "formal_reports": evidence,
            "fold_assignment": sha256_json(fold_by_id),
            "run_root_binding": sha256_json(str(run_root).casefold() if os.name == "nt" else str(run_root)),
        })
        existing_results = check_existing_output(output_results, request)
        existing_audit = check_existing_output(output_audit, request)
        if existing_results is not None and existing_audit is not None:
            return {
                "status": "PASS",
                "idempotent_reuse": True,
                "results_sha256": existing_results[1],
                "audit_sha256": existing_audit[1],
            }
        if (existing_results is None) != (existing_audit is None):
            fail("T1GR_U6_EVAL_PARTIAL_OUTPUT")
        try:
            import ultralytics
            from ultralytics.cfg import DEFAULT_CFG, get_cfg
            from ultralytics.data.utils import check_det_dataset
        except Exception:
            fail("T1GR_U6_EVAL_IMPORT_FAIL")
        if str(ultralytics.__version__) != recipe["environment"]["ultralytics_version"]:
            fail("T1GR_U6_EVAL_ULTRALYTICS_DRIFT")
        environment = environment_probe()
        assert_same_server_environment(environment, preflight.get("server_environment") or {})
        eval_root = run_root / "T1GR_U6_SERVER_DEV_EVAL"
        if eval_root.exists():
            fail("T1GR_U6_EVAL_DIRECTORY_ALREADY_EXISTS")
        eval_root.mkdir(parents=True, mode=0o700)
        overrides = dict(recipe["eval_args"])
        overrides.update({
            "task": "detect",
            "mode": "val",
            "data": str(view["dataset_yaml"]),
            "model": recipe["model_yaml"],
            "device": recipe["runtime"]["device"],
            "batch": int(recipe["train_args"]["batch"]),
            "imgsz": int(recipe["train_args"]["imgsz"]),
            "workers": 16,
        })
        cfg = get_cfg(DEFAULT_CFG, overrides=overrides)
        data = check_det_dataset(str(view["dataset_yaml"]))
        data["channels"] = 6
        subset_files = {}
        for fold in range(5):
            keep = [sid for sid in view["ids"]["dev"] if fold_by_id[sid] != fold]
            path = eval_root / f"lofo_{fold}.private.txt"
            path.write_text("".join(view["image_maps"]["dev"][sid] + "\n" for sid in keep), encoding="utf-8")
            subset_files[fold] = (path, keep)
        domain_files = {}
        for key, kind in (
            ("metric_png", "METRIC_UINT16_PNG"),
            ("unknown_jpg", "UNKNOWN_SCALE_JPG_QUARANTINED"),
        ):
            keep = [sid for sid in view["ids"]["dev"] if view["depth_kind_maps"]["dev"][sid] == kind]
            if not keep:
                fail("T1GR_U6_EVAL_EMPTY_DEPTH_DOMAIN")
            path = eval_root / f"depth_domain_{key}.private.txt"
            path.write_text("".join(view["image_maps"]["dev"][sid] + "\n" for sid in keep), encoding="utf-8")
            domain_files[key] = (path, keep)

        result_rows, offline, permissions = [], {}, {}
        evaluation_count = 0
        phase = "load_models"
        try:
            with ultralytics_offline_guard(bypass_amp_download_check=False) as network, private_umask() as modes:
                offline.update(network)
                permissions.update(modes)
                for position, row in enumerate(launch_rows()):
                    seed, arm = int(row["seed"]), str(row["arm"])
                    run_report, report_path = formal[(seed, arm)]
                    last = run_root / run_name("formal", seed, arm) / "weights" / "last.pt"
                    if not last.is_file() or sha256_file(last, deadline) != run_report.get("last_pt_sha256"):
                        fail("T1GR_U6_EVAL_CHECKPOINT_DRIFT")
                    checkpoint_obj = torch.load(last, map_location="cpu", weights_only=False)
                    model = (checkpoint_obj.get("ema") or checkpoint_obj.get("model")).float().eval()
                    stem_counts = _stem_gate(model, arm)

                    phase = f"native_full_{position}"
                    native = _evaluate(
                        model,
                        cfg=cfg,
                        data=data,
                        img_path=str(data["val"]),
                        view=view,
                        arm=arm,
                        seed=seed,
                        ir_condition="ARM_NATIVE",
                        depth_condition="NATIVE",
                        save_dir=eval_root / f"s{seed}_{arm.replace('-', '_')}_native_full",
                        timeout=timeout,
                    )
                    evaluation_count += 1
                    if native["sample_count"] != 198 or native["ids_commitment"] != recipe["ids_commitments"]["dev"]:
                        fail("T1GR_U6_EVAL_FULL_DEV_ID_DRIFT")

                    if arm == "G0-N":
                        rgb_only = native
                        rgb_evidence = "REUSED_IDENTICAL_G0_NATIVE_INPUT"
                    else:
                        phase = f"rgb_only_full_{position}"
                        rgb_only = _evaluate(
                            model,
                            cfg=cfg,
                            data=data,
                            img_path=str(data["val"]),
                            view=view,
                            arm=arm,
                            seed=seed,
                            ir_condition="ZERO",
                            depth_condition="ZERO",
                            save_dir=eval_root / f"s{seed}_{arm.replace('-', '_')}_rgb_only_full",
                            timeout=timeout,
                        )
                        evaluation_count += 1
                        rgb_evidence = "EXECUTED"

                    if arm in {"G1-P", "G2-S"}:
                        paired_ir = native
                        paired_evidence = "REUSED_IDENTICAL_NATIVE_INPUT"
                    else:
                        phase = f"paired_ir_zero_depth_full_{position}"
                        paired_ir = _evaluate(
                            model,
                            cfg=cfg,
                            data=data,
                            img_path=str(data["val"]),
                            view=view,
                            arm=arm,
                            seed=seed,
                            ir_condition="PAIRED",
                            depth_condition="ZERO",
                            save_dir=eval_root / f"s{seed}_{arm.replace('-', '_')}_paired_ir_zero_depth_full",
                            timeout=timeout,
                        )
                        evaluation_count += 1
                        paired_evidence = "EXECUTED"

                    wrong = None
                    if arm in WRONG_DIAGNOSTIC_ARMS:
                        phase = f"wrong_ir_zero_depth_full_{position}"
                        wrong = _evaluate(
                            model,
                            cfg=cfg,
                            data=data,
                            img_path=str(data["val"]),
                            view=view,
                            arm=arm,
                            seed=seed,
                            ir_condition="WRONG",
                            depth_condition="ZERO",
                            save_dir=eval_root / f"s{seed}_{arm.replace('-', '_')}_wrong_ir_zero_depth_full",
                            timeout=timeout,
                        )
                        evaluation_count += 1

                    lofo = None
                    lofo_evidence = None
                    if arm in LOFO_ARMS:
                        lofo, lofo_evidence = {}, {}
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
                                ir_condition="ARM_NATIVE",
                                depth_condition="NATIVE",
                                save_dir=eval_root / f"s{seed}_{arm.replace('-', '_')}_lofo_{fold}",
                                timeout=timeout,
                            )
                            evaluation_count += 1
                            expected_count = 198 - fold_counts[fold]
                            if metric["sample_count"] != expected_count:
                                fail("T1GR_U6_EVAL_LOFO_COUNT_DRIFT")
                            lofo[f"fold_{fold}"] = metric["map50_95"]
                            lofo_evidence[f"fold_{fold}"] = {
                                "sample_count": metric["sample_count"],
                                "ids_commitment": metric["ids_commitment"],
                            }

                    domains = None
                    domain_evidence = None
                    if arm in DOMAIN_ARMS:
                        domains, domain_evidence = {}, {}
                        for domain, (domain_path, keep) in domain_files.items():
                            phase = f"domain_{position}_{domain}"
                            metric = _evaluate(
                                model,
                                cfg=cfg,
                                data=data,
                                img_path=str(domain_path),
                                view=view,
                                arm=arm,
                                seed=seed,
                                ir_condition="ARM_NATIVE",
                                depth_condition="NATIVE",
                                save_dir=eval_root / f"s{seed}_{arm.replace('-', '_')}_domain_{domain}",
                                timeout=timeout,
                            )
                            evaluation_count += 1
                            if metric["sample_count"] != len(keep):
                                fail("T1GR_U6_EVAL_DEPTH_DOMAIN_COUNT_DRIFT")
                            domains[domain] = metric["map50_95"]
                            domain_evidence[domain] = {
                                "sample_count": metric["sample_count"],
                                "ids_commitment": metric["ids_commitment"],
                            }

                    result_rows.append({
                        "seed": seed,
                        "arm": arm,
                        "native_map50_95": native["map50_95"],
                        "native_map50": native["map50"],
                        "rgb_only_map50_95": rgb_only["map50_95"],
                        "paired_ir_zero_depth_map50_95": paired_ir["map50_95"],
                        "wrong_ir_zero_depth_map50_95": None if wrong is None else wrong["map50_95"],
                        "lofo_native_map50_95": lofo,
                        "depth_domain_native_map50_95": domains,
                        "run_manifest_sha256": sha256_file(report_path, deadline),
                        "last_checkpoint_sha256": sha256_file(last, deadline),
                        "final_stem_channel_nonzero_counts": stem_counts,
                        "full_dev_ids_commitment": native["ids_commitment"],
                        "rgb_only_evidence": rgb_evidence,
                        "paired_ir_zero_depth_evidence": paired_evidence,
                        "lofo_evidence": lofo_evidence,
                        "depth_domain_evidence": domain_evidence,
                    })
        except BaseException as exc:
            try:
                write_private_failure_report(eval_root, exc, phase, int(recipe["runtime"]["private_traceback_max_bytes"]))
            except Exception:
                pass
            raise

        results = {
            "schema": SCHEMA_RESULTS,
            "authority": "DEV_ONLY_T1_U6_FOUR_ARM_EVALUATION",
            "metric": "mAP50-95",
            "checkpoint": "last.pt",
            "max_det": 100,
            "conditions": {
                "native": "arm deployment input",
                "rgb_only": "zero IR, zero Depth, zero mask",
                "paired_ir_zero_depth": "paired IR, zero Depth, zero mask",
                "wrong_ir_zero_depth": "fixed DEV derangement at epoch zero, zero Depth, zero mask",
            },
            "fold_count": 5,
            "fold_unit": "E3_LEAKAGE_COMPONENT",
            "rows": result_rows,
            "legacy_primary_g_suite_unchanged": True,
            "legacy_g3_fe_included": False,
            "final_holdout_accessed": False,
            "final_holdout_open_authorized": False,
        }
        audit = {
            "schema": SCHEMA_EVAL_AUDIT,
            "script_version": SCRIPT_VERSION,
            "preflight_public_sha256": sha256_file(preflight_path, deadline),
            "smoke_audit_sha256": sha256_file(smoke_audit_path, deadline),
            "u6_view_manifest_private_sha256": view_sha,
            "dev_component_map_private_sha256": sha256_file(component_map_path, deadline),
            "formal_run_evidence": evidence,
            "implementation_source_hashes": sources,
            "legacy_primary_suite_source_hashes": upstream,
            "server_environment": environment,
            "run_count": len(result_rows),
            "evaluation_count": evaluation_count,
            "expected_evaluation_count": 90,
            "full_dev_count": 198,
            "fold_counts": {f"fold_{fold}": count for fold, count in fold_counts.items()},
            "depth_domain_counts": {key: len(value[1]) for key, value in domain_files.items()},
            "fold_assignment_commitment": sha256_json(fold_by_id),
            "external_network_integrations": offline,
            "private_artifact_permissions": permissions,
            "eval_gate_passed": len(result_rows) == 12 and evaluation_count == 90,
            "legacy_primary_g_suite_unchanged": True,
            "final_holdout_ids_available_to_evaluator": False,
            "final_holdout_open_authorized": False,
            "production_go": False,
            "next_action": "run t1gr_u6_server_summarize.py and apply the frozen IR and Depth operational selectors",
        }
        assert_public_safe(results)
        assert_public_safe(audit)
        if audit["eval_gate_passed"] is not True:
            fail("T1GR_U6_EVAL_MATRIX_INCOMPLETE")
        results_sha, _ = atomic_json_write(output_results, results, private=False, request_fingerprint=request)
        audit_sha, _ = atomic_json_write(output_audit, audit, private=False, request_fingerprint=request)
        return {
            "status": "PASS",
            "idempotent_reuse": False,
            "results_sha256": results_sha,
            "audit_sha256": audit_sha,
            "final_holdout_open_authorized": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--u6-view-manifest", required=True)
    parser.add_argument("--dev-component-map", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": safe_error_message(exc)}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
