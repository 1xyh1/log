#!/usr/bin/env python3
"""Run one authorized T1-U6 smoke or formal job."""
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
    ARM_POLICIES,
    SCHEMA_PREFLIGHT,
    SCHEMA_RUN,
    SCHEMA_SMOKE_AUDIT,
    implementation_source_hashes,
    launch_rows,
    payload_ok,
    run_name,
    run_report_rel,
    validate_spec,
)
from multimodal.t1gr_u6_model import build_t1gr_u6_model, first_conv, stem_contract  # noqa: E402
from multimodal.t1gr_u6_runtime import (  # noqa: E402
    T1GRU6Trainer,
    assert_same_server_environment,
    verify_u6_view,
)
from multimodal.t1gr_e5_core import (  # noqa: E402
    FROZEN_E5_SECURITY_POLICY_SHA256,
    SCHEMA_RECIPE,
    effective_args_mismatch,
    environment_probe,
    optimizer_fingerprint,
    parse_utc,
    payload_ok as e5_payload_ok,
    private_umask,
    results_csv_epoch_count,
    state_dict_sha256,
    ultralytics_offline_guard,
    utc_now,
    wall_clock_watchdog,
    write_private_failure_report,
)
from multimodal.t1gr_g_core import SEEDS  # noqa: E402
from multimodal.t1gr_g_runtime import implementation_source_hashes as primary_source_hashes  # noqa: E402
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

SCRIPT_VERSION = "t1gr-u6-server-run-one-v1"


def _private_run_root(path: Path, repo: Path) -> Path:
    value = path.expanduser().resolve(strict=False)
    if is_within(value, repo) or not value.parent.is_dir() or not os.access(value.parent, os.W_OK):
        fail("T1GR_U6_RUN_ROOT_INVALID")
    if value.exists() and not value.is_dir():
        fail("T1GR_U6_RUN_ROOT_NOT_DIRECTORY")
    return value


def _expected(recipe: dict, mode: str, seed: int) -> dict:
    out = dict(recipe["train_args"])
    out["seed"] = int(seed)
    out["epochs"] = 1 if mode == "smoke" else 80
    out.update(recipe["eval_args"])
    out.update({"resume": False, "profile": False, "verbose": True, "pretrained": False, "exist_ok": False})
    return out


def _identity(preflight: dict, seed: int, arm: str) -> dict:
    matches = [
        row for row in preflight.get("model_identity_rows") or []
        if int(row.get("seed", -1)) == int(seed) and row.get("arm") == arm
    ]
    if len(matches) != 1:
        fail("T1GR_U6_PREFLIGHT_IDENTITY_MISSING")
    return matches[0]


def _final_stem(model) -> dict:
    conv = first_conv(model)
    weights = conv.weight.detach().float().cpu()
    return {
        "physical_in_channels": int(conv.in_channels),
        "channel_nonzero_counts": [int(torch.count_nonzero(weights[:, index]).item()) for index in range(6)],
        "channel_l2_norms": [float(torch.linalg.vector_norm(weights[:, index]).item()) for index in range(6)],
        "complete_weight_sha256": state_dict_sha256({"stem.weight": weights.contiguous()}),
    }


def _assert_lane_predecessors(
    *,
    repo: Path,
    security: dict,
    mode: str,
    seed: int,
    arm: str,
    run_root: Path,
    view_sha: str,
    checkpoint_sha: str,
    environment: dict,
    sources: dict,
    deadline: Deadline,
) -> None:
    lane = [row for row in launch_rows() if int(row["seed"]) == int(seed)]
    matches = [index for index, row in enumerate(lane) if row["arm"] == arm]
    if len(matches) != 1:
        fail("T1GR_U6_LANE_ARM_LOOKUP_FAIL")
    expected_epochs = 1 if mode == "smoke" else 80
    for row in lane[: matches[0]]:
        predecessor = str(row["arm"])
        path = ensure_repo_input(repo, run_report_rel(mode, seed, predecessor), "reports/step4_t1gr")
        report = read_json_bounded(path, int(security["max_public_json_bytes"]), SCHEMA_RUN)
        last = run_root / run_name(mode, seed, predecessor) / "weights" / "last.pt"
        if (
            not payload_ok(report)
            or report.get("run_gate_passed") is not True
            or report.get("mode") != mode
            or report.get("arm") != predecessor
            or int(report.get("seed", -1)) != int(seed)
            or int(report.get("epochs_completed", -1)) != expected_epochs
            or report.get("u6_view_manifest_private_sha256") != view_sha
            or report.get("base_checkpoint_sha256") != checkpoint_sha
            or report.get("environment") != environment
            or report.get("implementation_source_hashes") != sources
            or not last.is_file()
            or sha256_file(last, deadline) != report.get("last_pt_sha256")
        ):
            fail("T1GR_U6_LANE_PREDECESSOR_NOT_COMPLETE", predecessor)


def run(args) -> dict:
    if args.mode not in {"smoke", "formal"} or args.arm not in ARMS or int(args.seed) not in SEEDS:
        fail("T1GR_U6_RUN_REQUEST_INVALID")
    launch_match = [
        row for row in launch_rows()
        if int(row["seed"]) == int(args.seed) and row["arm"] == args.arm
    ]
    if len(launch_match) != 1:
        fail("T1GR_U6_RUN_LAUNCH_ROW_FAIL")
    launch_row = launch_match[0]
    repo = ROOT.resolve(strict=True)
    security_path = ensure_repo_input(repo, "config/t1gr_e5_security_policy.json", "config")
    if sha256_file(security_path) != FROZEN_E5_SECURITY_POLICY_SHA256:
        fail("E5_SECURITY_POLICY_SHA_DRIFT")
    security = read_json_bounded(security_path, 1 << 20, "t1gr-e5-security-policy-v2")
    spec_path = ensure_repo_input(repo, "config/t1gr_u6_design.frozen.json", "config")
    recipe_path = ensure_repo_input(repo, "reports/step4_t1gr/e5_v2_step1_recipe_public.json", "reports/step4_t1gr")
    preflight_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_u6_server_preflight_public.json", "reports/step4_t1gr")
    view_public_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_u6_view_public.json", "reports/step4_t1gr")
    smoke_audit_path = None
    if args.mode == "formal":
        smoke_audit_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_u6_server_smoke_audit_public.json", "reports/step4_t1gr")
    view_manifest_path = ensure_private_input(Path(args.u6_view_manifest), repo)
    checkpoint = ensure_private_input(Path(args.base_checkpoint), repo)
    run_root = _private_run_root(Path(args.run_root), repo)
    output = ensure_public_output(repo, run_report_rel(args.mode, int(args.seed), args.arm), security["public_output_prefix"])
    recipe = read_json_bounded(recipe_path, int(security["max_public_json_bytes"]), SCHEMA_RECIPE)
    spec = read_json_bounded(spec_path, int(security["max_public_json_bytes"]))
    preflight = read_json_bounded(preflight_path, int(security["max_public_json_bytes"]), SCHEMA_PREFLIGHT)
    view_public = read_json_bounded(view_public_path, int(security["max_public_json_bytes"]))
    validate_spec(spec)
    if not e5_payload_ok(recipe) or not payload_ok(preflight) or not payload_ok(view_public):
        fail("T1GR_U6_RUN_PUBLIC_INTEGRITY_FAIL")
    if preflight.get("preflight_gate_passed") is not True or preflight.get("smoke_training_authorized") is not True:
        fail("T1GR_U6_PREFLIGHT_NOT_AUTHORIZED")
    smoke_audit = None
    if smoke_audit_path is not None:
        smoke_audit = read_json_bounded(smoke_audit_path, int(security["max_public_json_bytes"]), SCHEMA_SMOKE_AUDIT)
        if not payload_ok(smoke_audit) or smoke_audit.get("formal_training_authorized") is not True:
            fail("T1GR_U6_FORMAL_NOT_AUTHORIZED")
    timeout = float(recipe["runtime"]["smoke_timeout_seconds"] if args.mode == "smoke" else recipe["runtime"]["formal_timeout_seconds"])
    deadline = Deadline(timeout)
    with file_lock(output.with_suffix(output.suffix + ".lock"), 5.0, 900.0):
        sources = implementation_source_hashes(repo)
        upstream = primary_source_hashes(repo)
        if sources != preflight.get("implementation_source_hashes") or upstream != preflight.get("legacy_primary_suite_source_hashes"):
            fail("T1GR_U6_IMPLEMENTATION_CHANGED_AFTER_PREFLIGHT")
        if smoke_audit is not None and sources != smoke_audit.get("implementation_source_hashes"):
            fail("T1GR_U6_IMPLEMENTATION_CHANGED_AFTER_SMOKE")
        view_sha = sha256_file(view_manifest_path, deadline)
        if view_sha != preflight.get("u6_view_manifest_private_sha256") or view_sha != view_public.get("u6_view_manifest_private_sha256"):
            fail("T1GR_U6_VIEW_SHA_DRIFT")
        view = verify_u6_view(view_manifest_path, recipe, deadline=deadline)
        if sha256_file(view["dataset_yaml"], deadline) != preflight.get("u6_dataset_yaml_private_sha256"):
            fail("T1GR_U6_DATASET_YAML_SHA_DRIFT")
        checkpoint_sha = sha256_file(checkpoint, deadline)
        if checkpoint_sha != preflight.get("base_checkpoint_sha256") or checkpoint_sha != recipe.get("base_checkpoint_sha256"):
            fail("T1GR_U6_CHECKPOINT_SHA_DRIFT")
        environment = environment_probe()
        assert_same_server_environment(environment, preflight.get("server_environment") or {})
        _assert_lane_predecessors(
            repo=repo,
            security=security,
            mode=args.mode,
            seed=int(args.seed),
            arm=args.arm,
            run_root=run_root,
            view_sha=view_sha,
            checkpoint_sha=checkpoint_sha,
            environment=environment,
            sources=sources,
            deadline=deadline,
        )
        expected_identity = _identity(preflight, int(args.seed), args.arm)
        inputs = {
            "script": SCRIPT_VERSION,
            "mode": args.mode,
            "arm": args.arm,
            "seed": int(args.seed),
            "suite_position_zero_based": int(launch_row["position"]),
            "lane_position_zero_based": int(launch_row["lane_position"]),
            "spec": sha256_file(spec_path, deadline),
            "recipe": sha256_file(recipe_path, deadline),
            "preflight": sha256_file(preflight_path, deadline),
            "smoke_audit": sha256_file(smoke_audit_path, deadline) if smoke_audit_path else None,
            "u6_view": view_sha,
            "checkpoint": checkpoint_sha,
            "environment": environment,
            "sources": sources,
            "run_root_binding": sha256_json(str(run_root).casefold() if os.name == "nt" else str(run_root)),
        }
        request = sha256_json(inputs)
        run_dir = run_root / run_name(args.mode, int(args.seed), args.arm)
        existing = check_existing_output(output, request)
        if existing is not None:
            last = run_dir / "weights" / "last.pt"
            if existing[0].get("run_gate_passed") is not True or not last.is_file() or sha256_file(last, deadline) != existing[0].get("last_pt_sha256"):
                fail("T1GR_U6_EXISTING_RUN_NOT_REUSABLE")
            return {"status": "PASS", "idempotent_reuse": True, "public_output_sha256": existing[1]}
        if run_dir.exists():
            fail("T1GR_U6_RUN_DIRECTORY_ALREADY_EXISTS")
        run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        start = utc_now()
        if not parse_utc(recipe["recipe_frozen_at_utc"]) < parse_utc(start):
            fail("T1GR_U6_RECIPE_NOT_BEFORE_TRAINING")
        try:
            import ultralytics
            import yaml
        except Exception:
            fail("T1GR_U6_TRAIN_IMPORT_FAIL")
        if str(ultralytics.__version__) != recipe["environment"]["ultralytics_version"]:
            fail("T1GR_U6_ULTRALYTICS_DRIFT")
        model, identity = build_t1gr_u6_model(checkpoint, recipe, arm=args.arm, seed=int(args.seed))
        for key in ("complete_initial_state_sha256", "reference_initial_state_sha256", "total_parameter_count", "trainable_parameter_count", "stem"):
            if identity.get(key) != expected_identity.get(key):
                fail("T1GR_U6_MODEL_INITIAL_STATE_DRIFT", key)
        overrides = dict(recipe["train_args"])
        overrides["seed"] = int(args.seed)
        overrides["epochs"] = 1 if args.mode == "smoke" else 80
        overrides.update(recipe["eval_args"])
        overrides.update({
            "task": "detect",
            "mode": "train",
            "model": recipe["model_yaml"],
            "data": str(view["dataset_yaml"]),
            "device": recipe["runtime"]["device"],
            "project": str(run_root),
            "name": run_name(args.mode, int(args.seed), args.arm),
            "exist_ok": False,
            "pretrained": False,
            "resume": False,
            "profile": False,
            "verbose": True,
            "time": None,
        })
        expected = _expected(recipe, args.mode, int(args.seed))
        start_state, optimizer, offline, permissions = {}, {}, {}, {}
        phase = "trainer_setup"
        try:
            with ultralytics_offline_guard(bypass_amp_download_check=bool(expected["amp"])) as network, private_umask() as modes:
                offline.update(network)
                permissions.update(modes)
                trainer = T1GRU6Trainer(
                    overrides=overrides,
                    arm=args.arm,
                    seed=int(args.seed),
                    view=view,
                    trace_dir=run_dir / "t1gr_u6_private_trace",
                )
                mismatch = effective_args_mismatch(trainer.args, expected)
                if mismatch:
                    fail("T1GR_U6_EFFECTIVE_ARGS_PREFLIGHT_MISMATCH", f"count={len(mismatch)}")
                trainer.model = model
                trainer.model.args = trainer.args

                def runtime_check(current):
                    deadline.check("T1GR_U6_TRAINING_TIMEOUT")
                    if int(current.batch_size) != 4 or bool(current.amp) is not bool(expected["amp"]):
                        fail("T1GR_U6_BATCH_OR_AMP_DRIFT")
                    if int(getattr(current.train_loader, "num_workers", -1)) != int(expected["workers"]):
                        fail("T1GR_U6_TRAIN_WORKER_DRIFT")
                    if int(getattr(current.test_loader, "num_workers", -1)) != int(expected["workers"]) * 2:
                        fail("T1GR_U6_VAL_WORKER_DRIFT")
                    if first_conv(current.model).in_channels != 6:
                        fail("T1GR_U6_RUNTIME_STEM_DRIFT")
                    if type(getattr(current.train_loader, "sampler", None)).__name__ != "RecipientEpochSampler":
                        fail("T1GR_U6_TRAIN_SAMPLER_DRIFT")

                def on_start(current):
                    runtime_check(current)
                    start_state["sha256"] = state_dict_sha256(current.model.state_dict())
                    start_state["stem"] = stem_contract(current.model)
                    if start_state["sha256"] != identity["complete_initial_state_sha256"]:
                        fail("T1GR_U6_TRAINING_START_STATE_DRIFT")
                    optimizer.update(optimizer_fingerprint(current.optimizer))
                    if "musgd" not in str(optimizer.get("class_name", "")).lower():
                        fail("T1GR_U6_OPTIMIZER_CLASS_DRIFT")

                trainer.add_callback("on_train_start", on_start)
                trainer.add_callback("on_train_epoch_start", lambda current: current.begin_epoch())
                trainer.add_callback("on_train_batch_end", runtime_check)
                trainer.add_callback("on_train_epoch_end", lambda current: (current.finish_epoch(), runtime_check(current)))
                phase = "trainer_train"
                with wall_clock_watchdog(timeout, "T1GR_U6_TRAINING_TIMEOUT"):
                    trainer.train()
                phase = "post_train_runtime_check"
                runtime_check(trainer)
        except BaseException as exc:
            try:
                write_private_failure_report(run_dir, exc, phase, int(recipe["runtime"]["private_traceback_max_bytes"]))
            except Exception:
                pass
            try:
                if run_dir.exists():
                    (run_dir / "T1GR_U6_INCOMPLETE.txt").write_text("No PASS issued; inspect E5_PRIVATE_FAILURE.json.\n", encoding="utf-8")
            except Exception:
                pass
            raise
        last, best = run_dir / "weights" / "last.pt", run_dir / "weights" / "best.pt"
        args_yaml, results_csv = run_dir / "args.yaml", run_dir / "results.csv"
        trace_pairs = run_dir / "t1gr_u6_private_trace" / "source_pairs.private.jsonl"
        trace_depth = run_dir / "t1gr_u6_private_trace" / "depth_records.private.jsonl"
        trace_summary = run_dir / "t1gr_u6_private_trace" / "epoch_summaries.private.json"
        if any(not path.is_file() for path in (last, args_yaml, results_csv, trace_pairs, trace_depth, trace_summary)):
            fail("T1GR_U6_RUN_ARTIFACT_MISSING")
        post_args = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
        mismatch = effective_args_mismatch(post_args, expected)
        if mismatch:
            fail("T1GR_U6_EFFECTIVE_ARGS_POSTRUN_MISMATCH", f"count={len(mismatch)}")
        expected_epochs = 1 if args.mode == "smoke" else 80
        completed = results_csv_epoch_count(results_csv)
        if completed != expected_epochs or len(trainer.t1gr_epoch_summaries) != expected_epochs:
            fail("T1GR_U6_EPOCH_COUNT_DRIFT")
        checkpoint_obj = torch.load(last, map_location="cpu", weights_only=False)
        checkpoint_model = checkpoint_obj.get("ema") or checkpoint_obj.get("model")
        if checkpoint_model is None:
            fail("T1GR_U6_CHECKPOINT_MODEL_MISSING")
        final_stem = _final_stem(checkpoint_model)
        counts = final_stem["channel_nonzero_counts"]
        if final_stem["physical_in_channels"] != 6:
            fail("T1GR_U6_FINAL_STEM_CHANNEL_FAIL")
        if args.arm == "G0-N" and counts[3:] != [0, 0, 0]:
            fail("T1GR_U6_G0_AUX_STEM_CHANGED")
        if args.arm in {"G1-P", "G2-S"} and (counts[3] <= 0 or counts[4:] != [0, 0]):
            fail("T1GR_U6_IR_ARM_STEM_CONTRACT_FAIL")
        if args.arm == "G3-D" and any(value <= 0 for value in counts[3:]):
            fail("T1GR_U6_G3_AUX_STEM_DID_NOT_LEARN")
        finish = utc_now()
        report = {
            "schema": SCHEMA_RUN,
            "script_version": SCRIPT_VERSION,
            "mode": args.mode,
            "arm": args.arm,
            "seed": int(args.seed),
            "suite_position_zero_based": int(launch_row["position"]),
            "lane_position_zero_based": int(launch_row["lane_position"]),
            "training_started_at_utc": start,
            "training_finished_at_utc": finish,
            "recipe_freeze_precedes_training": parse_utc(recipe["recipe_frozen_at_utc"]) < parse_utc(start),
            "spec_file_sha256": sha256_file(spec_path, deadline),
            "recipe_public_sha256": sha256_file(recipe_path, deadline),
            "preflight_public_sha256": sha256_file(preflight_path, deadline),
            "smoke_audit_sha256": sha256_file(smoke_audit_path, deadline) if smoke_audit_path else None,
            "u6_view_manifest_private_sha256": view_sha,
            "base_checkpoint_sha256": checkpoint_sha,
            "environment": environment,
            "implementation_source_hashes": sources,
            "legacy_primary_suite_source_hashes": upstream,
            "complete_initial_state_sha256": identity["complete_initial_state_sha256"],
            "training_start_state_sha256": start_state.get("sha256"),
            "initial_stem": identity["stem"],
            "final_stem": final_stem,
            "optimizer": optimizer,
            "physical_first_conv_in_channels": 6,
            "input_condition": ARM_POLICIES[args.arm]["meaning"],
            "unknown_scale_jpg_treatment": "QUARANTINE_AS_MISSING",
            "epochs_completed": completed,
            "primary_checkpoint": "last.pt",
            "best_checkpoint_role": "DIAGNOSTIC_ONLY",
            "last_pt_sha256": sha256_file(last, deadline),
            "best_pt_sha256": sha256_file(best, deadline) if best.is_file() else None,
            "effective_args_yaml_sha256": sha256_file(args_yaml, deadline),
            "results_csv_sha256": sha256_file(results_csv, deadline),
            "source_pairs_private_sha256": sha256_file(trace_pairs, deadline),
            "depth_records_private_sha256": sha256_file(trace_depth, deadline),
            "trace_summary_sha256": sha256_file(trace_summary, deadline),
            "epoch_modality_contract_passed": all(row.get("modality_contract_passed") is True for row in trainer.t1gr_epoch_summaries),
            "external_network_integrations": offline,
            "private_artifact_permissions": permissions,
            "run_gate_passed": True,
            "formal_training_authorized": False,
            "dev_suite_eval_authorized": args.mode == "formal",
            "legacy_primary_g_suite_unchanged": True,
            "final_holdout_open_authorized": False,
            "production_go": False,
        }
        assert_public_safe(report)
        digest, _ = atomic_json_write(output, report, private=False, request_fingerprint=request)
        return {
            "status": "PASS",
            "idempotent_reuse": False,
            "public_output_sha256": digest,
            "run_dir": str(run_dir),
            "epochs_completed": completed,
            "final_holdout_open_authorized": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--u6-view-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--run-root", required=True)
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
