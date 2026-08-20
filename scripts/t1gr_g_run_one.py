#!/usr/bin/env python3
"""Execute one suite-authorized T1-GR smoke or formal run."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_e5_core import (  # noqa: E402
    FROZEN_E5_SECURITY_POLICY_SHA256,
    SCHEMA_RECIPE,
    compare_environment,
    effective_args_mismatch,
    environment_probe,
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
from multimodal.t1gr_g_core import ARMS, SEEDS, payload_ok  # noqa: E402
from multimodal.t1gr_g_impl_core import (  # noqa: E402
    SCHEMA_PREFLIGHT,
    SCHEMA_RUN,
    SCHEMA_SMOKE_AUDIT,
    SCHEMA_SUITE_STATE,
    optimizer_contract_snapshot,
    payload_ok as impl_payload_ok,
)
from multimodal.t1gr_g_model import build_t1gr_g_model  # noqa: E402
from multimodal.t1gr_g_runtime import (  # noqa: E402
    T1GRDetectionTrainer,
    frozen_launch_rows,
    implementation_source_hashes,
    read_json,
    run_name,
    run_report_rel,
    validate_frozen_chain,
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

SCRIPT_VERSION = "t1gr-g-run-one-v1"


def _private_run_root(path: Path, repo: Path) -> Path:
    value = path.expanduser().resolve(strict=False)
    if is_within(value, repo):
        fail("T1GR_G_RUN_ROOT_INSIDE_REPO")
    if not value.parent.is_dir() or not os.access(value.parent, os.W_OK):
        fail("T1GR_G_RUN_PARENT_NOT_WRITABLE")
    if value.exists() and not value.is_dir():
        fail("T1GR_G_RUN_ROOT_NOT_DIRECTORY")
    return value


def _validate_suite_state(state: dict, *, mode: str, arm: str, seed: int, run_root: Path) -> dict:
    if state.get("schema") != SCHEMA_SUITE_STATE or not impl_payload_ok(state):
        fail("T1GR_G_SUITE_STATE_INTEGRITY_FAIL")
    if state.get("mode") != mode or state.get("status") != "IN_PROGRESS":
        fail("T1GR_G_SUITE_STATE_MODE_FAIL")
    binding = sha256_json(str(run_root).casefold() if os.name == "nt" else str(run_root))
    if state.get("run_root_binding") != binding:
        fail("T1GR_G_SUITE_RUN_ROOT_BINDING_FAIL")
    rows = state.get("rows")
    current = int(state.get("current_position", -1))
    completed = state.get("completed")
    if not isinstance(rows, list) or len(rows) != 9 or not isinstance(completed, list):
        fail("T1GR_G_SUITE_STATE_MATRIX_FAIL")
    if current < 0 or current >= len(rows) or len(completed) != current:
        fail("T1GR_G_SUITE_STATE_POSITION_FAIL")
    row = rows[current]
    if int(row.get("seed", -1)) != int(seed) or row.get("arm") != arm or int(row.get("position", -1)) != current:
        fail("T1GR_G_SUITE_REQUEST_OUT_OF_ORDER")
    expected_completed = list(range(current))
    if [int(item.get("position", -1)) for item in completed] != expected_completed:
        fail("T1GR_G_SUITE_COMPLETION_ORDER_FAIL")
    return row


def _expected_args(recipe: dict, mode: str, seed: int) -> dict:
    expected = dict(recipe["train_args"])
    expected["seed"] = int(seed)
    expected["epochs"] = 1 if mode == "smoke" else 80
    expected.update(recipe["eval_args"])
    expected.update({
        "resume": False,
        "profile": False,
        "verbose": True,
        "pretrained": False,
        "exist_ok": False,
    })
    return expected


def _preflight_seed_hash(preflight: dict, seed: int) -> str:
    rows = ((preflight.get("model_probe") or {}).get("seed_identity_rows") or [])
    matches = [row for row in rows if int(row.get("seed", -1)) == int(seed)]
    if len(matches) != 1:
        fail("T1GR_G_PREFLIGHT_SEED_IDENTITY_MISSING")
    value = matches[0].get("complete_initial_state_sha256")
    if not isinstance(value, str) or len(value) != 64:
        fail("T1GR_G_PREFLIGHT_INITIAL_HASH_FAIL")
    return value


def run(args) -> dict:
    if args.mode not in {"smoke", "formal"} or args.arm not in ARMS or int(args.seed) not in SEEDS:
        fail("T1GR_G_RUN_REQUEST_INVALID")
    repo = ROOT.resolve(strict=True)
    security_path = ensure_repo_input(repo, "config/t1gr_e5_security_policy.json", "config")
    if sha256_file(security_path) != FROZEN_E5_SECURITY_POLICY_SHA256:
        fail("E5_SECURITY_POLICY_SHA_DRIFT")
    security = read_json_bounded(security_path, 1 << 20, "t1gr-e5-security-policy-v2")
    recipe_path = ensure_repo_input(repo, "reports/step4_t1gr/e5_v2_step1_recipe_public.json", "reports/step4_t1gr")
    design_path = ensure_repo_input(repo, "config/t1gr_g_design.frozen.json", "config")
    spec_path = ensure_repo_input(repo, "config/t1gr_g_implementation_spec.frozen.json", "config")
    preflight_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_g_implementation_preflight_public.json", "reports/step4_t1gr")
    view_public_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_g_multimodal_view_public.json", "reports/step4_t1gr")
    view_manifest_path = ensure_private_input(Path(args.view_manifest), repo)
    suite_state_path = ensure_private_input(Path(args.suite_state), repo)
    checkpoint = Path(args.base_checkpoint).expanduser().resolve(strict=False)
    if not checkpoint.is_file():
        fail("BASE_CHECKPOINT_NOT_FOUND")
    run_root = _private_run_root(Path(args.run_root), repo)
    output = ensure_public_output(repo, run_report_rel(args.mode, args.seed, args.arm), security["public_output_prefix"])
    recipe = read_json_bounded(recipe_path, int(security["max_public_json_bytes"]), SCHEMA_RECIPE)
    design = read_json(design_path)
    spec = read_json(spec_path)
    preflight = read_json_bounded(preflight_path, int(security["max_public_json_bytes"]), SCHEMA_PREFLIGHT)
    view_public = read_json(view_public_path)
    suite_state = read_json(suite_state_path)
    if not all((e5_payload_ok(recipe), e5_payload_ok(preflight), e5_payload_ok(view_public))):
        fail("T1GR_G_RUN_PUBLIC_INPUT_INTEGRITY_FAIL")
    validate_frozen_chain(design, recipe, spec)
    if (
        sha256_file(design_path) != preflight.get("design_file_sha256")
        or sha256_file(spec_path) != preflight.get("implementation_spec_sha256")
        or sha256_file(recipe_path) != preflight.get("recipe_public_sha256")
    ):
        fail("T1GR_G_FROZEN_INPUT_CHANGED_AFTER_PREFLIGHT")
    if preflight.get("preflight_gate_passed") is not True or preflight.get("smoke_training_authorized") is not True:
        fail("T1GR_G_PREFLIGHT_NOT_PASS")
    run_row = _validate_suite_state(
        suite_state, mode=args.mode, arm=args.arm, seed=int(args.seed), run_root=run_root
    )
    if suite_state.get("rows") != frozen_launch_rows(design):
        fail("T1GR_G_SUITE_STATE_FROZEN_ORDER_DRIFT")
    smoke_audit_path = None
    smoke_audit = None
    if args.mode == "formal":
        smoke_audit_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_g_smoke_audit_public.json", "reports/step4_t1gr")
        smoke_audit = read_json_bounded(smoke_audit_path, int(security["max_public_json_bytes"]), SCHEMA_SMOKE_AUDIT)
        if (
            not e5_payload_ok(smoke_audit)
            or smoke_audit.get("smoke_gate_passed") is not True
            or smoke_audit.get("multiseed_training_authorized") is not True
            or smoke_audit.get("final_holdout_open_authorized") is not False
        ):
            fail("T1GR_G_FORMAL_NOT_AUTHORIZED")
    timeout = float(
        recipe["runtime"]["smoke_timeout_seconds"]
        if args.mode == "smoke"
        else recipe["runtime"]["formal_timeout_seconds"]
    )
    deadline = Deadline(timeout)
    with file_lock(
        output.with_suffix(output.suffix + ".lock"),
        float(recipe["runtime"]["lock_wait_seconds"]),
        float(recipe["runtime"]["lock_stale_seconds"]),
    ):
        view_manifest_sha = sha256_file(view_manifest_path, deadline)
        if view_manifest_sha != view_public.get("view_manifest_private_sha256"):
            fail("T1GR_G_RUN_VIEW_PRIVATE_SHA_DRIFT")
        view = verify_multimodal_view(view_manifest_path, recipe, deadline=deadline)
        checkpoint_sha = sha256_file(checkpoint, deadline)
        if checkpoint_sha != recipe["base_checkpoint_sha256"]:
            fail("T1GR_G_RUN_CHECKPOINT_SHA_DRIFT")
        source_hashes = implementation_source_hashes(repo)
        if source_hashes != preflight.get("implementation_source_hashes"):
            fail("T1GR_G_IMPLEMENTATION_CHANGED_AFTER_PREFLIGHT")
        environment = environment_probe()
        compare_environment(environment, recipe["environment"])
        initial_expected = _preflight_seed_hash(preflight, int(args.seed))
        report_inputs = {
            "script": SCRIPT_VERSION,
            "mode": args.mode,
            "arm": args.arm,
            "seed": int(args.seed),
            "recipe": sha256_file(recipe_path, deadline),
            "design": sha256_file(design_path, deadline),
            "spec": sha256_file(spec_path, deadline),
            "preflight": sha256_file(preflight_path, deadline),
            "view": view_manifest_sha,
            "checkpoint": checkpoint_sha,
            "suite_state": sha256_file(suite_state_path, deadline),
            "smoke_audit": sha256_file(smoke_audit_path, deadline) if smoke_audit_path else None,
            "initial_state": initial_expected,
            "run_root_binding": sha256_json(str(run_root).casefold() if os.name == "nt" else str(run_root)),
        }
        request_fingerprint = sha256_json(report_inputs)
        existing = check_existing_output(output, request_fingerprint)
        run_dir = run_root / run_name(args.mode, int(args.seed), args.arm)
        if existing is not None:
            report, digest = existing
            last = run_dir / "weights" / "last.pt"
            if report.get("run_gate_passed") is not True or not last.is_file():
                fail("T1GR_G_EXISTING_RUN_NOT_REUSABLE")
            if sha256_file(last, deadline) != report.get("last_pt_sha256"):
                fail("T1GR_G_EXISTING_LAST_SHA_DRIFT")
            return {"status": "PASS", "idempotent_reuse": True, "public_output_sha256": digest}
        if run_dir.exists():
            fail("T1GR_G_RUN_DIRECTORY_ALREADY_EXISTS")
        run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        start = utc_now()
        if not parse_utc(recipe["recipe_frozen_at_utc"]) < parse_utc(start):
            fail("T1GR_G_RECIPE_NOT_BEFORE_TRAINING")
        try:
            import ultralytics
            import yaml
        except Exception:
            fail("T1GR_G_TRAINER_IMPORT_FAIL")
        if str(ultralytics.__version__) != recipe["environment"]["ultralytics_version"]:
            fail("T1GR_G_ULTRALYTICS_VERSION_DRIFT")
        model, identity = build_t1gr_g_model(checkpoint, recipe, arm=args.arm, seed=int(args.seed))
        if identity["complete_initial_state_sha256"] != initial_expected:
            fail("T1GR_G_MODEL_INITIAL_STATE_DRIFT")
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
        expected = _expected_args(recipe, args.mode, int(args.seed))
        optimizer_public: dict = {}
        start_state: dict = {}
        offline_state: dict = {}
        permission_state: dict = {}
        phase = "trainer_setup"
        try:
            with ultralytics_offline_guard(bypass_amp_download_check=bool(expected["amp"])) as offline, private_umask() as permissions:
                offline_state.update(offline)
                permission_state.update(permissions)
                trainer = T1GRDetectionTrainer(
                    overrides=overrides,
                    arm=args.arm,
                    seed=int(args.seed),
                    view=view,
                    trace_dir=run_dir / "t1gr_private_trace",
                )
                phase = "effective_args_preflight"
                mismatches = effective_args_mismatch(trainer.args, expected)
                if mismatches:
                    fail("T1GR_G_EFFECTIVE_ARGS_PREFLIGHT_MISMATCH", f"count={len(mismatches)}")
                trainer.model = model
                trainer.model.args = trainer.args

                def runtime_check(current):
                    deadline.check("T1GR_G_TRAINING_TIMEOUT")
                    if int(current.batch_size) != 4:
                        fail("T1GR_G_BATCH_AUTO_REDUCED")
                    if bool(current.amp) is not True:
                        fail("T1GR_G_AMP_DRIFT")
                    if int(getattr(current.train_loader, "num_workers", -1)) != 8:
                        fail("T1GR_G_TRAIN_WORKER_DRIFT")
                    if int(getattr(current.test_loader, "num_workers", -1)) != 16:
                        fail("T1GR_G_VALIDATION_WORKER_DRIFT")
                    if type(getattr(current.train_loader, "sampler", None)).__name__ != "RecipientEpochSampler":
                        fail("T1GR_G_TRAIN_SAMPLER_DRIFT")

                def on_start(current):
                    runtime_check(current)
                    start_state["sha256"] = state_dict_sha256(current.model.state_dict())
                    if start_state["sha256"] != initial_expected:
                        fail("T1GR_G_TRAINING_START_STATE_DRIFT")
                    contract = optimizer_contract_snapshot(current.model, current.optimizer)
                    private_path = run_dir / "t1gr_private_trace" / "optimizer_contract.private.json"
                    private_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    private_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    optimizer_public.update({
                        "class_name": contract["class_name"],
                        "contract_sha256": contract["contract_sha256"],
                        "group_count": len(contract["groups"]),
                        "all_parameter_count": len(contract["all_parameter_names"]),
                        "trainable_parameter_count": len(contract["trainable_parameter_names"]),
                    })
                    if "musgd" not in contract["class_name"].lower():
                        fail("T1GR_G_OPTIMIZER_CLASS_DRIFT")

                trainer.add_callback("on_train_start", on_start)
                trainer.add_callback("on_train_epoch_start", lambda current: current.begin_epoch())
                trainer.add_callback("on_train_batch_end", runtime_check)
                trainer.add_callback("on_train_epoch_end", lambda current: (current.finish_epoch(), runtime_check(current)))
                phase = "trainer_train"
                with wall_clock_watchdog(timeout, "T1GR_G_TRAINING_TIMEOUT"):
                    trainer.train()
                phase = "post_train_runtime_check"
                runtime_check(trainer)
        except BaseException as exc:
            try:
                write_private_failure_report(
                    run_dir, exc, phase, int(recipe["runtime"]["private_traceback_max_bytes"])
                )
            except Exception:
                pass
            try:
                if run_dir.exists():
                    (run_dir / "T1GR_G_INCOMPLETE.txt").write_text(
                        "No PASS issued. Inspect the private failure and archive this directory explicitly before rerun.\n",
                        encoding="utf-8",
                    )
            except Exception:
                pass
            raise
        if not start_state or not optimizer_public:
            fail("T1GR_G_RUNTIME_CAPTURE_MISSING")
        last = run_dir / "weights" / "last.pt"
        best = run_dir / "weights" / "best.pt"
        args_yaml = run_dir / "args.yaml"
        results_csv = run_dir / "results.csv"
        trace_raw = run_dir / "t1gr_private_trace" / "source_pairs.private.jsonl"
        trace_summary = run_dir / "t1gr_private_trace" / "epoch_summaries.private.json"
        required = (last, args_yaml, results_csv, trace_raw, trace_summary)
        if any(not path.is_file() for path in required):
            fail("T1GR_G_RUN_ARTIFACT_MISSING")
        post = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
        mismatches = effective_args_mismatch(post, expected)
        if mismatches:
            fail("T1GR_G_EFFECTIVE_ARGS_POSTRUN_MISMATCH", f"count={len(mismatches)}")
        completed_epochs = results_csv_epoch_count(results_csv)
        expected_epochs = 1 if args.mode == "smoke" else 80
        if completed_epochs != expected_epochs or len(trainer.t1gr_epoch_summaries) != expected_epochs:
            fail("T1GR_G_EPOCH_COUNT_DRIFT")
        if not all(row["source_condition_passed"] for row in trainer.t1gr_epoch_summaries):
            fail("T1GR_G_SOURCE_TRACE_NOT_PASS")
        mechanism = trainer.model.mechanism_stats()
        forward_trace = trainer.model.last_forward_trace
        mechanism_ok = bool(
            forward_trace.get("p3_direct_injection_count") == 0
            and forward_trace.get("p4_direct_injection_count") == 0
            and forward_trace.get("p5_direct_injection_count") == 1
            and (
                (args.arm == "G0-N" and forward_trace.get("t0_loss_connected_aux") is False and mechanism.get("used_rms") == 0.0)
                or (args.arm in {"G1-P", "G2-S"} and forward_trace.get("t0_loss_connected_aux") is True)
            )
        )
        if not mechanism_ok:
            fail("T1GR_G_MECHANISM_RUNTIME_FAIL")
        finish = utc_now()
        report = {
            "schema": SCHEMA_RUN,
            "script_version": SCRIPT_VERSION,
            "mode": args.mode,
            "arm": args.arm,
            "seed": int(args.seed),
            "suite_position_zero_based": int(run_row["position"]),
            "status": "SMOKE_COMPLETE" if args.mode == "smoke" else "FORMAL_TRAIN_COMPLETE",
            "training_started_at_utc": start,
            "training_finished_at_utc": finish,
            "recipe_public_sha256": sha256_file(recipe_path, deadline),
            "design_file_sha256": sha256_file(design_path, deadline),
            "implementation_spec_sha256": sha256_file(spec_path, deadline),
            "preflight_public_sha256": sha256_file(preflight_path, deadline),
            "smoke_audit_sha256": sha256_file(smoke_audit_path, deadline) if smoke_audit_path else None,
            "view_manifest_private_sha256": view_manifest_sha,
            "base_checkpoint_sha256": checkpoint_sha,
            "environment": environment,
            "model_class": identity["model_class"],
            "model_treatment_id": identity["model_treatment_id"],
            "complete_initial_state_sha256": identity["complete_initial_state_sha256"],
            "training_start_state_sha256": start_state["sha256"],
            "initialization_claim": "AUDITABLE_INITIALIZATION_ONLY",
            "numerical_repeatability_claimed": False,
            "optimizer": optimizer_public,
            "effective_args_yaml_sha256": sha256_file(args_yaml, deadline),
            "effective_training_args_sha256": sha256_json(expected),
            "results_csv_sha256": sha256_file(results_csv, deadline),
            "last_pt_sha256": sha256_file(last, deadline),
            "best_pt_sha256": sha256_file(best, deadline) if best.is_file() else None,
            "source_pairs_private_sha256": sha256_file(trace_raw, deadline),
            "source_trace_summary_private_sha256": sha256_file(trace_summary, deadline),
            "epoch_trace_summaries": trainer.t1gr_epoch_summaries,
            "mechanism": mechanism,
            "mechanism_runtime_passed": mechanism_ok,
            "epochs_expected": expected_epochs,
            "epochs_completed": completed_epochs,
            "actual_batch_size": int(trainer.batch_size),
            "actual_train_workers": int(trainer.train_loader.num_workers),
            "actual_validation_workers": int(trainer.test_loader.num_workers),
            "actual_train_sampler": type(trainer.train_loader.sampler).__name__,
            "primary_checkpoint": "last.pt",
            "best_checkpoint_role": "DIAGNOSTIC_ONLY",
            "validation_ir": "ZERO_IR",
            "implementation_source_hashes": source_hashes,
            "external_network_integrations": offline_state,
            "private_artifact_permissions": permission_state,
            "final_holdout_ids_available_to_runner": False,
            "final_holdout_open_authorized": False,
            "run_gate_passed": True,
            "formal_suite_authorized_by_this_run": False,
            "dev_suite_eval_authorized": args.mode == "formal",
        }
        assert_public_safe(report)
        digest, _ = atomic_json_write(output, report, private=False, request_fingerprint=request_fingerprint)
        return {
            "status": "PASS",
            "idempotent_reuse": False,
            "public_output_sha256": digest,
            "mode": args.mode,
            "arm": args.arm,
            "seed": int(args.seed),
            "epochs_completed": completed_epochs,
            "final_holdout_open_authorized": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("smoke", "formal"))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--view-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--suite-state", required=True)
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
