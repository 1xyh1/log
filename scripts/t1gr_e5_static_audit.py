#!/usr/bin/env python3
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
OPERATIONAL={
"t1gr_e5_freeze_recipe.py","t1gr_e5_build_rgb_view.py","t1gr_e5_preflight.py",
"t1gr_e5_run_step1.py","t1gr_e5_eval_step1.py","t1gr_e5_final_audit.py",
}
files={p.name:p.read_text(encoding="utf-8") for p in (ROOT/"scripts").glob("t1gr_e5_*.py") if p.name in OPERATIONAL}
runner=files["t1gr_e5_run_step1.py"];view=files["t1gr_e5_build_rgb_view.py"];recipe=files["t1gr_e5_freeze_recipe.py"]
pre=files["t1gr_e5_preflight.py"];eva=files["t1gr_e5_eval_step1.py"];final=files["t1gr_e5_final_audit.py"]
core=(ROOT/"src/multimodal/t1gr_e5_core.py").read_text(encoding="utf-8")
secure=(ROOT/"src/multimodal/t1gr_secure_io.py").read_text(encoding="utf-8")
checks={}
def add(k,v):checks[k]=bool(v)
for name,text in files.items():
    add(name+"_safe_error","safe_error_message" in text)
    add(name+"_no_holdout_sealed_arg","final-holdout-sealed" not in text and "final_holdout_sealed" not in text)
add("optimizer_auto_forbidden","E5_OPTIMIZER_AUTO_FORBIDDEN" in core)
add("training_spec_review_required","REVIEWED_FROZEN" in core)
add("ultralytics_8456_pinned",'!=spec["expected_ultralytics_version"]' in recipe and '"8.4.56"' in core)
add("full_zip_sha_recipe","formal_zip_sha256" in recipe and "sha256_file(zp" in recipe)
add("zip_metadata_pin","formal_zip_metadata_commitment" in recipe)
add("labels_commitment_pin","labels_commitment" in recipe)
add("e4_train_dev_only","E4_TRAIN_DEV_SCHEMA" in core and "E4_TRAIN_DEV_EXPOSES_HOLDOUT_IDS" in core)
add("view_copy_only","write_private_file" in view and "os.replace(tmp,out_root)" in view)
add("view_atomic_directory_commit","os.replace(tmp,out_root)" in view)
add("view_outside_repo","E5_VIEW_ROOT_INSIDE_REPO" in view)
add("view_full_reverify","verify_view_tree" in view)
add("view_detects_extra_files","E5_VIEW_EXTRA_OR_MISSING_FILE" in core)
add("view_hashes_files","E5_VIEW_FILE_SHA_DRIFT" in core)
add("run_outside_repo","E5_RUN_ROOT_INSIDE_REPO" in runner)
add("runner_no_data_cli",'ap.add_argument("--data"' not in runner)
add("runner_no_device_cli",'ap.add_argument("--device"' not in runner)
add("runner_no_epochs_cli",'ap.add_argument("--epochs"' not in runner)
add("runner_no_optimizer_cli",'ap.add_argument("--optimizer"' not in runner)
add("runner_no_project_cli",'ap.add_argument("--project"' not in runner)
add("runner_no_name_cli",'ap.add_argument("--name"' not in runner)
add("runner_fixed_public_output",'ap.add_argument("--public-out"' not in runner)
add("runner_requires_fixed_smoke",'reports/step4_t1gr/e5_step1_smoke_public.json' in runner and 'if a.mode=="formal"' in runner)
add("runner_effective_preflight","E5_EFFECTIVE_ARGS_PREFLIGHT_MISMATCH" in runner)
add("runner_effective_postrun","E5_EFFECTIVE_ARGS_POSTRUN_MISMATCH" in runner)
add("runner_epoch_exact","E5_EPOCH_COUNT_DRIFT" in runner)
add("runner_optimizer_capture","optimizer_fingerprint" in runner)
add("runner_timeout_callback","on_train_batch_end" in runner and "E5_TRAINING_TIMEOUT" in runner)
add("runner_incomplete_marker","E5_INCOMPLETE.txt" in runner)
add("runner_idempotent_artifact_recheck","E5_EXISTING_LAST_PT_SHA_DRIFT" in runner)
add("preflight_physical_nc","E5_PHYSICAL_HEAD_NC_FAIL" in pre)
add("preflight_partial_transfer","transferred_state_keys" in pre)
add("eval_dev_only",'split="val"' in eva and "DEV_ONLY_DIAGNOSTIC_BASELINE" in eva)
add("eval_timeout_callback","on_val_batch_end" in eva and "E5_EVAL_TIMEOUT" in eva)
add("eval_no_checkpoint_cli",'ap.add_argument("--checkpoint"' not in eva)
add("final_holdout_never_opened","all_holdout_access_false" in final and "holdout_never_opened" in final)
add("final_only_design_entry",'"t1gr_multiseed_training_authorized":False' in final)
add("secure_atomic","os.replace" in secure)
add("secure_lock","O_EXCL" in secure)
add("secure_private_guard","PRIVATE_INPUT_INSIDE_REPO" in secure and "PRIVATE_OUTPUT_INSIDE_REPO" in secure)
add("secure_public_scanner","PUBLIC_SENSITIVE_KEY" in secure and "PUBLIC_PATH_STRING_FORBIDDEN" in secure)

frozen=json.loads((ROOT/"config/t1gr_e5_training_spec.frozen.json").read_text())
sec=json.loads((ROOT/"config/t1gr_e5_security_policy.json").read_text())
add("frozen_spec_reviewed",frozen.get("status")=="REVIEWED_FROZEN")
add("frozen_spec_optimizer_explicit",frozen.get("train_args",{}).get("optimizer")=="MuSGD")
add("frozen_spec_device_single",str(frozen.get("runtime",{}).get("device"))=="0")
add("security_disables_integrations",sec.get("ultralytics_external_integrations")=="DISABLED_DURING_TRAIN_AND_EVAL")
add("security_disables_analytics",sec.get("ultralytics_usage_analytics")=="DISABLED_FOR_E5_PROCESS")
add("security_amp_network_probe_bypassed",str(sec.get("amp_network_probe","")).startswith("BYPASSED"))
add("offline_guard_core","ultralytics_offline_guard" in core and "ul_events.enabled = False" in core)
add("amp_probe_patch","trainer_mod.check_amp = lambda model: True" in core)
add("wall_clock_watchdog_runner","wall_clock_watchdog" in runner)
add("wall_clock_watchdog_eval","wall_clock_watchdog" in eva)
add("runtime_batch_gate","E5_BATCH_AUTO_REDUCED" in runner)
add("runtime_amp_gate","E5_EFFECTIVE_AMP_DRIFT" in runner)
add("runtime_workers_gate","E5_EFFECTIVE_WORKERS_DRIFT" in runner)
add("posix_umask","os.umask(0o077)" in core)
add("view_request_binds_root","out_root_binding" in view)
add("run_request_binds_root","run_root_binding" in runner)
add("eval_request_binds_root","run_root_binding" in eva)
add("eval_private_project",'project=str(run_root)' in eva and 'name="STEP1_RGB_DEV_EVAL"' in eva)
add("ultralytics_source_hash_pin","ultralytics_source_sha256" in core)
add("frozen_config_hash_constants","FROZEN_E5_TRAINING_SPEC_SHA256" in core and "FROZEN_E5_SECURITY_POLICY_SHA256" in core)
for name,text in files.items():
    add(name+"_keyboardinterrupt_safe","except KeyboardInterrupt" in text and "USER_INTERRUPT" in text)
public_override_tokens=("--forensic-public","--taxonomy-public","--e4-freeze-public","--e4-verification-public",
 "--training-spec","--recipe","--view-public","--preflight-public","--formal-run-public","--smoke-report")
for name,text in files.items():
    add(name+"_no_mutable_public_input_cli",not any(f'ap.add_argument("{x}"' in text for x in public_override_tokens))

passed=sum(checks.values())
report={"schema":"t1gr-e5-static-audit-v1","checks":checks,"passed":passed,"total":len(checks),"all_passed":passed==len(checks)}
out=ROOT/"reports/step4_t1gr/e5_static_audit.json";out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
if not report["all_passed"]:raise SystemExit(2)
