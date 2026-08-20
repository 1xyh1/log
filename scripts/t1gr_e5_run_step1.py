#!/usr/bin/env python3
"""Hardened Step1 RGB smoke/formal trainer. Scientific args come ONLY from frozen recipe."""
from __future__ import annotations

import argparse,json,os,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from multimodal.t1gr_secure_io import (
 Deadline,assert_public_safe,atomic_json_write,ensure_private_input,ensure_public_output,
 ensure_repo_input,fail,file_lock,is_within,read_json_bounded,safe_error_message,sha256_file,sha256_json
)
from multimodal.t1gr_e5_core import (
 FROZEN_E5_SECURITY_POLICY_SHA256,FROZEN_E5_TRAINING_SPEC_SHA256,
 SCHEMA_PREFLIGHT,SCHEMA_RECIPE,SCHEMA_RUN,SCHEMA_VIEW_PUBLIC,
 build_seeded_model,compare_environment,effective_args_mismatch,environment_probe,optimizer_fingerprint,
 parse_utc,payload_ok,private_umask,results_csv_epoch_count,ultralytics_offline_guard,
 state_dict_sha256,utc_now,verify_view_tree,wall_clock_watchdog,write_private_failure_report
)

SCRIPT_VERSION="t1gr-e5-v2-step1-trainer-hardened-v2"

def private_run_root(path:Path,repo:Path)->Path:
    p=path.expanduser().resolve(strict=False)
    if is_within(p,repo):fail("E5_RUN_ROOT_INSIDE_REPO")
    if not p.parent.is_dir():fail("E5_RUN_PARENT_NOT_FOUND")
    if not os.access(p.parent,os.W_OK):fail("E5_RUN_PARENT_NOT_WRITABLE")
    if p.exists() and not p.is_dir():fail("E5_RUN_ROOT_NOT_DIRECTORY")
    if p.exists() and os.name!="nt":
        try: os.chmod(p,0o700)
        except OSError: fail("E5_RUN_ROOT_PERMISSION_HARDEN_FAIL")
    return p

def expected_effective(recipe:dict,mode:str)->dict:
    d=dict(recipe["train_args"])
    if mode=="smoke":
        d["epochs"]=int(recipe["runtime"]["smoke_epochs"])
    d.update(recipe["eval_args"])
    return d

def validate_smoke(smoke:dict,recipe_sha:str,view_sha:str,ck_sha:str,initial_state_sha:str):
    if smoke.get("schema")!=SCHEMA_RUN or smoke.get("mode")!="smoke" or smoke.get("run_gate_passed") is not True:
        fail("E5_SMOKE_REPORT_NOT_PASS")
    if smoke.get("recipe_public_sha256")!=recipe_sha or smoke.get("view_manifest_private_sha256")!=view_sha:
        fail("E5_SMOKE_PROVENANCE_DRIFT")
    if smoke.get("base_checkpoint_sha256")!=ck_sha:fail("E5_SMOKE_CHECKPOINT_DRIFT")
    if smoke.get("model_initial_state_sha256")!=initial_state_sha:fail("E5_SMOKE_INITIAL_STATE_DRIFT")
    if smoke.get("formal_step1_authorized_after_smoke") is not True:fail("E5_SMOKE_DID_NOT_AUTHORIZE_FORMAL")

def run(a):
    repo=ROOT.resolve(strict=True)
    secp=ensure_repo_input(repo,"config/t1gr_e5_security_policy.json","config")
    if sha256_file(secp)!=FROZEN_E5_SECURITY_POLICY_SHA256: fail("E5_SECURITY_POLICY_SHA_DRIFT")
    sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v2")
    rp=ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_recipe_public.json","reports/step4_t1gr")
    vpubp=ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_view_public.json","reports/step4_t1gr")
    pfp=ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_preflight_public.json","reports/step4_t1gr")
    td_p=ensure_private_input(Path(a.train_dev_access),repo)
    vm_p=ensure_private_input(Path(a.view_manifest),repo)
    recipe=read_json_bounded(rp,int(sec["max_public_json_bytes"]),SCHEMA_RECIPE)
    vpub=read_json_bounded(vpubp,int(sec["max_public_json_bytes"]),SCHEMA_VIEW_PUBLIC)
    pre=read_json_bounded(pfp,int(sec["max_public_json_bytes"]),SCHEMA_PREFLIGHT)
    td=read_json_bounded(td_p,int(sec["max_private_json_bytes"]))
    if int(sec.get("private_failure_traceback_max_bytes",-1))!=int(recipe["runtime"]["private_traceback_max_bytes"]):
        fail("E5_PRIVATE_TRACEBACK_POLICY_DRIFT")
    if not all(payload_ok(x) for x in (recipe,vpub,pre,td)):fail("E5_RUN_INPUT_INTEGRITY_FAIL")
    if pre.get("preflight_gate_passed") is not True or pre.get("smoke_authorized") is not True:fail("E5_PREFLIGHT_NOT_PASS")
    if a.mode not in {"smoke","formal"}:fail("E5_RUN_MODE_INVALID")
    run_root=private_run_root(Path(a.run_root),repo)
    run_name="STEP1_RGB_SMOKE_V2" if a.mode=="smoke" else "STEP1_RGB_BASELINE_V2"
    run_dir=run_root/run_name
    out_rel=("reports/step4_t1gr/e5_v2_step1_smoke_public.json" if a.mode=="smoke"
             else "reports/step4_t1gr/e5_v2_step1_formal_run_public.json")
    out=ensure_public_output(repo,out_rel,sec["public_output_prefix"])
    ck=Path(a.base_checkpoint).expanduser().resolve(strict=False)
    if not ck.is_file():fail("BASE_CHECKPOINT_NOT_FOUND")
    timeout=float(recipe["runtime"]["smoke_timeout_seconds"] if a.mode=="smoke" else recipe["runtime"]["formal_timeout_seconds"])
    deadline=Deadline(timeout)

    with file_lock(out.with_suffix(out.suffix+".lock"),float(recipe["runtime"]["lock_wait_seconds"]),float(recipe["runtime"]["lock_stale_seconds"])):
        ck_sha=sha256_file(ck,deadline)
        recipe_sha=sha256_file(rp,deadline);view_sha=sha256_file(vm_p,deadline)
        if ck_sha!=recipe["base_checkpoint_sha256"]:fail("E5_RUN_CHECKPOINT_SHA_DRIFT")
        if view_sha!=vpub.get("view_manifest_private_sha256"):fail("E5_RUN_VIEW_PRIVATE_SHA_DRIFT")
        vr=verify_view_tree(vm_p,recipe,td,deadline)
        env=environment_probe();compare_environment(env,recipe["environment"])
        if pre.get("recipe_public_sha256")!=recipe_sha or pre.get("view_manifest_private_sha256")!=view_sha:
            fail("E5_RUN_PREFLIGHT_PROVENANCE_DRIFT")
        if pre.get("base_checkpoint_sha256")!=ck_sha:fail("E5_RUN_PREFLIGHT_CHECKPOINT_DRIFT")
        pre_model=pre.get("model_preflight") or {}
        initial_state_sha=pre_model.get("model_initial_state_sha256")
        untransferred_state_sha=pre_model.get("untransferred_initial_state_sha256")
        if not isinstance(initial_state_sha,str) or len(initial_state_sha)!=64:
            fail("E5_PREFLIGHT_INITIAL_STATE_PIN_MISSING")
        if not isinstance(untransferred_state_sha,str) or len(untransferred_state_sha)!=64:
            fail("E5_PREFLIGHT_UNTRANSFERRED_STATE_PIN_MISSING")

        smoke_sha=None
        if a.mode=="formal":
            sp=ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_smoke_public.json","reports/step4_t1gr")
            smoke=read_json_bounded(sp,int(sec["max_public_json_bytes"]),SCHEMA_RUN)
            if not payload_ok(smoke):fail("E5_SMOKE_REPORT_INTEGRITY_FAIL")
            validate_smoke(smoke,recipe_sha,view_sha,ck_sha,initial_state_sha)
            if smoke.get("untransferred_initial_state_sha256")!=untransferred_state_sha:
                fail("E5_SMOKE_UNTRANSFERRED_STATE_DRIFT")
            smoke_sha=sha256_file(sp,deadline)

        run_root_binding=sha256_json(str(run_root).casefold() if os.name=="nt" else str(run_root))
        request_fp=sha256_json({"script":SCRIPT_VERSION,"mode":a.mode,"recipe":recipe_sha,
                               "view":view_sha,"preflight":sha256_file(pfp,deadline),
                               "checkpoint":ck_sha,"smoke_report":smoke_sha,
                               "model_initial_state_sha256":initial_state_sha,
                               "run_root_binding":run_root_binding})
        # A completed same-request public report is idempotent; do not retrain.
        from multimodal.t1gr_secure_io import check_existing_output
        existing=check_existing_output(out,request_fp)
        if existing is not None:
            obj,sh=existing
            if obj.get("run_gate_passed") is not True:fail("E5_EXISTING_RUN_NOT_PASS")
            if obj.get("model_initial_state_sha256")!=initial_state_sha:fail("E5_EXISTING_INITIAL_STATE_DRIFT")
            if not run_dir.is_dir(): fail("E5_EXISTING_RUN_DIRECTORY_MISSING")
            last0=run_dir/"weights"/"last.pt";args0=run_dir/"args.yaml";csv0=run_dir/"results.csv"
            if not last0.is_file() or not args0.is_file() or not csv0.is_file(): fail("E5_EXISTING_RUN_ARTIFACT_MISSING")
            if sha256_file(last0,deadline)!=obj.get("last_pt_sha256"): fail("E5_EXISTING_LAST_PT_SHA_DRIFT")
            if sha256_file(args0,deadline)!=obj.get("effective_args_yaml_sha256"): fail("E5_EXISTING_ARGS_SHA_DRIFT")
            if sha256_file(csv0,deadline)!=obj.get("results_csv_sha256"): fail("E5_EXISTING_RESULTS_SHA_DRIFT")
            if results_csv_epoch_count(csv0)!=int(obj.get("epochs_completed",-1)): fail("E5_EXISTING_EPOCH_COUNT_DRIFT")
            return {"status":"PASS","idempotent_reuse":True,"public_output_sha256":sh}

        if run_dir.exists():
            fail("E5_RUN_DIRECTORY_ALREADY_EXISTS")
        run_root.mkdir(parents=True,exist_ok=True,mode=0o700)
        if os.name!="nt":
            try: os.chmod(run_root,0o700)
            except OSError: fail("E5_RUN_ROOT_PERMISSION_HARDEN_FAIL")
        start=utc_now()
        if not parse_utc(recipe["e4_freeze_timestamp_utc"]) < parse_utc(start):
            fail("E5_E4_FREEZE_NOT_BEFORE_TRAINING")
        if not parse_utc(recipe["recipe_frozen_at_utc"]) < parse_utc(start):
            fail("E5_RECIPE_NOT_BEFORE_TRAINING")

        try:
            import ultralytics,yaml
            from ultralytics.models.yolo.detect.train import DetectionTrainer
        except Exception:fail("E5_TRAINER_IMPORT_FAIL")
        if str(ultralytics.__version__)!=recipe["environment"]["ultralytics_version"]:fail("E5_TRAIN_ULTRALYTICS_VERSION_DRIFT")

        model,model_info=build_seeded_model(ck,recipe)
        if model_info["model_initial_state_sha256"]!=initial_state_sha:
            fail("E5_MODEL_INITIAL_STATE_NOT_REPRODUCIBLE")
        if model_info["untransferred_initial_state_sha256"]!=untransferred_state_sha:
            fail("E5_MODEL_UNTRANSFERRED_STATE_NOT_REPRODUCIBLE")
        head=model.model[-1]
        if int(getattr(head,"nc",-1))!=12:fail("E5_TRAIN_PHYSICAL_HEAD_NC_FAIL")
        if bool(getattr(head,"end2end",getattr(model,"end2end",False))) is not True:fail("E5_TRAIN_HEAD_MODE_FAIL")

        overrides=dict(recipe["train_args"])
        if a.mode=="smoke":overrides["epochs"]=int(recipe["runtime"]["smoke_epochs"])
        overrides.update(recipe["eval_args"])
        fixed_args={
            "task":"detect","mode":"train","model":recipe["model_yaml"],
            "data":str(vr["dataset_yaml"]),"device":recipe["runtime"]["device"],
            "project":str(run_root),"name":run_name,"exist_ok":False,"pretrained":False,
            "resume":False,"profile":False,"verbose":True,"time":None,
        }
        overrides.update(fixed_args)
        expected=expected_effective(recipe,a.mode)
        expected.update({"resume":False,"profile":False,"verbose":True,"pretrained":False,"exist_ok":False})
        optimizer_capture={}
        training_start_state={}
        offline_state={}
        permission_state={}
        phase="trainer_setup"
        try:
            with ultralytics_offline_guard(bypass_amp_download_check=bool(expected["amp"])) as og, private_umask() as pg:
                offline_state.update(og); permission_state.update(pg)
                trainer=DetectionTrainer(overrides=overrides)
                phase="effective_args_preflight"
                mm=effective_args_mismatch(trainer.args,expected)
                if mm:fail("E5_EFFECTIVE_ARGS_PREFLIGHT_MISMATCH",f"count={len(mm)}")
                trainer.model=model;trainer.model.args=trainer.args
                def runtime_check(t):
                    deadline.check("E5_TRAINING_TIMEOUT")
                    if int(t.batch_size)!=int(expected["batch"]): fail("E5_BATCH_AUTO_REDUCED")
                    if bool(t.amp)!=bool(expected["amp"]): fail("E5_EFFECTIVE_AMP_DRIFT")
                    actual_workers=int(getattr(t.train_loader,"num_workers",-1))
                    if actual_workers!=int(expected["workers"]): fail("E5_EFFECTIVE_WORKERS_DRIFT")
                def optimizer_cb(t):
                    runtime_check(t)
                    if not training_start_state:
                        training_start_state["sha256"]=state_dict_sha256(t.model.state_dict())
                        if training_start_state["sha256"]!=initial_state_sha:
                            fail("E5_TRAINING_START_STATE_DRIFT")
                    if not optimizer_capture:
                        optimizer_capture.update(optimizer_fingerprint(t.optimizer))
                        expected_name=str(recipe["train_args"]["optimizer"]).lower()
                        if expected_name not in str(optimizer_capture["class_name"]).lower():
                            fail("E5_OPTIMIZER_CLASS_DRIFT")
                trainer.add_callback("on_train_batch_end",runtime_check)
                trainer.add_callback("on_train_epoch_end",runtime_check)
                trainer.add_callback("on_train_start",optimizer_cb)
                phase="trainer_train"
                with wall_clock_watchdog(timeout,"E5_TRAINING_TIMEOUT"):
                    trainer.train()
                phase="post_train_runtime_check"
                runtime_check(trainer)
                if not training_start_state:fail("E5_TRAINING_START_STATE_NOT_CAPTURED")
        except BaseException as exc:
            # Keep a private traceback and partial run for diagnosis; public stderr stays sanitized.
            try:
                write_private_failure_report(
                    run_dir,exc,phase,int(recipe["runtime"]["private_traceback_max_bytes"])
                )
            except Exception:
                pass
            try:
                if run_dir.exists():
                    (run_dir/"E5_INCOMPLETE.txt").write_text(
                        "E5 v2 PASS not issued. Inspect E5_PRIVATE_FAILURE.json locally; "
                        "archive or remove this run directory explicitly before rerun.\n",encoding="utf-8"
                    )
            except Exception:pass
            raise

        last=run_dir/"weights"/"last.pt";best=run_dir/"weights"/"best.pt"
        args_yaml=run_dir/"args.yaml";results_csv=run_dir/"results.csv"
        if not last.is_file() or not args_yaml.is_file() or not results_csv.is_file():fail("E5_RUN_ARTIFACT_MISSING")
        post=yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
        mm=effective_args_mismatch(post,expected)
        if mm:fail("E5_EFFECTIVE_ARGS_POSTRUN_MISMATCH",f"count={len(mm)}")
        expected_epochs=int(expected["epochs"])
        completed=results_csv_epoch_count(results_csv)
        if completed!=expected_epochs:fail("E5_EPOCH_COUNT_DRIFT",f"completed={completed}")
        finish=utc_now()
        report={
            "schema":SCHEMA_RUN,"script_version":SCRIPT_VERSION,"mode":a.mode,
            "status":"STEP1_SMOKE_COMPLETE" if a.mode=="smoke" else "STEP1_FORMAL_TRAIN_COMPLETE",
            "training_started_at_utc":start,"training_finished_at_utc":finish,
            "e4_freeze_precedes_training":parse_utc(recipe["e4_freeze_timestamp_utc"])<parse_utc(start),
            "recipe_freeze_precedes_training":parse_utc(recipe["recipe_frozen_at_utc"])<parse_utc(start),
            "recipe_public_sha256":recipe_sha,"view_public_sha256":sha256_file(vpubp,deadline),
            "view_manifest_private_sha256":view_sha,"preflight_public_sha256":sha256_file(pfp,deadline),
            "smoke_report_sha256":smoke_sha,
            "base_checkpoint_sha256":ck_sha,"environment":env,
            "train_count":vr["train_count"],"dev_count":vr["dev_count"],
            "ids_commitments":recipe["ids_commitments"],
            "physical_head_nc":12,"head_end2end":True,
            "pretrained_transfer_key_count":model_info["transferred_state_keys"],
            "pretrained_destination_key_count":model_info["destination_state_keys"],
            "untransferred_state_key_count":model_info["untransferred_state_keys"],
            "model_initialization_effective_seed":model_info["model_initialization_effective_seed"],
            "model_initial_state_sha256":model_info["model_initial_state_sha256"],
            "untransferred_initial_state_sha256":model_info["untransferred_initial_state_sha256"],
            "training_start_state_sha256":training_start_state.get("sha256"),
            "optimizer":optimizer_capture,
            "external_network_integrations":offline_state,
            "private_artifact_permissions":permission_state,
            "actual_batch_size":int(trainer.batch_size),
            "actual_amp":bool(trainer.amp),
            "actual_train_workers":int(getattr(trainer.train_loader,"num_workers",-1)),
            "effective_args_yaml_sha256":sha256_file(args_yaml,deadline),
            "results_csv_sha256":sha256_file(results_csv,deadline),
            "last_pt_sha256":sha256_file(last,deadline),
            "best_pt_sha256":sha256_file(best,deadline) if best.is_file() else None,
            "epochs_expected":expected_epochs,"epochs_completed":completed,
            "timeout_seconds":timeout,
            "final_holdout_ids_available_to_runner":False,
            "final_holdout_open_authorized":False,
            "run_gate_passed":True,
            "formal_step1_authorized_after_smoke":a.mode=="smoke",
            "dev_eval_authorized":a.mode=="formal",
        }
        assert_public_safe(report)
        sh,_=atomic_json_write(out,report,private=False,request_fingerprint=request_fp)
        return {"status":"PASS","idempotent_reuse":False,"public_output_sha256":sh,
                "mode":a.mode,"epochs_completed":completed}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",required=True,choices=["smoke","formal"])
    ap.add_argument("--train-dev-access",required=True)
    ap.add_argument("--view-manifest",required=True)
    ap.add_argument("--base-checkpoint",required=True)
    ap.add_argument("--run-root",required=True)
    a=ap.parse_args()
    try:print(json.dumps(run(a),ensure_ascii=False,indent=2))
    except KeyboardInterrupt:
        print(json.dumps({"status":"FAIL","error":"USER_INTERRUPT"},ensure_ascii=False),file=sys.stderr);raise SystemExit(130)
    except Exception as e:
        print(json.dumps({"status":"FAIL","error":safe_error_message(e)},ensure_ascii=False),file=sys.stderr);raise SystemExit(2)
if __name__=="__main__":main()
