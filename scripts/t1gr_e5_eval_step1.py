#!/usr/bin/env python3
"""DEV-only evaluator for the formal Step1 baseline. FINAL HOLDOUT is not an accepted input."""
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
 SCHEMA_EVAL,SCHEMA_RECIPE,SCHEMA_RUN,SCHEMA_VIEW_PUBLIC,compare_environment,
 environment_probe,payload_ok,private_umask,ultralytics_offline_guard,verify_view_tree,wall_clock_watchdog
)

SCRIPT_VERSION="t1gr-e5-step1-dev-eval-hardened-v1"

def private_run_root(path:Path,repo:Path)->Path:
    p=path.expanduser().resolve(strict=False)
    if is_within(p,repo):fail("E5_RUN_ROOT_INSIDE_REPO")
    if not p.is_dir():fail("E5_RUN_ROOT_NOT_FOUND")
    return p

def run(a):
    repo=ROOT.resolve(strict=True)
    secp=ensure_repo_input(repo,"config/t1gr_e5_security_policy.json","config")
    if sha256_file(secp)!=FROZEN_E5_SECURITY_POLICY_SHA256: fail("E5_SECURITY_POLICY_SHA_DRIFT")
    sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v1")
    rp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_recipe_public.json","reports/step4_t1gr")
    vpubp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_view_public.json","reports/step4_t1gr")
    runp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_formal_run_public.json","reports/step4_t1gr")
    td_p=ensure_private_input(Path(a.train_dev_access),repo);vm_p=ensure_private_input(Path(a.view_manifest),repo)
    out=ensure_public_output(repo,"reports/step4_t1gr/e5_step1_eval_public.json",sec["public_output_prefix"])
    recipe=read_json_bounded(rp,int(sec["max_public_json_bytes"]),SCHEMA_RECIPE)
    vpub=read_json_bounded(vpubp,int(sec["max_public_json_bytes"]),SCHEMA_VIEW_PUBLIC)
    rr=read_json_bounded(runp,int(sec["max_public_json_bytes"]),SCHEMA_RUN)
    td=read_json_bounded(td_p,int(sec["max_private_json_bytes"]))
    if not all(payload_ok(x) for x in (recipe,vpub,rr,td)):fail("E5_EVAL_INPUT_INTEGRITY_FAIL")
    if vpub.get("recipe_public_sha256")!=sha256_file(rp): fail("E5_EVAL_VIEW_RECIPE_DRIFT")
    if rr.get("mode")!="formal" or rr.get("run_gate_passed") is not True or rr.get("dev_eval_authorized") is not True:
        fail("E5_FORMAL_RUN_NOT_EVALUABLE")
    run_root=private_run_root(Path(a.run_root),repo)
    run_dir=run_root/"STEP1_RGB_BASELINE";last=run_dir/"weights"/"last.pt"
    if not last.is_file():fail("E5_FORMAL_LAST_PT_MISSING")
    deadline=Deadline(float(recipe["runtime"]["eval_timeout_seconds"]))
    with file_lock(out.with_suffix(out.suffix+".lock"),float(recipe["runtime"]["lock_wait_seconds"]),float(recipe["runtime"]["lock_stale_seconds"])):
        vr=verify_view_tree(vm_p,recipe,td,deadline)
        if sha256_file(last,deadline)!=rr["last_pt_sha256"]:fail("E5_EVAL_LAST_PT_SHA_DRIFT")
        if rr["recipe_public_sha256"]!=sha256_file(rp,deadline) or rr["view_manifest_private_sha256"]!=sha256_file(vm_p,deadline):
            fail("E5_EVAL_RUN_PROVENANCE_DRIFT")
        env=environment_probe();compare_environment(env,recipe["environment"])
        run_root_binding=sha256_json(str(run_root).casefold() if os.name=="nt" else str(run_root))
        request_fp=sha256_json({"script":SCRIPT_VERSION,"recipe":sha256_file(rp,deadline),
                               "view":sha256_file(vm_p,deadline),"formal_run":sha256_file(runp,deadline),
                               "last_pt":rr["last_pt_sha256"],"run_root_binding":run_root_binding})
        from multimodal.t1gr_secure_io import check_existing_output
        ex=check_existing_output(out,request_fp)
        if ex is not None:
            obj,sh=ex
            if obj.get("eval_gate_passed") is not True:fail("E5_EXISTING_EVAL_NOT_PASS")
            return {"status":"PASS","idempotent_reuse":True,"public_output_sha256":sh}
        try:
            import ultralytics
            from ultralytics import YOLO
        except Exception:fail("E5_EVAL_IMPORT_FAIL")
        if str(ultralytics.__version__)!=recipe["environment"]["ultralytics_version"]:fail("E5_EVAL_ULTRALYTICS_DRIFT")
        eval_dir=run_root/"STEP1_RGB_DEV_EVAL"
        if eval_dir.exists(): fail("E5_EVAL_DIRECTORY_ALREADY_EXISTS")
        offline_state={};permission_state={}
        with ultralytics_offline_guard(bypass_amp_download_check=False) as og, private_umask() as pg:
            offline_state.update(og);permission_state.update(pg)
            y=YOLO(str(last))
        head=y.model.model[-1]
        nc=int(getattr(head,"nc",-1));e2e=bool(getattr(head,"end2end",getattr(y.model,"end2end",False)))
        if nc!=12 or not e2e:fail("E5_EVAL_HEAD_DRIFT")
        def val_deadline(_):
            deadline.check("E5_EVAL_TIMEOUT")
        try:
            y.add_callback("on_val_batch_end",val_deadline)
            y.add_callback("on_val_end",val_deadline)
        except Exception:
            fail("E5_EVAL_CALLBACK_REGISTRATION_FAIL")
        ea=dict(recipe["eval_args"])
        if ea.pop("split")!="val":fail("E5_EVAL_SPLIT_DRIFT")
        with ultralytics_offline_guard(bypass_amp_download_check=False) as og, private_umask() as pg:
            offline_state.update(og);permission_state.update(pg)
            with wall_clock_watchdog(float(recipe["runtime"]["eval_timeout_seconds"]),"E5_EVAL_TIMEOUT"):
                result=y.val(data=str(vr["dataset_yaml"]),split="val",device=recipe["runtime"]["device"],
                             project=str(run_root),name="STEP1_RGB_DEV_EVAL",exist_ok=False,verbose=False,**ea)
        box=getattr(result,"box",None)
        if box is None:fail("E5_EVAL_BOX_METRICS_MISSING")
        maps=[float(x) for x in getattr(box,"maps",[])]
        if len(maps)!=12:fail("E5_EVAL_PER_CLASS_METRIC_COUNT_DRIFT")
        report={
            "schema":SCHEMA_EVAL,"script_version":SCRIPT_VERSION,
            "authority":"DEV_ONLY_DIAGNOSTIC_BASELINE",
            "recipe_public_sha256":sha256_file(rp,deadline),
            "view_public_sha256":sha256_file(vpubp,deadline),
            "view_manifest_private_sha256":sha256_file(vm_p,deadline),
            "formal_run_public_sha256":sha256_file(runp,deadline),
            "last_pt_sha256":rr["last_pt_sha256"],
            "environment":env,"actual_eval_count":vr["dev_count"],
            "actual_eval_ids_commitment":recipe["ids_commitments"]["dev"],
            "physical_head_nc":nc,"head_end2end":e2e,
            "eval_args":recipe["eval_args"],
            "external_network_integrations":offline_state,
            "private_artifact_permissions":permission_state,
            "dev_metrics":{"map50_95":float(box.map),"map50":float(box.map50),
                           "per_class_map50_95":{str(i):maps[i] for i in range(12)}},
            "final_holdout_ids_available_to_evaluator":False,
            "final_holdout_open_authorized":False,
            "eval_gate_passed":True,
        }
        assert_public_safe(report)
        sh,_=atomic_json_write(out,report,private=False,request_fingerprint=request_fp)
        return {"status":"PASS","idempotent_reuse":False,"public_output_sha256":sh,
                "dev_map50_95":report["dev_metrics"]["map50_95"]}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-dev-access",required=True)
    ap.add_argument("--view-manifest",required=True)
    ap.add_argument("--run-root",required=True)
    a=ap.parse_args()
    try:print(json.dumps(run(a),ensure_ascii=False,indent=2))
    except KeyboardInterrupt:
        print(json.dumps({"status":"FAIL","error":"USER_INTERRUPT"},ensure_ascii=False),file=sys.stderr);raise SystemExit(130)
    except Exception as e:print(json.dumps({"status":"FAIL","error":safe_error_message(e)},ensure_ascii=False),file=sys.stderr);raise SystemExit(2)
if __name__=="__main__":main()
