#!/usr/bin/env python3
"""E5 preflight: verify recipe/view/checkpoint/runtime and physical YOLO26 head before any training."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from multimodal.t1gr_secure_io import (
 Deadline,assert_public_safe,atomic_json_write,ensure_private_input,ensure_public_output,
 ensure_repo_input,fail,file_lock,read_json_bounded,safe_error_message,sha256_file,sha256_json
)
from multimodal.t1gr_e5_core import (
 FROZEN_E5_SECURITY_POLICY_SHA256,FROZEN_E5_TRAINING_SPEC_SHA256,
 SCHEMA_PREFLIGHT,SCHEMA_RECIPE,SCHEMA_VIEW_PUBLIC,compare_environment,environment_probe,payload_ok,verify_view_tree
)

SCRIPT_VERSION="t1gr-e5-step1-preflight-hardened-v1"

def build_model(checkpoint:Path,recipe:dict):
    try:
        import torch
        from ultralytics.nn.tasks import DetectionModel,yaml_model_load
    except Exception:fail("E5_MODEL_IMPORT_FAIL")
    try:
        d=yaml_model_load(recipe["model_yaml"]);d["nc"]=12;d["end2end"]=True
        model=DetectionModel(d,ch=3,nc=12,verbose=False)
        ckpt=torch.load(checkpoint,map_location="cpu",weights_only=False)
        src=ckpt["model"].float().state_dict()
        dst=model.state_dict()
        compatible={k:v for k,v in src.items() if k in dst and tuple(v.shape)==tuple(dst[k].shape)}
        if not compatible:fail("E5_PRETRAIN_TRANSFER_EMPTY")
        model.load_state_dict(compatible,strict=False)
    except KeyError:fail("E5_CHECKPOINT_MODEL_KEY_MISSING")
    except Exception as e:
        if hasattr(e,"code"):raise
        fail("E5_MODEL_BUILD_OR_LOAD_FAIL")
    head=model.model[-1]
    return model,{
        "physical_nc":int(getattr(head,"nc",-1)),
        "end2end":bool(getattr(head,"end2end",getattr(model,"end2end",False))),
        "destination_state_keys":len(dst),"source_state_keys":len(src),
        "transferred_state_keys":len(compatible),
        "transfer_fraction_of_destination":len(compatible)/max(1,len(dst)),
    }

def run(a):
    repo=ROOT.resolve(strict=True)
    secp=ensure_repo_input(repo,"config/t1gr_e5_security_policy.json","config")
    if sha256_file(secp)!=FROZEN_E5_SECURITY_POLICY_SHA256: fail("E5_SECURITY_POLICY_SHA_DRIFT")
    sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v1")
    rp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_recipe_public.json","reports/step4_t1gr")
    vpubp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_view_public.json","reports/step4_t1gr")
    td_p=ensure_private_input(Path(a.train_dev_access),repo)
    vm_p=ensure_private_input(Path(a.view_manifest),repo)
    out=ensure_public_output(repo,"reports/step4_t1gr/e5_step1_preflight_public.json",sec["public_output_prefix"])
    ck=Path(a.base_checkpoint).expanduser().resolve(strict=False)
    if not ck.is_file():fail("BASE_CHECKPOINT_NOT_FOUND")
    deadline=Deadline(float(a.timeout_seconds or sec["view_verify_timeout_seconds"]))
    with file_lock(out.with_suffix(out.suffix+".lock"),5.0,900.0):
        recipe=read_json_bounded(rp,int(sec["max_public_json_bytes"]),SCHEMA_RECIPE)
        vpub=read_json_bounded(vpubp,int(sec["max_public_json_bytes"]),SCHEMA_VIEW_PUBLIC)
        td=read_json_bounded(td_p,int(sec["max_private_json_bytes"]))
        if not payload_ok(recipe) or not payload_ok(vpub) or not payload_ok(td):fail("E5_PREFLIGHT_INPUT_INTEGRITY_FAIL")
        if vpub.get("view_gate_passed") is not True:fail("E5_VIEW_GATE_NOT_PASS")
        if sha256_file(vm_p,deadline)!=vpub.get("view_manifest_private_sha256"):fail("E5_VIEW_PRIVATE_SHA_DRIFT")
        vr=verify_view_tree(vm_p,recipe,td,deadline)
        ck_sha=sha256_file(ck,deadline)
        if ck_sha!=recipe["base_checkpoint_sha256"]:fail("E5_CHECKPOINT_SHA_DRIFT")
        env=environment_probe();compare_environment(env,recipe["environment"])
        _,model_info=build_model(ck,recipe)
        if model_info["physical_nc"]!=12:fail("E5_PHYSICAL_HEAD_NC_FAIL")
        if model_info["end2end"] is not True:fail("E5_HEAD_END2END_FAIL")
        request_fp=sha256_json({"script":SCRIPT_VERSION,"recipe":sha256_file(rp,deadline),
                               "view_public":sha256_file(vpubp,deadline),"view_private":sha256_file(vm_p,deadline),
                               "train_dev":sha256_file(td_p,deadline),"checkpoint":ck_sha,"environment":env})
        report={
            "schema":SCHEMA_PREFLIGHT,"script_version":SCRIPT_VERSION,
            "recipe_public_sha256":sha256_file(rp,deadline),
            "view_public_sha256":sha256_file(vpubp,deadline),
            "view_manifest_private_sha256":sha256_file(vm_p,deadline),
            "train_dev_access_private_sha256":sha256_file(td_p,deadline),
            "base_checkpoint_sha256":ck_sha,
            "environment":env,"model_preflight":model_info,
            "train_count":vr["train_count"],"dev_count":vr["dev_count"],
            "ids_commitments":recipe["ids_commitments"],
            "final_holdout_ids_available_to_preflight":False,
            "preflight_gate_passed":True,
            "smoke_authorized":True,
            "formal_step1_authorized":False,
        }
        assert_public_safe(report)
        sh,reuse=atomic_json_write(out,report,private=False,request_fingerprint=request_fp)
        return {"status":"PASS","public_output_sha256":sh,"idempotent_reuse":reuse,"smoke_authorized":True}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-dev-access",required=True)
    ap.add_argument("--view-manifest",required=True)
    ap.add_argument("--base-checkpoint",required=True)
    ap.add_argument("--timeout-seconds",type=float,default=None)
    a=ap.parse_args()
    try:print(json.dumps(run(a),ensure_ascii=False,indent=2))
    except KeyboardInterrupt:
        print(json.dumps({"status":"FAIL","error":"USER_INTERRUPT"},ensure_ascii=False),file=sys.stderr);raise SystemExit(130)
    except Exception as e:print(json.dumps({"status":"FAIL","error":safe_error_message(e)},ensure_ascii=False),file=sys.stderr);raise SystemExit(2)
if __name__=="__main__":main()
