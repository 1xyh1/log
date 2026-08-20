#!/usr/bin/env python3
"""Final E5 auditor. Opens no FINAL-HOLDOUT artifact and authorizes only T1-GR design entry."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from multimodal.t1gr_secure_io import (
 assert_public_safe,atomic_json_write,ensure_public_output,ensure_repo_input,fail,file_lock,
 read_json_bounded,safe_error_message,sha256_file,sha256_json
)
from multimodal.t1gr_e5_core import (
 FROZEN_E5_SECURITY_POLICY_SHA256,FROZEN_E5_TRAINING_SPEC_SHA256,
 SCHEMA_EVAL,SCHEMA_FINAL,SCHEMA_PREFLIGHT,SCHEMA_RECIPE,SCHEMA_RUN,SCHEMA_VIEW_PUBLIC,payload_ok,parse_utc
)
SCRIPT_VERSION="t1gr-e5-v2-final-audit-hardened-v2"

def run(a):
 repo=ROOT.resolve(strict=True)
 secp=ensure_repo_input(repo,"config/t1gr_e5_security_policy.json","config")
 if sha256_file(secp)!=FROZEN_E5_SECURITY_POLICY_SHA256: fail("E5_SECURITY_POLICY_SHA_DRIFT")
 sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v2")
 paths={
  "e4_freeze":ensure_repo_input(repo,"reports/step4_t1gr/e4_split_freeze_public.json","reports/step4_t1gr"),
  "e4_verify":ensure_repo_input(repo,"reports/step4_t1gr/e4_seal_verification_public.json","reports/step4_t1gr"),
  "recipe":ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_recipe_public.json","reports/step4_t1gr"),
  "view":ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_view_public.json","reports/step4_t1gr"),
  "preflight":ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_preflight_public.json","reports/step4_t1gr"),
  "smoke":ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_smoke_public.json","reports/step4_t1gr"),
  "formal":ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_formal_run_public.json","reports/step4_t1gr"),
  "eval":ensure_repo_input(repo,"reports/step4_t1gr/e5_v2_step1_eval_public.json","reports/step4_t1gr"),
 }
 out=ensure_public_output(repo,"reports/step4_t1gr/e5_v2_final_audit_public.json",sec["public_output_prefix"])
 with file_lock(out.with_suffix(out.suffix+".lock"),5.0,900.0):
  obj={k:read_json_bounded(p,int(sec["max_public_json_bytes"])) for k,p in paths.items()}
  if obj["recipe"].get("schema")!=SCHEMA_RECIPE or obj["view"].get("schema")!=SCHEMA_VIEW_PUBLIC or obj["preflight"].get("schema")!=SCHEMA_PREFLIGHT:
   fail("E5_FINAL_SCHEMA_FAIL")
  if obj["smoke"].get("schema")!=SCHEMA_RUN or obj["formal"].get("schema")!=SCHEMA_RUN or obj["eval"].get("schema")!=SCHEMA_EVAL:
   fail("E5_FINAL_SCHEMA_FAIL")
  if not all(payload_ok(x) for x in obj.values()):fail("E5_FINAL_INPUT_INTEGRITY_FAIL")
  recipe_sha=sha256_file(paths["recipe"])
  view_sha=sha256_file(paths["view"]);pre_sha=sha256_file(paths["preflight"]);smoke_sha=sha256_file(paths["smoke"]);formal_sha=sha256_file(paths["formal"])
  checks={
   "e4_seal_pass":obj["e4_freeze"].get("seal_gate_passed") is True,
   "e4_verify_pass":obj["e4_verify"].get("seal_verification_passed") is True and obj["e4_verify"].get("e5_entry_authorized") is True,
   "e4_holdout_closed":obj["e4_verify"].get("final_holdout_open_authorized") is False,
   "recipe_after_e4":parse_utc(obj["e4_freeze"]["freeze_timestamp_utc"])<parse_utc(obj["recipe"]["recipe_frozen_at_utc"]),
   "view_pass":obj["view"].get("view_gate_passed") is True and obj["view"].get("final_holdout_ids_available_to_view") is False,
   "view_recipe_pin":obj["view"].get("recipe_public_sha256")==recipe_sha,
   "preflight_pass":obj["preflight"].get("preflight_gate_passed") is True and obj["preflight"].get("smoke_authorized") is True,
   "preflight_recipe_pin":obj["preflight"].get("recipe_public_sha256")==recipe_sha,
   "preflight_view_pin":obj["preflight"].get("view_public_sha256")==view_sha,
   "smoke_recipe_pin":obj["smoke"].get("recipe_public_sha256")==recipe_sha,
   "smoke_view_pin":obj["smoke"].get("view_public_sha256")==view_sha,
   "smoke_preflight_pin":obj["smoke"].get("preflight_public_sha256")==pre_sha,
   "smoke_pass":obj["smoke"].get("mode")=="smoke" and obj["smoke"].get("run_gate_passed") is True and obj["smoke"].get("formal_step1_authorized_after_smoke") is True,
   "formal_pass":obj["formal"].get("mode")=="formal" and obj["formal"].get("run_gate_passed") is True and obj["formal"].get("dev_eval_authorized") is True,
   "formal_recipe_pin":obj["formal"].get("recipe_public_sha256")==recipe_sha,
   "formal_view_pin":obj["formal"].get("view_public_sha256")==view_sha,
   "formal_preflight_pin":obj["formal"].get("preflight_public_sha256")==pre_sha,
   "formal_smoke_pin":obj["formal"].get("smoke_report_sha256")==smoke_sha,
   "initial_state_preflight_smoke_pin":isinstance(obj["preflight"].get("model_preflight",{}).get("model_initial_state_sha256"),str) and obj["preflight"].get("model_preflight",{}).get("model_initial_state_sha256")==obj["smoke"].get("model_initial_state_sha256"),
   "initial_state_preflight_formal_pin":isinstance(obj["preflight"].get("model_preflight",{}).get("model_initial_state_sha256"),str) and obj["preflight"].get("model_preflight",{}).get("model_initial_state_sha256")==obj["formal"].get("model_initial_state_sha256"),
   "smoke_training_start_state_pin":obj["smoke"].get("model_initial_state_sha256")==obj["smoke"].get("training_start_state_sha256"),
   "formal_training_start_state_pin":obj["formal"].get("model_initial_state_sha256")==obj["formal"].get("training_start_state_sha256"),
   "untransferred_state_preflight_smoke_pin":isinstance(obj["preflight"].get("model_preflight",{}).get("untransferred_initial_state_sha256"),str) and obj["preflight"].get("model_preflight",{}).get("untransferred_initial_state_sha256")==obj["smoke"].get("untransferred_initial_state_sha256"),
   "untransferred_state_preflight_formal_pin":isinstance(obj["preflight"].get("model_preflight",{}).get("untransferred_initial_state_sha256"),str) and obj["preflight"].get("model_preflight",{}).get("untransferred_initial_state_sha256")==obj["formal"].get("untransferred_initial_state_sha256"),
   "formal_exact_epochs":obj["formal"].get("epochs_completed")==obj["formal"].get("epochs_expected")==obj["recipe"]["train_args"]["epochs"],
   "eval_pass":obj["eval"].get("eval_gate_passed") is True and obj["eval"].get("authority")=="DEV_ONLY_DIAGNOSTIC_BASELINE",
   "eval_recipe_pin":obj["eval"].get("recipe_public_sha256")==recipe_sha,
   "eval_view_pin":obj["eval"].get("view_public_sha256")==view_sha,
   "eval_formal_pin":obj["eval"].get("formal_run_public_sha256")==formal_sha,
   "dev_commitment_pin":obj["eval"].get("actual_eval_ids_commitment")==obj["recipe"]["ids_commitments"]["dev"],
   "all_holdout_access_false":all(x.get(k) is False for x,k in (
      (obj["view"],"final_holdout_ids_available_to_view"),
      (obj["preflight"],"final_holdout_ids_available_to_preflight"),
      (obj["smoke"],"final_holdout_ids_available_to_runner"),
      (obj["formal"],"final_holdout_ids_available_to_runner"),
      (obj["eval"],"final_holdout_ids_available_to_evaluator"),
   )),
   "holdout_never_opened":all(x.get("final_holdout_open_authorized") is False for x in (obj["smoke"],obj["formal"],obj["eval"])),
  }
  passed=all(checks.values())
  request_fp=sha256_json({"script":SCRIPT_VERSION,"inputs":{k:sha256_file(v) for k,v in paths.items()}})
  report={
   "schema":SCHEMA_FINAL,"script_version":SCRIPT_VERSION,
   "checks":checks,"passed_count":sum(checks.values()),"total_count":len(checks),
   "e5_gate_passed":passed,
   "step1_baseline_status":"ACCEPTED_DEV_ONLY" if passed else "HOLD",
   "step1_dev_map50_95":obj["eval"].get("dev_metrics",{}).get("map50_95") if passed else None,
   "final_holdout_open_authorized":False,
   "t1gr_design_entry_authorized":passed,
   "t1gr_multiseed_training_authorized":False,
   "next_action":"freeze T1-GR G0-G1-G2 multi-seed design; FINAL HOLDOUT remains sealed" if passed else "fix E5 evidence gates",
  }
  assert_public_safe(report)
  sh,reuse=atomic_json_write(out,report,private=False,request_fingerprint=request_fp)
  if not passed:fail("E5_FINAL_GATE_HOLD")
  return {"status":"PASS","public_output_sha256":sh,"idempotent_reuse":reuse,"t1gr_design_entry_authorized":True}
def main():
 ap=argparse.ArgumentParser()
 a=ap.parse_args()
 try:print(json.dumps(run(a),ensure_ascii=False,indent=2))
 except KeyboardInterrupt:
  print(json.dumps({"status":"FAIL","error":"USER_INTERRUPT"},ensure_ascii=False),file=sys.stderr);raise SystemExit(130)
 except Exception as e:print(json.dumps({"status":"FAIL","error":safe_error_message(e)},ensure_ascii=False),file=sys.stderr);raise SystemExit(2)
if __name__=="__main__":main()
