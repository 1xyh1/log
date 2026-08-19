#!/usr/bin/env python3
"""Verify E4 seal without opening FINAL HOLDOUT to downstream tooling.

The verifier necessarily reads the sealed artifact internally to validate its commitment,
but emits only non-ID public evidence and never returns IDs.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from multimodal.t1gr_secure_io import (
 Deadline,assert_public_safe,atomic_json_write,ensure_private_input,ensure_public_output,
 ensure_repo_input,fail,file_lock,read_json_bounded,require_keys,safe_error_message,
 sha256_file,sha256_json,validate_identifier
)

SCRIPT_VERSION="t1gr-e4-seal-verifier-v1"
POLICY_REL="config/t1gr_e4_seal_policy.json"

def payload_ok(o):
 c=o.get("payload_sha256")
 if not isinstance(c,str) or len(c)!=64:return False
 b=dict(o);b.pop("payload_sha256",None);b.pop("request_fingerprint",None)
 return sha256_json(b)==c

def ids(xs):
 if not isinstance(xs,list):fail("BAD_ID_LIST")
 out=[validate_identifier(x) for x in xs]
 if len(out)!=len(set(out)):fail("DUPLICATE_ID")
 return sorted(out)

def run(a):
 repo=ROOT.resolve(strict=True)
 pp=ensure_repo_input(repo,POLICY_REL,"config");p=read_json_bounded(pp,1<<20,"t1gr-e4-seal-policy-v1")
 sec=p["security"];dl=Deadline(float(a.timeout_seconds or sec["verify_timeout_seconds"]))
 pubp=ensure_repo_input(repo,a.freeze_public,"reports/step4_t1gr")
 td=ensure_private_input(Path(a.train_dev_access),repo)
 ho=ensure_private_input(Path(a.final_holdout_sealed),repo)
 rc=ensure_private_input(Path(a.seal_receipt_private),repo)
 out=ensure_public_output(repo,a.public_out,sec["public_output_prefix"])
 with file_lock(out.with_suffix(out.suffix+".lock"),float(sec["lock_wait_seconds"]),float(sec["lock_stale_seconds"])):
  public=read_json_bounded(pubp,int(sec["max_public_json_bytes"]),"t1gr-e4-split-freeze-public-v1")
  t=read_json_bounded(td,int(sec["max_private_json_bytes"]),"t1gr-e4-train-dev-access-private-v1")
  h=read_json_bounded(ho,int(sec["max_private_json_bytes"]),"t1gr-e4-final-holdout-sealed-private-v1")
  r=read_json_bounded(rc,int(sec["max_private_json_bytes"]),"t1gr-e4-seal-receipt-private-v1")
  if not all(payload_ok(x) for x in (public,t,h,r)):fail("E4_PAYLOAD_INTEGRITY_FAIL")
  fp=public.get("request_fingerprint")
  if not isinstance(fp,str) or any(x.get("request_fingerprint")!=fp for x in (t,h,r)):fail("E4_REQUEST_FINGERPRINT_MISMATCH")
  if public.get("seal_gate_passed") is not True or r.get("seal_gate_passed") is not True:fail("E4_SEAL_GATE_NOT_PASS")
  ts=public.get("freeze_timestamp_utc")
  if any(x.get("freeze_timestamp_utc")!=ts for x in (t,h,r)):fail("E4_FREEZE_TIMESTAMP_MISMATCH")
  if public.get("reviewed_e3_commit")!=p["reviewed_e3"]["commit_full"]:fail("E4_REVIEWED_COMMIT_DRIFT")
  if sha256_file(td,dl)!=public["train_dev_access_private_sha256"]:fail("TRAIN_DEV_ARTIFACT_SHA_DRIFT")
  if sha256_file(ho,dl)!=public["final_holdout_sealed_private_sha256"]:fail("HOLDOUT_ARTIFACT_SHA_DRIFT")
  if sha256_file(rc,dl)!=public["seal_receipt_private_sha256"]:fail("RECEIPT_ARTIFACT_SHA_DRIFT")
  train,dev,hold=ids(t["train_ids"]),ids(t["dev_ids"]),ids(h["final_holdout_ids"])
  if set(train)&set(dev) or set(train)&set(hold) or set(dev)&set(hold):fail("E4_SPLIT_OVERLAP")
  if len(set(train)|set(dev)|set(hold))!=2000:fail("E4_SPLIT_UNION_NOT_2000")
  got={"train":sha256_json(train),"dev":sha256_json(dev),"final_holdout":sha256_json(hold)}
  if got!=public["ids_commitments"] or got!=p["reviewed_e3"]["ids_commitments"]:fail("E4_ID_COMMITMENT_DRIFT")
  if {"train":len(train),"dev":len(dev),"final_holdout":len(hold)}!=public["sample_counts"]:fail("E4_COUNT_DRIFT")
  if t.get("final_holdout_ids_sha256")!=got["final_holdout"] or int(t.get("final_holdout_count",-1))!=len(hold):fail("TRAIN_DEV_HOLDOUT_COMMITMENT_DRIFT")
  if h.get("ids_sha256")!=got["final_holdout"] or int(h.get("count",-1))!=len(hold):fail("SEALED_HOLDOUT_COMMITMENT_DRIFT")
  if h.get("open_policy")!=p["seal"]["final_holdout_open_policy"]:fail("HOLDOUT_OPEN_POLICY_DRIFT")
  request_fp=sha256_json({"script":SCRIPT_VERSION,"freeze_public_sha256":sha256_file(pubp,dl),"train_dev_sha256":sha256_file(td,dl),"holdout_sha256":sha256_file(ho,dl),"receipt_sha256":sha256_file(rc,dl)})
  report={
   "schema":"t1gr-e4-seal-verification-public-v1",
   "script_version":SCRIPT_VERSION,
   "freeze_timestamp_utc":ts,
   "reviewed_e3_commit":p["reviewed_e3"]["commit_full"],
   "sample_counts":public["sample_counts"],
   "ids_commitments":public["ids_commitments"],
   "split_disjoint":True,
   "split_union_equals_2000":True,
   "train_dev_artifact_matches_freeze":True,
   "final_holdout_artifact_matches_freeze":True,
   "final_holdout_open_authorized":False,
   "any_raw_sample_id_present":False,
   "seal_verification_passed":True,
   "e5_entry_authorized":True,
   "step1_training_authorized":False,
  }
  assert_public_safe(report)
  sh,reuse=atomic_json_write(out,report,private=False,request_fingerprint=request_fp)
  return {"status":"PASS","public_output_sha256":sh,"idempotent_reuse":reuse,"e5_entry_authorized":True}
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--freeze-public",default="reports/step4_t1gr/e4_split_freeze_public.json")
 ap.add_argument("--train-dev-access",required=True)
 ap.add_argument("--final-holdout-sealed",required=True)
 ap.add_argument("--seal-receipt-private",required=True)
 ap.add_argument("--public-out",default="reports/step4_t1gr/e4_seal_verification_public.json")
 ap.add_argument("--timeout-seconds",type=float,default=None)
 a=ap.parse_args()
 try:print(json.dumps(run(a),ensure_ascii=False,indent=2))
 except Exception as e:print(json.dumps({"status":"FAIL","error":safe_error_message(e)},ensure_ascii=False),file=sys.stderr);raise SystemExit(2)
if __name__=="__main__":main()
