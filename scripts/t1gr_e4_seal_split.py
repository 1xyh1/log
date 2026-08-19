#!/usr/bin/env python3
"""E4: seal exactly the reviewer-accepted E3 split.

Security model:
- consumes one exact E3 candidate, pinned by reviewed commit/report SHA/ID commitments;
- validates private split truth against public evidence and final leakage components;
- emits TRAIN/DEV access and FINAL-HOLDOUT IDs into SEPARATE private artifacts;
- public artifact contains commitments/counts only;
- no claim of human secrecy from raw data; this is a harness/access seal.

This script does NOT authorize Step1 training. It only authorizes E5 preparation after
the seal verifier passes.
"""
from __future__ import annotations
import argparse, datetime as dt, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from multimodal.t1gr_secure_io import (  # noqa:E402
    Deadline, assert_public_safe, atomic_json_write, check_existing_output,
    ensure_private_input, ensure_private_output, ensure_public_output, ensure_repo_input,
    fail, file_lock, read_json_bounded, require_dict, require_keys, require_list,
    require_unchanged, safe_error_message, sha256_file, sha256_json, stat_token,
    validate_identifier,
)

SCRIPT_VERSION="t1gr-e4-seal-hardened-v1"
POLICY_REL="config/t1gr_e4_seal_policy.json"
SPLITS=("train","dev","final_holdout")

def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00","Z")

def payload_ok(obj:dict)->bool:
    claimed=obj.get("payload_sha256")
    if not isinstance(claimed,str) or len(claimed)!=64:return False
    base=dict(obj);base.pop("payload_sha256",None);base.pop("request_fingerprint",None)
    return sha256_json(base)==claimed

def validate_ids(xs, code):
    xs=require_list(xs,code)
    out=[validate_identifier(x) for x in xs]
    if len(out)!=len(set(out)):fail("DUPLICATE_ID_IN_SPLIT")
    return sorted(out)

def git_head_and_ancestor(repo:Path, reviewed:str):
    try:
        head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True,stderr=subprocess.DEVNULL,timeout=10).strip()
        r=subprocess.run(["git","merge-base","--is-ancestor",reviewed,head],cwd=repo,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
    except (OSError,subprocess.SubprocessError):
        fail("GIT_PROVENANCE_CHECK_FAILED")
    if len(head)!=40 or any(c not in "0123456789abcdef" for c in head.lower()):fail("GIT_HEAD_INVALID")
    if r.returncode!=0:fail("REVIEWED_E3_COMMIT_NOT_ANCESTOR")
    return head

def load_policy(repo):
    pp=ensure_repo_input(repo,POLICY_REL,"config")
    p=read_json_bounded(pp,1<<20,"t1gr-e4-seal-policy-v1")
    require_keys(p,("reviewed_e3","seal","security"),"E4_POLICY_MISSING")
    return pp,p

def validate_publics(repo,p,deadline):
    r=p["reviewed_e3"]
    cp=ensure_repo_input(repo,r["closure_public_relpath"],"reports/step4_t1gr")
    sp=ensure_repo_input(repo,r["split_public_relpath"],"reports/step4_t1gr")
    ct,st=stat_token(cp),stat_token(sp)
    if sha256_file(cp,deadline)!=r["closure_public_sha256"]:fail("REVIEWED_CLOSURE_PUBLIC_SHA_DRIFT")
    if sha256_file(sp,deadline)!=r["split_public_sha256"]:fail("REVIEWED_SPLIT_PUBLIC_SHA_DRIFT")
    c=read_json_bounded(cp,1<<20,"t1gr-e3-final-components-public-v1.2")
    s=read_json_bounded(sp,1<<20,"t1gr-e3-split-candidate-public-v1.2")
    if not payload_ok(c) or not payload_ok(s):fail("E3_PUBLIC_PAYLOAD_INTEGRITY_FAIL")
    if c.get("request_fingerprint")!=r["closure_request_fingerprint"]:fail("CLOSURE_REQUEST_FINGERPRINT_DRIFT")
    if s.get("request_fingerprint")!=r["split_request_fingerprint"]:fail("SPLIT_REQUEST_FINGERPRINT_DRIFT")
    if c.get("component_gate_passed") is not True or c.get("e3_component_status")!="PASS":fail("CLOSURE_GATE_NOT_PASS")
    if s.get("hard_gate_passed") is not True or s.get("status")!="CANDIDATE_PASS_AWAIT_REVIEW":fail("SPLIT_GATE_NOT_PASS")
    if s.get("e4_seal_authorized") is not False or s.get("step1_authorized") is not False:fail("E3_AUTHORIZATION_PROVENANCE_DRIFT")
    if c.get("policy_sha256")!=r["e3_policy_sha256"] or s.get("policy_sha256")!=r["e3_policy_sha256"]:fail("E3_POLICY_SHA_DRIFT")
    if s.get("formal_zip_metadata_commitment")!=r["formal_zip_metadata_commitment"]:fail("FORMAL_ZIP_COMMITMENT_DRIFT")
    if s.get("labels_commitment")!=r["labels_commitment"]:fail("LABEL_COMMITMENT_DRIFT")
    if s.get("counts")!=r["counts"]:fail("REVIEWED_SPLIT_COUNTS_DRIFT")
    if s.get("ids_commitments")!=r["ids_commitments"]:fail("REVIEWED_ID_COMMITMENTS_DRIFT")
    if int(c.get("component_count",-1))!=int(r["component_count"]):fail("REVIEWED_COMPONENT_COUNT_DRIFT")
    if int(c.get("max_component_size",-1))!=int(r["max_component_size"]):fail("REVIEWED_MAX_COMPONENT_DRIFT")
    if int(c.get("force_train_id_count_after_component_propagation",-1))!=int(r["force_train_id_count"]):fail("REVIEWED_FORCE_TRAIN_COUNT_DRIFT")
    require_unchanged(cp,ct,"CLOSURE_PUBLIC_CHANGED_DURING_RUN")
    require_unchanged(sp,st,"SPLIT_PUBLIC_CHANGED_DURING_RUN")
    return cp,sp,c,s

def validate_private_split(split_obj, comp_obj, split_pub, closure_pub, p):
    if split_obj.get("schema")!="t1gr-e3-split-candidate-private-v1.2":fail("BAD_SPLIT_PRIVATE_SCHEMA")
    if comp_obj.get("schema")!="t1gr-e3-final-components-private-v1.2":fail("BAD_COMPONENTS_PRIVATE_SCHEMA")
    if not payload_ok(split_obj) or not payload_ok(comp_obj):fail("PRIVATE_PAYLOAD_INTEGRITY_FAIL")
    if split_obj.get("request_fingerprint")!=p["reviewed_e3"]["split_request_fingerprint"]:fail("SPLIT_PRIVATE_REQUEST_DRIFT")
    if split_obj.get("hard_gate_passed") is not True:fail("SPLIT_PRIVATE_GATE_NOT_PASS")
    if comp_obj.get("component_gate_passed") is not True:fail("COMPONENT_PRIVATE_GATE_NOT_PASS")
    if split_obj.get("policy_sha256")!=p["reviewed_e3"]["e3_policy_sha256"]:fail("SPLIT_PRIVATE_POLICY_DRIFT")
    if comp_obj.get("policy_sha256")!=p["reviewed_e3"]["e3_policy_sha256"]:fail("COMPONENT_PRIVATE_POLICY_DRIFT")
    ids_obj=require_dict(split_obj.get("ids"),"SPLIT_IDS_MISSING")
    if set(ids_obj)!=set(SPLITS):fail("SPLIT_KEYS_DRIFT")
    ids={sp:validate_ids(ids_obj[sp],"BAD_SPLIT_IDS") for sp in SPLITS}
    sets={sp:set(ids[sp]) for sp in SPLITS}
    if any(sets[a]&sets[b] for i,a in enumerate(SPLITS) for b in SPLITS[i+1:]):fail("SPLIT_OVERLAP")
    if len(set().union(*(sets[x] for x in SPLITS)))!=2000:fail("SPLIT_UNION_NOT_2000")
    if {sp:len(ids[sp]) for sp in SPLITS}!=p["reviewed_e3"]["counts"]:fail("PRIVATE_SPLIT_COUNTS_DRIFT")
    commits={sp:sha256_json(ids[sp]) for sp in SPLITS}
    if commits!=p["reviewed_e3"]["ids_commitments"] or commits!=split_pub["ids_commitments"]:fail("PRIVATE_ID_COMMITMENT_DRIFT")
    if split_obj.get("components_private_sha256")!=split_pub.get("components_private_sha256"):fail("COMPONENT_PRIVATE_SHA_LINK_DRIFT")
    comps=require_list(comp_obj.get("all_components"),"COMPONENTS_MISSING")
    if len(comps)!=int(closure_pub["component_count"]):fail("COMPONENT_COUNT_PRIVATE_PUBLIC_DRIFT")
    universe=set()
    norm=[]
    for c in comps:
        c=validate_ids(c,"BAD_COMPONENT")
        if not c:fail("EMPTY_COMPONENT")
        if universe.intersection(c):fail("COMPONENT_ID_DUPLICATED")
        universe.update(c);norm.append(c)
    if len(universe)!=2000:fail("COMPONENT_UNIVERSE_DRIFT")
    if sha256_json(norm)!=closure_pub["all_components_commitment"]:fail("COMPONENT_COMMITMENT_DRIFT")
    force=set(validate_ids(comp_obj.get("force_train_ids_after_component_propagation"),"FORCE_TRAIN_IDS_MISSING"))
    if sha256_json(sorted(force))!=closure_pub["force_train_ids_commitment"]:fail("FORCE_TRAIN_COMMITMENT_DRIFT")
    if not force<=sets["train"] or force&(sets["dev"]|sets["final_holdout"]):fail("HISTORICAL_QUARANTINE_VIOLATION")
    for c in norm:
        hits=sum(bool(set(c)&sets[sp]) for sp in SPLITS)
        if hits!=1:fail("COMPONENT_SPLIT_DETECTED")
    return ids,commits

def recover_freeze_timestamp(existing):
    ts=None
    for x in existing:
        if x is None:continue
        obj,_=x
        v=obj.get("freeze_timestamp_utc")
        if not isinstance(v,str) or not v:fail("EXISTING_SEAL_TIMESTAMP_MISSING")
        if ts is None:ts=v
        elif ts!=v:fail("EXISTING_SEAL_TIMESTAMP_CONFLICT")
    return ts or utc_now()

def run(args):
    repo=ROOT.resolve(strict=True)
    policy_path,p=load_policy(repo)
    sec=p["security"];deadline=Deadline(float(args.timeout_seconds or sec["seal_timeout_seconds"]))
    policy_token=stat_token(policy_path)
    head=git_head_and_ancestor(repo,p["reviewed_e3"]["commit_full"])
    closure_path,split_pub_path,closure_pub,split_pub=validate_publics(repo,p,deadline)

    split_private=ensure_private_input(Path(args.split_private),repo)
    comp_private=ensure_private_input(Path(args.components_private),repo)
    train_dev_out=ensure_private_output(Path(args.train_dev_access_out),repo,bool(sec["private_parent_must_preexist"]))
    holdout_out=ensure_private_output(Path(args.final_holdout_sealed_out),repo,bool(sec["private_parent_must_preexist"]))
    receipt_out=ensure_private_output(Path(args.seal_receipt_private_out),repo,bool(sec["private_parent_must_preexist"]))
    public_out=ensure_public_output(repo,args.public_out,sec["public_output_prefix"])
    outs=[train_dev_out,holdout_out,receipt_out,public_out]
    if len({str(x).lower() for x in outs})!=4:fail("OUTPUT_PATH_COLLISION")

    lock=public_out.with_suffix(public_out.suffix+".lock")
    with file_lock(lock,float(sec["lock_wait_seconds"]),float(sec["lock_stale_seconds"])):
        stoken,ctoken=stat_token(split_private),stat_token(comp_private)
        split_obj=read_json_bounded(split_private,int(sec["max_private_json_bytes"]))
        comp_obj=read_json_bounded(comp_private,int(sec["max_private_json_bytes"]))
        split_sha=sha256_file(split_private,deadline)
        comp_sha=sha256_file(comp_private,deadline)
        if split_sha!=split_pub["components_private_sha256"] and False:
            fail("UNREACHABLE")  # prevents accidental mistaken comparison during maintenance
        if comp_sha!=split_pub["components_private_sha256"]:fail("COMPONENTS_PRIVATE_SHA_DRIFT")
        ids,commits=validate_private_split(split_obj,comp_obj,split_pub,closure_pub,p)
        require_unchanged(split_private,stoken,"SPLIT_PRIVATE_CHANGED_DURING_RUN")
        require_unchanged(comp_private,ctoken,"COMPONENTS_PRIVATE_CHANGED_DURING_RUN")
        require_unchanged(policy_path,policy_token,"E4_POLICY_CHANGED_DURING_RUN")

        request_fp=sha256_json({
            "script":SCRIPT_VERSION,
            "reviewed_e3_commit":p["reviewed_e3"]["commit_full"],
            "e4_policy_sha256":sha256_file(policy_path,deadline),
            "split_private_sha256":split_sha,
            "components_private_sha256":comp_sha,
            "split_public_sha256":p["reviewed_e3"]["split_public_sha256"],
            "closure_public_sha256":p["reviewed_e3"]["closure_public_sha256"],
        })

        existing=[
            check_existing_output(train_dev_out,request_fp),
            check_existing_output(holdout_out,request_fp),
            check_existing_output(receipt_out,request_fp),
            check_existing_output(public_out,request_fp),
        ]
        if all(x is not None for x in existing):
            pub,_=existing[-1]
            if pub.get("seal_gate_passed") is not True:fail("EXISTING_E4_SEAL_NOT_PASS")
            return {"status":"PASS","request_fingerprint":request_fp,"idempotent_reuse":True}
        freeze_ts=recover_freeze_timestamp(existing)

        train_dev={
            "schema":"t1gr-e4-train-dev-access-private-v1",
            "script_version":SCRIPT_VERSION,
            "freeze_timestamp_utc":freeze_ts,
            "reviewed_e3_commit":p["reviewed_e3"]["commit_full"],
            "runtime_repo_head":head,
            "e4_policy_sha256":sha256_file(policy_path,deadline),
            "split_candidate_private_sha256":split_sha,
            "components_private_sha256":comp_sha,
            "formal_zip_metadata_commitment":p["reviewed_e3"]["formal_zip_metadata_commitment"],
            "labels_commitment":p["reviewed_e3"]["labels_commitment"],
            "train_ids":ids["train"],
            "dev_ids":ids["dev"],
            "train_ids_sha256":commits["train"],
            "dev_ids_sha256":commits["dev"],
            "final_holdout_count":len(ids["final_holdout"]),
            "final_holdout_ids_sha256":commits["final_holdout"],
            "access_policy":{
                "train":"ALLOWED_FOR_E5_PREPARATION_AND_TRAINING_AFTER_E5_GATE",
                "dev":"ALLOWED_FOR_MONITORING_MODEL_SELECTION_AND_BASELINE_EVAL",
                "final_holdout_access":"NOT_PRESENT_IN_THIS_ARTIFACT",
            }
        }
        td_sha,_=atomic_json_write(train_dev_out,train_dev,private=True,request_fingerprint=request_fp)

        holdout={
            "schema":"t1gr-e4-final-holdout-sealed-private-v1",
            "script_version":SCRIPT_VERSION,
            "freeze_timestamp_utc":freeze_ts,
            "reviewed_e3_commit":p["reviewed_e3"]["commit_full"],
            "e4_policy_sha256":sha256_file(policy_path,deadline),
            "split_candidate_private_sha256":split_sha,
            "final_holdout_ids":ids["final_holdout"],
            "count":len(ids["final_holdout"]),
            "ids_sha256":commits["final_holdout"],
            "open_policy":p["seal"]["final_holdout_open_policy"],
            "training_access":p["seal"]["final_holdout_training_access"],
            "tuning_access":p["seal"]["final_holdout_tuning_access"],
            "seed_selection_access":p["seal"]["final_holdout_seed_selection_access"],
            "early_stopping_access":p["seal"]["final_holdout_early_stopping_access"],
        }
        ho_sha,_=atomic_json_write(holdout_out,holdout,private=True,request_fingerprint=request_fp)

        receipt={
            "schema":"t1gr-e4-seal-receipt-private-v1",
            "script_version":SCRIPT_VERSION,
            "freeze_timestamp_utc":freeze_ts,
            "reviewed_e3_commit":p["reviewed_e3"]["commit_full"],
            "runtime_repo_head":head,
            "e4_policy_sha256":sha256_file(policy_path,deadline),
            "split_candidate_private_sha256":split_sha,
            "components_private_sha256":comp_sha,
            "train_dev_access_private_sha256":td_sha,
            "final_holdout_sealed_private_sha256":ho_sha,
            "sample_counts":p["reviewed_e3"]["counts"],
            "ids_commitments":commits,
            "seal_gate_passed":True,
        }
        rc_sha,_=atomic_json_write(receipt_out,receipt,private=True,request_fingerprint=request_fp)

        public={
            "schema":"t1gr-e4-split-freeze-public-v1",
            "script_version":SCRIPT_VERSION,
            "freeze_timestamp_utc":freeze_ts,
            "reviewed_e3_commit":p["reviewed_e3"]["commit_full"],
            "runtime_repo_head_at_freeze":head,
            "e4_policy_sha256":sha256_file(policy_path,deadline),
            "reviewed_closure_public_sha256":p["reviewed_e3"]["closure_public_sha256"],
            "reviewed_split_public_sha256":p["reviewed_e3"]["split_public_sha256"],
            "split_candidate_private_sha256":split_sha,
            "components_private_sha256":comp_sha,
            "train_dev_access_private_sha256":td_sha,
            "final_holdout_sealed_private_sha256":ho_sha,
            "seal_receipt_private_sha256":rc_sha,
            "sample_counts":p["reviewed_e3"]["counts"],
            "ids_commitments":commits,
            "formal_zip_metadata_commitment":p["reviewed_e3"]["formal_zip_metadata_commitment"],
            "labels_commitment":p["reviewed_e3"]["labels_commitment"],
            "historical_quarantine_count":p["reviewed_e3"]["force_train_id_count"],
            "any_raw_sample_id_present":False,
            "final_holdout_ids_exposed":False,
            "final_holdout_open_authorized":False,
            "seal_gate_passed":True,
            "e5_entry_authorized_after_seal_verification":True,
            "step1_training_authorized":False,
            "evidence_model":"repo nondisclosure + harness access seal; not a claim of human secrecy from raw data",
            "training_order_claim":"NOT_ASSERTED_HERE; future E5 runner must prove freeze_timestamp precedes training_start",
        }
        assert_public_safe(public)
        pub_sha,_=atomic_json_write(public_out,public,private=False,request_fingerprint=request_fp)
        return {
            "status":"PASS","request_fingerprint":request_fp,"freeze_timestamp_utc":freeze_ts,
            "public_output_sha256":pub_sha,"idempotent_reuse":False,
            "sample_counts":p["reviewed_e3"]["counts"]
        }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split-private",required=True)
    ap.add_argument("--components-private",required=True)
    ap.add_argument("--train-dev-access-out",required=True)
    ap.add_argument("--final-holdout-sealed-out",required=True)
    ap.add_argument("--seal-receipt-private-out",required=True)
    ap.add_argument("--public-out",default="reports/step4_t1gr/e4_split_freeze_public.json")
    ap.add_argument("--timeout-seconds",type=float,default=None)
    args=ap.parse_args()
    try:
        print(json.dumps(run(args),ensure_ascii=False,indent=2))
    except Exception as e:
        print(json.dumps({"status":"FAIL","error":safe_error_message(e)},ensure_ascii=False),file=sys.stderr)
        raise SystemExit(2)
if __name__=="__main__":main()
