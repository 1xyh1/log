#!/usr/bin/env python3
"""Freeze E5 Step1 recipe from accepted E2/E4 evidence and exact raw artifacts."""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from multimodal.t1gr_secure_io import (
    Deadline, assert_public_safe, atomic_json_write, ensure_private_input, ensure_public_output,
    ensure_repo_input, fail, file_lock, read_json_bounded, require_unchanged, safe_error_message,
    sha256_file, sha256_json, stat_token,
)
from multimodal.t1gr_e5_core import (
 FROZEN_E5_SECURITY_POLICY_SHA256,FROZEN_E5_TRAINING_SPEC_SHA256,
    CLASS_NAME_MAP, SCHEMA_RECIPE, environment_probe, scan_formal_zip,
    utc_now, validate_e2_evidence, validate_e4_evidence, validate_training_spec, parse_utc,
)

SCRIPT_VERSION="t1gr-e5-freeze-recipe-hardened-v1"

def git_head(repo:Path)->str:
    try:
        x=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True,stderr=subprocess.DEVNULL,timeout=10).strip()
    except Exception:
        fail("GIT_HEAD_CHECK_FAILED")
    if len(x)!=40:
        fail("GIT_HEAD_INVALID")
    return x

def run(a):
    repo=ROOT.resolve(strict=True)
    secp=ensure_repo_input(repo,"config/t1gr_e5_security_policy.json","config")
    if sha256_file(secp)!=FROZEN_E5_SECURITY_POLICY_SHA256: fail("E5_SECURITY_POLICY_SHA_DRIFT")
    sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v1")
    tsp=ensure_repo_input(repo,"config/t1gr_e5_training_spec.frozen.json","config")
    if sha256_file(tsp)!=FROZEN_E5_TRAINING_SPEC_SHA256: fail("E5_FROZEN_TRAINING_SPEC_SHA_DRIFT")
    forensic_p=ensure_repo_input(repo,"reports/step4_t1gr/zip_forensic_public.json","reports/step4_t1gr")
    taxonomy_p=ensure_repo_input(repo,"reports/step4_t1gr/label_error_taxonomy_public.json","reports/step4_t1gr")
    e4f_p=ensure_repo_input(repo,"reports/step4_t1gr/e4_split_freeze_public.json","reports/step4_t1gr")
    e4v_p=ensure_repo_input(repo,"reports/step4_t1gr/e4_seal_verification_public.json","reports/step4_t1gr")
    td_p=ensure_private_input(Path(a.train_dev_access),repo)
    out=ensure_public_output(repo,"reports/step4_t1gr/e5_step1_recipe_public.json",sec["public_output_prefix"])
    zp=Path(a.formal_zip).expanduser().resolve(strict=False)
    ck=Path(a.base_checkpoint).expanduser().resolve(strict=False)
    if not zp.is_file(): fail("FORMAL_ZIP_NOT_FOUND")
    if not ck.is_file(): fail("BASE_CHECKPOINT_NOT_FOUND")
    if not os.access(zp,os.R_OK) or not os.access(ck,os.R_OK): fail("READ_PERMISSION_DENIED")

    deadline=Deadline(float(a.timeout_seconds or sec["recipe_hash_timeout_seconds"]))
    with file_lock(out.with_suffix(out.suffix+".lock"),5.0,900.0):
        tokens={p:stat_token(p) for p in (secp,tsp,forensic_p,taxonomy_p,e4f_p,e4v_p,td_p,zp,ck)}
        forensic=read_json_bounded(forensic_p,int(sec["max_public_json_bytes"]))
        taxonomy=read_json_bounded(taxonomy_p,int(sec["max_public_json_bytes"]))
        e4f=read_json_bounded(e4f_p,int(sec["max_public_json_bytes"]))
        e4v=read_json_bounded(e4v_p,int(sec["max_public_json_bytes"]))
        td=read_json_bounded(td_p,int(sec["max_private_json_bytes"]))
        spec=read_json_bounded(tsp,int(sec["max_public_json_bytes"]),"t1gr-e5-training-spec-v1")
        validate_e2_evidence(forensic,taxonomy)
        validate_training_spec(spec)
        td_sha=sha256_file(td_p,deadline)
        e4=validate_e4_evidence(e4f,e4v,td,td_sha)

        zip_scan=scan_formal_zip(
            zp,deadline,max_members=int(sec["max_zip_members"]),
            max_label_member_bytes=int(sec["max_label_member_bytes"]),
            max_total_label_bytes=int(sec["max_total_label_bytes"]),
        )
        if zip_scan["metadata_commitment"]!=e4f.get("formal_zip_metadata_commitment"):
            fail("E5_FORMAL_ZIP_METADATA_DRIFT")
        if zip_scan["labels_commitment"]!=e4f.get("labels_commitment"):
            fail("E5_FORMAL_LABELS_COMMITMENT_DRIFT")
        full_zip_sha=sha256_file(zp,deadline)
        checkpoint_sha=sha256_file(ck,deadline)
        env=environment_probe()
        if env["ultralytics_version"]!=spec["expected_ultralytics_version"]:
            fail("E5_ULTRALYTICS_VERSION_DRIFT")
        if str(spec["runtime"]["device"]).lower() not in {"cpu","mps"} and not env["cuda_available"]:
            fail("E5_CUDA_REQUIRED_BUT_UNAVAILABLE")

        for p,t in tokens.items(): require_unchanged(p,t,"E5_RECIPE_INPUT_CHANGED_DURING_RUN")
        freeze_ts=utc_now()
        if not parse_utc(e4["freeze_timestamp_utc"]) < parse_utc(freeze_ts):
            fail("E5_RECIPE_NOT_AFTER_E4_FREEZE")
        input_shas={
            "forensic_public":sha256_file(forensic_p,deadline),
            "taxonomy_public":sha256_file(taxonomy_p,deadline),
            "e4_freeze_public":sha256_file(e4f_p,deadline),
            "e4_verification_public":sha256_file(e4v_p,deadline),
            "training_spec":sha256_file(tsp,deadline),
            "security_policy":sha256_file(secp,deadline),
        }
        request_fp=sha256_json({
            "script":SCRIPT_VERSION,"inputs":input_shas,"train_dev_private_sha256":td_sha,
            "formal_zip_sha256":full_zip_sha,"base_checkpoint_sha256":checkpoint_sha,"environment":env,
        })
        rec={
            "schema":SCHEMA_RECIPE,
            "script_version":SCRIPT_VERSION,
            "recipe_frozen_at_utc":freeze_ts,
            "runtime_repo_head":git_head(repo),
            "source_report_sha256":input_shas,
            "train_dev_access_private_sha256":td_sha,
            "formal_zip_sha256":full_zip_sha,
            "formal_zip_bytes":int(zp.stat().st_size),
            "formal_zip_metadata_commitment":zip_scan["metadata_commitment"],
            "labels_commitment":zip_scan["labels_commitment"],
            "base_checkpoint_sha256":checkpoint_sha,
            "environment":env,
            "e4_freeze_timestamp_utc":e4["freeze_timestamp_utc"],
            "sample_counts":e4["counts"],
            "ids_commitments":e4["commits"],
            "architecture":spec["architecture"],
            "model_yaml":spec["model_yaml"],
            "num_classes":12,
            "class_names":CLASS_NAME_MAP,
            "train_args":spec["train_args"],
            "eval_args":spec["eval_args"],
            "runtime":spec["runtime"],
            "view_policy":{
                "mode":"COPY_ONLY",
                "modalities":"VISIBLE_PLUS_LABELS",
                "contains":"TRAIN_DEV_ONLY",
                "final_holdout_ids_available_to_view_builder":False,
            },
            "pretrained_transfer_policy":"fresh physical nc=12 YOLO26s + shape-compatible partial transfer from pinned checkpoint",
            "final_holdout_access":"FORBIDDEN_UNTIL_T1GR_FINAL_ADJUDICATION",
            "smoke_authorized":False,
            "formal_step1_authorized":False,
        }
        # Self-pin recipe content excluding integrity fields.
        rec["recipe_sha256_self"]=sha256_json(rec)
        assert_public_safe(rec)
        sh,reuse=atomic_json_write(out,rec,private=False,request_fingerprint=request_fp)
        return {"status":"PASS","recipe_public_sha256":sh,"idempotent_reuse":reuse,
                "formal_zip_sha256":full_zip_sha,"base_checkpoint_sha256":checkpoint_sha}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-dev-access",required=True)
    ap.add_argument("--formal-zip",required=True)
    ap.add_argument("--base-checkpoint",required=True)
    ap.add_argument("--timeout-seconds",type=float,default=None)
    a=ap.parse_args()
    try: print(json.dumps(run(a),ensure_ascii=False,indent=2))
    except KeyboardInterrupt:
        print(json.dumps({"status":"FAIL","error":"USER_INTERRUPT"},ensure_ascii=False),file=sys.stderr)
        raise SystemExit(130)
    except Exception as e:
        print(json.dumps({"status":"FAIL","error":safe_error_message(e)},ensure_ascii=False),file=sys.stderr)
        raise SystemExit(2)
if __name__=="__main__":main()
