#!/usr/bin/env python3
"""Build atomic TRAIN/DEV-only RGB view directly from the pinned formal ZIP."""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, stat, sys, uuid, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from multimodal.t1gr_secure_io import (
    Deadline, assert_public_safe, atomic_json_write, check_existing_output, ensure_private_input,
    ensure_public_output, ensure_repo_input, fail, file_lock, is_within,
    read_json_bounded, require_unchanged, safe_error_message, sha256_file, sha256_json, stat_token,
)
from multimodal.t1gr_e5_core import (
 FROZEN_E5_SECURITY_POLICY_SHA256,FROZEN_E5_TRAINING_SPEC_SHA256,
    SCHEMA_RECIPE, SCHEMA_VIEW_PRIVATE, SCHEMA_VIEW_PUBLIC, canonical_ids_sha, payload_ok,
    scan_formal_zip, validate_e4_evidence, verify_view_tree,
)

SCRIPT_VERSION="t1gr-e5-build-rgb-view-hardened-v1"

def raw_sha(b:bytes)->str: return hashlib.sha256(b).hexdigest()

def write_private_file(path:Path,data:bytes):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): fail("E5_VIEW_TEMP_COLLISION")
    try:
        with open(path,"xb") as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        if os.name!="nt": os.chmod(path,stat.S_IRUSR|stat.S_IWUSR)
    except PermissionError: fail("WRITE_PERMISSION_DENIED")
    except OSError: fail("WRITE_IO_ERROR")

def run(a):
    repo=ROOT.resolve(strict=True)
    secp=ensure_repo_input(repo,"config/t1gr_e5_security_policy.json","config")
    if sha256_file(secp)!=FROZEN_E5_SECURITY_POLICY_SHA256: fail("E5_SECURITY_POLICY_SHA_DRIFT")
    sec=read_json_bounded(secp,1<<20,"t1gr-e5-security-policy-v1")
    rp=ensure_repo_input(repo,"reports/step4_t1gr/e5_step1_recipe_public.json","reports/step4_t1gr")
    fp=ensure_repo_input(repo,"reports/step4_t1gr/e4_split_freeze_public.json","reports/step4_t1gr")
    vp=ensure_repo_input(repo,"reports/step4_t1gr/e4_seal_verification_public.json","reports/step4_t1gr")
    td_p=ensure_private_input(Path(a.train_dev_access),repo)
    out_root=Path(a.out_root).expanduser().resolve(strict=False)
    if is_within(out_root,repo): fail("E5_VIEW_ROOT_INSIDE_REPO")
    if not out_root.parent.is_dir(): fail("E5_VIEW_PARENT_NOT_FOUND")
    if not os.access(out_root.parent,os.W_OK): fail("E5_VIEW_PARENT_NOT_WRITABLE")
    if out_root.exists() and not out_root.is_dir(): fail("E5_VIEW_ROOT_NOT_DIRECTORY")
    pub=ensure_public_output(repo,"reports/step4_t1gr/e5_step1_view_public.json",sec["public_output_prefix"])
    zp=Path(a.formal_zip).expanduser().resolve(strict=False)
    if not zp.is_file():fail("FORMAL_ZIP_NOT_FOUND")
    deadline=Deadline(float(a.timeout_seconds or sec["view_build_timeout_seconds"]))
    lock=out_root.parent/f".{out_root.name}.e5view.lock"
    with file_lock(lock,5.0,900.0):
        recipe=read_json_bounded(rp,int(sec["max_public_json_bytes"]),SCHEMA_RECIPE)
        if not payload_ok(recipe):fail("E5_RECIPE_INTEGRITY_FAIL")
        e4f=read_json_bounded(fp,int(sec["max_public_json_bytes"]))
        e4v=read_json_bounded(vp,int(sec["max_public_json_bytes"]))
        td=read_json_bounded(td_p,int(sec["max_private_json_bytes"]))
        td_sha=sha256_file(td_p,deadline)
        e4=validate_e4_evidence(e4f,e4v,td,td_sha)
        if td_sha!=recipe["train_dev_access_private_sha256"]:fail("E5_VIEW_TRAIN_DEV_RECIPE_PIN_FAIL")
        if recipe["ids_commitments"]!=e4["commits"] or recipe["sample_counts"]!=e4["counts"]:fail("E5_VIEW_E4_RECIPE_DRIFT")
        zip_token=stat_token(zp)
        full_zip_sha=sha256_file(zp,deadline)
        if full_zip_sha!=recipe["formal_zip_sha256"]:fail("E5_VIEW_FORMAL_ZIP_SHA_DRIFT")
        zscan=scan_formal_zip(zp,deadline,max_members=int(sec["max_zip_members"]),
                              max_label_member_bytes=int(sec["max_label_member_bytes"]),
                              max_total_label_bytes=int(sec["max_total_label_bytes"]))
        if zscan["metadata_commitment"]!=recipe["formal_zip_metadata_commitment"] or zscan["labels_commitment"]!=recipe["labels_commitment"]:
            fail("E5_VIEW_ZIP_COMMITMENT_DRIFT")
        require_unchanged(zp,zip_token,"E5_VIEW_ZIP_CHANGED_DURING_BUILD")
        recipe_sha=sha256_file(rp,deadline)
        out_root_binding=sha256_json(str(out_root).casefold() if os.name=="nt" else str(out_root))
        request_fp=sha256_json({"script":SCRIPT_VERSION,"recipe_sha256":recipe_sha,
                               "train_dev_sha256":td_sha,"formal_zip_sha256":full_zip_sha,
                               "out_root_binding":out_root_binding})
        manifest_path=out_root/"view_manifest.json"
        existing_pub=check_existing_output(pub,request_fp)
        if out_root.exists():
            if not manifest_path.is_file():fail("E5_VIEW_ROOT_NONEMPTY_UNMANAGED")
            vr=verify_view_tree(manifest_path,recipe,td,deadline)
            existing=read_json_bounded(manifest_path,int(sec["max_private_json_bytes"]),SCHEMA_VIEW_PRIVATE)
            if existing.get("request_fingerprint")!=request_fp:fail("OUTPUT_CONFLICT_DIFFERENT_REQUEST")
            if existing_pub is None:
                report={
                    "schema":SCHEMA_VIEW_PUBLIC,"script_version":SCRIPT_VERSION,
                    "recipe_public_sha256":recipe_sha,"view_manifest_private_sha256":sha256_file(manifest_path,deadline),
                    "dataset_yaml_sha256":existing["dataset_yaml_sha256"],
                    "train_count":vr["train_count"],"dev_count":vr["dev_count"],
                    "final_holdout_count":recipe["sample_counts"]["final_holdout"],
                    "ids_commitments":recipe["ids_commitments"],
                    "mapping_commitment":vr["mapping_commitment"],
                    "final_holdout_ids_available_to_view":False,
                    "any_raw_sample_id_present":False,"view_gate_passed":True,
                }
                assert_public_safe(report)
                sh,_=atomic_json_write(pub,report,private=False,request_fingerprint=request_fp)
                return {"status":"PASS","idempotent_reuse":True,"public_output_sha256":sh}
            return {"status":"PASS","idempotent_reuse":True,"public_output_sha256":existing_pub[1]}

        tmp=out_root.parent/f".{out_root.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        if tmp.exists():fail("E5_VIEW_TEMP_COLLISION")
        tmp.mkdir(mode=0o700)
        try:
            maps=zscan["maps"]; rows=[]; ext_counts={"train_jpg":0,"train_png":0,"dev_jpg":0,"dev_png":0}
            with zipfile.ZipFile(zp) as z:
                for sp,ids,folder in (("train",e4["train"],"train"),("dev",e4["dev"],"val")):
                    for sid in ids:
                        deadline.check("E5_VIEW_BUILD_TIMEOUT")
                        if sid not in maps["visible"] or sid not in maps["labels"]:fail("E5_VIEW_SOURCE_ID_MISSING")
                        ii,ll=maps["visible"][sid],maps["labels"][sid]
                        ext=Path(ii.filename).suffix.lower()
                        if ext not in {".jpg",".jpeg",".png"}:fail("E5_VIEW_IMAGE_EXTENSION_BAD")
                        try:ib=z.read(ii);lb=z.read(ll)
                        except Exception:fail("ZIP_MEMBER_READ_ERROR")
                        image_rel=f"images/{folder}/{sid}{ext}"
                        label_rel=f"labels/{folder}/{sid}.txt"
                        write_private_file(tmp/image_rel,ib);write_private_file(tmp/label_rel,lb)
                        ish,lsh=raw_sha(ib),raw_sha(lb)
                        rows.append({"sample_id":sid,"split":sp,"image_rel":image_rel,"label_rel":label_rel,
                                     "image_sha256":ish,"label_sha256":lsh,"image_bytes":len(ib),"label_bytes":len(lb)})
                        key=f"{sp}_{'jpg' if ext in {'.jpg','.jpeg'} else 'png'}";ext_counts[key]+=1
            final_abs=out_root.resolve(strict=False).as_posix()
            yaml_text=(f"path: {json.dumps(final_abs,ensure_ascii=False)}\n"
                       "train: images/train\nval: images/val\nnc: 12\nnames:\n"+
                       "".join(f"  {i}: {json.dumps(name,ensure_ascii=False)}\n" for i,name in enumerate(recipe["class_names"].values())))
            write_private_file(tmp/"dataset.yaml",yaml_text.encode("utf-8"))
            mapping_commit=sha256_json(sorted((r["split"],r["sample_id"],r["image_rel"],r["image_sha256"],r["label_rel"],r["label_sha256"]) for r in rows))
            manifest={
                "schema":SCHEMA_VIEW_PRIVATE,"script_version":SCRIPT_VERSION,
                "recipe_sha256":recipe["recipe_sha256_self"],"recipe_public_file_sha256":recipe_sha,
                "train_dev_access_private_sha256":td_sha,"formal_zip_sha256":full_zip_sha,
                "train_ids":e4["train"],"dev_ids":e4["dev"],
                "train_ids_sha256":canonical_ids_sha(e4["train"]),"dev_ids_sha256":canonical_ids_sha(e4["dev"]),
                "final_holdout_count":e4["holdout_count"],"final_holdout_ids_sha256":e4["commits"]["final_holdout"],
                "dataset_yaml_rel":"dataset.yaml","dataset_yaml_sha256":raw_sha(yaml_text.encode("utf-8")),
                "mappings":rows,"mapping_count":len(rows),"mapping_commitment":mapping_commit,
                "extension_counts":ext_counts,
                "final_holdout_ids_present":False,
            }
            # Private integrity fields are generated by hardened writer.
            atomic_json_write(tmp/"view_manifest.json",manifest,private=True,request_fingerprint=request_fp)
            if out_root.exists():fail("E5_VIEW_OUTPUT_APPEARED_CONCURRENTLY")
            os.replace(tmp,out_root)
            if os.name!="nt":
                try: os.chmod(out_root,0o700)
                except OSError: fail("E5_VIEW_PERMISSION_HARDEN_FAIL")
                try:
                    dfd=os.open(str(out_root.parent),os.O_RDONLY);os.fsync(dfd);os.close(dfd)
                except OSError:pass
        except Exception:
            shutil.rmtree(tmp,ignore_errors=True)
            raise
        vr=verify_view_tree(out_root/"view_manifest.json",recipe,td,deadline)
        report={
            "schema":SCHEMA_VIEW_PUBLIC,"script_version":SCRIPT_VERSION,
            "recipe_public_sha256":recipe_sha,
            "view_manifest_private_sha256":sha256_file(out_root/"view_manifest.json",deadline),
            "dataset_yaml_sha256":vr["manifest"]["dataset_yaml_sha256"],
            "train_count":vr["train_count"],"dev_count":vr["dev_count"],
            "final_holdout_count":recipe["sample_counts"]["final_holdout"],
            "ids_commitments":recipe["ids_commitments"],
            "extension_counts":vr["manifest"]["extension_counts"],
            "mapping_commitment":vr["mapping_commitment"],
            "final_holdout_ids_available_to_view":False,
            "any_raw_sample_id_present":False,"view_gate_passed":True,
        }
        assert_public_safe(report)
        sh,_=atomic_json_write(pub,report,private=False,request_fingerprint=request_fp)
        return {"status":"PASS","idempotent_reuse":False,"public_output_sha256":sh,
                "train_count":vr["train_count"],"dev_count":vr["dev_count"]}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-dev-access",required=True)
    ap.add_argument("--formal-zip",required=True)
    ap.add_argument("--out-root",required=True)
    ap.add_argument("--timeout-seconds",type=float,default=None)
    a=ap.parse_args()
    try:print(json.dumps(run(a),ensure_ascii=False,indent=2))
    except KeyboardInterrupt:
        print(json.dumps({"status":"FAIL","error":"USER_INTERRUPT"},ensure_ascii=False),file=sys.stderr);raise SystemExit(130)
    except Exception as e:print(json.dumps({"status":"FAIL","error":safe_error_message(e)},ensure_ascii=False),file=sys.stderr);raise SystemExit(2)
if __name__=="__main__":main()
