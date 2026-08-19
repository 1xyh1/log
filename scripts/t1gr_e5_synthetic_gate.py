#!/usr/bin/env python3
"""Non-formal E5 synthetic gate: real 2000-ID ZIP scan + 1702-file TRAIN/DEV view verification."""
from __future__ import annotations
import hashlib,json,shutil,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from multimodal.t1gr_secure_io import Deadline,GateError,atomic_json_write,file_lock,sha256_json
from multimodal.t1gr_e5_core import *

def h(b):return hashlib.sha256(b).hexdigest()
def payload(o,fp):
    x=dict(o);x["payload_sha256"]=sha256_json(o);x["request_fingerprint"]=fp;return x

def main():
    checks={}
    with tempfile.TemporaryDirectory() as td0:
        td=Path(td0)
        zp=td/"formal.zip"
        train=[f"t{i:04d}" for i in range(1504)]
        dev=[f"d{i:04d}" for i in range(198)]
        hold=[f"h{i:04d}" for i in range(298)]
        allids=train+dev+hold
        img=b"tiny";lab=b"0 0.5 0.5 0.2 0.2\n"
        with zipfile.ZipFile(zp,"w",zipfile.ZIP_STORED) as z:
            for sid in allids:
                for m in ("visible","infrared","depth"):
                    z.writestr(f"{m}/{sid}.png",img)
                z.writestr(f"labels/{sid}.txt",lab)
        scan=scan_formal_zip(zp,Deadline(60))
        checks["zip_4x2000"]=all(v==2000 for v in scan["member_counts"].values())
        commits={"train":canonical_ids_sha(train),"dev":canonical_ids_sha(dev),"final_holdout":canonical_ids_sha(hold)}
        recipe0={
            "schema":SCHEMA_RECIPE,"recipe_sha256_self":"r"*64,
            "train_dev_access_private_sha256":"tdsha","formal_zip_sha256":h(zp.read_bytes()),
            "sample_counts":{"train":1504,"dev":198,"final_holdout":298},"ids_commitments":commits,
        }
        train_dev0={"schema":E4_TRAIN_DEV_SCHEMA,"train_ids":train,"dev_ids":dev,
                    "train_ids_sha256":commits["train"],"dev_ids_sha256":commits["dev"],
                    "final_holdout_count":298,"final_holdout_ids_sha256":commits["final_holdout"]}
        train_dev=payload(train_dev0,"td")
        view=td/"view";view.mkdir()
        rows=[]
        for sp,ids,folder in (("train",train,"train"),("dev",dev,"val")):
            for sid in ids:
                ip=view/f"images/{folder}/{sid}.png";lp=view/f"labels/{folder}/{sid}.txt"
                ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
                ip.write_bytes(img);lp.write_bytes(lab)
                rows.append({"sample_id":sid,"split":sp,"image_rel":str(ip.relative_to(view)).replace("\\\\","/"),
                             "label_rel":str(lp.relative_to(view)).replace("\\\\","/"),
                             "image_sha256":h(img),"label_sha256":h(lab)})
        y=(view/"dataset.yaml");y.write_text("path: synthetic\ntrain: images/train\nval: images/val\nnc: 12\n",encoding="utf-8")
        mc=sha256_json(sorted((r["split"],r["sample_id"],r["image_rel"],r["image_sha256"],r["label_rel"],r["label_sha256"]) for r in rows))
        m={"schema":SCHEMA_VIEW_PRIVATE,"recipe_sha256":recipe0["recipe_sha256_self"],
           "train_dev_access_private_sha256":"tdsha","formal_zip_sha256":recipe0["formal_zip_sha256"],
           "train_ids":train,"dev_ids":dev,"dataset_yaml_rel":"dataset.yaml","dataset_yaml_sha256":h(y.read_bytes()),
           "mappings":rows,"mapping_commitment":mc}
        _,reuse1=atomic_json_write(view/"view_manifest.json",m,private=True,request_fingerprint="vf")
        _,reuse2=atomic_json_write(view/"view_manifest.json",m,private=True,request_fingerprint="vf")
        checks["same_request_idempotent"]=reuse1 is False and reuse2 is True
        lockp=td/"concurrency.lock"
        try:
            with file_lock(lockp,0.1,900):
                try:
                    with file_lock(lockp,0.0,900):
                        pass
                    checks["concurrent_writer_rejected"]=False
                except GateError as e:
                    checks["concurrent_writer_rejected"]=e.code=="CONCURRENT_RUN_LOCKED"
        except GateError:
            checks["concurrent_writer_rejected"]=False
        vr=verify_view_tree(view/"view_manifest.json",recipe0,train_dev,Deadline(120))
        checks["view_1504_198"]=vr["train_count"]==1504 and vr["dev_count"]==198
        checks["holdout_ids_not_in_train_dev_artifact"]="final_holdout_ids" not in train_dev
        checks["holdout_ids_not_in_view_manifest"]="final_holdout_ids" not in vr["manifest"]
        # Injection attack must fail.
        (view/"images/val/evil.png").write_bytes(img)
        try:
            verify_view_tree(view/"view_manifest.json",recipe0,train_dev,Deadline(120))
            checks["extra_file_rejected"]=False
        except GateError:
            checks["extra_file_rejected"]=True
        (view/"images/val/evil.png").unlink()
        # File tamper must fail.
        victim=view/f"images/train/{train[0]}.png";victim.write_bytes(b"tamper")
        try:
            verify_view_tree(view/"view_manifest.json",recipe0,train_dev,Deadline(120))
            checks["file_tamper_rejected"]=False
        except GateError:
            checks["file_tamper_rejected"]=True
    report={"schema":"t1gr-e5-synthetic-gate-v1","checks":checks,"passed":sum(checks.values()),"total":len(checks),"all_passed":all(checks.values()),"formal":False}
    out=ROOT/"reports/step4_t1gr/e5_synthetic_gate.json";out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not report["all_passed"]:raise SystemExit(2)
if __name__=="__main__":main()
