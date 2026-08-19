#!/usr/bin/env python3
"""Hardened deterministic E3 split proposal.

Consumes ONLY:
  1) formal training ZIP,
  2) PRIVATE finalized leakage components v1.2,
  3) frozen E3 policy v1.2.

It does not consume old val6, model metrics, or FINAL-HOLDOUT content. It produces one
candidate only; E4 sealing remains separately unauthorized.
"""
from __future__ import annotations

import argparse, collections, json, math, random, sys, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from multimodal.t1gr_secure_io import (  # noqa:E402
    Deadline, assert_public_safe, atomic_json_write, check_existing_output, ensure_private_input, ensure_private_output, ensure_public_output, ensure_repo_input,
    fail, file_lock, read_json_bounded, require_dict, require_keys, require_list, safe_error_message,
    require_unchanged, sha256_file, sha256_json, stat_token, validate_identifier, validate_zip_name,
)

SCRIPT_VERSION="t1gr-e3-split-candidate-hardened-v1.2"
FROZEN_POLICY_SHA256="9f60361ae393983b1f972b428005f354429f2b9abd71c2ee50459706b7ebe9ae"
NAMES=["person","boat","animal","seat","sign","bicycle","car","ball","light","garbage can","uav","tricycle"]
SPLITS=("train","dev","final_holdout")


def mod(name):
 p=[x for x in name.replace("\\","/").split("/") if x]
 return p[-2].lower() if len(p)>=2 else None

def sid(name): return Path(name).stem

def parse_label(raw):
 box=[0]*12;present=set();rows=[]
 try:text=raw.decode("utf-8-sig")
 except UnicodeDecodeError:fail("LABEL_UTF8_DRIFT")
 for line in text.splitlines():
  if not line.strip():continue
  parts=line.split()
  if len(parts)!=5:fail("LABEL_SCHEMA_DRIFT")
  try:vals=[float(x) for x in parts]
  except ValueError:fail("LABEL_NONNUMERIC_DRIFT")
  if not all(math.isfinite(x) for x in vals):fail("LABEL_NONFINITE_DRIFT")
  cf,cx,cy,w,h=vals
  if not cf.is_integer():fail("LABEL_CLASS_NONINTEGER_DRIFT")
  c=int(cf)
  if not 0<=c<12:fail("LABEL_CLASS_DRIFT")
  if w<=0 or h<=0:fail("LABEL_WH_DRIFT")
  if min(cx,cy,w,h)<-0.01 or max(cx,cy,w,h)>1.01:fail("LABEL_ULTRALYTICS_TOLERANCE_DRIFT")
  rows.append((c,cx,cy,w,h))
 for c,cx,cy,w,h in sorted(set(rows)):
  box[c]+=1;present.add(c)
 return box,[1 if c in present else 0 for c in range(12)]


def zip_metadata_commitment(zp: Path, sec: dict, deadline: Deadline) -> tuple[str,list[zipfile.ZipInfo]]:
 try:z=zipfile.ZipFile(zp)
 except FileNotFoundError:fail("FORMAL_ZIP_NOT_FOUND")
 except PermissionError:fail("READ_PERMISSION_DENIED")
 except zipfile.BadZipFile:fail("FORMAL_ZIP_BAD")
 with z:
  infos=z.infolist()
  if len(infos)>int(sec["max_zip_members"]):fail("ZIP_MEMBER_COUNT_EXCEEDED")
  meta=[];seen_names=set()
  for i in infos:
   deadline.check("ZIP_SCAN_TIMEOUT")
   if i.is_dir():continue
   validate_zip_name(i.filename)
   norm=i.filename.replace("\\","/")
   if norm in seen_names:fail("ZIP_DUPLICATE_MEMBER_NAME")
   seen_names.add(norm)
   if bool(sec["reject_encrypted_zip_members"]) and (i.flag_bits & 0x1):fail("ZIP_ENCRYPTED_MEMBER_FORBIDDEN")
   meta.append((norm,int(i.CRC),int(i.file_size),int(i.compress_size)))
  return sha256_json(sorted(meta)), infos


def load_stats(zp: Path, sec: dict, deadline: Deadline) -> tuple[dict,str,str]:
 metadata_commit, _ = zip_metadata_commitment(zp,sec,deadline)
 try:z=zipfile.ZipFile(zp)
 except Exception:fail("FORMAL_ZIP_OPEN_FAILED")
 labels={};ext={};label_hashes=[];total_label_bytes=0
 with z:
  for i in z.infolist():
   deadline.check("ZIP_SCAN_TIMEOUT")
   if i.is_dir():continue
   m=mod(i.filename);s=sid(i.filename)
   if m not in {"labels","visible"}:continue
   validate_identifier(s)
   target=labels if m=="labels" else ext
   if s in target:fail("ZIP_DUPLICATE_SAMPLE_ID")
   if m=="labels":
    if i.file_size<0 or i.file_size>int(sec["max_label_member_bytes"]):fail("LABEL_MEMBER_SIZE_EXCEEDED")
    total_label_bytes+=int(i.file_size)
    if total_label_bytes>int(sec["max_total_label_bytes"]):fail("TOTAL_LABEL_BYTES_EXCEEDED")
    labels[s]=i
   else:ext[s]=Path(i.filename).suffix.lower()
  if set(labels)!=set(ext) or len(labels)!=2000:fail("ZIP_PAIRING_DRIFT")
  st={}
  for s in sorted(labels):
   deadline.check("LABEL_READ_TIMEOUT")
   try:raw=z.read(labels[s])
   except RuntimeError:fail("ZIP_MEMBER_READ_ERROR")
   except OSError:fail("ZIP_MEMBER_READ_ERROR")
   label_hashes.append((s,sha256_json([raw.hex()])))
   b,im=parse_label(raw);st[s]={"boxes":b,"images":im,"jpg":int(ext[s] in {".jpg",".jpeg"})}
 return st,metadata_commit,sha256_json(label_hashes)


def vec_add(a,b):return [x+y for x,y in zip(a,b)]
def unit_stats(ids,st):
 boxes=[0]*12;images=[0]*12;jpg=0
 for s in ids:
  if s not in st:fail("COMPONENT_ID_NOT_IN_FORMAL_ZIP")
  boxes=vec_add(boxes,st[s]["boxes"]);images=vec_add(images,st[s]["images"]);jpg+=st[s]["jpg"]
 return {"n":len(ids),"jpg":jpg,"images":images,"boxes":boxes}

def objective(assign,units,targets,w):
 score=0.0
 for sp in SPLITS:
  cur={"n":0,"jpg":0,"images":[0]*12,"boxes":[0]*12}
  for ui in assign[sp]:
   u=units[ui]["stats"];cur["n"]+=u["n"];cur["jpg"]+=u["jpg"]
   cur["images"]=vec_add(cur["images"],u["images"]);cur["boxes"]=vec_add(cur["boxes"],u["boxes"])
  for key,ww in (("n",w["sample_count"]),("jpg",w["jpg_count"])):
   t=targets[sp][key];score+=ww*((cur[key]-t)/max(1,t))**2
  for key,ww in (("images",w["class_image_count"]),("boxes",w["class_box_count"])):
   for c in range(12):
    t=targets[sp][key][c];score+=ww*((cur[key][c]-t)/max(1,t))**2
 return score


def validate_policy(p):
 require_keys(p,("schema","split","security"),"POLICY_MISSING_FIELDS")
 if p["schema"]!="t1gr-e3-closure-split-policy-v1.2":fail("BAD_POLICY_SCHEMA")
 s=require_dict(p["split"],"BAD_SPLIT_POLICY");require_keys(s,("train_fraction","dev_fraction","final_holdout_fraction","split_seed","hard_min_class_images","objective_weights","n_local_passes"),"SPLIT_POLICY_MISSING")
 vals=[float(s[k]) for k in ("train_fraction","dev_fraction","final_holdout_fraction")]
 if any(v<=0 or v>=1 for v in vals) or abs(sum(vals)-1)>1e-9:fail("BAD_SPLIT_FRACTIONS")
 if not isinstance(s["split_seed"],int):fail("BAD_SPLIT_SEED")
 sec=require_dict(p["security"],"BAD_SECURITY_POLICY");require_keys(sec,("private_parent_must_preexist","public_output_prefix","lock_wait_seconds","lock_stale_seconds","split_timeout_seconds","max_private_json_bytes","max_zip_members","max_label_member_bytes","max_total_label_bytes"),"SECURITY_POLICY_MISSING")


def validate_components(c,p,deadline):
 require_keys(c,("schema","policy_sha256","all_components","force_train_seed_ids","force_train_ids_after_component_propagation","component_gate_passed","component_count"),"COMPONENTS_MISSING_FIELDS")
 if c["schema"]!="t1gr-e3-final-components-private-v1.2":fail("BAD_COMPONENTS_SCHEMA")
 if not c["component_gate_passed"]:fail("COMPONENT_GATE_NOT_PASS")
 comps=require_list(c["all_components"],"BAD_COMPONENTS_TYPE")
 if len(comps)!=int(c["component_count"]):fail("COMPONENT_COUNT_MISMATCH")
 universe=set();norm=[]
 for comp in comps:
  deadline.check();comp=require_list(comp,"BAD_COMPONENT_TYPE")
  if not comp:fail("EMPTY_COMPONENT")
  lc=[]
  for x in comp:
   sx=validate_identifier(x)
   if sx in universe:fail("DUPLICATE_ID_ACROSS_COMPONENTS")
   universe.add(sx);lc.append(sx)
  norm.append(sorted(lc))
 if len(universe)!=2000:fail("COMPONENT_UNIVERSE_COUNT_DRIFT")
 seed=set(validate_identifier(x) for x in require_list(c["force_train_seed_ids"],"BAD_FORCE_SEED_TYPE"))
 force=set(validate_identifier(x) for x in require_list(c["force_train_ids_after_component_propagation"],"BAD_FORCE_PROP_TYPE"))
 if len(seed)!=len(c["force_train_seed_ids"]):fail("DUPLICATE_FORCE_SEED_ID")
 if len(force)!=len(c["force_train_ids_after_component_propagation"]):fail("DUPLICATE_FORCE_PROP_ID")
 if not seed<=universe or not force<=universe or not seed<=force:fail("FORCE_TRAIN_RELATION_INVALID")
 recomputed=set()
 for comp in norm:
  if seed.intersection(comp):recomputed.update(comp)
 if recomputed!=force:fail("FORCE_TRAIN_PROPAGATION_DRIFT")
 return norm,seed,force


def run(args):
 repo=ROOT.resolve(strict=True);pol_path=ensure_repo_input(repo,args.policy,"config")
 policy_token=stat_token(pol_path)
 p=read_json_bounded(pol_path,1<<20);validate_policy(p);sec=p["security"]
 deadline=Deadline(float(args.timeout_seconds or sec["split_timeout_seconds"]))
 cp=ensure_private_input(Path(args.components_private),repo)
 private_out=ensure_private_output(Path(args.private_out),repo,bool(sec["private_parent_must_preexist"]))
 public_out=ensure_public_output(repo,args.public_out,sec["public_output_prefix"])
 lock=public_out.with_suffix(public_out.suffix+".lock")
 with file_lock(lock,float(sec["lock_wait_seconds"]),float(sec["lock_stale_seconds"])):
  components_token=stat_token(cp);zip_token=stat_token(Path(args.formal_zip).expanduser().resolve(strict=False))
  c=read_json_bounded(cp,int(sec["max_private_json_bytes"]),"t1gr-e3-final-components-private-v1.2")
  policy_sha=sha256_file(pol_path,deadline)
  if policy_sha!=FROZEN_POLICY_SHA256:fail("FROZEN_POLICY_SHA_DRIFT")
  if c["policy_sha256"]!=policy_sha:fail("COMPONENT_POLICY_SHA_DRIFT")
  comps,seed,force=validate_components(c,p,deadline)
  zp=Path(args.formal_zip).expanduser().resolve(strict=False)
  stats,zip_meta_commit,label_commit=load_stats(zp,sec,deadline)
  components_sha=sha256_file(cp,deadline)
  require_unchanged(cp,components_token,"COMPONENTS_PRIVATE_CHANGED_DURING_RUN")
  require_unchanged(zp,zip_token,"FORMAL_ZIP_CHANGED_DURING_RUN")
  require_unchanged(pol_path,policy_token,"POLICY_CHANGED_DURING_RUN")
  request_fp=sha256_json({"script":SCRIPT_VERSION,"components_private_sha256":components_sha,"policy_sha256":policy_sha,"formal_zip_metadata_commitment":zip_meta_commit,"label_commitment":label_commit})
  existing_private=check_existing_output(private_out,request_fp)
  existing_public=check_existing_output(public_out,request_fp)
  if existing_private is not None and existing_public is not None:
   pub_obj,pub_sha=existing_public;_,priv_sha=existing_private
   if not bool(pub_obj.get("hard_gate_passed")):fail("SPLIT_HARD_GATE_HOLD")
   return {"status":"PASS","request_fingerprint":request_fp,"private_output_sha256":priv_sha,"public_output_sha256":pub_sha,"idempotent_reuse":True}
  units=[]
  for idx,ids in enumerate(comps):
   deadline.check();units.append({"id":idx,"ids":ids,"force_train":bool(force.intersection(ids)),"stats":unit_stats(ids,stats)})
  if sum(u["stats"]["n"] for u in units)!=2000:fail("UNIT_COVERAGE_FAIL")
  total=unit_stats(sorted(stats),stats);s=p["split"]
  frac={"train":float(s["train_fraction"]),"dev":float(s["dev_fraction"]),"final_holdout":float(s["final_holdout_fraction"])}
  targets={}
  for sp in SPLITS:
   f=frac[sp];targets[sp]={"n":total["n"]*f,"jpg":total["jpg"]*f,"images":[x*f for x in total["images"]],"boxes":[x*f for x in total["boxes"]]}
  assign={x:[] for x in SPLITS};free=[]
  for i,u in enumerate(units):(assign["train"] if u["force_train"] else free).append(i)
  rare=[1/max(1,x) for x in total["images"]]
  free.sort(key=lambda i:(-(units[i]["stats"]["n"]+20*sum(a*b for a,b in zip(units[i]["stats"]["images"],rare))),i))
  w=s["objective_weights"]
  for ui in free:
   deadline.check("SPLIT_GREEDY_TIMEOUT");best=None
   for sp in SPLITS:
    assign[sp].append(ui);sc=objective(assign,units,targets,w);assign[sp].pop();k=(sc,SPLITS.index(sp))
    if best is None or k<best[0]:best=(k,sp)
   assign[best[1]].append(ui)
  rng=random.Random(int(s["split_seed"]));current=objective(assign,units,targets,w)
  for _ in range(int(s["n_local_passes"])):
   deadline.check("SPLIT_LOCAL_SEARCH_TIMEOUT");order=[(sp,ui) for sp in SPLITS for ui in assign[sp] if not units[ui]["force_train"]];rng.shuffle(order);changed=False
   for src,ui in order:
    deadline.check("SPLIT_LOCAL_SEARCH_TIMEOUT")
    if ui not in assign[src]:continue
    best=(current,src)
    for dst in SPLITS:
     if dst==src:continue
     assign[src].remove(ui);assign[dst].append(ui);sc=objective(assign,units,targets,w);assign[dst].remove(ui);assign[src].append(ui)
     if sc+1e-12<best[0]:best=(sc,dst)
    if best[1]!=src:assign[src].remove(ui);assign[best[1]].append(ui);current=best[0];changed=True
   if not changed:break
  ids_by={sp:sorted(x for ui in assign[sp] for x in units[ui]["ids"]) for sp in SPLITS};sets={sp:set(v) for sp,v in ids_by.items()}
  overlap=any(sets[a]&sets[b] for i,a in enumerate(SPLITS) for b in SPLITS[i+1:]);coverage=set().union(*(sets[sp] for sp in SPLITS))==set(stats)
  force_ok=force<=sets["train"] and not force&(sets["dev"]|sets["final_holdout"])
  comp_ok=all(sum(bool(set(comp)&sets[sp]) for sp in SPLITS)==1 for comp in comps)
  support={sp:unit_stats(ids_by[sp],stats) for sp in SPLITS};mins=s["hard_min_class_images"];class_gate={sp:[support[sp]["images"][c]>=int(mins[sp]) for c in range(12)] for sp in SPLITS}
  nonempty=all(len(ids_by[sp])>0 for sp in SPLITS)
  hard=nonempty and not overlap and coverage and force_ok and comp_ok and all(all(v) for v in class_gate.values())
  private={"schema":"t1gr-e3-split-candidate-private-v1.2","script_version":SCRIPT_VERSION,"policy_sha256":policy_sha,"components_private_sha256":components_sha,"formal_zip_metadata_commitment":zip_meta_commit,"labels_commitment":label_commit,"ids":ids_by,"component_indices":{sp:sorted(assign[sp]) for sp in SPLITS},"hard_gate_passed":hard,"objective":current,"note":"CANDIDATE ONLY. Not sealed. No training authorized."}
  public={"schema":"t1gr-e3-split-candidate-public-v1.2","script_version":SCRIPT_VERSION,"policy_sha256":policy_sha,"components_private_sha256":components_sha,"formal_zip_metadata_commitment":zip_meta_commit,"labels_commitment":label_commit,"fractions":frac,"split_seed":s["split_seed"],"objective":current,"counts":{sp:support[sp]["n"] for sp in SPLITS},"jpg_counts":{sp:support[sp]["jpg"] for sp in SPLITS},"png_counts":{sp:support[sp]["n"]-support[sp]["jpg"] for sp in SPLITS},"class_image_counts":{sp:{str(c):support[sp]["images"][c] for c in range(12)} for sp in SPLITS},"class_box_counts":{sp:{str(c):support[sp]["boxes"][c] for c in range(12)} for sp in SPLITS},"hard_min_class_images":mins,"class_min_gate":{sp:{str(c):class_gate[sp][c] for c in range(12)} for sp in SPLITS},"all_splits_nonempty":nonempty,"sample_overlap_empty":not overlap,"union_equals_2000":coverage,"historical_component_quarantine_respected":force_ok,"components_not_split":comp_ok,"hard_gate_passed":hard,"ids_commitments":{sp:sha256_json(ids_by[sp]) for sp in SPLITS},"private_ids_in_public_report":False,"status":"CANDIDATE_PASS_AWAIT_REVIEW" if hard else "CANDIDATE_HOLD","e4_seal_authorized":False,"step1_authorized":False}
  assert_public_safe(public)
  psha,preuse=atomic_json_write(private_out,private,private=True,request_fingerprint=request_fp)
  usha,ureuse=atomic_json_write(public_out,public,private=False,request_fingerprint=request_fp)
  if not hard:fail("SPLIT_HARD_GATE_HOLD")
  return {"status":"PASS","request_fingerprint":request_fp,"private_output_sha256":psha,"public_output_sha256":usha,"idempotent_reuse":bool(preuse and ureuse)}


def main():
 ap=argparse.ArgumentParser();ap.add_argument("--formal-zip",required=True);ap.add_argument("--components-private",required=True);ap.add_argument("--policy",default="config/t1gr_e3_closure_split_policy.json");ap.add_argument("--private-out",required=True);ap.add_argument("--public-out",default="reports/step4_t1gr/split_candidate_public.json");ap.add_argument("--timeout-seconds",type=float,default=None);args=ap.parse_args()
 try:print(json.dumps(run(args),ensure_ascii=False,indent=2))
 except Exception as e:print(json.dumps({"status":"FAIL","error":safe_error_message(e)},ensure_ascii=False),file=sys.stderr);raise SystemExit(2)
if __name__=="__main__":main()
