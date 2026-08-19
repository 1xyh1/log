from __future__ import annotations
import importlib.util, json, os, sys, tempfile, time
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from multimodal import t1gr_secure_io as sec

def load(name,path):
 s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
f=load("f","scripts/t1gr_finalize_leakage_components.py")
p=load("p","scripts/t1gr_propose_component_split.py")
POL=json.loads((ROOT/"config/t1gr_e3_closure_split_policy.json").read_text())

def base_visual():
 ids=[f"s{i:04d}" for i in range(2000)]
 # Synthetic shape pins the same accepted provenance counts: 1880 comps, 25 nontrivial, max 27.
 # Use 120 merge edges distributed inside 25 components, then duplicate-free extra edges to total 808.
 sizes=[27,24,20,11,10,6,5,5,3,3,3]+[2]*14
 comps=[];pos=0
 for z in sizes: comps.append(ids[pos:pos+z]);pos+=z
 comps += [[x] for x in ids[pos:]]
 strong=[];pairs=set()
 for c in comps:
  if len(c)>1:
   for a,b in zip(c,c[1:]):
    pair=tuple(sorted((a,b)));pairs.add(pair);strong.append({"a":a,"b":b,"class":"STRONG_NEAR_DUPLICATE"})
 # Add redundant within-component edges until exactly 808 without changing components.
 for c in comps[:25]:
  if len(strong)>=808: break
  for i,a in enumerate(c):
   if len(strong)>=808: break
   for b in c[i+1:]:
    pair=tuple(sorted((a,b)))
    if pair not in pairs:
     pairs.add(pair);strong.append({"a":a,"b":b,"class":"STRONG_NEAR_DUPLICATE"})
     if len(strong)>=808: break
 if len(strong)!=808: raise RuntimeError("synthetic edge construction failed")
 review=[];rp=set();i=1000
 while len(review)<180:
  a,b=ids[i%2000],ids[(i+317)%2000];i+=1;pair=tuple(sorted((a,b)))
  if a!=b and pair not in pairs and pair not in rp:
   rp.add(pair);review.append({"a":a,"b":b,"class":"REVIEW_NEAR_DUPLICATE"})
 return {"schema":"t1gr-visual-leakage-private-v1","policy_sha256":POL["source_visual_leakage"]["source_policy_sha256"],
  "internal_components_all":comps,"internal_strong_edges":strong,"internal_review_edges":review,
  "force_train_ids_due_historical_contamination":ids[:18]}

def test_null_required_field_rejected():
 v=base_visual();v["internal_review_edges"]=None
 with pytest.raises(sec.GateError):f.validate_visual_private(v,POL,sec.Deadline(5))

def test_empty_component_rejected():
 v=base_visual();v["internal_components_all"][3]=[]
 with pytest.raises(sec.GateError):f.validate_visual_private(v,POL,sec.Deadline(5))

def test_duplicate_id_across_components_rejected():
 v=base_visual();v["internal_components_all"][1]=[v["internal_components_all"][0][0]]
 with pytest.raises(sec.GateError):f.validate_visual_private(v,POL,sec.Deadline(5))

def test_self_edge_rejected():
 v=base_visual();v["internal_strong_edges"][0]["b"]=v["internal_strong_edges"][0]["a"]
 with pytest.raises(sec.GateError):f.validate_visual_private(v,POL,sec.Deadline(5))

def test_duplicate_edge_rejected():
 v=base_visual();v["internal_strong_edges"][1]=dict(v["internal_strong_edges"][0])
 with pytest.raises(sec.GateError):f.validate_visual_private(v,POL,sec.Deadline(5))

def test_force_id_outside_universe_rejected():
 v=base_visual();v["force_train_ids_due_historical_contamination"][0]="not_in_universe"
 with pytest.raises(sec.GateError):f.validate_visual_private(v,POL,sec.Deadline(5))

def test_private_output_inside_repo_rejected(tmp_path):
 with pytest.raises(sec.GateError):sec.ensure_private_output(ROOT/"reports"/"x.json",ROOT,False)

def test_public_escape_rejected():
 with pytest.raises(sec.GateError):sec.ensure_public_output(ROOT,"../leak.json","reports/step4_t1gr")

def test_public_wrong_prefix_rejected():
 with pytest.raises(sec.GateError):sec.ensure_public_output(ROOT,"config/leak.json","reports/step4_t1gr")

def test_public_sensitive_key_rejected():
 with pytest.raises(sec.GateError):sec.assert_public_safe({"ids":["secret"]})

def test_public_path_string_rejected():
 with pytest.raises(sec.GateError):sec.assert_public_safe({"note":"E:/google/private/file.json"})

def test_public_commitment_allowed():
 sec.assert_public_safe({"force_train_ids_commitment":"a"*64,"private_ids_in_public_report":False})

def test_atomic_same_request_idempotent(tmp_path):
 pth=tmp_path/"x.json";obj={"schema":"x","n":1}
 a,re1=sec.atomic_json_write(pth,obj,private=False,request_fingerprint="r1")
 b,re2=sec.atomic_json_write(pth,obj,private=False,request_fingerprint="r1")
 assert a==b and not re1 and re2

def test_atomic_different_request_conflict(tmp_path):
 pth=tmp_path/"x.json";sec.atomic_json_write(pth,{"schema":"x"},private=False,request_fingerprint="r1")
 with pytest.raises(sec.GateError):sec.atomic_json_write(pth,{"schema":"x"},private=False,request_fingerprint="r2")

def test_lock_blocks_concurrent(tmp_path):
 lock=tmp_path/"x.lock"
 with sec.file_lock(lock,0.1,3600):
  with pytest.raises(sec.GateError):
   with sec.file_lock(lock,0.1,3600): pass

def test_stale_dead_lock_recovers(tmp_path):
 lock=tmp_path/"x.lock";lock.write_text(json.dumps({"pid":99999999,"created_unix":0}))
 old=time.time()-10000;os.utime(lock,(old,old))
 with sec.file_lock(lock,0.2,1): assert lock.exists()
 assert not lock.exists()

def test_deadline_expires():
 d=sec.Deadline(0.01);time.sleep(0.02)
 with pytest.raises(sec.GateError):d.check()

def test_read_json_size_limit(tmp_path):
 pth=tmp_path/"x.json";pth.write_text(json.dumps({"a":"x"*100}))
 with pytest.raises(sec.GateError):sec.read_json_bounded(pth,10)

def test_parse_label_dedups_exact_rows():
 b,i=p.parse_label(b"0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")
 assert b[0]==1 and i[0]==1

def test_parse_label_rejects_noninteger_class():
 with pytest.raises(sec.GateError):p.parse_label(b"0.5 0.5 0.5 0.2 0.2\n")

def test_parse_label_rejects_null_bytes_or_nonnumeric():
 with pytest.raises(sec.GateError):p.parse_label(b"0 a 0.5 0.2 0.2\n")

def test_validate_components_force_propagation():
 comps=[["a","b"],["c"]]+[[f"x{i}"] for i in range(1997)]
 allids=[x for c in comps for x in c]
 c={"schema":"t1gr-e3-final-components-private-v1.2","policy_sha256":"x","all_components":comps,"force_train_seed_ids":["b"],"force_train_ids_after_component_propagation":["a","b"],"component_gate_passed":True,"component_count":len(comps)}
 norm,seed,force=p.validate_components(c,POL,sec.Deadline(5));assert force=={"a","b"}

def test_validate_components_bad_propagation_rejected():
 comps=[["a","b"]]+[[f"x{i}"] for i in range(1998)]
 c={"schema":"t1gr-e3-final-components-private-v1.2","policy_sha256":"x","all_components":comps,"force_train_seed_ids":["b"],"force_train_ids_after_component_propagation":["b"],"component_gate_passed":True,"component_count":len(comps)}
 with pytest.raises(sec.GateError):p.validate_components(c,POL,sec.Deadline(5))

def test_policy_no_seed_shopping():
 assert isinstance(POL["split"]["split_seed"],int) and POL["split"]["candidate_generation"].startswith("DETERMINISTIC")


def test_private_input_inside_repo_rejected():
 with pytest.raises(sec.GateError):sec.ensure_private_input(ROOT/"config"/"t1gr_e3_closure_split_policy.json",ROOT)

def test_repo_policy_escape_rejected(tmp_path):
 with pytest.raises(sec.GateError):sec.ensure_repo_input(ROOT,"../outside.json","config")

def test_strong_list_review_class_rejected():
 v=base_visual();v["internal_strong_edges"][0]["class"]="REVIEW_NEAR_DUPLICATE"
 with pytest.raises(sec.GateError):f.validate_visual_private(v,POL,sec.Deadline(5))

def test_strong_component_provenance_mismatch_rejected():
 v=base_visual();v["internal_components_all"]=[[f"s{i:04d}"] for i in range(2000)]
 with pytest.raises(sec.GateError):f.validate_visual_private(v,POL,sec.Deadline(5))


def test_existing_output_integrity_detects_tamper(tmp_path):
 pth=tmp_path/"x.json"
 sec.atomic_json_write(pth,{"schema":"x","n":1},private=False,request_fingerprint="r1")
 obj=json.loads(pth.read_text());obj["n"]=2;pth.write_text(json.dumps(obj))
 with pytest.raises(sec.GateError):sec.check_existing_output(pth,"r1")

def test_union_uses_split_values_not_dict_keys():
 sets={"train":{"a"},"dev":{"b"},"final_holdout":{"c"}}
 coverage=set().union(*(sets[sp] for sp in p.SPLITS))=={"a","b","c"}
 assert coverage


def test_private_atomic_mode_posix(tmp_path):
 pth=tmp_path/"private.json"
 sec.atomic_json_write(pth,{"schema":"x"},private=True,request_fingerprint="r")
 if os.name!="nt":
  assert (pth.stat().st_mode & 0o077)==0

def test_exception_redaction_generic():
 assert sec.safe_error_message(RuntimeError("E:/secret/sample_001"))=="UNHANDLED_INTERNAL_ERROR"

def test_exception_redaction_gate_safe_only():
 assert sec.safe_error_message(sec.GateError("SAFE_CODE","count=3"))=="SAFE_CODE:count=3"

def test_zip_unsafe_member_name_rejected():
 with pytest.raises(sec.GateError):sec.validate_zip_name("../labels/x.txt")

def test_stat_token_change_detected(tmp_path):
 pth=tmp_path/"x";pth.write_text("a");tok=sec.stat_token(pth);time.sleep(0.001);pth.write_text("bb")
 with pytest.raises(sec.GateError):sec.require_unchanged(pth,tok)

def test_frozen_policy_sha_matches_file():
 import hashlib
 h=hashlib.sha256((ROOT/"config/t1gr_e3_closure_split_policy.json").read_bytes()).hexdigest()
 assert h==f.FROZEN_POLICY_SHA256==p.FROZEN_POLICY_SHA256
