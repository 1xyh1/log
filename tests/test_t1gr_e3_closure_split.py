from pathlib import Path
import importlib.util, json, tempfile
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
f=load("f","scripts/t1gr_finalize_leakage_components.py")
p=load("p","scripts/t1gr_propose_component_split.py")

def test_dsu_review_merge():
 d=f.DSU(["a","b","c"]);d.union("a","b");d.union("b","c")
 assert sorted(len(x) for x in d.comps())==[3]

def test_component_quarantine_propagation_logic():
 comps=[["a","b"],["c"],["d","e"]];force={"b"}
 got=[c for c in comps if force.intersection(c)]
 assert got==[["a","b"]]

def test_parse_label():
 b,i=p.parse_label(b"0 0.5 0.5 0.2 0.2\n11 0.4 0.4 0.1 0.1\n")
 assert b[0]==1 and b[11]==1 and i[0]==1 and i[11]==1

def test_unit_stats():
 st={"a":{"boxes":[1]+[0]*11,"images":[1]+[0]*11,"jpg":1},
     "b":{"boxes":[2]+[0]*11,"images":[1]+[0]*11,"jpg":0}}
 u=p.unit_stats(["a","b"],st);assert u["n"]==2 and u["jpg"]==1 and u["boxes"][0]==3

def test_policy_fractions():
 pol=json.loads((ROOT/"config/t1gr_e3_closure_split_policy.json").read_text())
 s=pol["split"];assert abs(s["train_fraction"]+s["dev_fraction"]+s["final_holdout_fraction"]-1)<1e-9

def test_policy_review_all():
 pol=json.loads((ROOT/"config/t1gr_e3_closure_split_policy.json").read_text())
 assert pol["review_edge_adjudication"]=="MERGE_ALL_CONSERVATIVELY"

def test_final_min_support():
 pol=json.loads((ROOT/"config/t1gr_e3_closure_split_policy.json").read_text())
 assert pol["split"]["hard_min_class_images"]["final_holdout"]>=3

def test_force_train_override():
 pol=json.loads((ROOT/"config/t1gr_e3_closure_split_policy.json").read_text())
 assert pol["split"]["force_train_overrides_fraction"] is True

def test_parse_label_dedups_exact_rows():
 x=b"0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n"
 b,i=p.parse_label(x)
 assert b[0]==1 and i[0]==1

def test_parse_label_rejects_noninteger_class():
 try:
  p.parse_label(b"0.5 0.5 0.5 0.2 0.2\n")
 except RuntimeError as e:
  assert "NONINTEGER" in str(e)
 else:
  raise AssertionError("expected rejection")
