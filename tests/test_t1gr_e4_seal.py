from pathlib import Path
import importlib.util,json,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
def load(n,p):
 s=importlib.util.spec_from_file_location(n,ROOT/p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load("seal","scripts/t1gr_e4_seal_split.py")
v=load("verify","scripts/t1gr_e4_verify_seal.py")
from multimodal.t1gr_secure_io import GateError,assert_public_safe,sha256_json

def fake_ids():
 return {"train":[f"t{i:04d}" for i in range(1504)],"dev":[f"d{i:04d}" for i in range(198)],"final_holdout":[f"h{i:04d}" for i in range(298)]}

def test_validate_ids_rejects_null():
 try:m.validate_ids(None,"X")
 except GateError:pass
 else:raise AssertionError

def test_validate_ids_rejects_duplicate():
 try:m.validate_ids(["a","a"],"X")
 except GateError:pass
 else:raise AssertionError

def test_commitment_changes_on_id_change():
 x=fake_ids();a=sha256_json(sorted(x["train"]));x["train"][0]="z";b=sha256_json(sorted(x["train"]));assert a!=b

def test_public_scanner_rejects_raw_ids():
 try:assert_public_safe({"train_ids":["x"]})
 except GateError:pass
 else:raise AssertionError

def test_public_scanner_allows_commitments():
 assert_public_safe({"ids_commitments":{"train":"a"*64}})

def test_payload_integrity():
 o={"a":1};o["payload_sha256"]=sha256_json({"a":1});o["request_fingerprint"]="x";assert m.payload_ok(o)

def test_payload_integrity_detects_tamper():
 o={"a":2,"payload_sha256":sha256_json({"a":1}),"request_fingerprint":"x"};assert not m.payload_ok(o)

def test_freeze_timestamp_recovery_same():
 a=({"freeze_timestamp_utc":"2026-01-01T00:00:00Z"},"x");assert m.recover_freeze_timestamp([a,None])=="2026-01-01T00:00:00Z"

def test_freeze_timestamp_conflict():
 a=({"freeze_timestamp_utc":"a"},"x");b=({"freeze_timestamp_utc":"b"},"y")
 try:m.recover_freeze_timestamp([a,b])
 except GateError:pass
 else:raise AssertionError

def test_policy_no_human_secrecy_claim():
 p=json.loads((ROOT/"config/t1gr_e4_seal_policy.json").read_text());assert p["seal"]["human_secrecy_claimed"] is False

def test_policy_training_not_authorized():
 p=json.loads((ROOT/"config/t1gr_e4_seal_policy.json").read_text());assert p["seal"]["step1_training_authorized_by_e4_alone"] is False

def test_policy_holdout_forbidden_for_tuning():
 p=json.loads((ROOT/"config/t1gr_e4_seal_policy.json").read_text());assert p["seal"]["final_holdout_tuning_access"]=="FORBIDDEN"

def test_policy_holdout_forbidden_for_seed_selection():
 p=json.loads((ROOT/"config/t1gr_e4_seal_policy.json").read_text());assert p["seal"]["final_holdout_seed_selection_access"]=="FORBIDDEN"

def test_reviewed_counts_exact():
 p=json.loads((ROOT/"config/t1gr_e4_seal_policy.json").read_text());assert sum(p["reviewed_e3"]["counts"].values())==2000

def test_reviewed_commit_exact():
 p=json.loads((ROOT/"config/t1gr_e4_seal_policy.json").read_text());assert p["reviewed_e3"]["commit_full"]=="9835262acd8a23aa86ff7076909abbceb18060dc"

def test_three_commitments_64hex():
 p=json.loads((ROOT/"config/t1gr_e4_seal_policy.json").read_text())
 for x in p["reviewed_e3"]["ids_commitments"].values():assert len(x)==64 and all(c in "0123456789abcdef" for c in x)

def test_train_dev_schema_has_no_holdout_id_list_literal():
 s=(ROOT/"scripts/t1gr_e4_seal_split.py").read_text()
 block=s.split("train_dev={",1)[1].split("holdout={",1)[0]
 assert '"final_holdout_ids":' not in block

def test_holdout_schema_does_have_holdout_ids():
 s=(ROOT/"scripts/t1gr_e4_seal_split.py").read_text()
 block=s.split("holdout={",1)[1].split("receipt={",1)[0]
 assert '"final_holdout_ids":ids["final_holdout"]' in block

def test_verifier_never_prints_ids():
 s=(ROOT/"scripts/t1gr_e4_verify_seal.py").read_text()
 assert "print(train" not in s and "print(hold" not in s

def test_public_e5_only_after_verification():
 s=(ROOT/"scripts/t1gr_e4_verify_seal.py").read_text();assert '"e5_entry_authorized":True' in s

def test_seal_public_e5_worded_after_verification():
 s=(ROOT/"scripts/t1gr_e4_seal_split.py").read_text();assert '"e5_entry_authorized_after_seal_verification":True' in s

def test_output_collision_check_present():
 assert "OUTPUT_PATH_COLLISION" in (ROOT/"scripts/t1gr_e4_seal_split.py").read_text()

def test_git_ancestor_provenance_present():
 s=(ROOT/"scripts/t1gr_e4_seal_split.py").read_text();assert "merge-base" in s and "--is-ancestor" in s

def test_reviewed_candidate_not_any_pass_candidate():
 p=json.loads((ROOT/"config/t1gr_e4_seal_policy.json").read_text())
 assert p["reviewed_e3"]["split_request_fingerprint"]=="da1a0b1a57484d10a82cafb63b06538efbf755b28703e83fff8bbe0435637289"

def test_final_holdout_open_policy():
 p=json.loads((ROOT/"config/t1gr_e4_seal_policy.json").read_text());assert p["seal"]["final_holdout_open_policy"].startswith("DO_NOT_OPEN")

def test_final_holdout_count_298():
 p=json.loads((ROOT/"config/t1gr_e4_seal_policy.json").read_text());assert p["reviewed_e3"]["counts"]["final_holdout"]==298

def test_secure_io_private_outside_repo_guard_exists():
 s=(ROOT/"src/multimodal/t1gr_secure_io.py").read_text();assert "PRIVATE_OUTPUT_INSIDE_REPO" in s

def test_secure_io_private_input_outside_repo_guard_exists():
 s=(ROOT/"src/multimodal/t1gr_secure_io.py").read_text();assert "PRIVATE_INPUT_INSIDE_REPO" in s

def test_secure_io_atomic_replace_exists():
 s=(ROOT/"src/multimodal/t1gr_secure_io.py").read_text();assert "os.replace" in s

def test_secure_io_lock_exists():
 s=(ROOT/"src/multimodal/t1gr_secure_io.py").read_text();assert "O_EXCL" in s

def test_public_no_path_string():
 assert_public_safe({"evidence_model":"repo nondisclosure + harness access seal"})

def test_public_rejects_path():
 try:assert_public_safe({"foo":"E:/private/x.json"})
 except GateError:pass
 else:raise AssertionError

def test_public_boolean_no_raw_id_statement_is_safe():
 assert_public_safe({"any_raw_sample_id_present": False})

def test_raw_ids_still_rejected():
 try:assert_public_safe({"train_ids":["sample_001"]})
 except GateError:pass
 else:raise AssertionError

def test_public_false_boolean_does_not_use_ids_suffix():
 assert not "any_raw_sample_id_present".endswith("_ids")
