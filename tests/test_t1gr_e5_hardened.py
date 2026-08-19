from pathlib import Path
import importlib.util,json,sys,tempfile,copy,hashlib,os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from multimodal.t1gr_secure_io import GateError,assert_public_safe,atomic_json_write,sha256_json
from multimodal.t1gr_e5_core import *

def candidate():
    x=json.loads((ROOT/"config/t1gr_e5_training_spec.candidate.json").read_text())
    x["status"]="REVIEWED_FROZEN"
    return x

def with_payload(o,fp="x"):
    b=dict(o);b["payload_sha256"]=sha256_json(o);b["request_fingerprint"]=fp;return b

def fake_e2():
    forensic={"schema":FORENSIC_SCHEMA,"common_id_count":2000,"id_sets_equal":True,
              "member_counts":{"visible":2000,"infrared":2000,"depth":2000,"labels":2000},
              "extension_counts":{"visible":{".jpg":149,".png":1851},"infrared":{".jpg":149,".png":1851},"depth":{".jpg":149,".png":1851},"labels":{".txt":2000}},
              "duplicate_id_counts":{"visible":0,"infrared":0,"depth":0,"labels":0},
              "triplet_extension_mismatch_count":0,"triplet_header_dimension_mismatch_count":0,
              "header_error_count":0,"labels":{"class_names":CLASS_NAMES}}
    tax={"schema":TAXONOMY_SCHEMA,"n_label_visible_pairs":2000,"class_names":CLASS_NAMES,
         "sample_category_counts":{"CLEAN":1667,"DERIVED_CORNER_OVERFLOW":332,"DUPLICATE_ROWS":2,"STRICT_[0,1]_ONLY":3},
         "adjudication_counts":{"hard_schema_or_class_samples":0,"ultralytics_8_4_56_reject_samples":0,"corner_overflow_only_samples":328,"duplicate_row_samples":2}}
    return forensic,tax

def fake_e4():
    train=[f"t{i:04d}" for i in range(1504)];dev=[f"d{i:04d}" for i in range(198)];hold=[f"h{i:04d}" for i in range(298)]
    commits={"train":canonical_ids_sha(train),"dev":canonical_ids_sha(dev),"final_holdout":canonical_ids_sha(hold)}
    freeze=with_payload({"schema":E4_FREEZE_SCHEMA,"seal_gate_passed":True,"step1_training_authorized":False,
                         "final_holdout_open_authorized":False,"freeze_timestamp_utc":"2026-08-20T00:00:00Z",
                         "sample_counts":{"train":1504,"dev":198,"final_holdout":298},"ids_commitments":commits,
                         "train_dev_access_private_sha256":"x"})
    verify=with_payload({"schema":E4_VERIFY_SCHEMA,"seal_verification_passed":True,"e5_entry_authorized":True,
                         "step1_training_authorized":False,"final_holdout_open_authorized":False,
                         "sample_counts":freeze["sample_counts"],"ids_commitments":commits})
    td0={"schema":E4_TRAIN_DEV_SCHEMA,"freeze_timestamp_utc":"2026-08-20T00:00:00Z","train_ids":train,"dev_ids":dev,
         "train_ids_sha256":commits["train"],"dev_ids_sha256":commits["dev"],
         "final_holdout_count":298,"final_holdout_ids_sha256":commits["final_holdout"]}
    td=with_payload(td0,"td")
    return freeze,verify,td,hold

def test_candidate_requires_review():
    x=json.loads((ROOT/"config/t1gr_e5_training_spec.candidate.json").read_text())
    try:validate_training_spec(x)
    except GateError as e: assert e.code=="E5_TRAINING_SPEC_NOT_REVIEWED"
    else:raise AssertionError

def test_reviewed_candidate_validates():
    validate_training_spec(candidate())

def test_optimizer_auto_rejected():
    x=candidate();x["train_args"]["optimizer"]="auto"
    try:validate_training_spec(x)
    except GateError as e:assert e.code=="E5_OPTIMIZER_AUTO_FORBIDDEN"
    else:raise AssertionError

def test_null_workers_rejected():
    x=candidate();x["train_args"]["workers"]=None
    try:validate_training_spec(x)
    except GateError:pass
    else:raise AssertionError

def test_e2_valid():
    validate_e2_evidence(*fake_e2())

def test_e2_ul_reject_blocks():
    f,t=fake_e2();t["adjudication_counts"]["ultralytics_8_4_56_reject_samples"]=1
    try:validate_e2_evidence(f,t)
    except GateError as e:assert e.code=="E2_ULTRALYTICS_LABEL_FAIL"
    else:raise AssertionError

def test_e4_train_dev_rejects_holdout_list():
    f,v,td,h=fake_e4();td["final_holdout_ids"]=h
    # payload now invalid first, refresh so semantic gate is tested.
    td0={k:v for k,v in td.items() if k not in ("payload_sha256","request_fingerprint")}
    td=with_payload(td0,"td")
    try:validate_e4_evidence(f,v,td,"x")
    except GateError as e:assert e.code in {"E4_TRAIN_DEV_EXPOSES_HOLDOUT_IDS","E4_TRAIN_DEV_SHA_DRIFT"}
    else:raise AssertionError

def test_public_scanner_rejects_path():
    try:assert_public_safe({"x":"E:/secret/view"})
    except GateError:pass
    else:raise AssertionError

def test_public_scanner_rejects_ids():
    try:assert_public_safe({"train_ids":["x"]})
    except GateError:pass
    else:raise AssertionError

def test_class_map_is_12():
    assert len(CLASS_NAMES)==12 and CLASS_NAMES[11]=="tricycle"

def test_canonical_commitment_order_independent():
    assert canonical_ids_sha(["b","a"])==canonical_ids_sha(["a","b"])

def test_payload_ok():
    assert payload_ok(with_payload({"a":1}))

def test_payload_tamper_detected():
    x=with_payload({"a":1});x["a"]=2;assert not payload_ok(x)

def test_runtime_smoke_one_epoch():
    assert candidate()["runtime"]["smoke_epochs"]==1

def test_end2end_true():
    assert candidate()["train_args"]["end2end"] is True

def test_formal_epochs_80_candidate():
    assert candidate()["train_args"]["epochs"]==80

def test_batch4_candidate():
    assert candidate()["train_args"]["batch"]==4

def test_seed_candidate():
    assert candidate()["train_args"]["seed"]==20260812

def test_optimizer_candidate_explicit():
    assert candidate()["train_args"]["optimizer"]=="MuSGD"

def test_eval_conf_explicit():
    assert candidate()["eval_args"]["conf"]==0.001

def test_eval_max_det_explicit():
    assert candidate()["eval_args"]["max_det"]==300

def test_effective_args_mismatch():
    class X: a=1
    assert effective_args_mismatch(X(),{"a":1})=={}
    assert "a" in effective_args_mismatch(X(),{"a":2})

def test_optimizer_fingerprint_minimal():
    class O:
        param_groups=[{"lr":0.1,"initial_lr":0.1,"weight_decay":0.0,"params":[1,2]}]
    o=O();x=optimizer_fingerprint(o);assert x["param_group_count"]==1

def test_parse_utc_aware():
    assert parse_utc("2026-08-20T00:00:00Z").tzinfo is not None

def test_parse_utc_naive_rejected():
    try:parse_utc("2026-08-20T00:00:00")
    except GateError:pass
    else:raise AssertionError

def test_security_policy_no_holdout_input():
    p=json.loads((ROOT/"config/t1gr_e5_security_policy.json").read_text())
    assert p["final_holdout_sealed_artifact_is_not_an_E5_input"] is True

def test_runner_has_no_scientific_cli_overrides():
    s=(ROOT/"scripts/t1gr_e5_run_step1.py").read_text()
    for arg in ("--data","--device","--epochs","--batch","--optimizer","--project","--name"):
        assert f'ap.add_argument("{arg}"' not in s

def test_runner_formal_requires_fixed_smoke():
    s=(ROOT/"scripts/t1gr_e5_run_step1.py").read_text()
    assert 'reports/step4_t1gr/e5_step1_smoke_public.json' in s
    assert 'if a.mode=="formal"' in s

def test_eval_no_holdout_arg():
    s=(ROOT/"scripts/t1gr_e5_eval_step1.py").read_text()
    assert "final-holdout-sealed" not in s

def test_final_multiseed_not_directly_authorized():
    s=(ROOT/"scripts/t1gr_e5_final_audit.py").read_text()
    assert '"t1gr_multiseed_training_authorized":False' in s

def test_recipe_has_full_zip_hash():
    assert "formal_zip_sha256" in (ROOT/"scripts/t1gr_e5_freeze_recipe.py").read_text()

def test_view_is_copy_not_symlink():
    s=(ROOT/"scripts/t1gr_e5_build_rgb_view.py").read_text()
    assert "symlink" not in s.lower()
    assert "write_private_file" in s

def test_run_root_outside_repo():
    assert "E5_RUN_ROOT_INSIDE_REPO" in (ROOT/"scripts/t1gr_e5_run_step1.py").read_text()

def test_view_root_outside_repo():
    assert "E5_VIEW_ROOT_INSIDE_REPO" in (ROOT/"scripts/t1gr_e5_build_rgb_view.py").read_text()

def test_args_yaml_private_run():
    s=(ROOT/"scripts/t1gr_e5_run_step1.py").read_text()
    assert 'run_dir/"args.yaml"' in s

def test_exact_epoch_gate():
    assert "E5_EPOCH_COUNT_DRIFT" in (ROOT/"scripts/t1gr_e5_run_step1.py").read_text()

def test_timeout_callback_gate():
    s=(ROOT/"scripts/t1gr_e5_run_step1.py").read_text()
    assert "on_train_batch_end" in s and "E5_TRAINING_TIMEOUT" in s

def test_view_extra_file_gate():
    assert "E5_VIEW_EXTRA_OR_MISSING_FILE" in (ROOT/"src/multimodal/t1gr_e5_core.py").read_text()

def test_no_holdout_sealed_in_operational_e5_scripts():
    names={
      "t1gr_e5_freeze_recipe.py","t1gr_e5_build_rgb_view.py","t1gr_e5_preflight.py",
      "t1gr_e5_run_step1.py","t1gr_e5_eval_step1.py","t1gr_e5_final_audit.py",
    }
    for name in names:
        s=(ROOT/"scripts"/name).read_text()
        assert "final-holdout-sealed" not in s and "final_holdout_sealed" not in s


def test_frozen_spec_validates():
    x=json.loads((ROOT/"config/t1gr_e5_training_spec.frozen.json").read_text())
    validate_training_spec(x)

def test_multi_gpu_device_rejected():
    x=candidate();x["runtime"]["device"]="0,1"
    try: validate_training_spec(x)
    except GateError as e: assert e.code=="E5_MULTI_GPU_FORBIDDEN"
    else: raise AssertionError

def test_bool_type_rejected():
    x=candidate();x["train_args"]["amp"]=1
    try: validate_training_spec(x)
    except GateError as e: assert e.code=="E5_TRAINING_SPEC_BOOL_TYPE_FAIL"
    else: raise AssertionError

def test_offline_guard_exists():
    s=(ROOT/"src/multimodal/t1gr_e5_core.py").read_text()
    assert "ultralytics_offline_guard" in s and "ul_events.enabled = False" in s

def test_amp_network_probe_bypassed_by_runner():
    s=(ROOT/"scripts/t1gr_e5_run_step1.py").read_text()
    assert "bypass_amp_download_check=bool(expected" in s

def test_wall_clock_watchdog_runner_and_eval():
    assert "wall_clock_watchdog" in (ROOT/"scripts/t1gr_e5_run_step1.py").read_text()
    assert "wall_clock_watchdog" in (ROOT/"scripts/t1gr_e5_eval_step1.py").read_text()

def test_runtime_batch_amp_workers_checked():
    s=(ROOT/"scripts/t1gr_e5_run_step1.py").read_text()
    for code in ("E5_BATCH_AUTO_REDUCED","E5_EFFECTIVE_AMP_DRIFT","E5_EFFECTIVE_WORKERS_DRIFT"):
        assert code in s

def test_frozen_config_sha_pins_present():
    s=(ROOT/"src/multimodal/t1gr_e5_core.py").read_text()
    assert FROZEN_E5_TRAINING_SPEC_SHA256 in s
    assert FROZEN_E5_SECURITY_POLICY_SHA256 in s

def test_operational_public_inputs_fixed():
    names={
      "t1gr_e5_freeze_recipe.py","t1gr_e5_build_rgb_view.py","t1gr_e5_preflight.py",
      "t1gr_e5_run_step1.py","t1gr_e5_eval_step1.py","t1gr_e5_final_audit.py",
    }
    forbidden=("--forensic-public","--taxonomy-public","--e4-freeze-public","--e4-verification-public",
               "--training-spec","--recipe","--view-public","--preflight-public","--formal-run-public",
               "--smoke-report","--e4-freeze","--e4-verify","--view","--preflight","--smoke","--formal","--eval")
    for name in names:
        x=(ROOT/"scripts"/name).read_text()
        for arg in forbidden:
            assert f'ap.add_argument("{arg}"' not in x

def test_user_interrupt_sanitized_all_operational():
    names={
      "t1gr_e5_freeze_recipe.py","t1gr_e5_build_rgb_view.py","t1gr_e5_preflight.py",
      "t1gr_e5_run_step1.py","t1gr_e5_eval_step1.py","t1gr_e5_final_audit.py",
    }
    for name in names:
        x=(ROOT/"scripts"/name).read_text()
        assert "except KeyboardInterrupt" in x and "USER_INTERRUPT" in x

def test_run_and_view_path_bound_in_request():
    assert "run_root_binding" in (ROOT/"scripts/t1gr_e5_run_step1.py").read_text()
    assert "out_root_binding" in (ROOT/"scripts/t1gr_e5_build_rgb_view.py").read_text()

def test_posix_private_umask_present():
    s=(ROOT/"src/multimodal/t1gr_e5_core.py").read_text()
    assert "os.umask(0o077)" in s

def test_environment_pins_ultralytics_source_hashes():
    s=(ROOT/"src/multimodal/t1gr_e5_core.py").read_text()
    assert "ultralytics_source_sha256" in s and "trainer_py" in s and "default_yaml" in s
