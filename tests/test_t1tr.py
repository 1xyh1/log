from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1tr_training_source import (  # noqa: E402
    FORMAL_BATCH, FORMAL_EPOCHS, FORMAL_SEED, U2_RUN_NAME,
    balanced_derangement_map, contrast_label,
    decide_training_source_specificity, epoch_shift,
    schedule_balance, verify_epoch_mapping,
)

IDS = tuple(f"s{i}" for i in range(11))
END = ("val6_zero", "train11_zero", "all17_zero")

def vec(a,b,c):
    return dict(zip(END, [a,b,c]))

def test_protocol_constants():
    assert FORMAL_SEED == 20260812
    assert FORMAL_EPOCHS == 80
    assert FORMAL_BATCH == 4
    assert U2_RUN_NAME == "U2-S_P5_FULL_BALANCED_SHUFFLED_seed20260812"

@pytest.mark.parametrize("epoch,shift", [(0,1),(1,2),(9,10),(10,1),(79,10)])
def test_epoch_shift(epoch, shift):
    assert epoch_shift(epoch, 11) == shift

@pytest.mark.parametrize("epoch", [0,1,2,9,10,37,79])
def test_mapping_is_derangement_bijection(epoch):
    m = balanced_derangement_map(IDS, epoch)
    assert set(m) == set(IDS)
    assert set(m.values()) == set(IDS)
    assert all(m[x] != x for x in IDS)

@pytest.mark.parametrize("epoch", [0,9,10,79])
def test_verify_epoch_mapping_pass(epoch):
    m = balanced_derangement_map(IDS, epoch)
    r = verify_epoch_mapping(IDS, epoch, m)
    assert r["passed"] is True
    assert r["self_matches"] == 0

def test_verify_epoch_mapping_rejects_self():
    m = balanced_derangement_map(IDS, 0)
    m["s0"] = "s0"
    assert verify_epoch_mapping(IDS, 0, m)["passed"] is False

def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        balanced_derangement_map(["a","a","b"], 0)

def test_schedule_balance_80_epochs():
    b = schedule_balance(IDS, 80)
    assert b["passed"] is True
    assert b["expected_each_nonself_pair"] == 8
    assert set(b["shift_counts"]) == set(range(1,11))
    assert all(v == 8 for v in b["shift_counts"].values())
    assert all(v == 8 for v in b["nonself_counts"].values())
    assert all(v == 0 for v in b["self_counts"].values())

def test_schedule_balance_non_multiple_fails():
    assert schedule_balance(IDS, 79)["passed"] is False

@pytest.mark.parametrize(
    "new,base,label",
    [
        (vec(2,2,2), vec(1,1,1), "STABLE_POSITIVE"),
        (vec(0,0,0), vec(1,1,1), "STABLE_NEGATIVE"),
        (vec(1,1,1), vec(1,1,1), "EXACT_TIE"),
        (vec(2,0,2), vec(1,1,1), "MIXED"),
    ],
)
def test_contrast_labels(new, base, label):
    assert contrast_label(new, base)["label"] == label

def test_contrast_keys_must_match():
    with pytest.raises(ValueError):
        contrast_label({"a":1}, {"b":1})

def test_branch_shuffled_outperforms_paired_priority():
    d = decide_training_source_specificity(
        vec(1,1,1), vec(2,2,2), vec(3,3,3)
    )
    assert d["branch"] == "SHUFFLED_TRAINING_OUTPERFORMS_PAIRED"
    assert d["replication_seed_go"] is False

def test_branch_paired_source_specificity():
    d = decide_training_source_specificity(
        vec(1,1,1), vec(3,3,3), vec(2,2,2)
    )
    assert d["branch"] == "PAIRED_TRAINING_SOURCE_SPECIFICITY_SUPPORTED_SINGLE_SEED"
    assert d["replication_seed_go"] is True

def test_branch_generic_regularization():
    d = decide_training_source_specificity(
        vec(1,1,1), vec(3,3,3), vec(2.5,3.5,2.5)
    )
    assert d["branch"] == "GENERIC_TRAINING_REGULARIZATION_SOURCE_IDENTITY_UNPROVEN"
    assert d["replication_seed_go"] is False

def test_branch_shuffled_gain_paired_inconclusive():
    # S is stable positive; P is mixed; Q is mixed -> C cannot preempt D.
    d = decide_training_source_specificity(
        vec(1,1,1), vec(2,0.5,2), vec(2,2,2)
    )
    assert d["branch"] == "SHUFFLED_TRAINING_HAS_GAIN_PAIRED_ADVANTAGE_INCONCLUSIVE"

def test_branch_paired_advantage_inconclusive():
    # P is stable positive; S and Q are mixed -> B/C/D cannot preempt E.
    d = decide_training_source_specificity(
        vec(1,1,1), vec(2,2,2), vec(3,0.5,0.5)
    )
    assert d["branch"] == "PAIRED_TRAINING_ADVANTAGE_INCONCLUSIVE"

def test_branch_treatment_gain_not_stable():
    d = decide_training_source_specificity(
        vec(1,1,1), vec(2,0.5,2), vec(0.5,0.5,0.5)
    )
    assert d["branch"] == "TRAINING_TREATMENT_GAIN_NOT_STABLE"

@pytest.mark.parametrize(
    "path,needles",
    [
        ("docs/step4_t1tr/DESIGN_FREEZE.md", [
            "U0-N", "U1-P", "U2-S", "ZERO", "each of its 10 non-self donors",
            "NO Depth", "NO centering", "G18",
        ]),
        ("scripts/run_t1tr.py", [
            "balanced_derangement_map", "aux_id_map=aux_map",
            "T1TR_INITIAL_IDENTITY_FAIL", "T1TR_OPTIMIZER_MISMATCH",
            "T1TR_RUNTIME_SELF_DONOR", 'group="C1-I"',
        ]),
        ("scripts/eval_t1tr.py", [
            "zeros_like", "T1TR_T0_ZERO_NATIVE_ANCHOR_FAIL",
            "T1TR_T1_ZERO_NUMERIC_ANCHOR_FAIL", "U0-N", "U1-P", "U2-S",
        ]),
        ("scripts/summarize_t1tr.py", [
            "no_arbitrary_ap_margin", '"depth_go": False',
            '"production_go": False', "loo_sensitivity",
        ]),
        ("scripts/verify_t1tr_run.py", [
            "T1TR_U2_FORMAL_RUN_PASS", "shift_balance",
            "results_rows", "last_pt_bytes",
        ]),
    ],
)
def test_source_contracts(path, needles):
    text = (ROOT/path).read_text(encoding="utf-8")
    for needle in needles:
        assert needle in text

@pytest.mark.parametrize(
    "file",
    [
        "scripts/run_t1tr.py",
        "scripts/eval_t1tr.py",
        "scripts/summarize_t1tr.py",
        "scripts/audit_t1tr.py",
        "src/multimodal/t1tr_training_source.py",
    ],
)
def test_no_depth_training_code(file):
    text = (ROOT/file).read_text(encoding="utf-8")
    assert 'group="C2-D"' not in text
    assert 'aux_mode = "depth"' not in text

def test_no_dataset_modification_file_in_bundle():
    # Bundle must not modify the frozen dataset module. On a full checkout the
    # file legitimately exists, so verify against the bundle payload manifest
    # (test-harness environment fix, mirrors A3 FakeGate precedent).
    import json
    val = json.loads(
        (ROOT/"T1TR_IMPLEMENTATION_VALIDATION.json").read_text(encoding="utf-8")
    )
    payload = val["payload"]["files"]
    assert "src/multimodal/trimodal_dataset.py" not in payload

def test_static_audit_package_only(tmp_path):
    out = tmp_path/"audit.json"
    cp = subprocess.run(
        [sys.executable, str(ROOT/"scripts/audit_t1tr.py"),
         "--phase", "static", "--out", str(out)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    obj = json.loads(out.read_text(encoding="utf-8"))
    assert obj["schema"] == "step4-t1tr-pretraining-audit-v1"
    assert obj["phase"] == "static"
    assert obj["all_passed"] is True
    assert obj["failed"] == []

def test_summary_replication_only_one_branch():
    text = (ROOT/"src/multimodal/t1tr_training_source.py").read_text(encoding="utf-8")
    # One assignment to True, for the pre-registered paired-specificity branch only.
    assert text.count("replication = True") == 1

def test_primary_endpoints_are_three_zero_endpoints():
    text = (ROOT/"scripts/summarize_t1tr.py").read_text(encoding="utf-8")
    for x in ("val6_zero", "train11_zero", "all17_zero"):
        assert x in text

def test_eval_refuses_overwrite():
    text = (ROOT/"scripts/eval_t1tr.py").read_text(encoding="utf-8")
    assert "T1TR_REFUSE_OVERWRITE" in text

def test_runner_formal_is_one_arm_only():
    text = (ROOT/"scripts/run_t1tr.py").read_text(encoding="utf-8")
    assert "U2-S" in text
    assert "--treatment" not in text  # no accidental T0/T1 retraining selector

def test_schedule_donor_balance_exact_ordered_pairs():
    b = schedule_balance(IDS, 80)
    assert len(b["nonself_counts"]) == 110
    assert sum(b["nonself_counts"].values()) == 880
