from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multimodal.t1gr_e2e5 import *  # noqa: E402,F403


def test_canonical_ids_order_independent():
    assert canonical_ids_sha(["b", "a"]) == canonical_ids_sha(["a", "b"])


def test_duplicate_groups():
    assert exact_duplicate_groups({"a": "x", "b": "x", "c": "y"}) == [["a", "b"]]


def test_label_exact_five_fields(tmp_path):
    p = tmp_path / "x.txt"; p.write_text("0 0.5 0.5 0.2 0.3 9\n")
    r = parse_yolo_label(p, 2)
    assert any(e["reason"] == "field_count_not_exact" for e in r["errors"])


def test_label_edges_checked(tmp_path):
    p = tmp_path / "x.txt"; p.write_text("0 0.05 0.5 0.2 0.2\n")
    r = parse_yolo_label(p, 2)
    assert any(e["reason"] == "bbox_edges_outside_image" for e in r["errors"])


def test_label_good(tmp_path):
    p = tmp_path / "x.txt"; p.write_text("1 0.5 0.5 0.2 0.2\n")
    assert not parse_yolo_label(p, 2)["errors"]


def test_group_regex_executes():
    ids = ["scene01_f1", "scene02_f2"]
    got = group_map(ids, {x: x for x in ids}, {"type": "regex", "regex": "^(scene\\d+)_", "regex_group": 1}, Path("."))
    assert got == {"scene01_f1": "scene01", "scene02_f2": "scene02"}


def test_unresolved_group_holds():
    with pytest.raises(ValueError, match="GROUP_RULE_UNRESOLVED"):
        group_map(["a"], {"a": "a"}, {"type": None}, Path("."))


def test_group_split_requires_three_groups():
    gs = {"g1": {"n_images": 1, "image_counts": [1], "box_counts": [1]}, "g2": {"n_images": 1, "image_counts": [1], "box_counts": [1]}}
    with pytest.raises(ValueError, match="THREE_GROUPS"):
        group_stratified_split({"g1": ["a"], "g2": ["b"]}, gs, {"train": .6, "dev": .2, "final_holdout": .2}, 1)


def test_group_stratified_split_nonempty_disjoint():
    g2i = {f"g{i}": [f"s{i}"] for i in range(6)}
    gs = {g: {"n_images": 1, "image_counts": [1 if i % 2 == 0 else 0, 1 if i % 2 else 0], "box_counts": [1 if i % 2 == 0 else 0, 1 if i % 2 else 0]} for i, g in enumerate(g2i)}
    s = group_stratified_split(g2i, gs, {"train": .5, "dev": .25, "final_holdout": .25}, 7)
    assert all(s[x] for x in SPLITS)
    assert classify_overlap(s)["passed"]


def test_coverage_audit_fails_missing_class():
    support = {s: {"image_counts": [3, 0], "box_counts": [3, 0], "n_images": 3} for s in SPLITS}
    policy = {"min_image_count_by_split": {s: 1 for s in SPLITS}, "min_box_count_by_split": {s: 1 for s in SPLITS}, "exempt_classes": []}
    r = coverage_audit(support, policy, 2)
    assert not r["passed"] and any(x["class_id"] == 1 for x in r["failures"])


def test_coverage_exemption_requires_reason():
    support = {s: {"image_counts": [1], "box_counts": [1], "n_images": 1} for s in SPLITS}
    policy = {"min_image_count_by_split": {s: 1 for s in SPLITS}, "min_box_count_by_split": {s: 1 for s in SPLITS}, "exempt_classes": [{"class_id": 0, "reason": ""}]}
    with pytest.raises(ValueError, match="REASON_REQUIRED"):
        coverage_audit(support, policy, 1)


def test_cross_split_duplicate_detects_each_kind():
    dup = {"rgb": [["a", "b"]], "ir": [], "depth": [], "triplet": []}
    r = cross_split_duplicate_audit(dup, {"a": "train", "b": "dev"})
    assert r["passed"] is False and r["by_kind"]["rgb"]


def test_public_freeze_has_no_id_fields_in_source():
    t = (ROOT / "scripts/t1gr_freeze_split.py").read_text(encoding="utf-8")
    assert '"contains_any_sample_ids": False' in t
    assert '"frozen_before_training": True' not in t
    assert "PUBLIC_FREEZE_EXPOSES_SAMPLE_IDS" in t


def test_contract_full_hash_has_no_optional_bypass():
    t = (ROOT / "scripts/t1gr_build_contract.py").read_text(encoding="utf-8")
    assert "Formal contract ALWAYS hashes every paired file" in t
    assert "--full-hash" not in t
    assert '"full_hash_mode": True' in t


def test_contract_format_in_gate():
    t = (ROOT / "scripts/t1gr_build_contract.py").read_text(encoding="utf-8")
    assert '"format_valid": not format_failures' in t
    assert '"cross_modal_hw_valid": not spatial_failures' in t
    assert "contract_gate_passed = all(gates.values())" in t


def test_split_proposal_is_private_only():
    t = (ROOT / "scripts/t1gr_propose_split.py").read_text(encoding="utf-8")
    assert "--out-private" in t and "PRIVATE_SPLIT_PROPOSAL_MUST_BE_OUTSIDE_REPO" in t
    assert "proposal_gate_passed" in t and "class_coverage_audit" in t


def test_runner_accepts_view_manifest_not_data():
    t = (ROOT / "scripts/t1gr_run_step1_baseline.py").read_text(encoding="utf-8")
    assert 'add_argument("--view-manifest"' in t
    assert 'add_argument("--data"' not in t
    assert "VIEW_ACTUAL_DEV_IDS_FAIL" in t
    assert "BASE_CHECKPOINT_SHA_DRIFT" in t


def test_evaluator_accepts_view_manifest_not_data():
    t = (ROOT / "scripts/t1gr_eval_step1_baseline.py").read_text(encoding="utf-8")
    assert 'add_argument("--view-manifest"' in t
    assert 'add_argument("--data"' not in t
    assert "ACTUAL_VAL_IDS_NOT_FROZEN_DEV" in t


def test_recipe_requires_optimizer_aug_and_eval_keys():
    t = (ROOT / "scripts/t1gr_build_step1_recipe.py").read_text(encoding="utf-8")
    for needle in ("optimizer", "lr0", "nbs", "warmup_epochs", "mosaic", "close_mosaic", "max_det", "ultralytics_version"):
        assert needle in t


def test_runner_postrun_effective_args_gate():
    t = (ROOT / "scripts/t1gr_run_step1_baseline.py").read_text(encoding="utf-8")
    assert "STEP1_EFFECTIVE_ARGS_PREFLIGHT_MISMATCH" in t
    assert "STEP1_EFFECTIVE_ARGS_POSTRUN_MISMATCH" in t
    assert "effective_args_frozen_keys_match" in t


def test_freeze_sequence_is_future_runner_derived():
    t = (ROOT / "scripts/t1gr_freeze_split.py").read_text(encoding="utf-8")
    r = (ROOT / "scripts/t1gr_run_step1_baseline.py").read_text(encoding="utf-8")
    assert "training_precedes_freeze_claim" in t
    assert "freeze_precedes_training_derived" in r


def test_no_t1gr_training_arm_runner_yet():
    names = {p.name.lower() for p in (ROOT / "scripts").glob("*.py")}
    assert not any("g0" in n or "g1_p" in n or "g2_s" in n for n in names)


def test_synthetic_gate_declares_all_required_cases():
    t = (ROOT / "scripts/t1gr_synthetic_integration_gate.py").read_text(encoding="utf-8")
    for key in ("bad_depth_contract_fails", "formal_full_hash_unconditional", "rare_class_coverage_blocks_split", "forged_holdout_in_view_fails", "checkpoint_same_path_content_change_fails"):
        assert key in t


def test_synthetic_integration_gate_runs(tmp_path):
    out = tmp_path / "gate.json"
    cp = subprocess.run([sys.executable, str(ROOT / "scripts/t1gr_synthetic_integration_gate.py"), "--out", str(out)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert cp.returncode == 0, cp.stdout
    obj = json.loads(out.read_text())
    assert obj["all_passed"] is True
    assert all(obj["cases"].values())
