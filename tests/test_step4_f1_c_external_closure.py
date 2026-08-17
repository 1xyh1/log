"""F1-C external runtime dependency closure (reviewer 2026-08-17 P0).

对抗测试:base checkpoint / 原始数据 / dataset.yaml 的篡改必须被校验函数
拒绝。全部走 tmp_path 与可注入路径参数,绝不触碰真实的
E:/odin/yolo26s.pt 与 D:/pycharm/.../sample_multimodal 数据。

模板参照 tests/test_step3_recovery_contract.py:
  写假产物 -> 记录 before hash -> 篡改 -> 断言失败串。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.step4_f1_c_readiness import (  # noqa: E402
    EXPECTED_BASE_CHECKPOINT_SHA256,
    AUDIT_TARGETS,
    MANIFEST_PIN_TARGETS,
    check_initial_state_equality,
    class_names_sha256,
    sha256_file,
    verify_base_checkpoint,
    verify_data_yaml,
    verify_raw_data_freshness,
)

DOCUMENTED_BASE_SHA = (
    "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b")


def _fake_contract(root: Path, sids=("000001", "000002")) -> dict:
    """在 tmp_path 下构造最小但结构完整的 contract + 数据目录。

    file_hashes 用真实写入的文件回填,all17_ids == train_ids+val_ids,
    无被排除组(与真实 contract 的 00000008 情形不同,保持最小)。"""
    raw = root / "sample_multimodal"
    labels = raw / "labels"
    for sub in ("visible", "infrared", "depth"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    for i, sid in enumerate(sids):
        entry = {}
        for kind, sub in (("visible", "visible"), ("infrared", "infrared"),
                          ("depth", "depth")):
            p = raw / sub / f"{sid}.png"
            p.write_bytes(f"{kind}-{sid}-v{i}".encode())
            entry[kind] = {"file": p.name, "sha256": sha256_file(p)}
        lp = labels / f"{sid}.txt"
        lp.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        entry["label"] = {"file": lp.name, "sha256": sha256_file(lp)}
        file_hashes[sid] = entry
    return {
        "_raw_dir": str(raw),
        "_labels_dir": str(labels),
        "file_hashes": file_hashes,
        "all17_ids": list(sids),
        "train_ids": [sids[0]],
        "val_ids": [sids[1]],
    }


# ---- base checkpoint ----

def test_base_checkpoint_accepts_identical_file(tmp_path):
    w = tmp_path / "yolo26s.pt"
    w.write_bytes(b"weights-v1")
    expected = sha256_file(w)
    r = verify_base_checkpoint(w, expected)
    assert r["passed"] is True
    assert r["sha256"] == expected
    assert r["errors"] == []


def test_base_checkpoint_aborts_on_modified_weights(tmp_path):
    # 对抗测试 1:记录 before -> 篡改 -> expected 不变 -> 必须失败
    w = tmp_path / "yolo26s.pt"
    w.write_bytes(b"weights-v1")
    expected = sha256_file(w)
    w.write_bytes(b"weights-v2-tampered")
    r = verify_base_checkpoint(w, expected)
    assert not r["passed"]
    assert "BASE_CHECKPOINT_STALE" in r["errors"]
    assert r["sha256"] != expected


def test_base_checkpoint_aborts_on_missing_file(tmp_path):
    r = verify_base_checkpoint(tmp_path / "nope.pt", "x" * 64)
    assert not r["passed"]
    assert "BASE_CHECKPOINT_MISSING" in r["errors"]
    assert r["sha256"] is None


# ---- raw data freshness ----

def test_raw_data_freshness_accepts_pristine_contract(tmp_path):
    contract = _fake_contract(tmp_path)
    r = verify_raw_data_freshness(contract)
    assert r["passed"] is True
    assert r["expected_total"] == 8
    assert r["checked"] == {"visible": 2, "infrared": 2, "depth": 2, "label": 2}
    assert r["mismatches"] == []


def test_raw_data_freshness_aborts_on_any_tamper(tmp_path):
    # 对抗测试 2:改一个 visible 文件 -> 必须失败,错误串含 sid + kind
    contract = _fake_contract(tmp_path)
    p = Path(contract["_raw_dir"]) / "visible" / "000001.png"
    p.write_bytes(b"tampered")
    r = verify_raw_data_freshness(contract)
    assert not r["passed"]
    assert any("RAW_DATA_STALE" in e and "000001" in e and "visible" in e
               for e in r["errors"])
    assert r["mismatches"][0]["sample_id"] == "000001"


def test_raw_data_freshness_aborts_on_missing_label(tmp_path):
    contract = _fake_contract(tmp_path)
    (Path(contract["_labels_dir"]) / "000002.txt").unlink()
    r = verify_raw_data_freshness(contract)
    assert any("RAW_DATA_MISSING" in e and "label" in e for e in r["errors"])


def test_raw_data_freshness_aborts_on_contract_structure_drift(tmp_path):
    contract = _fake_contract(tmp_path)
    contract["file_hashes"] = {sid: v for sid, v in
                               contract["file_hashes"].items() if sid != "000001"}
    r = verify_raw_data_freshness(contract)
    assert any("RAW_DATA_CONTRACT" in e for e in r["errors"])


# ---- dataset.yaml ----

def test_data_yaml_semantics_and_sha(tmp_path):
    y = tmp_path / "dataset.yaml"
    y.write_text("train: x\nval: y\nnames:\n  0: person\n  1: boat\n",
                 encoding="utf-8")
    r = verify_data_yaml(y, {0: "person", 1: "boat"})
    assert r["passed"] is True
    assert r["sha256"] == sha256_file(y)
    assert r["n_classes"] == 2
    assert r["names_matches_class_names"] is True


def test_data_yaml_aborts_on_class_names_drift(tmp_path):
    y = tmp_path / "dataset.yaml"
    y.write_text("train: x\nval: y\nnames:\n  0: person\n  1: cat\n",
                 encoding="utf-8")
    r = verify_data_yaml(y, {0: "person", 1: "boat"})
    assert not r["passed"]
    assert "DATA_YAML_SEMANTICS" in r["errors"]


def test_data_yaml_aborts_on_missing_file(tmp_path):
    r = verify_data_yaml(tmp_path / "nope.yaml", {0: "person"})
    assert not r["passed"]
    assert "DATA_YAML_MISSING" in r["errors"]


def test_data_yaml_n_classes_not_twelve_aborts(tmp_path):
    # 语义锁:names 数量必须 == 12(与 CLASS_NAMES 一致)
    y = tmp_path / "dataset.yaml"
    y.write_text("train: x\nval: y\nnames:\n  0: person\n  1: boat\n",
                 encoding="utf-8")
    r = verify_data_yaml(y, {i: f"c{i}" for i in range(12)})
    assert not r["passed"]
    assert "DATA_YAML_SEMANTICS" in r["errors"]


# ---- initial-state equality (pure dict) ----

def test_initial_state_equality_accepts_identical():
    base = {"initial_model_state_sha256": "a" * 64,
            "initial_gate_sha256": "b" * 64}
    r = check_initial_state_equality(dict(base), base)
    assert r["passed"] is True
    assert r["mismatches"] == {}


def test_initial_state_equality_reports_each_mismatch():
    base = {"initial_model_state_sha256": "a" * 64,
            "initial_rgb_backbone_sha256": "b" * 64}
    r = check_initial_state_equality(
        {"initial_model_state_sha256": "c" * 64,
         "initial_rgb_backbone_sha256": "b" * 64},
        base)
    assert not r["passed"]
    assert set(r["mismatches"]) == {"initial_model_state_sha256"}
    assert r["mismatches"]["initial_model_state_sha256"] == {
        "expected": "a" * 64, "actual": "c" * 64}


# ---- class names canonicalization ----

def test_class_names_sha256_canonicalizes_mixed_keys():
    a = class_names_sha256({0: "person", 1: "boat"})
    b = class_names_sha256({"0": "person", "1": "boat"})
    assert a == b


# ---- wiring regression (runner / pin tables) ----

def test_runner_wires_all_external_dependency_aborts():
    src = (ROOT / "scripts" / "run_step4_f1_c.py").read_text(
        encoding="utf-8")
    for code in (
        "ABORT_BASE_CHECKPOINT_STALE",
        "ABORT_BASE_CHECKPOINT_MISSING",
        "ABORT_RAW_DATA_STALE",
        "ABORT_INITIAL_STATE_MISMATCH",
        "ABORT_DATA_YAML_STALE",
        "ABORT_DATA_YAML_SEMANTICS",
        "verify_raw_data_freshness",
        "verify_base_checkpoint",
        "check_initial_state_equality",
    ):
        assert code in src, f"runner missing wiring: {code}"


def test_builder_is_pinned_everywhere():
    assert "builder_source_sha256" in AUDIT_TARGETS
    assert "builder_source_sha256" in MANIFEST_PIN_TARGETS
    assert EXPECTED_BASE_CHECKPOINT_SHA256 == DOCUMENTED_BASE_SHA
