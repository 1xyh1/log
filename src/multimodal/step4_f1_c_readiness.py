"""F1-C smoke-readiness gate.

This module is deliberately torch-free.  It re-judges the three registered F1-C
smoke arms from their raw artifacts instead of trusting stored ``passed`` flags:

    F1C-C0 / F1C-I-fixed / F1C-I-magsoft

The fourth formal arm (F1C-I-soft, original gate) is a matched formal control and
does not require a separate smoke according to DESIGN_FREEZE.  Formal training is
allowed only when the current runner/audit/design/contract still match the smoke
artifacts used to build a step4-f1-c-smoke-readiness-v2 report.

v2 (reviewer 2026-08-17 P0): the external runtime dependency closure is part of
the freshness set — base checkpoint file SHA (EXPECTED_BASE_CHECKPOINT_SHA256),
the builder module (early_fusion_yolo26.py, pinned in AUDIT/MANIFEST tables),
a re-hash of the 17x4 raw data files against contract["file_hashes"], and the
dataset.yaml semantics (nc=12 + names == CLASS_NAMES).  Formal re-checks these
plus a bitwise initial-state equality after model construction.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from multimodal.step4_closeout import g8_check
from multimodal.step4_f1_b_corruption import sample_schedule, schedule_sha256

READINESS_SCHEMA = "step4-f1-c-smoke-readiness-v2"
AUDIT_SCHEMA = "step4-f1-c-audit-v3"
MANIFEST_SCHEMA = "step4-f1-c-manifest-v2"
FP32_SCHEMA = "step4-f1-c-fp32-rgb-v1"

SMOKE_SPECS = {
    "C0": {
        "group": "F1C-C0", "aux_mode": "zero", "gate_mode": "learned",
        "gate_module": "magnitude", "dataset_group": "C0-N",
    },
    "FIXED": {
        "group": "F1C-I-fixed", "aux_mode": "ir", "gate_mode": "fixed_one",
        "gate_module": "magnitude", "dataset_group": "C1-I",
    },
    "MAGSOFT": {
        "group": "F1C-I-magsoft", "aux_mode": "ir", "gate_mode": "learned",
        "gate_module": "magnitude", "dataset_group": "C1-I",
    },
}
APPROVED_FORMAL_GROUPS = (
    "F1C-C0", "F1C-I-fixed", "F1C-I-magsoft", "F1C-I-soft",
)

EXPECTED_BASE_CHECKPOINT_SHA256 = (
    "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
)
# F1-C formal/smoke 构模的外部 RGB anchor 权重 (E:/odin/yolo26s.pt)。
# 与 reports/checkpoint_audit.md 记录及审阅者上传文件一致;本机实测
# (mtime 2026-08-06, 20,422,725 字节)逐字节相同。任何换基座权重必须
# 改此常量并完整重跑 audit -> smoke -> readiness 链 (reviewer 2026-08-17)。

RAW_KIND_DIRS = {"visible": "visible", "infrared": "infrared", "depth": "depth"}
RAW_KINDS = ("visible", "infrared", "depth", "label")

AUDIT_TARGETS = {
    "corruption_source_sha256": "src/multimodal/step4_f1_b_corruption.py",
    "runner_source_sha256": "scripts/run_step4_f1_c.py",
    "audit_source_sha256": "scripts/audit_step4_f1_c.py",
    "gate_module_sha256": "src/multimodal/reliability_gate.py",
    "model_source_sha256": "src/multimodal/step4_f1_ir_gate_model.py",
    "builder_source_sha256": "src/multimodal/early_fusion_yolo26.py",
    "f1c_design_freeze_sha256": "docs/step4_f1_c/DESIGN_FREEZE.md",
    "a1_v2_last_sha256": "reports/step4_f1_c_agreement/descriptor_audit_v2_last.json",
    "a1_v2_best_sha256": "reports/step4_f1_c_agreement/descriptor_audit_v2_best.json",
    "b1_v22_summary_sha256": "runs/step4_f1_b_corruption/_summary_step4_f1_b.json",
}

MANIFEST_PIN_TARGETS = {
    "corruption_source_sha256": "src/multimodal/step4_f1_b_corruption.py",
    "reliability_gate_source_sha256": "src/multimodal/reliability_gate.py",
    "runner_source_sha256": "scripts/run_step4_f1_c.py",
    "model_source_sha256": "src/multimodal/step4_f1_ir_gate_model.py",
    "gate_source_sha256": "src/multimodal/reliability_gate.py",
    "builder_source_sha256": "src/multimodal/early_fusion_yolo26.py",
    "f0_model_source_sha256": "src/multimodal/step4_f0_model.py",
    "aux_encoder_source_sha256": "src/multimodal/aux_encoder.py",
    "feature_fusion_source_sha256": "src/multimodal/feature_fusion.py",
    "trainability_source_sha256": "src/multimodal/trainability.py",
    "dataset_source_sha256": "src/multimodal/trimodal_dataset.py",
    "preprocess_source_sha256": "src/multimodal/modality_preprocess.py",
    "quality_mask_source_sha256": "src/multimodal/modality_quality.py",
    "design_freeze_sha256": "docs/step4_f1_c/DESIGN_FREEZE.md",
    "f1_v4_summary_sha256": "runs/step4_f1_ir_gate/_summary_step4_f1.json",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(
        obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def class_names_sha256(names: dict) -> str:
    """规范化的类名 SHA:{str(k): v} 按 str(k) 排序后再 sha256_json。
    runner 与 readiness 共用,杜绝两端 int/str 键规范化漂移
    (json.dumps sort_keys 对混合键会 TypeError,必须先统一 str 键)。"""
    canonical = {
        str(k): v for k, v in sorted(names.items(), key=lambda kv: str(kv[0]))
    }
    return sha256_json(canonical)


def verify_base_checkpoint(weights_path: Path, expected_sha: str) -> dict:
    """base checkpoint 存在性 + SHA 复核(纯文件级,torch-free)。

    返回 {"passed", "sha256", "expected_sha256", "errors"}。
    errors 元素: BASE_CHECKPOINT_MISSING / BASE_CHECKPOINT_STALE。
    必须在 build_reference_3ch() 之前执行——权重缺失时给出明确错误码
    而不是 torch.load 的神秘报错。"""
    errors: list[str] = []
    path = Path(weights_path)
    if not path.exists():
        return {
            "passed": False, "sha256": None,
            "expected_sha256": expected_sha, "errors": ["BASE_CHECKPOINT_MISSING"],
        }
    actual = sha256_file(path)
    if actual != expected_sha:
        errors.append("BASE_CHECKPOINT_STALE")
    return {
        "passed": not errors, "sha256": actual,
        "expected_sha256": expected_sha, "errors": errors,
    }


def verify_raw_data_freshness(contract: dict) -> dict:
    """按 contract["file_hashes"] 重新 hash 磁盘原始文件并比对。

    目录映射:visible/infrared/depth 在 _raw_dir 下,label 在 _labels_dir 下
    (labels 带 s)。file_hashes 之外的磁盘文件(如被排除组)不校验。
    返回 {"passed", "errors", "checked", "expected_total", "mismatches"}。
    errors 元素:
      RAW_DATA_CONTRACT:{sid}:{kind} / RAW_DATA_CONTRACT:COUNT / RAW_DATA_CONTRACT:ID:{sid}
      RAW_DATA_MISSING:{sid}:{kind} / RAW_DATA_STALE:{sid}:{kind}"""
    errors: list[str] = []
    raw_dir = Path(contract["_raw_dir"])
    labels_dir = Path(contract["_labels_dir"])
    file_hashes = contract.get("file_hashes") or {}
    all_ids = set(contract.get("all17_ids", []))
    split_ids = set(contract.get("train_ids", [])) | set(contract.get("val_ids", []))
    if len(file_hashes) != len(all_ids):
        errors.append(f"RAW_DATA_CONTRACT:COUNT:{len(file_hashes)}!={len(all_ids)}")
    for sid in sorted(split_ids):
        if sid not in file_hashes:
            errors.append(f"RAW_DATA_CONTRACT:ID:{sid}")
    checked = {"visible": 0, "infrared": 0, "depth": 0, "label": 0}
    mismatches: list[dict] = []
    for sid in sorted(file_hashes):
        for kind in RAW_KINDS:
            entry = file_hashes[sid].get(kind)
            if entry is None or not isinstance(entry, dict):
                errors.append(f"RAW_DATA_CONTRACT:{sid}:{kind}")
                continue
            if kind == "label":
                path = labels_dir / entry["file"]
            else:
                path = raw_dir / RAW_KIND_DIRS[kind] / entry["file"]
            checked[kind] += 1
            if not path.exists():
                errors.append(f"RAW_DATA_MISSING:{sid}:{kind}")
                continue
            actual = sha256_file(path)
            if actual != entry["sha256"]:
                mismatches.append({
                    "sample_id": sid, "kind": kind,
                    "expected": entry["sha256"], "actual": actual,
                })
                errors.append(f"RAW_DATA_STALE:{sid}:{kind}")
    return {
        "passed": not errors,
        "errors": errors,
        "checked": checked,
        "expected_total": len(file_hashes) * len(RAW_KINDS),
        "mismatches": mismatches,
    }


def verify_data_yaml(data_yaml_path: Path, class_names: dict) -> dict:
    """dataset.yaml 存在性 + SHA + 语义锁(names 数量==12 且逐项==class_names)。

    返回 {"passed", "errors", "sha256", "names_sha256", "n_classes",
          "names_matches_class_names"}。
    errors 元素: DATA_YAML_MISSING / DATA_YAML_SEMANTICS。
    yaml 解析用函数内 import(同 _read_yaml 风格)。"""
    errors: list[str] = []
    path = Path(data_yaml_path)
    if not path.exists():
        return {
            "passed": False, "errors": ["DATA_YAML_MISSING"],
            "sha256": None, "names_sha256": None, "n_classes": None,
            "names_matches_class_names": False,
        }
    obj = _read_yaml(path)
    names = obj.get("names") or {}
    expected = {str(k): v for k, v in class_names.items()}
    actual = {str(k): v for k, v in names.items()}
    # 语义锁:数量与内容都等于 class_names(runner 传 CLASS_NAMES 即 nc=12)
    semantic_ok = len(actual) == len(class_names) and actual == expected
    if not semantic_ok:
        errors.append("DATA_YAML_SEMANTICS")
    return {
        "passed": not errors,
        "errors": errors,
        "sha256": sha256_file(path),
        "names_sha256": class_names_sha256(names),
        "n_classes": len(actual),
        "names_matches_class_names": semantic_ok,
    }


def check_initial_state_equality(actual_shas: dict, expected_shas: dict) -> dict:
    """formal 构模后 5 个 initial SHA 与 readiness 冻结值比对(纯 dict,torch-free)。

    返回 {"passed", "mismatches"};mismatches[key] = {expected, actual}。
    至少 full model state SHA 必须逐位相同;实现比对全部 5 个分量。"""
    mismatches: dict[str, dict] = {}
    for key in (
        "initial_rgb_backbone_sha256", "initial_aux_encoder_sha256",
        "initial_fusion_sha256", "initial_gate_sha256",
        "initial_model_state_sha256",
    ):
        expected = expected_shas.get(key)
        actual = actual_shas.get(key)
        if actual != expected:
            mismatches[key] = {"expected": expected, "actual": actual}
    return {"passed": not mismatches, "mismatches": mismatches}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_yaml(path: Path) -> dict:
    import yaml
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _record_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _audit_fresh(root: Path, audit_path: Path) -> dict:
    errors: list[str] = []
    if not audit_path.exists():
        return {"passed": False, "errors": ["AUDIT_MISSING"]}
    audit = _read_json(audit_path)
    _record_error(errors, audit.get("schema") == AUDIT_SCHEMA, "AUDIT_SCHEMA")
    _record_error(errors, audit.get("all_passed") is True, "AUDIT_NOT_PASSED")
    prov = audit.get("provenance") or {}
    current = {}
    for key, rel in AUDIT_TARGETS.items():
        path = root / rel
        cur = sha256_file(path) if path.exists() else None
        current[key] = cur
        if prov.get(key) != cur:
            errors.append(f"AUDIT_STALE:{key}")
    return {
        "passed": not errors,
        "errors": errors,
        "audit_sha256": sha256_file(audit_path),
        "current_pins": current,
    }


def _args_results_check(run_dir: Path, expected_epochs: int, batch: int, seed: int) -> dict:
    errors: list[str] = []
    args_path = run_dir / "args.yaml"
    results_path = run_dir / "results.csv"
    if not args_path.exists():
        errors.append("ARGS_MISSING")
        args = {}
    else:
        args = _read_yaml(args_path)
    expected = {
        "epochs": expected_epochs,
        "batch": batch,
        "seed": seed,
        "amp": False,
        "deterministic": True,
        "optimizer": "MuSGD",
    }
    for key, value in expected.items():
        if args.get(key) != value:
            errors.append(f"ARGS_MISMATCH:{key}:{args.get(key)!r}!={value!r}")
    rows = []
    if not results_path.exists():
        errors.append("RESULTS_MISSING")
    else:
        with results_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected_epochs:
            errors.append(f"RESULTS_ROW_COUNT:{len(rows)}!={expected_epochs}")
    return {"passed": not errors, "errors": errors, "results_rows": len(rows)}


def _manifest_check(
    root: Path,
    run_dir: Path,
    tag: str,
    audit_path: Path,
    contract_path: Path,
    expected_epochs: int,
    batch: int,
    seed: int,
    data_yaml_state: dict | None = None,
) -> dict:
    errors: list[str] = []
    path = run_dir / "manifest.json"
    if not path.exists():
        return {"passed": False, "errors": ["MANIFEST_MISSING"]}
    manifest = _read_json(path)
    spec = SMOKE_SPECS[tag]
    expected = {
        "schema": MANIFEST_SCHEMA,
        "group": spec["group"],
        "physical_run_name": run_dir.name,
        "run_kind": "smoke",
        "aux_mode": spec["aux_mode"],
        "gate_mode": spec["gate_mode"],
        "gate_module": spec["gate_module"],
        "gate_module_kind_from_model": spec["gate_module"],
        "dataset_group": spec["dataset_group"],
        "expected_epochs": expected_epochs,
        "requested_batch": batch,
        "seed": seed,
        "g8_evidence": "actual_dataloader_yield_v1",
        "g9_evidence": "actual_corruption_yield_v1",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            errors.append(
                f"MANIFEST_IDENTITY:{key}:{manifest.get(key)!r}!={value!r}"
            )
    if manifest.get("pretrain_audit_sha256") != sha256_file(audit_path):
        errors.append("MANIFEST_STALE:pretrain_audit_sha256")
    if manifest.get("contract_sha256") != sha256_file(contract_path):
        errors.append("MANIFEST_STALE:contract_sha256")
    if data_yaml_state is not None:
        # dataset.yaml 在 ROOT 外(绝对路径),不进 MANIFEST_PIN_TARGETS;
        # 其 SHA/语义由调用方重算后在此比对(v2 闭包,reviewer 2026-08-17)。
        for field, expected in (
            ("data_yaml_sha256", data_yaml_state.get("sha256")),
            ("data_yaml_names_sha256", data_yaml_state.get("names_sha256")),
            ("data_yaml_n_classes", data_yaml_state.get("n_classes")),
        ):
            if manifest.get(field) != expected:
                errors.append(f"MANIFEST_STALE:{field}")
    for field, rel in MANIFEST_PIN_TARGETS.items():
        target = root / rel
        current = sha256_file(target) if target.exists() else None
        if manifest.get(field) != current:
            errors.append(f"MANIFEST_STALE:{field}")
    return {
        "passed": not errors,
        "errors": errors,
        "manifest": manifest,
        "manifest_sha256": sha256_file(path),
    }


def _rejudge_g6(tag: str, run_dir: Path, expected_epochs: int) -> dict:
    errors: list[str] = []
    path = run_dir / "step4_update_gate.json"
    if not path.exists():
        return {"passed": False, "errors": ["G6_MISSING"]}
    gate = _read_json(path)
    rgb_ok = gate.get("rgb_backbone_unchanged") is True
    aux = float(gate.get("aux_encoder_global_rel_l2", float("nan")))
    proj = [float(v) for v in gate.get("proj_weight_norms", [])]
    q = gate.get("last_epoch_effective_q") or {}
    q_values = [q.get("mean"), q.get("min"), q.get("max")]
    q_ok = (
        int(q.get("count", 0)) > 0
        and all(v is not None and math.isfinite(float(v)) for v in q_values)
        and 0.0 <= float(q["min"]) <= float(q["max"]) <= 1.0
    )
    threshold = 1e-3 * (expected_epochs / 80.0)
    _record_error(errors, rgb_ok, "G6_RGB_CHANGED")
    _record_error(errors, len(proj) == 3, "G6_PROJ_COUNT")
    _record_error(errors, q_ok, "G6_Q_INVALID")
    if tag == "C0":
        _record_error(errors, math.isfinite(aux) and aux < threshold, "G6_C0_AUX")
        _record_error(errors, len(proj) == 3 and max(proj) == 0.0, "G6_C0_PROJ")
    elif tag == "FIXED":
        _record_error(errors, math.isfinite(aux) and aux > threshold, "G6_FIXED_AUX")
        _record_error(errors, len(proj) == 3 and min(proj) > 0.0, "G6_FIXED_PROJ")
        _record_error(
            errors,
            q.get("min") == 1.0 and q.get("max") == 1.0,
            "G6_FIXED_Q_NOT_ONE",
        )
    else:
        _record_error(errors, math.isfinite(aux) and aux > threshold, "G6_MAG_AUX")
        _record_error(errors, len(proj) == 3 and min(proj) > 0.0, "G6_MAG_PROJ")
        _record_error(
            errors,
            float(gate.get("gate_max_abs_change", 0.0)) > 0.0,
            "G6_MAG_GATE_NOT_MOVED",
        )
    return {
        "passed": not errors,
        "errors": errors,
        "threshold": threshold,
        "aux_encoder_global_rel_l2": aux,
        "proj_weight_norms": proj,
        "q": q,
        "stored_passed_ignored": gate.get("passed"),
    }


def _verify_fp32(tag: str, run_dir: Path, manifest: dict) -> dict:
    errors: list[str] = []
    path = run_dir / "step4_fp32_rgb_sha.json"
    if not path.exists():
        return {"passed": False, "errors": ["G10_7_MISSING"]}
    row = _read_json(path)
    expected_sha = manifest.get("initial_rgb_backbone_sha256")
    checks = {
        "schema": row.get("schema") == FP32_SCHEMA,
        "group": row.get("group") == SMOKE_SPECS[tag]["group"],
        "expected_matches_manifest": row.get("expected_initial_sha256") == expected_sha,
        "actual_matches_manifest": row.get("actual_final_sha256") == expected_sha,
        "match_flag": row.get("match") is True,
    }
    for key, ok in checks.items():
        if not ok:
            errors.append(f"G10_7:{key}")
    return {"passed": not errors, "errors": errors, "checks": checks}


def _rejudge_g9(
    run_dirs: dict[str, Path],
    contract: dict,
    expected_epochs: int,
    seed: int,
) -> dict:
    errors: list[str] = []
    train_ids = [str(x) for x in contract["train_ids"]]
    cross_expected: dict[int, set[str]] = {}
    for tag, run_dir in run_dirs.items():
        trace_path = run_dir / "step4_f1c_g9_trace.jsonl"
        records_path = run_dir / "step4_f1c_g9_records.jsonl"
        if not trace_path.exists() or not records_path.exists():
            errors.append(f"G9_MISSING:{tag}")
            continue
        trace = _read_jsonl(trace_path)
        records = _read_jsonl(records_path)
        if len(trace) != expected_epochs:
            errors.append(f"G9_TRACE_COUNT:{tag}:{len(trace)}")
        by_epoch: dict[int, list[dict]] = {}
        for row in records:
            by_epoch.setdefault(int(row["epoch"]), []).append(row)
        if set(by_epoch) != set(range(expected_epochs)):
            errors.append(f"G9_EPOCH_SET:{tag}")
        for epoch in range(expected_epochs):
            if epoch >= len(trace):
                continue
            recs = by_epoch.get(epoch, [])
            row = trace[epoch]
            ids = [str(r["sample_id"]) for r in recs]
            if len(recs) != len(train_ids):
                errors.append(f"G9_RECORD_COUNT:{tag}:e{epoch}")
            if sorted(ids) != sorted(train_ids) or len(set(ids)) != len(ids):
                errors.append(f"G9_ID_SET:{tag}:e{epoch}")
            canonical = [
                {
                    k: r[k]
                    for k in (
                        "epoch", "sample_id", "kind", "severity",
                        "ir_sha_before", "ir_sha_after", "rgb_unchanged",
                        "depth_unchanged", "labels_bboxes_same_object",
                    )
                }
                for r in recs
            ]
            if row.get("records_sha256") != sha256_json(
                sorted(canonical, key=lambda x: x["sample_id"])
            ):
                errors.append(f"G9_RECORDS_SHA:{tag}:e{epoch}")
            actual_sched = [
                {
                    "sample_id": str(r["sample_id"]),
                    "kind": r["kind"],
                    "severity": r["severity"],
                }
                for r in recs
            ]
            actual_sha = sha256_json(
                sorted(actual_sched, key=lambda x: x["sample_id"])
            )
            if row.get("actual_schedule_sha256") != actual_sha:
                errors.append(f"G9_ACTUAL_SHA:{tag}:e{epoch}")
            expected_sha = schedule_sha256(seed, epoch, train_ids)
            cross_expected.setdefault(epoch, set()).add(expected_sha)
            if row.get("expected_schedule_sha256") != expected_sha:
                errors.append(f"G9_EXPECTED_SHA:{tag}:e{epoch}")
            if row.get("expected_matches_actual") is not True or expected_sha != actual_sha:
                errors.append(f"G9_EXPECTED_ACTUAL:{tag}:e{epoch}")
            counts: dict[str, int] = {}
            for r in recs:
                sid = str(r["sample_id"])
                expected = sample_schedule(seed, epoch, sid)
                if r["kind"] != expected["kind"] or r["severity"] != expected["severity"]:
                    errors.append(f"G9_SCHEDULE:{tag}:e{epoch}:{sid}")
                if not (
                    r["rgb_unchanged"]
                    and r["depth_unchanged"]
                    and r["labels_bboxes_same_object"]
                ):
                    errors.append(f"G9_PROTECTED_CHANNEL:{tag}:e{epoch}:{sid}")
                ir_same = r["ir_sha_before"] == r["ir_sha_after"]
                if tag == "C0":
                    if not ir_same:
                        errors.append(f"G9_C0_IR_CHANGED:{tag}:e{epoch}:{sid}")
                elif ir_same != (r["kind"] == "clean"):
                    errors.append(f"G9_IR_SEMANTICS:{tag}:e{epoch}:{sid}")
                counts[r["kind"]] = counts.get(r["kind"], 0) + 1
            if row.get("kind_counts") != counts:
                errors.append(f"G9_KIND_COUNTS:{tag}:e{epoch}")
            if row.get("rgb_depth_labels_bboxes_unchanged") is not True:
                errors.append(f"G9_SUMMARY_PROTECTED:{tag}:e{epoch}")
            if row.get("ir_changed_for_corrupted_only") is not True:
                errors.append(f"G9_SUMMARY_IR:{tag}:e{epoch}")
    for epoch, values in cross_expected.items():
        if len(values) != 1:
            errors.append(f"G9_CROSS_GROUP_SCHEDULE:e{epoch}")
    return {"passed": not errors, "errors": errors}


def _artifact_provenance(root: Path, run_dirs: dict[str, Path], contract_path: Path) -> dict:
    paths: dict[str, Path] = {
        "readiness_module_sha256": root / "src/multimodal/step4_f1_c_readiness.py",
        "readiness_script_sha256": root / "scripts/audit_step4_f1_c_smoke_readiness.py",
        "runner_source_sha256": root / "scripts/run_step4_f1_c.py",
        "audit_source_sha256": root / "scripts/audit_step4_f1_c.py",
        "audit_report_sha256": root / "reports/step4_f1_c/pretrain_audit.json",
        "model_source_sha256": root / "src/multimodal/step4_f1_ir_gate_model.py",
        "gate_source_sha256": root / "src/multimodal/reliability_gate.py",
        "builder_source_sha256": root / "src/multimodal/early_fusion_yolo26.py",
        "corruption_source_sha256": root / "src/multimodal/step4_f1_b_corruption.py",
        "design_freeze_sha256": root / "docs/step4_f1_c/DESIGN_FREEZE.md",
        "contract_sha256": contract_path,
    }
    for tag, rd in run_dirs.items():
        for name in (
            "manifest.json", "args.yaml", "results.csv", "step4_update_gate.json",
            "step4_g8_trace.jsonl", "step4_f1c_g9_trace.jsonl",
            "step4_f1c_g9_records.jsonl", "step4_growth.jsonl",
            "step4_fp32_rgb_sha.json",
        ):
            paths[f"{tag}_{name.replace('.', '_')}_sha256"] = rd / name
    return {
        key: sha256_file(path) if path.exists() else None
        for key, path in paths.items()
    }


def evaluate_smoke_readiness(
    root: Path,
    smoke_runs: dict[str, Path],
    contract_path: Path,
    *,
    expected_epochs: int = 1,
    batch: int = 4,
    seed: int = 20260812,
    data_yaml_path: Path | None = None,
    base_checkpoint_path: Path | None = None,
    class_names: dict | None = None,
) -> dict:
    root = Path(root).resolve()
    contract_path = Path(contract_path).resolve()
    errors: list[str] = []
    if set(smoke_runs) != set(SMOKE_SPECS):
        return {
            "all_passed": False,
            "errors": ["SMOKE_TAG_SET"],
            "evidence": {},
            "provenance": {},
        }
    run_dirs = {tag: Path(path).resolve() for tag, path in smoke_runs.items()}
    for tag, rd in run_dirs.items():
        try:
            rd.relative_to(root)
        except ValueError:
            errors.append(f"SMOKE_OUTSIDE_REPO:{tag}")
    audit_path = root / "reports/step4_f1_c/pretrain_audit.json"
    audit = _audit_fresh(root, audit_path)
    if not audit["passed"]:
        errors.extend(audit["errors"])

    if contract_path.exists():
        contract = _read_json(contract_path)
    else:
        contract = None
        errors.append("CONTRACT_MISSING")

    data_yaml_state: dict | None = None
    if data_yaml_path is not None and class_names is not None:
        data_yaml_state = verify_data_yaml(data_yaml_path, class_names)
        if not data_yaml_state["passed"]:
            errors.extend(f"DATA_YAML:{x}" for x in data_yaml_state["errors"])

    manifests = {}
    args_results = {}
    g6 = {}
    fp32 = {}
    for tag, rd in run_dirs.items():
        if not rd.exists():
            errors.append(f"RUN_DIR_MISSING:{tag}")
            continue
        manifests[tag] = _manifest_check(
            root, rd, tag, audit_path, contract_path,
            expected_epochs, batch, seed,
            data_yaml_state=data_yaml_state,
        )
        args_results[tag] = _args_results_check(rd, expected_epochs, batch, seed)
        g6[tag] = _rejudge_g6(tag, rd, expected_epochs)
        for block_name, block in (
            ("manifest", manifests[tag]), ("args_results", args_results[tag]),
            ("g6", g6[tag]),
        ):
            if not block["passed"]:
                errors.extend(f"{tag}:{block_name}:{x}" for x in block["errors"])
        if manifests[tag]["passed"]:
            fp32[tag] = _verify_fp32(tag, rd, manifests[tag]["manifest"])
            if not fp32[tag]["passed"]:
                errors.extend(f"{tag}:fp32:{x}" for x in fp32[tag]["errors"])

    initial_state_equal = {}
    initial_state_frozen = {}
    if len(manifests) == 3 and all(m["passed"] for m in manifests.values()):
        first_manifest = manifests["C0"]["manifest"]
        for key in (
            "initial_rgb_backbone_sha256", "initial_aux_encoder_sha256",
            "initial_fusion_sha256", "initial_gate_sha256",
            "initial_model_state_sha256",
        ):
            values = {manifests[tag]["manifest"].get(key) for tag in manifests}
            initial_state_equal[key] = len(values) == 1 and None not in values
            if not initial_state_equal[key]:
                errors.append(f"INITIAL_STATE_MISMATCH:{key}")
        # 冻结三组全等的 initial SHA 供 formal 构模后逐位比对
        # (reviewer 2026-08-17 P0: formal 此刻构造的模型必须 == r3 验证过的)。
        initial_state_frozen = {
            key: first_manifest.get(key) for key in (
                "initial_rgb_backbone_sha256", "initial_aux_encoder_sha256",
                "initial_fusion_sha256", "initial_gate_sha256",
                "initial_model_state_sha256",
            )
        }
        initial_state_frozen["passed"] = all(initial_state_equal.values())

    try:
        g8 = g8_check(run_dirs, expected_epochs)
    except Exception as exc:  # artifact parser failures are readiness failures
        g8 = {"passed": False, "error": f"{type(exc).__name__}:{exc}"}
    if not g8.get("passed"):
        errors.append("G8_REJUDGE_FAIL")

    if contract is not None:
        try:
            g9 = _rejudge_g9(run_dirs, contract, expected_epochs, seed)
        except Exception as exc:
            g9 = {"passed": False, "errors": [f"{type(exc).__name__}:{exc}"]}
    else:
        g9 = {"passed": False, "errors": ["CONTRACT_MISSING"]}
    if not g9["passed"]:
        errors.extend(f"G9:{x}" for x in g9["errors"])

    # ---- external runtime dependency closure (reviewer 2026-08-17 P0) ----
    # ① 原始数据 freshness:重 hash 磁盘 17x4 文件 vs contract["file_hashes"]
    data_state = {"passed": True, "errors": [], "checked": {},
                  "expected_total": 0, "mismatches": []}
    if contract is not None:
        data_state = verify_raw_data_freshness(contract)
        if not data_state["passed"]:
            errors.extend(f"DATA:{x}" for x in data_state["errors"])
    # ② base checkpoint:磁盘当前 SHA == 文档常量;三组 manifest 记录一致
    base_state = {"passed": True, "errors": [], "sha256": None,
                  "expected_sha256": EXPECTED_BASE_CHECKPOINT_SHA256}
    if base_checkpoint_path is not None:
        base_state = verify_base_checkpoint(
            base_checkpoint_path, EXPECTED_BASE_CHECKPOINT_SHA256)
        if not base_state["passed"]:
            errors.extend(f"BASE:{x}" for x in base_state["errors"])
    recorded_bc = {
        m.get("base_checkpoint_sha256")
        for m in (manifests[tag]["manifest"] for tag in manifests
                  if manifests[tag]["passed"])
    }
    if len(recorded_bc) != 1 or next(iter(recorded_bc)) is None:
        errors.append("BASE_CHECKPOINT_MISMATCH")
    elif next(iter(recorded_bc)) != EXPECTED_BASE_CHECKPOINT_SHA256:
        errors.append("BASE_CHECKPOINT_DOCUMENTED_MISMATCH")

    versions = {}
    for tag, block in manifests.items():
        if block.get("passed"):
            m = block["manifest"]
            versions[tag] = {
                "torch": m.get("torch_version"),
                "ultralytics": m.get("ultralytics_version"),
            }
    if len(versions) == 3 and len({json.dumps(v, sort_keys=True) for v in versions.values()}) != 1:
        errors.append("SMOKE_ENV_VERSION_MISMATCH")

    evidence = {
        "audit": audit,
        "manifests": {
            tag: {
                "passed": block.get("passed", False),
                "errors": block.get("errors", []),
            }
            for tag, block in manifests.items()
        },
        "args_results": args_results,
        "g6_rejudged": g6,
        "g8_rejudged": g8,
        "g9_rejudged": g9,
        "g10_7_fp32": fp32,
        "initial_state_equal": initial_state_equal,
        "initial_state_frozen": initial_state_frozen,
        "base_checkpoint": {
            "expected_sha256": EXPECTED_BASE_CHECKPOINT_SHA256,
            "smoke_recorded": (
                next(iter(recorded_bc)) if len(recorded_bc) == 1 else None
            ),
            "current_sha256": base_state.get("sha256"),
            "current_matches_documented": bool(
                base_state.get("sha256") == EXPECTED_BASE_CHECKPOINT_SHA256
            ),
            "passed": bool(
                base_state["passed"]
                and len(recorded_bc) == 1
                and next(iter(recorded_bc)) == EXPECTED_BASE_CHECKPOINT_SHA256
            ),
        },
        "data_freshness": {
            "checked": data_state.get("checked"),
            "expected_total": data_state.get("expected_total"),
            "mismatches": data_state.get("mismatches"),
            "passed": data_state["passed"],
        },
        "data_yaml": (
            None if data_yaml_state is None else {
                "path": str(Path(data_yaml_path).resolve()),
                "sha256": data_yaml_state.get("sha256"),
                "names_sha256": data_yaml_state.get("names_sha256"),
                "n_classes": data_yaml_state.get("n_classes"),
                "names_matches_class_names": data_yaml_state.get(
                    "names_matches_class_names"),
                "passed": data_yaml_state["passed"],
            }
        ),
        "versions": versions,
        "formal_protocol": {
            "epochs": 80, "batch": 4, "seed": 20260812, "amp": False,
            "smoke_groups": [SMOKE_SPECS[x]["group"] for x in SMOKE_SPECS],
            "formal_groups": list(APPROVED_FORMAL_GROUPS),
            "original_soft_smoke_required": False,
        },
    }
    provenance = _artifact_provenance(root, run_dirs, contract_path)
    return {
        "all_passed": not errors,
        "errors": errors,
        "evidence": evidence,
        "evidence_sha256": sha256_json(evidence),
        "provenance": provenance,
    }


def verify_readiness_report(
    root: Path,
    report_path: Path,
    contract_path: Path,
    *,
    requested_group: str,
    data_yaml_path: Path,
    base_checkpoint_path: Path,
    class_names: dict,
) -> dict:
    """Recompute readiness from raw smoke artifacts at formal-run time.

    data_yaml_path / base_checkpoint_path / class_names 必传 (v2 closure):
    重算 evaluate_smoke_readiness 必须使用与报告生成时相同的语义输入,
    否则 evidence_sha256 / provenance 整体比对会失败 (READINESS_EVIDENCE_STALE)。"""
    root = Path(root).resolve()
    report_path = Path(report_path).resolve()
    errors: list[str] = []
    if not report_path.exists():
        return {"passed": False, "errors": ["READINESS_REPORT_MISSING"]}
    report = _read_json(report_path)
    if report.get("schema") != READINESS_SCHEMA:
        errors.append("READINESS_SCHEMA")
    if report.get("all_passed") is not True:
        errors.append("READINESS_RECORDED_NOT_PASSED")
    if requested_group not in report.get("approved_formal_groups", []):
        errors.append(f"FORMAL_GROUP_NOT_APPROVED:{requested_group}")
    if report.get("original_soft_smoke_required") is not False:
        errors.append("ORIGINAL_SOFT_SMOKE_POLICY_DRIFT")

    raw_runs = report.get("smoke_runs")
    if not isinstance(raw_runs, dict) or set(raw_runs) != set(SMOKE_SPECS):
        errors.append("READINESS_SMOKE_RUNS_INVALID")
        return {"passed": False, "errors": errors}
    run_dirs = {}
    for tag, rel in raw_runs.items():
        path = (root / rel).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(f"READINESS_PATH_ESCAPE:{tag}")
        run_dirs[tag] = path

    current = evaluate_smoke_readiness(
        root, run_dirs, contract_path,
        data_yaml_path=data_yaml_path,
        base_checkpoint_path=base_checkpoint_path,
        class_names=class_names,
    )
    if not current["all_passed"]:
        errors.extend(f"CURRENT:{x}" for x in current["errors"])
    if report.get("evidence_sha256") != current.get("evidence_sha256"):
        errors.append("READINESS_EVIDENCE_STALE")
    if report.get("provenance") != current.get("provenance"):
        errors.append("READINESS_PROVENANCE_STALE")
    return {
        "passed": not errors,
        "errors": errors,
        "evidence": current.get("evidence"),
        "evidence_sha256": current.get("evidence_sha256"),
        "provenance": current.get("provenance"),
    }
