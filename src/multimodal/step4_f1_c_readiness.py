"""F1-C smoke-readiness gate.

This module is deliberately torch-free.  It re-judges the three registered F1-C
smoke arms from their raw artifacts instead of trusting stored ``passed`` flags:

    F1C-C0 / F1C-I-fixed / F1C-I-magsoft

The fourth formal arm (F1C-I-soft, original gate) is a matched formal control and
does not require a separate smoke according to DESIGN_FREEZE.  Formal training is
allowed only when the current runner/audit/design/contract still match the smoke
artifacts used to build a step4-f1-c-smoke-readiness-v1 report.
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

READINESS_SCHEMA = "step4-f1-c-smoke-readiness-v1"
AUDIT_SCHEMA = "step4-f1-c-audit-v2"
MANIFEST_SCHEMA = "step4-f1-c-manifest-v1"
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

AUDIT_TARGETS = {
    "corruption_source_sha256": "src/multimodal/step4_f1_b_corruption.py",
    "runner_source_sha256": "scripts/run_step4_f1_c.py",
    "audit_source_sha256": "scripts/audit_step4_f1_c.py",
    "gate_module_sha256": "src/multimodal/reliability_gate.py",
    "model_source_sha256": "src/multimodal/step4_f1_ir_gate_model.py",
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
    if len(manifests) == 3 and all(m["passed"] for m in manifests.values()):
        for key in (
            "initial_rgb_backbone_sha256", "initial_aux_encoder_sha256",
            "initial_fusion_sha256", "initial_gate_sha256",
            "initial_model_state_sha256",
        ):
            values = {manifests[tag]["manifest"].get(key) for tag in manifests}
            initial_state_equal[key] = len(values) == 1 and None not in values
            if not initial_state_equal[key]:
                errors.append(f"INITIAL_STATE_MISMATCH:{key}")

    try:
        g8 = g8_check(run_dirs, expected_epochs)
    except Exception as exc:  # artifact parser failures are readiness failures
        g8 = {"passed": False, "error": f"{type(exc).__name__}:{exc}"}
    if not g8.get("passed"):
        errors.append("G8_REJUDGE_FAIL")

    if contract_path.exists():
        contract = _read_json(contract_path)
        try:
            g9 = _rejudge_g9(run_dirs, contract, expected_epochs, seed)
        except Exception as exc:
            g9 = {"passed": False, "errors": [f"{type(exc).__name__}:{exc}"]}
    else:
        g9 = {"passed": False, "errors": ["CONTRACT_MISSING"]}
    if not g9["passed"]:
        errors.extend(f"G9:{x}" for x in g9["errors"])

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
) -> dict:
    """Recompute readiness from raw smoke artifacts at formal-run time."""
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

    current = evaluate_smoke_readiness(root, run_dirs, contract_path)
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
