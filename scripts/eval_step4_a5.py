#!/usr/bin/env python3
"""A5 evaluation-only Cross-scale AC Paired Interaction Audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import step3_eval_utils as evu  # noqa: E402
from multimodal.raw_sample_index import CLASS_NAMES, OUT_DEFAULT  # noqa: E402
from multimodal.step4_a3_common import (  # noqa: E402
    build_residual_cache_no_gate,
    forward_with_custom_residuals,
    state_sha256,
    tensor_sha256,
)
from multimodal.step4_a5_context import (  # noqa: E402
    CONTEXT_ORDER,
    CONTEXT_STATES,
    active_scales_for_context,
    build_context,
    validate_context_trace,
    validate_pair_isolation,
)
from multimodal.step4_a5_effects import (  # noqa: E402
    INTERACTION_COEFFICIENTS,
    classify_paired_effect,
    classify_shift,
    context_shifts,
    difference_of_effects,
    effect_from_results,
    interaction_effects,
    linear_effect_contrast,
    mechanism_flags,
    route_decision,
)
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402

SCHEMA = "step4-a5-summary-v1"
RUN_TAGS = ("FIXED", "SOFT")
EXPECTED_VAL_IDS = (
    "000003_013_00000085",
    "000004_013_00000081",
    "000004_014_00000001",
    "000016",
    "000016_001_00000001",
    "000016_042_suppl_00000164",
)
EXPECTED_DONOR_MAP_SHA256 = "c5cd8e852663eae3243bc5e4c263c6f2c26c7b2faa977ae3b60cb5c1ea122af5"
EXPECTED_A2_RESULT_SHA256 = "756093358153c5e203f485dce96e0f2a5e91881fb6c6e4b49c036cbfdc6d1c6b"
EXPECTED_A4_EXECUTION_COMMIT = "36221d2f827c411bddd66350729dfd05a3b48f49"
EXPECTED_A4_REVIEWER_HEAD = "b7ee0d6803d949a8c512b11defcb2125a3f4c8a1"
EXPECTED_A4_SUMMARY_SHA256 = "721198d04b4ce54caec3d0b5c97ef5c665c3c4e8bf44e8df82e4a50a33406781"
EXPECTED_A4_SUMMARY_CANONICAL_LF_SHA256 = "95f768289e2f04010013eeeac20a83d2bf9e71b153c2e748f5a1ab5941c10ea1"
EXPECTED_A4_STANDALONE_SHA256 = "e1d9c95b84af300feea0148bfecde678fec257e6a2ba709cd4e26215265d90e5"
EXPECTED_A4_CONDITIONAL_SHA256 = "c0ce169c33d835c5e5d262f0b05029f8d63a2180acdec277158b2fc4450edcf7"
EXPECTED_A4_ADJUDICATION_SHA256 = "0ca9e6e7f3e8b8d2e0c4a8542bcfff5428139b77a61280357a0ae9f4a282c898"
EXPECTED_A4_FEEDBACK_SHA256 = "3bd2331d3e618f280b6c8a67699a93780aef1806a09c86bdad2b88ece8dd434a"
EXPECTED_A4_DECISION_AFTER_SHA256 = "50650e2b1a3679325a2cd1b7d95eccebfeb9044d801ffb48dc08b83c89ad95a2"
EXPECTED_A4_TESTS_AFTER_SHA256 = "ec9b294464237869b09c788dd1f23ebeeb2194cfb138514173934718e474cbc4"

SOURCE_PATHS = {
    "step3_eval_utils_sha256": "src/multimodal/step3_eval_utils.py",
    "model_source_sha256": "src/multimodal/step4_f1_ir_gate_model.py",
    "gate_source_sha256": "src/multimodal/reliability_gate.py",
    "trimodal_dataset_sha256": "src/multimodal/trimodal_dataset.py",
    "f0_model_source_sha256": "src/multimodal/step4_f0_model.py",
    "aux_encoder_source_sha256": "src/multimodal/aux_encoder.py",
    "feature_fusion_source_sha256": "src/multimodal/feature_fusion.py",
    "trainability_source_sha256": "src/multimodal/trainability.py",
    "causality_interventions_sha256": "src/multimodal/causality_interventions.py",
    "raw_sample_index_sha256": "src/multimodal/raw_sample_index.py",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def commit_json_bundle(payloads: dict[Path, dict]):
    temps: list[tuple[Path, Path]] = []
    try:
        for path, obj in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
            temps.append((tmp, path))
        for tmp, path in temps:
            tmp.replace(path)
    except Exception:
        for tmp, _ in temps:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
        raise


def verify_preexecution_audit(root: Path) -> dict:
    path = root / "reports/step4_a5/preexecution_audit.json"
    if not path.exists():
        raise RuntimeError(f"A5_PREEXECUTION_AUDIT_MISSING:{path}")
    obj = json_load(path)
    if obj.get("schema") != "step4-a5-preexecution-audit-v1" or obj.get("all_passed") is not True:
        raise RuntimeError("A5_PREEXECUTION_AUDIT_NOT_PASSING")
    if obj.get("mode") != "repo-full":
        raise RuntimeError("A5_PREEXECUTION_AUDIT_REQUIRES_REPO_FULL")
    targets = {
        "design_sha256": root / "docs/step4_a5/DESIGN_FREEZE.md",
        "context_sha256": root / "src/multimodal/step4_a5_context.py",
        "effects_sha256": root / "src/multimodal/step4_a5_effects.py",
        "evaluator_sha256": root / "scripts/eval_step4_a5.py",
        "tests_sha256": root / "tests/test_step4_a5.py",
        "audit_source_sha256": root / "scripts/audit_step4_a5.py",
    }
    current = {k: sha256_file(v) for k, v in targets.items()}
    recorded = obj.get("provenance") or {}
    stale = {k: {"recorded": recorded.get(k), "current": v} for k, v in current.items() if recorded.get(k) != v}
    if stale:
        raise RuntimeError(f"A5_PREEXECUTION_AUDIT_STALE:{stale}")
    return {
        "passed": True,
        "mode": "repo-full",
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "source_hashes": current,
    }


def verify_upstream(root: Path, contract_path: Path):
    paths = {
        "a2": root / "reports/step4_a2/scale_ir_residual_causality.json",
        "donor": root / "reports/step4_a2/val_donor_map.json",
        "a4_summary": root / "reports/step4_a4/a4_summary.json",
        "a4_standalone": root / "reports/step4_a4/ac_paired_standalone.json",
        "a4_conditional": root / "reports/step4_a4/ac_paired_conditional.json",
        "adjudication": root / "reports/step4_a4/reviewer_adjudication.json",
        "feedback": root / "docs/step4_a4/feedback/2026-08-19_formal-review.md",
    }
    expected = {
        "a2": EXPECTED_A2_RESULT_SHA256,
        "donor": EXPECTED_DONOR_MAP_SHA256,
        "a4_summary": EXPECTED_A4_SUMMARY_SHA256,
        "a4_standalone": EXPECTED_A4_STANDALONE_SHA256,
        "a4_conditional": EXPECTED_A4_CONDITIONAL_SHA256,
        "adjudication": EXPECTED_A4_ADJUDICATION_SHA256,
        "feedback": EXPECTED_A4_FEEDBACK_SHA256,
    }
    for key, path in paths.items():
        if not path.exists():
            raise RuntimeError(f"A5_UPSTREAM_ADJUDICATION_FAIL:MISSING:{key}:{path}")
        current = sha256_file(path)
        if current != expected[key]:
            raise RuntimeError(f"A5_UPSTREAM_ADJUDICATION_FAIL:SHA:{key}:{current}!={expected[key]}")
    if canonical_lf_sha256(paths["a4_summary"]) != EXPECTED_A4_SUMMARY_CANONICAL_LF_SHA256:
        raise RuntimeError("A5_UPSTREAM_ADJUDICATION_FAIL:A4_SUMMARY_CANONICAL")

    a2 = json_load(paths["a2"])
    donor = json_load(paths["donor"])
    a4 = json_load(paths["a4_summary"])
    standalone = json_load(paths["a4_standalone"])
    conditional = json_load(paths["a4_conditional"])
    adjudication = json_load(paths["adjudication"])

    if a2.get("schema") != "step4-a2-scale-ir-residual-causality-v2" or a2.get("all_gates_passed") is not True:
        raise RuntimeError("A5_UPSTREAM_ADJUDICATION_FAIL:A2")
    if donor != a2.get("donor_map"):
        raise RuntimeError("A5_DONOR_MAP_DRIFT:A2_CONTENT")
    if tuple(a2.get("protocol", {}).get("val_ids") or ()) != EXPECTED_VAL_IDS:
        raise RuntimeError("A5_UPSTREAM_ADJUDICATION_FAIL:A2_VAL_IDS")
    if a4.get("schema") != "step4-a4-summary-v1" or a4.get("all_gates_passed") is not True:
        raise RuntimeError("A5_UPSTREAM_ADJUDICATION_FAIL:A4_SUMMARY")
    if not a4.get("gates") or not all(bool(v) for v in a4["gates"].values()):
        raise RuntimeError("A5_UPSTREAM_ADJUDICATION_FAIL:A4_GATES")
    if tuple(a4.get("protocol", {}).get("val_ids") or ()) != EXPECTED_VAL_IDS:
        raise RuntimeError("A5_UPSTREAM_ADJUDICATION_FAIL:A4_VAL_IDS")
    if a4.get("protocol", {}).get("donor_map_sha256") != EXPECTED_DONOR_MAP_SHA256:
        raise RuntimeError("A5_DONOR_MAP_DRIFT:A4_PROTOCOL")

    required_adj = {
        "commit": EXPECTED_A4_EXECUTION_COMMIT,
        "upstream_commit": "4e15c1ec2cd64af39031d3fcfde200f2d248b65a",
        "experiment_result": "ACCEPTED_DIAGNOSTIC_COMPLETE",
        "gates_status": "G1-G14 ALL PASS",
        "rerun_required": False,
        "execution_artifacts_frozen": True,
        "corrected_branch": "MIXED_PAIRED_CONTEXT_NO_GO",
        "corrected_training_go": False,
        "a4t_status": "HOLD",
    }
    for key, value in required_adj.items():
        if adjudication.get(key) != value:
            raise RuntimeError(f"A5_UPSTREAM_ADJUDICATION_FAIL:{key}:{adjudication.get(key)}!={value}")
    machine = adjudication.get("machine_route_decision") or {}
    if machine.get("branch") != "CENTERING_TRAINING_GO" or machine.get("training_go") is not True or machine.get("status") != "REJECTED_INVALID":
        raise RuntimeError("A5_UPSTREAM_ADJUDICATION_FAIL:MACHINE_DECISION_HISTORY")
    change = adjudication.get("code_change_after_execution") or {}
    if change.get("declaration") != "corrected code is NOT the executed code" or change.get("audit_rerun") is not False:
        raise RuntimeError("A5_UPSTREAM_ADJUDICATION_FAIL:DUAL_TRACK_DECLARATION")
    if change.get("decision_py_sha256_after") != EXPECTED_A4_DECISION_AFTER_SHA256:
        raise RuntimeError("A5_UPSTREAM_ADJUDICATION_FAIL:DECISION_AFTER_SHA_RECORDED")
    if change.get("tests_sha256_after") != EXPECTED_A4_TESTS_AFTER_SHA256:
        raise RuntimeError("A5_UPSTREAM_ADJUDICATION_FAIL:TESTS_AFTER_SHA_RECORDED")
    if adjudication.get("a4_summary_sha256") != EXPECTED_A4_SUMMARY_SHA256:
        raise RuntimeError("A5_UPSTREAM_ADJUDICATION_FAIL:SUMMARY_SHA_LINK")

    # Current corrected code is deliberately different from execution-time provenance.
    current_decision = sha256_file(root / "src/multimodal/step4_a4_decision.py")
    current_tests = sha256_file(root / "tests/test_step4_a4.py")
    if current_decision != EXPECTED_A4_DECISION_AFTER_SHA256:
        raise RuntimeError(f"A5_FROZEN_DEPENDENCY_CLOSURE_FAIL:A4_DECISION_CORRECTED:{current_decision}")
    if current_tests != EXPECTED_A4_TESTS_AFTER_SHA256:
        raise RuntimeError(f"A5_FROZEN_DEPENDENCY_CLOSURE_FAIL:A4_TESTS_CORRECTED:{current_tests}")

    # Preserve execution-time dependency semantics for everything A5 reuses.
    errors: list[str] = []
    a4_prov = a4.get("provenance") or {}
    if sha256_file(contract_path) != a4_prov.get("contract_sha256"):
        errors.append("CONTRACT_DRIFT")
    current_sources = {k: sha256_file(root / rel) for k, rel in SOURCE_PATHS.items()}
    expected_sources = a4_prov.get("dependency_source_hashes") or {}
    for key, current in current_sources.items():
        if expected_sources.get(key) != current:
            errors.append(f"SOURCE_DRIFT:{key}")
    a3_common = root / "src/multimodal/step4_a3_common.py"
    if sha256_file(a3_common) != a4_prov.get("a3_common_sha256"):
        errors.append("A3_COMMON_DRIFT")
    a4_dc = root / "src/multimodal/step4_a4_dc_ac.py"
    if sha256_file(a4_dc) != a4_prov.get("dc_ac_sha256"):
        errors.append("A4_DC_AC_DRIFT")
    a4_eval = root / "scripts/eval_step4_a4.py"
    if sha256_file(a4_eval) != a4_prov.get("evaluator_sha256"):
        errors.append("A4_EVALUATOR_DRIFT")
    a4_design = root / "docs/step4_a4/DESIGN_FREEZE.md"
    if sha256_file(a4_design) != a4_prov.get("design_sha256"):
        errors.append("A4_DESIGN_DRIFT")
    versions = {"torch_version": torch.__version__, "ultralytics_version": __import__("ultralytics").__version__}
    for key, value in versions.items():
        if a4_prov.get(key) != value:
            errors.append(f"VERSION_DRIFT:{key}")
    if errors:
        raise RuntimeError(f"A5_FROZEN_DEPENDENCY_CLOSURE_FAIL:{errors}")

    stock = bool((a4.get("frozen_dependency_closure") or {}).get("stock_eval_semantics_frozen"))
    if not stock:
        raise RuntimeError("A5_STOCK_EVAL_SEMANTICS_FAIL:UPSTREAM_FALSE")

    evidence = {
        "passed": True,
        "a4_execution_commit": EXPECTED_A4_EXECUTION_COMMIT,
        "a4_reviewer_head": EXPECTED_A4_REVIEWER_HEAD,
        "a2_result_sha256": EXPECTED_A2_RESULT_SHA256,
        "donor_map_sha256": EXPECTED_DONOR_MAP_SHA256,
        "a4_summary_sha256": EXPECTED_A4_SUMMARY_SHA256,
        "a4_summary_canonical_lf_sha256": EXPECTED_A4_SUMMARY_CANONICAL_LF_SHA256,
        "a4_standalone_sha256": EXPECTED_A4_STANDALONE_SHA256,
        "a4_conditional_sha256": EXPECTED_A4_CONDITIONAL_SHA256,
        "reviewer_adjudication_sha256": EXPECTED_A4_ADJUDICATION_SHA256,
        "feedback_sha256": EXPECTED_A4_FEEDBACK_SHA256,
        "current_corrected_a4_decision_sha256": current_decision,
        "current_corrected_a4_tests_sha256": current_tests,
        "contract_sha256": sha256_file(contract_path),
        "dependency_source_hashes": current_sources,
        "a3_common_sha256": sha256_file(a3_common),
        "a4_dc_ac_sha256": sha256_file(a4_dc),
        "a4_evaluator_sha256": sha256_file(a4_eval),
        "a4_design_sha256": sha256_file(a4_design),
        "versions": versions,
        "stock_eval_semantics_frozen": stock,
        "executed_vs_corrected_a4_dual_track": True,
    }
    return a2, donor, a4, standalone, conditional, adjudication, evidence


def load_model(root: Path, project: str, tag: str, a2: dict, device):
    sysrec = a2["systems"][tag]
    run_dir = root / project / Path(sysrec["run_dir"]).name
    manifest_path = run_dir / "manifest.json"
    ckpt_path = run_dir / "weights/last.pt"
    if sha256_file(ckpt_path) != sysrec["checkpoint_sha256"]:
        raise RuntimeError(f"A5_CHECKPOINT_SHA_MISMATCH:{tag}")
    if sha256_file(manifest_path) != sysrec["manifest_sha256"]:
        raise RuntimeError(f"A5_MANIFEST_SHA_MISMATCH:{tag}")
    manifest = json_load(manifest_path)
    for key, expected in sysrec["manifest_identity"].items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"A5_CHECKPOINT_IDENTITY:{tag}:{key}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = (ck.get("ema") or ck.get("model")).float().eval().to(device)
    if getattr(model, "_gate_override", None) is not None:
        model.set_gate_override(None)
    return model, run_dir, ckpt_path, manifest_path


def build_dataset(contract: dict):
    ds = TriModalDataset(contract, split="val", group="C1-I", augment=False)
    if tuple(ds.ids) != EXPECTED_VAL_IDS:
        raise RuntimeError(f"A5_VAL_SET_DRIFT:{list(ds.ids)}!={list(EXPECTED_VAL_IDS)}")
    return ds


def metric_from_stats(stats_by_id: dict, ids: list[str], names: dict) -> dict:
    from ultralytics.utils.metrics import DetMetrics
    metrics = DetMetrics(names={int(k): v for k, v in names.items()})
    for sid in ids:
        metrics.update_stats(stats_by_id[sid])
    metrics.process()
    res = metrics.results_dict
    return {
        "map50": float(res["metrics/mAP50(B)"]),
        "map50_95": float(res["metrics/mAP50-95(B)"]),
        "n_images": len(ids),
    }


def collect_ap_condition(model, dataset, device, names, forward_fn):
    validator = evu.make_detection_validator(model, device, names)
    stats, traces = {}, {}
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            sid = str(sample["sample_id"])
            batch = dataset.collate_fn([sample])
            batch = evu.move_step3_batch_to_device(batch, device)
            output, trace = forward_fn(sid, sample, batch)
            raw = evu.extract_detection_tensor(output).detach()
            trace = dict(trace)
            trace["detection_sha256"] = tensor_sha256(raw)
            preds = validator.postprocess(raw)
            if len(preds) != 1:
                raise RuntimeError(f"A5_EXPECTED_ONE_PREDICTION:{len(preds)}")
            pbatch = validator._prepare_batch(0, batch)
            pred = validator._prepare_pred(preds[0])
            cls_np = pbatch["cls"].detach().cpu().numpy()
            no_pred = pred["cls"].numel() == 0
            stat = validator._process_batch(pred, pbatch)
            stat.update(
                target_cls=cls_np,
                target_img=np.unique(cls_np),
                conf=np.zeros(0, dtype=np.float32) if no_pred else pred["conf"].detach().cpu().numpy(),
                pred_cls=np.zeros(0, dtype=np.float32) if no_pred else pred["cls"].detach().cpu().numpy(),
                im_name=sid,
            )
            stats[sid], traces[sid] = stat, trace
    ids = [str(x) for x in dataset.ids]
    return {
        "full": metric_from_stats(stats, ids, names),
        "loo": {held: metric_from_stats(stats, [sid for sid in ids if sid != held], names) for held in ids},
        "trace": traces,
    }


def residual_cache_with_gate_guard(model, dataset, device):
    calls = {"n": 0}
    handle = model.reliability_gate.register_forward_hook(
        lambda module, inputs, output: calls.__setitem__("n", calls["n"] + 1)
    )
    cache = build_residual_cache_no_gate(model, dataset, device)
    handle.remove()
    if calls["n"] != 0:
        raise RuntimeError(f"A5_CONTEXT_SEMANTICS_FAIL:DONOR_CACHE_GATE:{calls['n']}")
    return cache, {"gate_calls": 0, "passed": True}


def forward_a5_context(model, batch, *, sid: str, donor_id: str, cache: dict, context_id: str, role: str):
    built = build_context(
        cache,
        recipient_id=sid,
        donor_id=donor_id,
        context_id=context_id,
        p5_role=role,
    )
    out, tr = forward_with_custom_residuals(
        model,
        batch["img"],
        recipient_id=sid,
        active_scales=built.active_scales,
        replacements=built.replacements,
        source_ids=built.source_ids,
        condition_name=f"A5_{context_id}_P5_{role.upper()}",
    )
    tr["a5_context"] = context_id
    tr["a5_context_states"] = dict(built.states)
    tr["a5_p5_role"] = role
    tr["a5_p5_source_id"] = built.p5_source_id
    tr["a5_component_trace"] = built.component_trace
    return out, tr


def validate_q(tag: str, qref: dict, results: dict) -> bool:
    errors = []
    for context_id in CONTEXT_ORDER:
        for role in ("native", "donor"):
            for sid, tr in results[context_id][role]["trace"].items():
                if tr.get("q_native") != qref[sid]:
                    errors.append(f"{context_id}:{role}:{sid}:Q_DRIFT")
                if tag == "FIXED" and tr.get("q_native") != [1.0]:
                    errors.append(f"{context_id}:{role}:{sid}:FIXED_NOT_ONE")
    if errors:
        raise RuntimeError(f"A5_Q_FREEZE_FAIL:{tag}:{errors[:10]}")
    return True


def validate_context_matrix(results: dict, donor_map: dict) -> dict:
    if set(results) != set(CONTEXT_ORDER):
        raise RuntimeError(f"A5_CONTEXT_MATRIX_INCOMPLETE:{sorted(results)}")
    context_ok = True
    donor_ok = True
    isolation_ok = True
    effect_ok = True
    rows = []
    for context_id in CONTEXT_ORDER:
        if set(results[context_id]) < {"native", "donor", "paired_effect_native_minus_donor"}:
            raise RuntimeError(f"A5_CONTEXT_MATRIX_INCOMPLETE:{context_id}:ROLES")
        native = results[context_id]["native"]
        donor = results[context_id]["donor"]
        if set(native["trace"]) != set(donor["trace"]):
            raise RuntimeError(f"A5_P5_IDENTITY_ISOLATION_FAIL:{context_id}:TRACE_IDS")
        for sid in native["trace"]:
            donor_id = donor_map[sid]
            nv = validate_context_trace(
                native["trace"][sid], recipient_id=sid, donor_id=donor_id,
                context_id=context_id, p5_role="native",
            )
            dn = validate_context_trace(
                donor["trace"][sid], recipient_id=sid, donor_id=donor_id,
                context_id=context_id, p5_role="donor",
            )
            pair = validate_pair_isolation(native["trace"][sid], donor["trace"][sid])
            context_ok = context_ok and nv["passed"] and dn["passed"]
            donor_ok = donor_ok and (
                (donor["trace"][sid].get("a5_component_trace") or {}).get("P5", {}).get("residual_source_id") == donor_id
                and (donor["trace"][sid].get("a5_component_trace") or {}).get("P5", {}).get("mean_source_id") == donor_id
            )
            isolation_ok = isolation_ok and pair["passed"]
            rows.append({"context": context_id, "sid": sid, "native": nv, "donor": dn, "pair": pair})
        recomputed = effect_from_results(native, donor)
        recorded = results[context_id]["paired_effect_native_minus_donor"]
        effect_ok = effect_ok and recomputed == recorded
    if not context_ok:
        raise RuntimeError("A5_CONTEXT_SEMANTICS_FAIL")
    if not donor_ok:
        raise RuntimeError("A5_DONOR_AC_MEAN_SOURCE_FAIL")
    if not isolation_ok:
        raise RuntimeError("A5_P5_IDENTITY_ISOLATION_FAIL")
    if not effect_ok:
        raise RuntimeError("A5_PAIRED_EFFECT_SEMANTICS_FAIL")
    return {
        "context_semantics_ok": context_ok,
        "donor_p5_self_centering_ok": donor_ok,
        "p5_identity_isolation_ok": isolation_ok,
        "paired_effect_semantics_ok": effect_ok,
        "rows": rows,
    }


def _same_metric_block(a: dict, b: dict) -> bool:
    return a.get("full") == b.get("full") and a.get("loo") == b.get("loo")


def validate_anchor(context_result: dict, a4_p5: dict, *, context_id: str) -> dict:
    rows = {}
    for role, a4_key in (("native", "native_ac"), ("donor", "donor_ac")):
        a5 = context_result[role]
        old = a4_p5[a4_key]
        metric_equal = _same_metric_block(a5, old)
        trace_equal = True
        row = {}
        for sid in a5["trace"]:
            a5_sha = a5["trace"][sid].get("detection_sha256")
            a4_sha = old["trace"][sid].get("detection_sha256")
            eq = a5_sha == a4_sha
            row[sid] = {"a5_sha256": a5_sha, "a4_sha256": a4_sha, "bitwise_equal": eq}
            trace_equal = trace_equal and eq
        rows[role] = {"metric_equal": metric_equal, "trace_equal": trace_equal, "rows": row}
    passed = all(rows[r]["metric_equal"] and rows[r]["trace_equal"] for r in rows)
    if not passed:
        code = "A5_OO_ANCHOR_FAIL" if context_id == "OO" else "A5_FF_ANCHOR_FAIL"
        raise RuntimeError(code)
    return {"passed": True, "context": context_id, "roles": rows}


def run_contexts(model, dataset, device, names, tag: str, donor_map: dict, cache: dict):
    results = {}
    for context_id in CONTEXT_ORDER:
        results[context_id] = {}
        for role in ("native", "donor"):
            def fn(sid, sample, batch, context_id=context_id, role=role):
                return forward_a5_context(
                    model, batch, sid=sid, donor_id=donor_map[sid], cache=cache,
                    context_id=context_id, role=role,
                )
            results[context_id][role] = collect_ap_condition(model, dataset, device, names, fn)
        results[context_id]["paired_effect_native_minus_donor"] = effect_from_results(
            results[context_id]["native"], results[context_id]["donor"]
        )
    return results


def validate_interactions(effect_by_tag: dict, computed: dict) -> bool:
    for tag in RUN_TAGS:
        if set(computed[tag]) != set(INTERACTION_COEFFICIENTS):
            return False
        for name, coeffs in INTERACTION_COEFFICIENTS.items():
            if linear_effect_contrast(effect_by_tag[tag], coeffs) != computed[tag][name]:
                return False
    return True


def validate_provenance(prov: dict) -> bool:
    required = {
        "design_sha256", "context_sha256", "effects_sha256", "evaluator_sha256",
        "tests_sha256", "audit_source_sha256", "preexecution_audit_sha256",
        "a4_execution_commit", "a4_reviewer_head", "a4_summary_sha256",
        "a4_summary_canonical_lf_sha256", "a4_standalone_sha256", "a4_conditional_sha256",
        "reviewer_adjudication_sha256", "feedback_sha256", "a2_result_sha256",
        "donor_map_sha256", "contract_sha256", "FIXED_last_pt_sha256",
        "SOFT_last_pt_sha256", "FIXED_manifest_sha256", "SOFT_manifest_sha256",
        "torch_version", "ultralytics_version", "context_matrix_version",
        "interaction_formula_version", "condition_count_per_system", "total_condition_instances",
    }
    missing = sorted(k for k in required if prov.get(k) in (None, ""))
    if missing:
        raise RuntimeError(f"A5_PROVENANCE_INCOMPLETE:{missing}")
    if prov["condition_count_per_system"] != 18 or prov["total_condition_instances"] != 36:
        raise RuntimeError("A5_PROVENANCE_INCOMPLETE:CONDITION_COUNTS")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="runs/step4_f1_c")
    ap.add_argument("--contract", default=OUT_DEFAULT)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out-dir", default="reports/step4_a5")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out_dir = ROOT / a.out_dir
    outputs = [
        out_dir / "context_paired_effects.json",
        out_dir / "context_shifts.json",
        out_dir / "cross_scale_interactions.json",
        out_dir / "native_ap_secondary.json",
        out_dir / "a5_summary.json",
    ]
    existing = [str(p) for p in outputs if p.exists()]
    if existing and not a.overwrite:
        raise RuntimeError(f"A5_REFUSE_OVERWRITE:{existing}")

    audit = verify_preexecution_audit(ROOT)
    contract_path = Path(a.contract)
    contract = json_load(contract_path)
    a2, donor_map, a4, a4_standalone, a4_conditional, adjudication, dependency = verify_upstream(ROOT, contract_path)
    dataset = build_dataset(contract)

    devarg = str(a.device)
    if devarg == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    elif devarg.startswith("cuda:"):
        device = torch.device(devarg)
    else:
        device = torch.device(f"cuda:{devarg}")
    names = {int(k): v for k, v in CLASS_NAMES.items()} if isinstance(CLASS_NAMES, dict) else CLASS_NAMES

    models, identities, state_before, state_after = {}, {}, {}, {}
    caches, cache_guards, qrefs = {}, {}, {}
    context_results, validation = {}, {}
    anchors = {}

    for tag in RUN_TAGS:
        model, run_dir, ckpt_path, manifest_path = load_model(ROOT, a.project, tag, a2, device)
        models[tag] = model
        identities[tag] = {
            "run_dir": str(run_dir.relative_to(ROOT)),
            "last_pt": str(ckpt_path.relative_to(ROOT)),
            "last_pt_sha256": sha256_file(ckpt_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
        qrefs[tag] = {
            sid: a4["systems"][tag]["native_equivalence"]["rows"][sid]["q_native"]
            for sid in EXPECTED_VAL_IDS
        }
        state_before[tag] = state_sha256(model)
        caches[tag], cache_guards[tag] = residual_cache_with_gate_guard(model, dataset, device)
        context_results[tag] = run_contexts(model, dataset, device, names, tag, donor_map, caches[tag])
        validate_q(tag, qrefs[tag], context_results[tag])
        validation[tag] = validate_context_matrix(context_results[tag], donor_map)
        anchors[tag] = {
            "OO": validate_anchor(context_results[tag]["OO"], a4_standalone["systems"][tag]["P5"], context_id="OO"),
            "FF": validate_anchor(context_results[tag]["FF"], a4_conditional["systems"][tag]["P5"], context_id="FF"),
        }
        state_after[tag] = state_sha256(model)
        if state_after[tag] != state_before[tag]:
            raise RuntimeError(f"A5_PARAMETER_MUTATION:{tag}")

    effects = {
        tag: {c: context_results[tag][c]["paired_effect_native_minus_donor"] for c in CONTEXT_ORDER}
        for tag in RUN_TAGS
    }
    context_labels = {
        c: classify_paired_effect(effects["FIXED"][c], effects["SOFT"][c])
        for c in CONTEXT_ORDER
    }
    shifts = {tag: context_shifts(effects[tag]) for tag in RUN_TAGS}
    shift_labels = {
        c: classify_shift(shifts["FIXED"][c], shifts["SOFT"][c])
        for c in CONTEXT_ORDER
    }
    interactions = {tag: interaction_effects(effects[tag]) for tag in RUN_TAGS}
    interaction_labels = {
        name: classify_shift(interactions["FIXED"][name], interactions["SOFT"][name])
        for name in INTERACTION_COEFFICIENTS
    }
    interaction_ok = validate_interactions(effects, interactions)
    if not interaction_ok:
        raise RuntimeError("A5_INTERACTION_CONTRAST_FAIL")

    flags = mechanism_flags(context_labels, shift_labels, interaction_labels, effects["FIXED"], effects["SOFT"])
    route = route_decision(flags)
    if route.get("training_go") is not False:
        raise RuntimeError("A5_TRAINING_GO_FORBIDDEN")

    native_secondary = {}
    for tag in RUN_TAGS:
        native_secondary[tag] = {}
        ff = context_results[tag]["FF"]["native"]
        for c in CONTEXT_ORDER:
            native = context_results[tag][c]["native"]
            native_secondary[tag][c] = {
                "native_ap": {"full": native["full"], "loo": native["loo"]},
                "native_ap_minus_FF": {
                    "full": float(native["full"]["map50_95"] - ff["full"]["map50_95"]),
                    "loo": {
                        sid: float(native["loo"][sid]["map50_95"] - ff["loo"][sid]["map50_95"])
                        for sid in ff["loo"]
                    },
                },
            }

    context_complete = all(set(context_results[tag]) == set(CONTEXT_ORDER) for tag in RUN_TAGS)
    q_ok = all(validate_q(tag, qrefs[tag], context_results[tag]) for tag in RUN_TAGS)
    p5_isolation_ok = all(validation[tag]["p5_identity_isolation_ok"] for tag in RUN_TAGS)
    context_semantics_ok = all(validation[tag]["context_semantics_ok"] for tag in RUN_TAGS)
    donor_self_ok = all(validation[tag]["donor_p5_self_centering_ok"] for tag in RUN_TAGS)
    effect_semantics_ok = all(validation[tag]["paired_effect_semantics_ok"] for tag in RUN_TAGS)
    oo_anchor_ok = all(anchors[tag]["OO"]["passed"] for tag in RUN_TAGS)
    ff_anchor_ok = all(anchors[tag]["FF"]["passed"] for tag in RUN_TAGS)

    provenance = {
        "design_sha256": sha256_file(ROOT / "docs/step4_a5/DESIGN_FREEZE.md"),
        "context_sha256": sha256_file(ROOT / "src/multimodal/step4_a5_context.py"),
        "effects_sha256": sha256_file(ROOT / "src/multimodal/step4_a5_effects.py"),
        "evaluator_sha256": sha256_file(ROOT / "scripts/eval_step4_a5.py"),
        "tests_sha256": sha256_file(ROOT / "tests/test_step4_a5.py"),
        "audit_source_sha256": sha256_file(ROOT / "scripts/audit_step4_a5.py"),
        "preexecution_audit_sha256": audit["sha256"],
        "a4_execution_commit": dependency["a4_execution_commit"],
        "a4_reviewer_head": dependency["a4_reviewer_head"],
        "a4_summary_sha256": dependency["a4_summary_sha256"],
        "a4_summary_canonical_lf_sha256": dependency["a4_summary_canonical_lf_sha256"],
        "a4_standalone_sha256": dependency["a4_standalone_sha256"],
        "a4_conditional_sha256": dependency["a4_conditional_sha256"],
        "reviewer_adjudication_sha256": dependency["reviewer_adjudication_sha256"],
        "feedback_sha256": dependency["feedback_sha256"],
        "a2_result_sha256": dependency["a2_result_sha256"],
        "donor_map_sha256": dependency["donor_map_sha256"],
        "contract_sha256": dependency["contract_sha256"],
        "dependency_source_hashes": dependency["dependency_source_hashes"],
        "a3_common_sha256": dependency["a3_common_sha256"],
        "a4_dc_ac_sha256": dependency["a4_dc_ac_sha256"],
        "a4_evaluator_sha256": dependency["a4_evaluator_sha256"],
        "a4_design_sha256": dependency["a4_design_sha256"],
        "current_corrected_a4_decision_sha256": dependency["current_corrected_a4_decision_sha256"],
        "current_corrected_a4_tests_sha256": dependency["current_corrected_a4_tests_sha256"],
        "FIXED_last_pt_sha256": identities["FIXED"]["last_pt_sha256"],
        "SOFT_last_pt_sha256": identities["SOFT"]["last_pt_sha256"],
        "FIXED_manifest_sha256": identities["FIXED"]["manifest_sha256"],
        "SOFT_manifest_sha256": identities["SOFT"]["manifest_sha256"],
        "torch_version": dependency["versions"]["torch_version"],
        "ultralytics_version": dependency["versions"]["ultralytics_version"],
        "val_ids": list(EXPECTED_VAL_IDS),
        "q_native": qrefs,
        "context_matrix_version": "A5_CONTEXT_3X3_V1=OO,FO,OF,FF,AO,OA,AF,FA,AA",
        "interaction_formula_version": "A5_INTERACTIONS_V1=D3F,D4F,IFF,D3A,D4A,IAA,IAF,IFA",
        "condition_count_per_system": 18,
        "total_condition_instances": 36,
        "executed_vs_corrected_a4_dual_track": dependency["executed_vs_corrected_a4_dual_track"],
        "anchor_identity": {
            "OO": "A4 AC_ALL P5 standalone native/donor",
            "FF": "A4 AC_ALL P5 conditional native/donor",
        },
    }
    provenance_ok = validate_provenance(provenance)

    gates = {
        "G1_upstream_reviewer_adjudication": (
            adjudication.get("corrected_training_go") is False
            and adjudication.get("corrected_branch") == "MIXED_PAIRED_CONTEXT_NO_GO"
            and adjudication.get("a4t_status") == "HOLD"
            and adjudication.get("rerun_required") is False
        ),
        "G2_frozen_dependency_closure": dependency["passed"],
        "G3_eval_only_state_unchanged": all(state_before[t] == state_after[t] for t in RUN_TAGS),
        "G4_q_freeze": q_ok,
        "G5_donor_map_freeze": sha256_file(ROOT / "reports/step4_a2/val_donor_map.json") == EXPECTED_DONOR_MAP_SHA256,
        "G6_p5_only_identity_intervention": p5_isolation_ok,
        "G7_p3_p4_context_semantics": context_semantics_ok and all(cache_guards[t]["passed"] for t in RUN_TAGS),
        "G8_donor_p5_self_centering": donor_self_ok,
        "G9_oo_a4_anchor_closure": oo_anchor_ok,
        "G10_ff_a4_anchor_closure": ff_anchor_ok,
        "G11_exact_context_completeness": context_complete,
        "G12_paired_effect_semantics": effect_semantics_ok,
        "G13_interaction_contrast_correctness": interaction_ok,
        "G14_stock_eval_semantics": dependency["stock_eval_semantics_frozen"],
        "G15_provenance_complete": bool(provenance_ok),
    }
    if not all(gates.values()):
        raise RuntimeError(f"A5_ABORT:{gates}")

    interpretation = {
        "context_shift_is_not_sign_flip": True,
        "native_ap_rescue_is_not_paired_restoration": True,
        "ac_utility_is_not_paired_causality": True,
        "a5_identifies_context_mechanism_not_training_efficacy": True,
        "a5_never_grants_training_go": route["training_go"] is False,
        "p3_p4_context_always_recipient_native": context_semantics_ok,
        "only_p5_identity_is_manipulated": p5_isolation_ok,
        "fixed_primary_soft_replication": True,
        "val6_limits_population_generalization": True,
    }

    systems_summary = {
        tag: {
            **identities[tag],
            "state_sha256_before": state_before[tag],
            "state_sha256_after": state_after[tag],
            "q_native": qrefs[tag],
            "residual_cache_no_gate": cache_guards[tag],
            "runtime_validation": {
                k: v for k, v in validation[tag].items() if k != "rows"
            },
        }
        for tag in RUN_TAGS
    }

    context_payload = {
        "schema": "step4-a5-context-paired-effects-v1",
        "contexts": {c: CONTEXT_STATES[c] for c in CONTEXT_ORDER},
        "systems": context_results,
        "cross_system_labels": context_labels,
        "anchors": anchors,
    }
    shifts_payload = {
        "schema": "step4-a5-context-shifts-v1",
        "baseline": "OO",
        "systems": shifts,
        "cross_system_labels": shift_labels,
    }
    interactions_payload = {
        "schema": "step4-a5-cross-scale-interactions-v1",
        "coefficients": INTERACTION_COEFFICIENTS,
        "systems": interactions,
        "cross_system_labels": interaction_labels,
        "mechanism_flags": flags,
    }
    native_payload = {
        "schema": "step4-a5-native-ap-secondary-v1",
        "authority": "secondary_diagnostic_only",
        "systems": native_secondary,
    }
    summary = {
        "schema": SCHEMA,
        "protocol": {
            "evaluation_only": True,
            "primary_system": "F1C-I-fixed/last.pt",
            "replication_system": "F1C-I-soft/last.pt",
            "primary_target": "P5 AC_ALL recipient-vs-donor identity",
            "context_states": {c: CONTEXT_STATES[c] for c in CONTEXT_ORDER},
            "val_ids": list(EXPECTED_VAL_IDS),
            "donor_map_sha256": EXPECTED_DONOR_MAP_SHA256,
            "q_rule": "untouched recipient native q before all P3/P4 context and P5 identity interventions",
            "training_go_forbidden": True,
        },
        "preexecution_audit": audit,
        "upstream": {
            "a4_execution_commit": EXPECTED_A4_EXECUTION_COMMIT,
            "a4_reviewer_head": EXPECTED_A4_REVIEWER_HEAD,
            "a4_experiment_result": adjudication["experiment_result"],
            "a4_corrected_branch": adjudication["corrected_branch"],
            "a4_corrected_training_go": adjudication["corrected_training_go"],
            "a4t_status": adjudication["a4t_status"],
            "execution_artifacts_frozen": adjudication["execution_artifacts_frozen"],
        },
        "frozen_dependency_closure": dependency,
        "systems": systems_summary,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "context_labels": context_labels,
        "context_shift_labels_from_OO": shift_labels,
        "interaction_labels": interaction_labels,
        "mechanism_flags": flags,
        "native_ap_secondary_authority": "secondary_diagnostic_only",
        "next_branch": route,
        "training_go": False,
        "provenance": provenance,
        "interpretation_discipline": interpretation,
    }

    if summary["training_go"] is not False or summary["next_branch"]["training_go"] is not False:
        raise RuntimeError("A5_TRAINING_GO_FORBIDDEN")

    commit_json_bundle({
        out_dir / "context_paired_effects.json": context_payload,
        out_dir / "context_shifts.json": shifts_payload,
        out_dir / "cross_scale_interactions.json": interactions_payload,
        out_dir / "native_ap_secondary.json": native_payload,
        out_dir / "a5_summary.json": summary,
    })
    print(json.dumps({
        "schema": SCHEMA,
        "all_gates_passed": True,
        "context_labels": context_labels,
        "mechanism_flags": {k: v for k, v in flags.items() if isinstance(v, bool)},
        "next_branch": route,
        "training_go": False,
        "conditions_per_system": 18,
        "total_condition_instances": 36,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
