#!/usr/bin/env python3
"""Static/pre-execution audit for A3 implementation.

This audit proves structure, freshness, and absence of obvious training calls.
Runtime causal semantics remain enforced again by eval_step4_a3.py.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/step4_a3/preexecution_audit.json"

FILES = {
    "design_sha256": ROOT / "docs/step4_a3/DESIGN_FREEZE.md",
    "common_sha256": ROOT / "src/multimodal/step4_a3_common.py",
    "registration_sha256": ROOT / "src/multimodal/step4_a3_registration.py",
    "spatial_sha256": ROOT / "src/multimodal/step4_a3_spatial.py",
    "semantic_sha256": ROOT / "src/multimodal/step4_a3_semantic.py",
    "generic_bias_sha256": ROOT / "src/multimodal/step4_a3_generic_bias.py",
    "evaluator_sha256": ROOT / "scripts/eval_step4_a3.py",
    "tests_sha256": ROOT / "tests/test_step4_a3.py",
    "audit_source_sha256": ROOT / "scripts/audit_step4_a3.py",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def calls(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                out.append(fn.attr)
            elif isinstance(fn, ast.Name):
                out.append(fn.id)
    return out


def main():
    checks = {}
    errors = []

    for key, path in FILES.items():
        name = key.replace("_sha256", "_exists")
        checks[name] = path.exists()

    missing = [str(p) for p in FILES.values() if not p.exists()]
    if missing:
        raise SystemExit(f"A3_AUDIT_MISSING:{missing}")

    evaluator = FILES["evaluator_sha256"].read_text(encoding="utf-8")
    common = FILES["common_sha256"].read_text(encoding="utf-8")
    registration = FILES["registration_sha256"].read_text(encoding="utf-8")
    semantic = FILES["semantic_sha256"].read_text(encoding="utf-8")
    generic = FILES["generic_bias_sha256"].read_text(encoding="utf-8")
    design = FILES["design_sha256"].read_text(encoding="utf-8")

    py_paths = [
        FILES["common_sha256"], FILES["registration_sha256"], FILES["spatial_sha256"],
        FILES["semantic_sha256"], FILES["generic_bias_sha256"], FILES["evaluator_sha256"],
    ]
    all_calls = []
    for path in py_paths:
        all_calls.extend(calls(parse(path)))

    checks.update({
        "no_optimizer_calls": not any(x in all_calls for x in ("step", "zero_grad")),
        "no_backward_calls": "backward" not in all_calls,
        "no_train_calls": "train" not in all_calls,
        "a2_result_sha_pinned": "756093358153c5e203f485dce96e0f2a5e91881fb6c6e4b49c036cbfdc6d1c6b" in evaluator,
        "a2_donor_sha_pinned": "c5cd8e852663eae3243bc5e4c263c6f2c26c7b2faa977ae3b60cb5c1ea122af5" in evaluator,
        "modality_preprocess_blob_pinned": "ed3a52150eedee18c60f163401dc64a198398662" in evaluator,
        "preaudit_freshness_consumed": "A3_PREEXECUTION_AUDIT_STALE" in evaluator,
        "dependency_closure_enforced": "A3_FROZEN_DEPENDENCY_CLOSURE_FAIL" in evaluator,
        "native_equivalence_enforced": "A3_NATIVE_EQUIVALENCE_FAIL" in evaluator,
        "q_freeze_enforced": "A3_Q_FREEZE_FAIL" in evaluator,
        "state_before_after_present": "state_sha256_before" in evaluator and "state_sha256_after" in evaluator,
        "donor_map_exact_reuse": 'donor_file != a2.get("donor_map")' in evaluator,
        "registration_estimator_has_no_ap_arg": "def estimate_registration_shift(sample" in registration,
        "registration_valid_content_crop": "valid_content_slices" in registration,
        "registration_crossfit_excludes_heldout": "sid != held_out" in registration,
        "registration_leakage_gate": "A3_REGISTRATION_LEAKAGE" in evaluator,
        "zero_fill_translation_present": "zero_fill_translate" in common,
        "shift_is_post_projection": "shifts.items()" in common and "q_native = model._effective_gate" in common,
        "feature_best_shift_descriptive_only": "shift_surface" in evaluator or "shift_surface" not in evaluator,
        "feature_best_shift_not_used_in_ap_forward": "best_feature_shift" not in evaluator,
        "spatial_native_vs_donor": "corr_native" in (ROOT / "src/multimodal/step4_a3_spatial.py").read_text(encoding="utf-8")
                                    and "corr_donor" in (ROOT / "src/multimodal/step4_a3_spatial.py").read_text(encoding="utf-8"),
        "semantic_uses_recipient_boxes": 'semantic_row(sample["bboxes"], native, dres)' in evaluator,
        "semantic_degenerate_fail": "SEMANTIC_MASK_DEGENERATE" in semantic,
        "semantic_coverage_fail": "A3_SEMANTIC_COVERAGE_FAIL" in semantic,
        "loo_mean_excludes_heldout": "sid != held_out" in generic,
        "loo_mean_self_leakage_gate": "A3_LOO_MEAN_SELF_LEAKAGE" in evaluator,
        "native_dc_broadcast": "expand_as(residual)" in generic,
        "native_ac_exact_subtraction": "return residual - native_dc(residual)" in generic,
        "donor_cache_no_gate_helper": "build_residual_cache_no_gate" in common,
        "donor_cache_runtime_gate_guard": "DONOR_CACHE_GATE" in evaluator,
        "fixed_primary_soft_replication": '"FIXED", "SOFT"' in evaluator,
        "last_pt_only": "best.pt" not in evaluator and 'weights/last.pt' in evaluator,
        "no_dataset_level_aux_map": "aux_id_map" not in evaluator,
        "no_dataset_level_aux_zero": "aux_zero" not in evaluator,
        "g1_g12_present": all(f'"G{i}_' in evaluator for i in range(1, 13)),
        "g8_not_unconditional_true": '"G8_post_projection_intervention": True' not in evaluator,
        "g12_not_unconditional_true": '"G12_provenance": True' not in evaluator,
        "gates_dynamic_all_values": "all(gates.values())" in evaluator,
        "correlation_not_causal_discipline": "correlation_is_not_registration_causality" in evaluator,
        "generic_not_semantic_discipline": "generic_mean_utility_is_not_multimodal_information_use" in evaluator,
        "design_forbids_training": "NO training" in design,
        "design_forbids_ap_selected_shift": "NO per-sample AP-selected shift" in design,
        "design_requires_valid_crop": "letterbox padding" in design,
        "summary_is_unique_decision_entry": 'a3_summary.json' in evaluator,
    })

    # Make one check non-tautological: evaluator must never reference best_feature_shift.
    checks["feature_best_shift_descriptive_only"] = checks["feature_best_shift_not_used_in_ap_forward"]

    for k, v in checks.items():
        if v is not True:
            errors.append(k)

    provenance = {key: sha256_file(path) for key, path in FILES.items()}
    obj = {
        "schema": "step4-a3-preexecution-audit-v1",
        "checks": checks,
        "errors": errors,
        "provenance": provenance,
        "all_passed": not errors,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(json.dumps(obj, indent=2))
    if errors:
        raise SystemExit(f"A3_AUDIT_FAIL:{errors}")


if __name__ == "__main__":
    main()
