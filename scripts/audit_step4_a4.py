#!/usr/bin/env python3
"""Static/pre-execution audit for A4 implementation."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/step4_a4/preexecution_audit.json"
FILES = {
    "design_sha256": ROOT / "docs/step4_a4/DESIGN_FREEZE.md",
    "dc_ac_sha256": ROOT / "src/multimodal/step4_a4_dc_ac.py",
    "content_mask_sha256": ROOT / "src/multimodal/step4_a4_content_mask.py",
    "decision_sha256": ROOT / "src/multimodal/step4_a4_decision.py",
    "evaluator_sha256": ROOT / "scripts/eval_step4_a4.py",
    "tests_sha256": ROOT / "tests/test_step4_a4.py",
    "audit_source_sha256": ROOT / "scripts/audit_step4_a4.py",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def call_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
        checks[key.replace("_sha256", "_exists")] = path.exists()
    missing = [str(p) for p in FILES.values() if not p.exists()]
    if missing:
        raise SystemExit(f"A4_AUDIT_MISSING:{missing}")

    design = FILES["design_sha256"].read_text(encoding="utf-8")
    dcac = FILES["dc_ac_sha256"].read_text(encoding="utf-8")
    content = FILES["content_mask_sha256"].read_text(encoding="utf-8")
    decision = FILES["decision_sha256"].read_text(encoding="utf-8")
    evaluator = FILES["evaluator_sha256"].read_text(encoding="utf-8")
    py_paths = [FILES["dc_ac_sha256"], FILES["content_mask_sha256"], FILES["decision_sha256"], FILES["evaluator_sha256"]]
    calls = [c for p in py_paths for c in call_names(p)]

    checks.update({
        "no_optimizer_calls": not any(x in calls for x in ("step", "zero_grad")),
        "no_backward_calls": "backward" not in calls,
        "no_train_calls": "train" not in calls,
        "a2_result_sha_pinned": "756093358153c5e203f485dce96e0f2a5e91881fb6c6e4b49c036cbfdc6d1c6b" in evaluator,
        "a3_summary_raw_sha_pinned": "121dacc0ed50f5d24a8108ea3710e981c3c0314210729c80ed339652ea579839" in evaluator,
        "a3_summary_canonical_sha_pinned": "3523cb526d7a0fde3b0f0f121f73f29326aa88167bf6ad60d0505d7fed50d9ed" in evaluator,
        "donor_sha_pinned": "c5cd8e852663eae3243bc5e4c263c6f2c26c7b2faa977ae3b60cb5c1ea122af5" in evaluator,
        "modality_preprocess_blob_pinned": "ed3a52150eedee18c60f163401dc64a198398662" in evaluator,
        "preaudit_freshness_consumed": "A4_PREEXECUTION_AUDIT_STALE" in evaluator,
        "a3_all_gates_consumed": 'a3.get("all_gates_passed") is True' in evaluator and 'all(bool(v) for v in a3["gates"].values())' in evaluator,
        "dependency_closure_enforced": "A4_FROZEN_DEPENDENCY_CLOSURE_FAIL" in evaluator,
        "native_equivalence_enforced": "A4_NATIVE_EQUIVALENCE_FAIL" in evaluator,
        "q_freeze_enforced": "A4_Q_FREEZE_FAIL" in evaluator,
        "state_before_after_enforced": "A4_PARAMETER_MUTATION" in evaluator,
        "donor_map_exact_reuse": 'donor != a2.get("donor_map")' in evaluator,
        "a3_common_source_pinned": "A3_COMMON_DRIFT" in evaluator,
        "dc_all_exact_mean": "residual.mean(dim=(-2, -1)" in dcac,
        "ac_all_exact_subtraction": "return residual - full_map_dc(residual)" in dcac,
        "dc_content_fractional_weighting": "adaptive_avg_pool2d" in content and "weighted_dc" in dcac,
        "content_mask_only_ori_ratio": '"source": "ori_shape+ratio_pad"' in content,
        "content_mask_no_gt_dependency": "bboxes" not in content and "cls" not in content,
        "content_mask_no_prediction_dependency": "prediction" not in content.lower() or "never reads" in content.lower(),
        "donor_self_centering_runtime_gate": "A4_DONOR_AC_MEAN_SOURCE_FAIL" in design and "donor_self_centering_ok" in evaluator,
        "donor_mean_source_traced": '"mean_source_id"' in dcac,
        "donor_mask_source_traced": '"content_mask_source_id"' in dcac,
        "component_reconstruction_checked": "reconstruction_max_abs_error" in dcac,
        "ac_all_zero_mean_checked": "ac_full_map_channel_mean_abs_max" in dcac,
        "ac_content_weighted_zero_mean_checked": "ac_content_weighted_channel_mean_abs_max" in dcac,
        "post_projection_uses_a3_forward": "forward_with_custom_residuals" in evaluator,
        "donor_cache_no_gate": "build_residual_cache_no_gate" in evaluator and "DONOR_CACHE_GATE" in evaluator,
        "standalone_block_present": '"standalone"' in evaluator,
        "conditional_block_present": '"conditional"' in evaluator,
        "paired_effect_native_minus_donor": "paired_effect_native_minus_donor" in evaluator,
        "centering_rescue_separate": "centering_rescue_native_ac_minus_full_native" in evaluator,
        "ac_content_diagnostic_only": '"diagnostic_only": True' in evaluator and '"training_go_allowed": False' in evaluator,
        "factorial_eight_cells": "FACTORIAL_CELLS" in evaluator and "C000" in decision and "C111" in decision,
        "factorial_c000_native_bitwise": "C000_NATIVE" in evaluator and "detection_sha256" in evaluator,
        "factorial_three_way": "I345" in decision,
        "p5_joint_decision_only_ac_all": "joint_p5_decision(paired_labels[\"P5\"], rescue_labels[\"P5\"])" in evaluator,
        "content_absent_from_joint_decision": "AC_CONTENT is intentionally absent" in decision,
        "paired_positive_not_utility": "ac_utility_is_not_ac_paired_causality" in evaluator,
        "rescue_not_paired": "centering_rescue_is_not_paired_restoration" in evaluator,
        "dc_not_projection_bias": "dc_harm_is_not_projection_bias_parameter_harm" in evaluator,
        "p5_primary_p3_secondary_p4_control": "p5_primary_p3_secondary_p4_control" in evaluator,
        "last_pt_only": "best.pt" not in evaluator and 'weights/last.pt' in evaluator,
        "no_dataset_level_shuffle": "aux_id_map" not in evaluator,
        "no_dataset_aux_zero": "aux_zero" not in evaluator,
        "all_g1_g14_present": all(f'"G{i}_' in evaluator for i in range(1, 15)),
        "g7_dynamic": '"G7_donor_ac_self_centering": donor_self_ok' in evaluator,
        "g8_dynamic": '"G8_full_map_dc_ac_semantics": dc_ac_ok' in evaluator,
        "g9_dynamic": '"G9_content_mask_provenance": content_prov_ok' in evaluator,
        "g10_dynamic": '"G10_content_dc_coverage": content_coverage_ok' in evaluator,
        "g11_dynamic": '"G11_post_projection_intervention": post_projection_ok' in evaluator,
        "g12_dynamic": '"G12_factorial_completeness": factorial_ok' in evaluator,
        "g14_dynamic": '"G14_provenance_complete": bool(provenance_ok)' in evaluator,
        "all_gates_runtime_abort": "if not all(gates.values())" in evaluator and "A4_ABORT" in evaluator,
        "transactional_output": "commit_json_bundle" in evaluator and ".tmp" in evaluator,
        "summary_unique_decision_entry": 'a4_summary.json' in evaluator,
        "design_evaluation_only": "EVALUATION-ONLY" in design,
        "design_ac_utility_distinction": "AC utility" in design and "AC paired causality" in design,
        "design_content_diagnostic_only": "diagnostic-only" in design.lower(),
        "design_forbids_training": "NO training" in design,
        "design_forbids_q3": "NO q3/q4/q5" in design,
        "design_forbids_recipient_mean_for_donor": "NO donor residual centered by recipient mean" in design,
        "design_forbids_gt_content_mask": "NO GT-derived content mask" in design,
        "design_forbids_projection_bias_claim": "NO calling DC harm" in design,
    })

    for key, value in checks.items():
        if value is not True:
            errors.append(key)
    provenance = {key: sha256_file(path) for key, path in FILES.items()}
    obj = {
        "schema": "step4-a4-preexecution-audit-v1",
        "checks": checks,
        "errors": errors,
        "provenance": provenance,
        "all_passed": not errors,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(json.dumps(obj, indent=2))
    if errors:
        raise SystemExit(f"A4_AUDIT_FAIL:{errors}")


if __name__ == "__main__":
    main()
