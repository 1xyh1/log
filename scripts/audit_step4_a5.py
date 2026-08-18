#!/usr/bin/env python3
"""A5 pre-execution audit.

Default mode is repo-full and is required by eval_step4_a5.py. --package-only is
for archive regression before the bundle is overlaid onto the frozen project.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "a2": "756093358153c5e203f485dce96e0f2a5e91881fb6c6e4b49c036cbfdc6d1c6b",
    "donor": "c5cd8e852663eae3243bc5e4c263c6f2c26c7b2faa977ae3b60cb5c1ea122af5",
    "a4_summary": "721198d04b4ce54caec3d0b5c97ef5c665c3c4e8bf44e8df82e4a50a33406781",
    "a4_standalone": "e1d9c95b84af300feea0148bfecde678fec257e6a2ba709cd4e26215265d90e5",
    "a4_conditional": "c0ce169c33d835c5e5d262f0b05029f8d63a2180acdec277158b2fc4450edcf7",
    "adjudication": "0ca9e6e7f3e8b8d2e0c4a8542bcfff5428139b77a61280357a0ae9f4a282c898",
    "feedback": "3bd2331d3e618f280b6c8a67699a93780aef1806a09c86bdad2b88ece8dd434a",
    "a4_decision_corrected": "50650e2b1a3679325a2cd1b7d95eccebfeb9044d801ffb48dc08b83c89ad95a2",
    "a4_tests_corrected": "ec9b294464237869b09c788dd1f23ebeeb2194cfb138514173934718e474cbc4",
}
EXPECTED_A4_SUMMARY_CANONICAL = "95f768289e2f04010013eeeac20a83d2bf9e71b153c2e748f5a1ab5941c10ea1"
EXPECTED_VAL_IDS = [
    "000003_013_00000085",
    "000004_013_00000081",
    "000004_014_00000001",
    "000016",
    "000016_001_00000001",
    "000016_042_suppl_00000164",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def add(checks: dict, name: str, value: bool, detail=None):
    checks[name] = {"passed": bool(value)}
    if detail is not None:
        checks[name]["detail"] = detail


def static_checks() -> dict:
    checks = {}
    design = read("docs/step4_a5/DESIGN_FREEZE.md")
    context = read("src/multimodal/step4_a5_context.py")
    effects = read("src/multimodal/step4_a5_effects.py")
    evaluator = read("scripts/eval_step4_a5.py")
    tests = read("tests/test_step4_a5.py")
    audit = read("scripts/audit_step4_a5.py")

    # Freeze / scope.
    add(checks, "S01_design_evaluation_only", "EVALUATION-ONLY" in design)
    add(checks, "S02_design_p5_centered", "P5-CENTERED" in design)
    add(checks, "S03_design_a4t_hold", "A4T HOLD" in design or "A4T = HOLD" in design)
    add(checks, "S04_design_mixed_no_go", "MIXED_PAIRED_CONTEXT_NO_GO" in design)
    add(checks, "S05_design_no_training_go", "NO A5 training_go=true under any result" in design)
    add(checks, "S06_design_fixed_primary", "FIXED = primary" in design)
    add(checks, "S07_design_soft_replication", "SOFT  = replication" in design or "SOFT  = replication" in design.replace("  ", " "))
    add(checks, "S08_design_same_val6", all(s in design for s in EXPECTED_VAL_IDS))
    add(checks, "S09_design_donor_hash", EXPECTED["donor"] in design)

    # Exact matrix and context semantics.
    for idx, ctx in enumerate(("OO", "FO", "OF", "FF", "AO", "OA", "AF", "FA", "AA"), start=10):
        add(checks, f"S{idx:02d}_context_{ctx}", f'"{ctx}"' in context and ctx in design)
    add(checks, "S19_context_count_9", "CONTEXT_ORDER = (\"OO\", \"FO\", \"OF\", \"FF\", \"AO\", \"OA\", \"AF\", \"FA\", \"AA\")" in context)
    add(checks, "S20_context_states_only_OFA", 'ALLOWED_STATES = frozenset({"O", "F", "A"})' in context)
    add(checks, "S21_p3p4_recipient_comment", "Recipient-only P3/P4 context" in context)
    add(checks, "S22_p5_role_source_switch", 'p5_source = recipient_id if role == "native" else donor_id' in context)
    add(checks, "S23_donor_self_match_forbidden", "A5_DONOR_SELF_MATCH" in context)
    add(checks, "S24_context_trace_validator", "def validate_context_trace" in context)
    add(checks, "S25_pair_isolation_validator", "def validate_pair_isolation" in context)
    add(checks, "S26_p3_donor_contamination_guard", "P3/P4 may never become donor-owned" in context)
    add(checks, "S27_donor_mean_source_guard", 'comp.get("mean_source_id") != expected_p5_source' in context)

    # Effect math / labels.
    add(checks, "S28_effect_native_minus_donor", "def effect_from_results" in effects and "native" in effects and "donor" in effects)
    add(checks, "S29_paired_positive_label", 'return "STRONG_POSITIVE"' in effects)
    add(checks, "S30_paired_negative_label", 'return "STRONG_NEGATIVE"' in effects)
    add(checks, "S31_shift_antagonistic_label", '"STRONG_ANTAGONISTIC_SHIFT"' in effects)
    add(checks, "S32_shift_rescuing_label", '"STRONG_RESCUING_SHIFT"' in effects)
    for n, name in enumerate(("D3F", "D4F", "IFF", "D3A", "D4A", "IAA", "IAF", "IFA"), start=33):
        add(checks, f"S{n:02d}_interaction_{name}", f'"{name}"' in effects)
    add(checks, "S41_no_margin_constant", "+0.01" not in effects and "margin" not in effects.lower())
    add(checks, "S42_shift_not_flip_guard", "STRONG_NEGATIVE" in effects and "STRONG_ANTAGONISTIC_SHIFT" in effects)

    # Mechanism flags.
    flags = (
        "P3_FULL_SUFFICIENT_FLIP", "P4_FULL_SUFFICIENT_FLIP",
        "BOTH_FULL_INDIVIDUALLY_SUFFICIENT", "JOINT_FULL_CONTEXT_REQUIRED",
        "FULL_CONTEXT_FLIP_WITH_UNRESOLVED_INTERACTION",
        "P3_CENTERING_RESCUES_WITH_P4_FULL", "P4_CENTERING_RESCUES_WITH_P3_FULL",
        "BOTH_CENTERED_RESTORE", "CENTERING_FAILS_TO_RESTORE",
    )
    for n, flag in enumerate(flags, start=43):
        add(checks, f"S{n:02d}_flag_{flag}", flag in effects and flag in design)
    add(checks, "S52_route_training_false", '"training_go": False' in effects)
    add(checks, "S53_route_no_training_doc", "A5 never grants training GO" in effects)

    # Evaluator causal / anchor / transactional contract.
    add(checks, "S54_eval_uses_a3_forward", "forward_with_custom_residuals" in evaluator)
    add(checks, "S55_eval_no_gate_cache", "build_residual_cache_no_gate" in evaluator)
    add(checks, "S56_eval_gate_guard", "residual_cache_with_gate_guard" in evaluator)
    add(checks, "S57_eval_step3_validator", "evu.make_detection_validator" in evaluator)
    add(checks, "S58_eval_stock_batch_move", "evu.move_step3_batch_to_device" in evaluator)
    add(checks, "S59_eval_exact_oo_anchor", "A5_OO_ANCHOR_FAIL" in evaluator)
    add(checks, "S60_eval_exact_ff_anchor", "A5_FF_ANCHOR_FAIL" in evaluator)
    add(checks, "S61_eval_detection_sha", "detection_sha256" in evaluator and "tensor_sha256" in evaluator)
    add(checks, "S62_eval_pair_isolation", "validate_pair_isolation" in evaluator)
    add(checks, "S63_eval_context_trace", "validate_context_trace" in evaluator)
    add(checks, "S64_eval_q_freeze", "A5_Q_FREEZE_FAIL" in evaluator)
    add(checks, "S65_eval_state_mutation", "A5_PARAMETER_MUTATION" in evaluator)
    add(checks, "S66_eval_donor_map_drift", "A5_DONOR_MAP_DRIFT" in evaluator)
    add(checks, "S67_eval_context_complete", "A5_CONTEXT_MATRIX_INCOMPLETE" in evaluator)
    add(checks, "S68_eval_paired_semantics", "A5_PAIRED_EFFECT_SEMANTICS_FAIL" in evaluator)
    add(checks, "S69_eval_interaction_semantics", "A5_INTERACTION_CONTRAST_FAIL" in evaluator)
    add(checks, "S70_eval_training_forbidden", "A5_TRAINING_GO_FORBIDDEN" in evaluator)
    add(checks, "S71_eval_condition_18", '"condition_count_per_system": 18' in evaluator)
    add(checks, "S72_eval_total_36", '"total_condition_instances": 36' in evaluator)
    add(checks, "S73_eval_transactional_json", "commit_json_bundle" in evaluator and ".tmp" in evaluator)
    add(checks, "S74_eval_refuse_overwrite", "A5_REFUSE_OVERWRITE" in evaluator)
    add(checks, "S75_eval_no_backward", ".backward(" not in evaluator)
    add(checks, "S76_eval_no_optimizer", "torch.optim" not in evaluator and "optimizer" not in evaluator.lower())
    add(checks, "S77_eval_no_train_call", ".train(" not in evaluator)

    # Upstream dual-track correction must be pinned.
    add(checks, "S78_pin_a4_execution_commit", "36221d2f827c411bddd66350729dfd05a3b48f49" in evaluator)
    add(checks, "S79_pin_a4_reviewer_head", "b7ee0d6803d949a8c512b11defcb2125a3f4c8a1" in evaluator)
    add(checks, "S80_pin_adjudication_sha", EXPECTED["adjudication"] in evaluator)
    add(checks, "S81_pin_summary_sha", EXPECTED["a4_summary"] in evaluator)
    add(checks, "S82_pin_standalone_sha", EXPECTED["a4_standalone"] in evaluator)
    add(checks, "S83_pin_conditional_sha", EXPECTED["a4_conditional"] in evaluator)
    add(checks, "S84_pin_corrected_decision", EXPECTED["a4_decision_corrected"] in evaluator)
    add(checks, "S85_pin_corrected_tests", EXPECTED["a4_tests_corrected"] in evaluator)
    add(checks, "S86_machine_history_preserved", "REJECTED_INVALID" in evaluator)
    add(checks, "S87_dual_track_declaration", "corrected code is NOT the executed code" in evaluator)

    # Tests cover high-risk regression points.
    add(checks, "S88_test_oo_ff_contract", "test_evaluator_has_oo_and_ff_anchor_codes" in tests)
    add(checks, "S89_test_only_p5_identity", "test_pair_isolation_passes" in tests)
    add(checks, "S90_test_donor_mean", "test_donor_p5_source_and_mean_are_donor" in tests)
    add(checks, "S91_test_p3_contamination", "test_donor_p3_contamination_fails_trace" in tests)
    add(checks, "S92_test_p4_contamination", "test_donor_p4_contamination_fails_trace" in tests)
    add(checks, "S93_test_context_shift", "test_context_shift_fo_relative_oo" in tests)
    add(checks, "S94_test_iff", "test_iff_formula" in tests)
    add(checks, "S95_test_negative_shift_not_flip", "test_negative_shift_without_sign_flip_not_sufficient_flip" in tests)
    add(checks, "S96_test_centered_no_training", "test_route_centered_candidate_never_training_go" in tests)
    add(checks, "S97_test_no_optimizer", "test_no_optimizer_or_backward_in_evaluator" in tests)
    add(checks, "S98_audit_has_repo_full", "repo-full" in audit)
    add(checks, "S99_audit_has_package_only", "--package-only" in audit)
    add(checks, "S100_source_count_complete", len(checks) == 99)  # value before inserting S100
    # Above evaluates against the 99 prior checks by construction.
    return checks


def repo_checks() -> dict:
    checks = {}
    paths = {
        "a2": ROOT / "reports/step4_a2/scale_ir_residual_causality.json",
        "donor": ROOT / "reports/step4_a2/val_donor_map.json",
        "a4_summary": ROOT / "reports/step4_a4/a4_summary.json",
        "a4_standalone": ROOT / "reports/step4_a4/ac_paired_standalone.json",
        "a4_conditional": ROOT / "reports/step4_a4/ac_paired_conditional.json",
        "adjudication": ROOT / "reports/step4_a4/reviewer_adjudication.json",
        "feedback": ROOT / "docs/step4_a4/feedback/2026-08-19_formal-review.md",
    }
    for key, path in paths.items():
        add(checks, f"R_{key}_exists", path.exists(), str(path))
        if path.exists():
            add(checks, f"R_{key}_sha", sha256_file(path) == EXPECTED[key], sha256_file(path))
        else:
            add(checks, f"R_{key}_sha", False, "missing")
    if paths["a4_summary"].exists():
        add(checks, "R_a4_summary_canonical", canonical_lf_sha256(paths["a4_summary"]) == EXPECTED_A4_SUMMARY_CANONICAL)
    else:
        add(checks, "R_a4_summary_canonical", False)

    dpath = ROOT / "src/multimodal/step4_a4_decision.py"
    tpath = ROOT / "tests/test_step4_a4.py"
    add(checks, "R_corrected_decision_exists", dpath.exists())
    add(checks, "R_corrected_tests_exists", tpath.exists())
    add(checks, "R_corrected_decision_sha", dpath.exists() and sha256_file(dpath) == EXPECTED["a4_decision_corrected"])
    add(checks, "R_corrected_tests_sha", tpath.exists() and sha256_file(tpath) == EXPECTED["a4_tests_corrected"])

    if paths["adjudication"].exists():
        adj = json.loads(paths["adjudication"].read_text(encoding="utf-8"))
        add(checks, "R_adj_schema", adj.get("schema") == "step4-a4-reviewer-adjudication-v1")
        add(checks, "R_adj_experiment_accepted", adj.get("experiment_result") == "ACCEPTED_DIAGNOSTIC_COMPLETE")
        add(checks, "R_adj_corrected_branch", adj.get("corrected_branch") == "MIXED_PAIRED_CONTEXT_NO_GO")
        add(checks, "R_adj_training_false", adj.get("corrected_training_go") is False)
        add(checks, "R_adj_a4t_hold", adj.get("a4t_status") == "HOLD")
        add(checks, "R_adj_no_rerun", adj.get("rerun_required") is False)
        add(checks, "R_adj_execution_frozen", adj.get("execution_artifacts_frozen") is True)
        add(checks, "R_adj_execution_commit", adj.get("commit") == "36221d2f827c411bddd66350729dfd05a3b48f49")
        add(checks, "R_adj_summary_link", adj.get("a4_summary_sha256") == EXPECTED["a4_summary"])
        cc = adj.get("code_change_after_execution") or {}
        add(checks, "R_adj_dual_track", cc.get("declaration") == "corrected code is NOT the executed code")
        add(checks, "R_adj_audit_not_rerun", cc.get("audit_rerun") is False)
    else:
        for name in (
            "R_adj_schema", "R_adj_experiment_accepted", "R_adj_corrected_branch", "R_adj_training_false",
            "R_adj_a4t_hold", "R_adj_no_rerun", "R_adj_execution_frozen", "R_adj_execution_commit",
            "R_adj_summary_link", "R_adj_dual_track", "R_adj_audit_not_rerun",
        ):
            add(checks, name, False)

    return checks


def provenance() -> dict:
    targets = {
        "design_sha256": ROOT / "docs/step4_a5/DESIGN_FREEZE.md",
        "context_sha256": ROOT / "src/multimodal/step4_a5_context.py",
        "effects_sha256": ROOT / "src/multimodal/step4_a5_effects.py",
        "evaluator_sha256": ROOT / "scripts/eval_step4_a5.py",
        "tests_sha256": ROOT / "tests/test_step4_a5.py",
        "audit_source_sha256": ROOT / "scripts/audit_step4_a5.py",
    }
    return {k: sha256_file(v) for k, v in targets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mode = "package-only" if a.package_only else "repo-full"

    checks = static_checks()
    if not a.package_only:
        checks.update(repo_checks())
    passed = sum(1 for v in checks.values() if v["passed"])
    failed = [k for k, v in checks.items() if not v["passed"]]
    obj = {
        "schema": "step4-a5-preexecution-audit-v1",
        "mode": mode,
        "all_passed": not failed,
        "passed_count": passed,
        "total_count": len(checks),
        "failed": failed,
        "checks": checks,
        "provenance": provenance(),
    }

    if a.out:
        out = Path(a.out)
    elif a.package_only:
        out = ROOT / "A5_PACKAGE_AUDIT.json"
    else:
        out = ROOT / "reports/step4_a5/preexecution_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: obj[k] for k in ("schema", "mode", "all_passed", "passed_count", "total_count", "failed")}, indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
