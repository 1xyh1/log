#!/usr/bin/env python3
"""Pre-execution static gate for A2 Scale-wise IR Residual Causality Audit v2."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.step4_a2_residual_interventions import (  # noqa: E402
    MASK_CONDITIONS, SCALES, gain_conditions, mask_conditions, shuffle_conditions,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    design = ROOT / "docs/step4_a2/DESIGN_FREEZE.md"
    engine = ROOT / "src/multimodal/step4_a2_residual_interventions.py"
    evaluator = ROOT / "scripts/eval_step4_a2_scale_causality.py"
    tests = ROOT / "tests/test_step4_a2_residual_causality.py"
    files = [design, engine, evaluator, tests]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise RuntimeError(f"A2_AUDIT_MISSING:{missing}")

    eval_src = evaluator.read_text(encoding="utf-8")
    engine_src = engine.read_text(encoding="utf-8")
    forbidden_calls = []
    forbidden_names = {"backward", "step", "zero_grad"}
    for source_name, source in (("evaluator", eval_src), ("engine", engine_src)):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else (
                    fn.id if isinstance(fn, ast.Name) else None)
                if name in forbidden_names:
                    forbidden_calls.append(f"{source_name}:{name}")
    checks = {
        "design_exists": design.exists(),
        "engine_exists": engine.exists(),
        "evaluator_exists": evaluator.exists(),
        "tests_exist": tests.exists(),
        "eight_masks_exact": tuple(c.name for c in mask_conditions()) == MASK_CONDITIONS,
        "three_scales_exact": SCALES == ("P3", "P4", "P5"),
        "shuffle_six_conditions": len(shuffle_conditions()) == 6,
        "fixed_gain_15_conditions": len(gain_conditions(include_native=False)) == 15,
        "soft_gain_18_conditions": len(gain_conditions(include_native=True)) == 18,
        "no_train_optimizer_backward_calls": not forbidden_calls,
        "last_pt_only": '"last.pt"' in eval_src and '"best.pt"' not in eval_src,
        "no_dataset_level_aux_map": "aux_id_map" not in eval_src,
        "no_dataset_level_aux_zero": "aux_zero" not in eval_src,
        "q_before_shuffle_in_engine": (
            engine_src.index("q_native = model._effective_gate")
            < engine_src.index('if condition.kind in {"shuffle_cond", "shuffle_only"}')
        ),
        "donor_cache_uses_projection_not_gate": (
            "project_native_residuals(model, x_aux)" in engine_src
            and "build_residual_cache" in engine_src
        ),
        "native_equivalence_probe_present": "native_equivalence_probe" in eval_src,
        "state_sha_before_after_present": (
            "state_sha256_before" in eval_src and "state_sha256_after" in eval_src
        ),
        "q_freeze_runtime_gate_present": "_assert_q_freeze" in eval_src,
        "fixed_primary_soft_replication": (
            '"FIXED"' in eval_src and '"SOFT"' in eval_src
            and '"primary" if tag == "FIXED" else "replication"' in eval_src
        ),
        "frozen_summary_sha_pinned": (
            "EXPECTED_F1C_SUMMARY_SHA256" in eval_src
            and "d4e64b86e221b102143bd98cc6056f8e84d7913680cad3c8c5826af4cf88942f" in eval_src
        ),
        "frozen_dependency_closure_present": "verify_frozen_dependency_closure" in eval_src,
        "contract_freshness_enforced": "CONTRACT_STALE" in eval_src,
        "source_freshness_enforced": "SOURCE_STALE" in eval_src,
        "version_freshness_enforced": "VERSION_STALE" in eval_src,
        "val6_identity_frozen": "A2_VAL_SET_DRIFT" in eval_src and "FROZEN_VAL_IDS_COUNT" in eval_src,
        "manifest_freshness_enforced": "MANIFEST_STALE" in eval_src,
        "audit_freshness_consumed": "verify_preexecution_audit" in eval_src and "A2_PREEXECUTION_AUDIT_STALE" in eval_src,
        "donor_map_exact_deterministic": (
            "expected_donor_map = bijective_derangement(val_ids)" in eval_src
            and "A2_DONOR_MAP_NOT_FROZEN_DETERMINISTIC" in eval_src
        ),
        "causality_source_in_provenance": '"causality_interventions_sha256"' in eval_src,
        "raw_index_source_in_provenance": '"raw_sample_index_sha256"' in eval_src,
        "g8_not_unconditional_true": '"G8_stock_validator_semantics": True' not in eval_src,
        "g9_not_unconditional_true": '"G9_provenance_complete": True' not in eval_src,
    }
    report = {
        "schema": "step4-a2-preexecution-audit-v2",
        "checks": checks,
        "errors": [k for k, v in checks.items() if not v],
        "provenance": {
            "design_sha256": sha(design),
            "engine_sha256": sha(engine),
            "evaluator_sha256": sha(evaluator),
            "tests_sha256": sha(tests),
            "audit_source_sha256": sha(Path(__file__)),
        },
    }
    report["all_passed"] = not report["errors"]
    out = ROOT / "reports/step4_a2/preexecution_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["all_passed"]:
        raise RuntimeError(f"A2_PREEXECUTION_AUDIT_FAIL:{report['errors']}")


if __name__ == "__main__":
    main()
