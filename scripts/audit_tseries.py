#!/usr/bin/env python3
"""Static/formal pretraining audit for T-series."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CONTRACT = "D:/pycharm/Python Develop/YOLO_1/step3_data_contract.json"

# These constants are duplicated deliberately so `--phase static` remains a true
# package-only audit with no dependency on the host repository.
A5_ACCEPTED_COMMIT = "f154c1ff9af6d31e60bc2c9a2c4fd5baafc3d8b8"
A5_SUMMARY_RAW_SHA256 = "f1dbd1bc828b55674406337a12add25dc0a1cdd3ee96ad46c3c9976014cb7950"
A5_SUMMARY_CANONICAL_LF_SHA256 = "0e3ebb5cc64362ee44a0de68899885f36842ac8f8c40556d26cd541002347915"
A4_ORIGINAL_FEEDBACK_SHA256 = "3bd2331d3e618f280b6c8a67699a93780aef1806a09c86bdad2b88ece8dd434a"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical_lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()

SCHEMA = "step4-tseries-pretraining-audit-v1"

SOURCE_PATHS = {
    "design_sha256": "docs/step4_tseries/TRAINING_DESIGN_FREEZE.md",
    "model_sha256": "src/multimodal/tseries_p5_model.py",
    "core_sha256": "src/multimodal/tseries_core.py",
    "runtime_sha256": "src/multimodal/tseries_runtime.py",
    "runner_sha256": "scripts/run_tseries.py",
    "suite_sha256": "scripts/smoke_tseries_suite.py",
    "audit_sha256": "scripts/audit_tseries.py",
    "posttrain_eval_sha256": "scripts/eval_tseries_posttrain.py",
    "paired_eval_sha256": "scripts/eval_tseries_paired.py",
    "formal_suite_sha256": "scripts/run_tseries_formal_suite.py",
    "summary_sha256": "scripts/summarize_tseries.py",
    "implementation_adjudication_sha256": "docs/step4_tseries/IMPLEMENTATION_ADJUDICATION.md",
    "tests_sha256": "tests/test_tseries.py",
    "readme_sha256": "T_SERIES_README.md",
}

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def add(checks, name, passed, detail=None):
    row = {"passed": bool(passed)}
    if detail is not None:
        row["detail"] = detail
    checks[name] = row

def _git_is_ancestor(commit: str) -> bool:
    try:
        p = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return p.returncode == 0
    except Exception:
        return False

def package_checks() -> dict:
    checks = {}
    design = (ROOT / SOURCE_PATHS["design_sha256"]).read_text(encoding="utf-8")
    model = (ROOT / SOURCE_PATHS["model_sha256"]).read_text(encoding="utf-8")
    runner = (ROOT / SOURCE_PATHS["runner_sha256"]).read_text(encoding="utf-8")
    paired = (ROOT / SOURCE_PATHS["paired_eval_sha256"]).read_text(encoding="utf-8")
    post = (ROOT / SOURCE_PATHS["posttrain_eval_sha256"]).read_text(encoding="utf-8")
    impl = (ROOT / SOURCE_PATHS["implementation_adjudication_sha256"]).read_text(encoding="utf-8")
    summary = (ROOT / SOURCE_PATHS["summary_sha256"]).read_text(encoding="utf-8")

    predicates = {
        "S01_design_frozen": "TRAINING DESIGN FROZEN" in design,
        "S02_p5_only_term": "P5-only direct IR injection" in design,
        "S03_no_p3_direct": "NO P3 direct IR injection" in design,
        "S04_no_p4_direct": "NO P4 direct IR injection" in design,
        "S05_no_gate": "NO reliability gate" in design,
        "S06_no_depth": "NO Depth" in design,
        "S07_t0_present": "T0-N" in design,
        "S08_t1_present": "T1-F" in design,
        "S09_t2_present": "T2-A" in design,
        "S10_ac_all_primary": "AC_ALL" in design,
        "S11_no_ac_content": "NO AC_CONTENT" in design,
        "S12_g18_present": "G18" in design,
        "S13_model_has_one_p5_fusion": "self.p5_fusion = ZeroInitResidualFusion" in model,
        "S14_model_no_reliability_gate": "reliability_gate" not in model,
        "S15_model_p3_count_zero": '"p3_direct_injection_count": 0' in model,
        "S16_model_p4_count_zero": '"p4_direct_injection_count": 0' in model,
        "S17_model_p5_count_one": '"p5_direct_injection_count": 1' in model,
        "S18_model_y10_handoff": "y[P5_TAP] = fused5" in model,
        "S19_model_x_y10_handoff": "x = y[P5_TAP]" in model,
        "S20_t0_no_grad": "with torch.no_grad()" in model,
        "S21_t0_direct_r5": 'fused5 = r5 if self.treatment_id == "T0-N"' in model,
        "S22_t0_no_zero_mul": "0.0 * delta" not in model and "0 * delta" not in model,
        "S23_t2_center_function": "center_full_map(delta)" in model,
        "S24_bias_true_inherited": "ZeroInitResidualFusion" in model,
        "S25_runner_explicit_weights": "build_tseries_model(Path(a.base_checkpoint)" in runner,
        "S26_runtime_explicit_builder_weights": 'build_reference_3ch(weights=str(base_checkpoint))' in (
            ROOT / "src/multimodal/tseries_runtime.py").read_text(encoding="utf-8"),
        "S27_runner_formal_80": "FORMAL_EPOCHS" in runner,
        "S28_runner_formal_seed": "FORMAL_SEED" in runner,
        "S29_runner_formal_batch": "FORMAL_BATCH" in runner,
        "S30_runner_refuse_existing_dir": "T_SERIES_RUN_DIR_EXISTS" in runner,
        "S31_gradient_probe": "gradient_probe" in runner,
        "S32_t2_bias_pre_optimizer_guard": "bias.grad.zero_()" in runner and "def optimizer_step" in runner,
        "S33_t2_bias_dust_logged": "t2_bias_pre_zero_grad_abs_max" in runner,
        "S34_t0_silent_update_gate": "T_SERIES_T0_SILENT_OPTIMIZER_UPDATE" in runner,
        "S35_t2_silent_bias_gate": "T_SERIES_T2_BIAS_OPTIMIZER_UPDATE" in runner,
        "S35_optimizer_manifest": "optimizer_manifest.json" in runner,
        "S36_mechanism_log": "tseries_mechanism.jsonl" in runner,
        "S37_data_order_log": "tseries_data_order.jsonl" in runner,
        "S38_paired_recipient_donor": "donor" in paired.lower() and "recipient" in paired.lower(),
        "S39_posttrain_train11": "train11" in post,
        "S40_posttrain_all17": "all17" in post,
        "S41_posttrain_late10": "late10" in post,
        "S42_bias_guard_adjudicated": "numerical-exactness guard" in impl,
        "S43_bias_guard_pre_optimizer": "immediately before the optimizer step" in impl,
        "S44_summary_no_margin": "no_arbitrary_ap_margin" in summary,
        "S45_summary_single_seed_not_replication": "single_seed_is_not_replication" in summary,
        "S46_formal_suite_exists": (ROOT / SOURCE_PATHS["formal_suite_sha256"]).exists(),
    }
    for name, passed in predicates.items():
        add(checks, name, passed)
    for key, rel in SOURCE_PATHS.items():
        add(checks, f"FILE_{key}", (ROOT / rel).exists(), rel)
    return checks

def upstream_checks(args) -> dict:
    checks = {}
    a5_path = ROOT / "reports/step4_a5/a5_summary.json"
    feedback = ROOT / "docs/step4_a4/feedback/2026-08-19_formal-review.md"
    erratum = ROOT / "docs/step4_a4/feedback/2026-08-19_erratum.md"
    add(checks, "R01_a5_summary_exists", a5_path.exists())
    if a5_path.exists():
        add(checks, "R02_a5_raw_sha", sha256_file(a5_path) == A5_SUMMARY_RAW_SHA256, sha256_file(a5_path))
        add(checks, "R03_a5_canonical_sha", canonical_lf_sha256(a5_path) == A5_SUMMARY_CANONICAL_LF_SHA256)
        a5 = load(a5_path)
        add(checks, "R04_a5_schema", a5.get("schema") == "step4-a5-summary-v1")
        add(checks, "R05_a5_all_gates", a5.get("all_gates_passed") is True)
        add(checks, "R06_a5_training_false", a5.get("training_go") is False)
        mf = a5.get("mechanism_flags") or {}
        add(checks, "R07_p3_flip", mf.get("P3_FULL_SUFFICIENT_FLIP") is True)
        add(checks, "R08_p4_flip", mf.get("P4_FULL_SUFFICIENT_FLIP") is True)
        add(checks, "R09_both_antagonistic", mf.get("BOTH_FULL_INDIVIDUALLY_SUFFICIENT") is True)
        add(checks, "R10_centering_fails", mf.get("CENTERING_FAILS_TO_RESTORE") is True)
    add(checks, "R11_a5_commit_is_ancestor", _git_is_ancestor(A5_ACCEPTED_COMMIT))
    add(checks, "R12_original_feedback_exists", feedback.exists())
    if feedback.exists():
        add(checks, "R13_original_feedback_sha", sha256_file(feedback) == A4_ORIGINAL_FEEDBACK_SHA256)
    add(checks, "R14_erratum_exists", erratum.exists())
    if erratum.exists():
        txt = erratum.read_text(encoding="utf-8")
        add(checks, "R15_erratum_1_6", "1/6 positive" in txt)
        add(checks, "R16_erratum_documentation_only", "documentation typo only" in txt.lower())

    # Host-repo imports are intentionally delayed until repo-full/formal mode.
    sys.path.insert(0, str(ROOT / "src"))
    from multimodal.raw_sample_index import CLASS_NAMES
    from multimodal.step4_f1_c_readiness import (
        EXPECTED_BASE_CHECKPOINT_SHA256, verify_base_checkpoint,
        verify_data_yaml, verify_raw_data_freshness,
    )

    contract_path = Path(args.contract)
    data_path = Path(args.data)
    base_path = Path(args.base_checkpoint)
    add(checks, "R17_contract_exists", contract_path.exists(), str(contract_path))
    if contract_path.exists():
        contract = load(contract_path)
        raw = verify_raw_data_freshness(contract)
        add(checks, "R18_raw_data_fresh", raw["passed"], raw.get("errors"))
    dy = verify_data_yaml(data_path, CLASS_NAMES)
    add(checks, "R19_data_yaml", dy["passed"], dy.get("errors"))
    bc = verify_base_checkpoint(base_path, EXPECTED_BASE_CHECKPOINT_SHA256)
    add(checks, "R20_base_checkpoint", bc["passed"], bc)
    return checks

def smoke_checks(path: Path) -> dict:
    checks = {}
    add(checks, "M01_smoke_exists", path.exists())
    if not path.exists():
        return checks
    obj = load(path)
    add(checks, "M02_smoke_schema", obj.get("schema") == "step4-tseries-pretraining-smoke-v1")
    add(checks, "M03_dynamic_passed", obj.get("all_dynamic_gates_passed") is True)
    gates = obj.get("gates") or {}
    for g in (
        "G2_identical_model_class", "G3_no_reliability_gate", "G4_p5_only_topology",
        "G5_neck_handoff", "G6_matched_initial_state", "G7_epoch0_prediction_equivalence",
        "G8_zero_init_projection", "G9_trainability_map_matched",
        "G10_optimizer_groups_matched", "G11_t0_null_loss_graph",
        "G12_t0_no_silent_optimizer_update", "G13_t2_bias_cancellation",
        "G14_t2_bias_optimizer_safety", "G15_bn_policy_matched",
        "G17_rng_data_order_closure", "G18_protocol_equality",
    ):
        add(checks, f"M_{g}", gates.get(g) is True)
    pins = obj.get("source_hashes") or {}
    smoke_source_map = {
        "model_sha256": SOURCE_PATHS["model_sha256"],
        "core_sha256": SOURCE_PATHS["core_sha256"],
        "runtime_sha256": SOURCE_PATHS["runtime_sha256"],
        "runner_sha256": SOURCE_PATHS["runner_sha256"],
        "suite_sha256": SOURCE_PATHS["suite_sha256"],
        "design_sha256": SOURCE_PATHS["design_sha256"],
    }
    for key, rel in smoke_source_map.items():
        current = sha256_file(ROOT / rel)
        add(checks, f"M_FRESH_{key}", pins.get(key) == current, {"recorded": pins.get(key), "current": current})
    return checks

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["static", "formal"], default="static")
    p.add_argument("--contract", default=DEFAULT_CONTRACT)
    p.add_argument("--data", default="D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml")
    p.add_argument("--base-checkpoint", default="E:/odin/yolo26s.pt")
    p.add_argument("--smoke-report", default="reports/step4_tseries/pretraining_smoke.json")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    checks = package_checks()
    if a.phase == "formal":
        checks.update(upstream_checks(a))
        checks.update(smoke_checks(ROOT / a.smoke_report))

    failed = [k for k, v in checks.items() if not v["passed"]]
    source_hashes = {k: sha256_file(ROOT / rel) for k, rel in SOURCE_PATHS.items()}
    report = {
        "schema": SCHEMA,
        "phase": a.phase,
        "all_passed": not failed,
        "passed_count": len(checks) - len(failed),
        "total_count": len(checks),
        "failed": failed,
        "checks": checks,
        "source_hashes": source_hashes,
        "gates": ({
            "G1": all(checks.get(x, {}).get("passed") for x in (
                "R05_a5_all_gates", "R07_p3_flip", "R08_p4_flip", "R10_centering_fails"
            )),
            "G2": checks.get("M_G2_identical_model_class", {}).get("passed") is True,
            "G3": checks.get("M_G3_no_reliability_gate", {}).get("passed") is True,
            "G4": checks.get("M_G4_p5_only_topology", {}).get("passed") is True,
            "G5": checks.get("M_G5_neck_handoff", {}).get("passed") is True,
            "G6": checks.get("M_G6_matched_initial_state", {}).get("passed") is True,
            "G7": checks.get("M_G7_epoch0_prediction_equivalence", {}).get("passed") is True,
            "G8": checks.get("M_G8_zero_init_projection", {}).get("passed") is True,
            "G9": checks.get("M_G9_trainability_map_matched", {}).get("passed") is True,
            "G10": checks.get("M_G10_optimizer_groups_matched", {}).get("passed") is True,
            "G11": checks.get("M_G11_t0_null_loss_graph", {}).get("passed") is True,
            "G12": checks.get("M_G12_t0_no_silent_optimizer_update", {}).get("passed") is True,
            "G13": checks.get("M_G13_t2_bias_cancellation", {}).get("passed") is True,
            "G14": checks.get("M_G14_t2_bias_optimizer_safety", {}).get("passed") is True,
            "G15": checks.get("M_G15_bn_policy_matched", {}).get("passed") is True,
            "G16": checks.get("R20_base_checkpoint", {}).get("passed") is True,
            "G17": checks.get("M_G17_rng_data_order_closure", {}).get("passed") is True,
            "G18": checks.get("M_G18_protocol_equality", {}).get("passed") is True,
        } if a.phase == "formal" else {}),
    }
    if a.phase == "formal" and not all(report["gates"].values()):
        report["all_passed"] = False
        for gate, passed in report["gates"].items():
            if not passed:
                report["failed"].append(f"FORMAL_GATE:{gate}")
        report["failed"] = sorted(set(report["failed"]))
        report["passed_count"] = report["total_count"] - len(
            [x for x in report["failed"] if not x.startswith("FORMAL_GATE:")]
        )

    if a.out is None:
        a.out = (
            "reports/step4_tseries/pretraining_audit.json"
            if a.phase == "formal"
            else "reports/step4_tseries/pretraining_static_audit.json"
        )
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "phase": a.phase,
        "all_passed": report["all_passed"],
        "passed": report["passed_count"],
        "total": report["total_count"],
        "failed": failed,
        "out": str(out),
    }, indent=2))
    if failed:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
