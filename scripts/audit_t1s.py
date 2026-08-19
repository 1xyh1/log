#!/usr/bin/env python3
"""Static/formal preexecution audit for T1-S source-specificity evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = "D:/pycharm/Python Develop/YOLO_1/step3_data_contract.json"

SCHEMA = "step4-t1s-preexecution-audit-v1"
T_SERIES_ACCEPTED_COMMIT = "1d318d0bfcbd9f6e2ebd88870c9ea571f984be2c"
EXPECTED_PERFORMANCE_SHA256 = "4b38f3ef8d7defac4b91b42e8651b667ca8320e990666af225dea4e6886ea93a"
EXPECTED_PAIRED_SHA256 = "249702b5542055d7bf36595b28cb1e5a0b42312d05c55d459a8b0890e3a2b5cd"
EXPECTED_T1_MANIFEST_SHA256 = "081afec392d96ee2d570a3424e5f015f05ee308297daed8900ece5584c707312"
EXPECTED_T1_LAST_SHA256 = "8380e21504fabd0d8c3715398739bbb0bed5aaafd9c822dfc14c9503af2daeee"
EXPECTED_DONOR_MAP_SHA256 = "c5cd8e852663eae3243bc5e4c263c6f2c26c7b2faa977ae3b60cb5c1ea122af5"
EXPECTED_TSERIES_SOURCE_SHA256 = {
    "src/multimodal/tseries_p5_model.py": "987c041d80d85cfc18626febad308529e927944d02cfea4bdaf9d0d63cc2af0d",
    "src/multimodal/tseries_core.py": "9f8f3ae1eade6feee5688fb13a3d91e1cfc33805fe71c3a3b5981e18df1863ec",
    "src/multimodal/tseries_runtime.py": "5daf47bdff07afdc1fe4e6790143d4f363d175b61d866a6f7d1ac4ec0b14230a",
    "scripts/eval_tseries_paired.py": "ebdf65012f21786c6f7e083f4fd74b41055f44a11e2f0b9b547632a410468d3e",
    "scripts/summarize_tseries.py": "3ae8d0d6da70074e744c096485214dd25da20c2fce7d09e10df9ccad17bd91a6",
}
VAL6_IDS = (
    "000003_013_00000085",
    "000004_013_00000081",
    "000004_014_00000001",
    "000016",
    "000016_001_00000001",
    "000016_042_suppl_00000164",
)
SOURCE_PATHS = {
    "design_sha256": "docs/step4_t1s/DESIGN_FREEZE.md",
    "core_sha256": "src/multimodal/t1s_source_specificity.py",
    "evaluator_sha256": "scripts/eval_t1s_source_specificity.py",
    "audit_sha256": "scripts/audit_t1s.py",
    "tests_sha256": "tests/test_t1s.py",
    "readme_sha256": "T1S_README.md",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def add(checks: dict, name: str, passed, detail=None) -> None:
    row = {"passed": bool(passed)}
    if detail is not None:
        row["detail"] = detail
    checks[name] = row


def static_checks() -> dict:
    checks = {}
    design = (ROOT / SOURCE_PATHS["design_sha256"]).read_text(encoding="utf-8")
    core = (ROOT / SOURCE_PATHS["core_sha256"]).read_text(encoding="utf-8")
    evaluator = (ROOT / SOURCE_PATHS["evaluator_sha256"]).read_text(encoding="utf-8")
    predicates = {
        "S01_design_frozen": "DESIGN FROZEN" in design,
        "S02_eval_only": "EVALUATION-ONLY" in design,
        "S03_t1_only": "T1-F_P5_FULL_seed20260812" in design,
        "S04_full_residual": "P5 FULL post-projection residual" in design,
        "S05_matrix_6x6": "6 recipients × 6 sources = 36" in design,
        "S06_zero": "ZERO residual condition" in design,
        "S07_265": "!6 = 265" in design,
        "S08_native_anchor": "T1S_NATIVE_ANCHOR_FAIL" in design,
        "S09_donor_anchor": "T1S_FIXED_DONOR_ANCHOR_FAIL" in design,
        "S10_alpha": "alpha = 0.05" in design,
        "S11_no_training": "NO model training" in design,
        "S12_no_depth": "NO Depth" in design,
        "S13_core_permutations": "permutations" in core,
        "S14_core_expected_265": "EXPECTED_DERANGEMENTS = 265" in core,
        "S15_core_priority_wrong_source": 'branch = "WRONG_SOURCE_TYPICALLY_OUTPERFORMS_NATIVE"' in core,
        "S16_core_zero_branch": 'branch = "INFERENCE_RESIDUAL_NOT_SUPPORTED_TRAINING_DYNAMICS_CANDIDATE"' in core,
        "S17_core_source_specificity_branch": 'branch = "PAIRED_SOURCE_SPECIFICITY_SUPPORTED_SINGLE_SEED"' in core,
        "S18_evaluator_matrix": "for source_id in ids" in evaluator,
        "S19_evaluator_no_265_forward": "assemble_metric(matrix_stats" in evaluator,
        "S20_evaluator_native_bitwise": "T1S_NATIVE_ANCHOR_FAIL" in evaluator,
        "S21_evaluator_donor_bitwise": "T1S_FIXED_DONOR_ANCHOR_FAIL" in evaluator,
        "S22_evaluator_zero": '"source_id": "ZERO"' in evaluator,
        "S23_evaluator_refuse_overwrite": "T1S_REFUSE_OVERWRITE" in evaluator,
        "S24_no_t2": 'RUN_NAMES["T2-A"]' not in evaluator,
        "S25_no_training_calls": ".train(" not in evaluator and "trainer" not in evaluator.lower(),
    }
    for k, v in predicates.items():
        add(checks, k, v)
    for key, rel in SOURCE_PATHS.items():
        add(checks, f"FILE_{key}", (ROOT / rel).is_file(), rel)
    return checks


def formal_checks(args) -> dict:
    checks = {}
    sys.path.insert(0, str(ROOT / "src"))
    from multimodal.t1s_source_specificity import is_derangement, verify_exact_derangement_family

    perf_path = ROOT / "reports/step4_tseries/posttrain_performance.json"
    paired_path = ROOT / "reports/step4_tseries/posttrain_paired.json"
    summary_path = ROOT / "reports/step4_tseries/tseries_summary.json"
    donor_path = ROOT / args.donor_map
    t1_run = ROOT / args.project / "T1-F_P5_FULL_seed20260812"
    manifest_path = t1_run / "manifest.json"
    ckpt_path = t1_run / "weights/last.pt"
    contract_path = Path(args.contract)

    for label, path in (
        ("R01_performance_exists", perf_path),
        ("R02_paired_exists", paired_path),
        ("R03_summary_exists", summary_path),
        ("R04_donor_exists", donor_path),
        ("R05_manifest_exists", manifest_path),
        ("R06_last_exists", ckpt_path),
        ("R07_contract_exists", contract_path),
    ):
        add(checks, label, path.is_file(), str(path))

    if perf_path.is_file():
        add(checks, "R08_performance_sha", sha256_file(perf_path) == EXPECTED_PERFORMANCE_SHA256, sha256_file(perf_path))
        perf = load(perf_path)
        add(checks, "R09_performance_schema", perf.get("schema") == "step4-tseries-posttrain-performance-v1")
    else:
        perf = {}

    if paired_path.is_file():
        add(checks, "R10_paired_sha", sha256_file(paired_path) == EXPECTED_PAIRED_SHA256, sha256_file(paired_path))
        paired = load(paired_path)
        add(checks, "R11_paired_schema", paired.get("schema") == "step4-tseries-posttrain-paired-v1")
        t1p = (paired.get("systems") or {}).get("T1-F") or {}
        add(checks, "R12_t1_paired_negative", t1p.get("single_seed_label") == "SEED20260812_NEGATIVE_PAIRED_EVIDENCE")
    else:
        paired = {}

    if summary_path.is_file():
        summary = load(summary_path)
        add(checks, "R13_summary_schema", summary.get("schema") == "step4-tseries-summary-v1")
        dec = summary.get("decision") or {}
        add(checks, "R14_summary_branch", dec.get("branch") == "T1_ARCHITECTURAL_GAIN_PAIRED_COMPLEMENTARITY_UNPROVEN")
        add(checks, "R15_summary_replication_hold", dec.get("replication_seed_go") is False)
        add(checks, "R16_summary_depth_hold", dec.get("depth_go") is False)
        c = (summary.get("contrasts") or {}).get("T1_minus_T0") or {}
        add(checks, "R17_t1_vs_t0_stable_positive", c.get("label") == "STABLE_POSITIVE")
    else:
        summary = {}

    if donor_path.is_file():
        add(checks, "R18_donor_sha", sha256_file(donor_path) == EXPECTED_DONOR_MAP_SHA256, sha256_file(donor_path))
        donor = {str(k): str(v) for k, v in load(donor_path).items()}
        add(checks, "R19_donor_semantics", is_derangement(donor, VAL6_IDS), donor)
    else:
        donor = {}

    if manifest_path.is_file():
        add(checks, "R20_manifest_sha", sha256_file(manifest_path) == EXPECTED_T1_MANIFEST_SHA256, sha256_file(manifest_path))
        manifest = load(manifest_path)
        add(checks, "R21_manifest_treatment", manifest.get("treatment_id") == "T1-F")
        add(checks, "R22_manifest_80ep", manifest.get("completed_epochs") == 80)
        add(checks, "R23_manifest_seed", manifest.get("seed") == 20260812)
    if ckpt_path.is_file():
        add(checks, "R24_last_sha", sha256_file(ckpt_path) == EXPECTED_T1_LAST_SHA256, sha256_file(ckpt_path))

    if contract_path.is_file():
        contract = load(contract_path)
        ids = tuple(str(x) for x in contract.get("val_ids", []))
        add(checks, "R25_val6_exact", ids == VAL6_IDS, ids)

    fam = verify_exact_derangement_family(VAL6_IDS)
    add(checks, "R26_derangement_family", fam["passed"], fam)

    for idx, (rel, expected) in enumerate(EXPECTED_TSERIES_SOURCE_SHA256.items(), start=27):
        path = ROOT / rel
        passed = path.is_file() and sha256_file(path) == expected
        add(checks, f"R{idx:02d}_source_fresh_{Path(rel).name}", passed, {
            "path": rel,
            "expected": expected,
            "current": sha256_file(path) if path.is_file() else None,
        })
    return checks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["static", "formal"], default="static")
    p.add_argument("--contract", default=DEFAULT_CONTRACT)
    p.add_argument("--project", default="runs/step4_tseries")
    p.add_argument("--donor-map", default="reports/step4_a2/val_donor_map.json")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    checks = static_checks()
    if a.phase == "formal":
        checks.update(formal_checks(a))

    failed = [k for k, v in checks.items() if not v["passed"]]
    source_hashes = {k: sha256_file(ROOT / rel) for k, rel in SOURCE_PATHS.items()}

    gates = {}
    if a.phase == "formal":
        gates = {
            "G1": checks.get("R13_summary_schema", {}).get("passed") is True,
            "G2": all(checks.get(x, {}).get("passed") is True for x in (
                "R14_summary_branch", "R17_t1_vs_t0_stable_positive", "R12_t1_paired_negative"
            )),
            "G3": checks.get("R08_performance_sha", {}).get("passed") is True,
            "G4": checks.get("R10_paired_sha", {}).get("passed") is True,
            "G5": checks.get("R20_manifest_sha", {}).get("passed") is True,
            "G6": checks.get("R24_last_sha", {}).get("passed") is True,
            "G7": checks.get("R25_val6_exact", {}).get("passed") is True,
            "G8": all(checks.get(x, {}).get("passed") is True for x in ("R18_donor_sha", "R19_donor_semantics")),
            "G9": all(v["passed"] for k, v in checks.items() if "source_fresh_" in k),
            "G10": checks.get("S05_matrix_6x6", {}).get("passed") is True,
            "G11": checks.get("S20_evaluator_native_bitwise", {}).get("passed") is True,
            "G12": checks.get("S21_evaluator_donor_bitwise", {}).get("passed") is True,
            "G13": checks.get("R26_derangement_family", {}).get("passed") is True,
            "G14": checks.get("S22_evaluator_zero", {}).get("passed") is True,
            "G15": all(checks.get(x, {}).get("passed") is True for x in (
                "S02_eval_only", "S11_no_training", "S12_no_depth", "S24_no_t2", "S25_no_training_calls"
            )),
        }

    all_passed = not failed and (a.phase != "formal" or all(gates.values()))
    report = {
        "schema": SCHEMA,
        "phase": a.phase,
        "all_passed": all_passed,
        "passed_count": len(checks) - len(failed),
        "total_count": len(checks),
        "failed": failed,
        "checks": checks,
        "gates": gates,
        "source_hashes": source_hashes,
        "upstream": {
            "tseries_accepted_commit": T_SERIES_ACCEPTED_COMMIT,
            "git_ancestry_not_used_as_evidence": True,
            "content_hashes_are_authoritative": True,
        },
    }
    if a.out is None:
        a.out = (
            "reports/step4_t1s/preexecution_audit.json"
            if a.phase == "formal"
            else "reports/step4_t1s/preexecution_static_audit.json"
        )
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "phase": a.phase,
        "all_passed": all_passed,
        "passed": report["passed_count"],
        "total": report["total_count"],
        "failed": failed,
        "gates": gates,
        "out": str(out),
    }, indent=2, ensure_ascii=False))
    if not all_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
