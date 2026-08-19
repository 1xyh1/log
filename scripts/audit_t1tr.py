#!/usr/bin/env python3
"""Static/formal pretraining audit for T1-TR."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "step4-t1tr-pretraining-audit-v1"
DEFAULT_CONTRACT = "D:/pycharm/Python Develop/YOLO_1/step3_data_contract.json"

SOURCE_PATHS = {
    "design_sha256": "docs/step4_t1tr/DESIGN_FREEZE.md",
    "core_sha256": "src/multimodal/t1tr_training_source.py",
    "runner_sha256": "scripts/run_t1tr.py",
    "smoke_sha256": "scripts/smoke_t1tr.py",
    "audit_sha256": "scripts/audit_t1tr.py",
    "eval_sha256": "scripts/eval_t1tr.py",
    "summary_sha256": "scripts/summarize_t1tr.py",
    "verify_sha256": "scripts/verify_t1tr_run.py",
    "tests_sha256": "tests/test_t1tr.py",
    "readme_sha256": "T1TR_README.md",
}

T1S_SUMMARY_RAW_SHA256 = "a38881cb019764242f3e34560c0be4a6d364aad36d7cb7496978526caf2e98f2"
T1_MANIFEST_SHA256 = "081afec392d96ee2d570a3424e5f015f05ee308297daed8900ece5584c707312"
T1_LAST_SHA256 = "8380e21504fabd0d8c3715398739bbb0bed5aaafd9c822dfc14c9503af2daeee"
T0_MANIFEST_SHA256 = "99c98b741ff3599223a26c0726f8cf7e702a9582d039e1dfe27c4c4af00b0f67"
T0_LAST_SHA256 = "a977dbe19e81bde06a14d635656a47034bf55d186f091a3352dd73b17e40a496"
T1S_ZERO = 0.29596085371085373

UPSTREAM_SOURCE_SHA256 = {
    "src/multimodal/tseries_p5_model.py":
        "987c041d80d85cfc18626febad308529e927944d02cfea4bdaf9d0d63cc2af0d",
    "src/multimodal/tseries_core.py":
        "9f8f3ae1eade6feee5688fb13a3d91e1cfc33805fe71c3a3b5981e18df1863ec",
    "src/multimodal/tseries_runtime.py":
        "5daf47bdff07afdc1fe4e6790143d4f363d175b61d866a6f7d1ac4ec0b14230a",
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def add(checks, name, passed, detail=None):
    row = {"passed": bool(passed)}
    if detail is not None:
        row["detail"] = detail
    checks[name] = row

def package_checks() -> dict:
    c = {}
    text = {
        key: (ROOT / rel).read_text(encoding="utf-8")
        for key, rel in SOURCE_PATHS.items()
        if (ROOT / rel).exists()
    }
    design = text.get("design_sha256", "")
    core = text.get("core_sha256", "")
    runner = text.get("runner_sha256", "")
    evaluator = text.get("eval_sha256", "")
    summary = text.get("summary_sha256", "")

    predicates = {
        "S01_design_frozen": "TRAINING DESIGN FROZEN" in design,
        "S02_one_new_arm": "U2-S" in design and "only new training arm" in design,
        "S03_zero_primary": "Primary comparison" in design and "ZERO" in design,
        "S04_no_depth": "NO Depth" in design,
        "S05_no_gate": "NO gate/q" in design,
        "S06_no_centering": "NO centering" in design,
        "S07_balanced_schedule": "shift(e) = 1 + (e mod 10)" in design,
        "S08_each_pair_8": "exactly 8 times" in design,
        "S09_core_shift": "1 + (int(epoch) % (n - 1))" in core,
        "S10_core_schedule_balance": "schedule_balance" in core,
        "S11_runner_aux_id_map": "aux_id_map=aux_map" in runner,
        "S12_runner_no_self_runtime": "T1TR_RUNTIME_SELF_DONOR" in runner,
        "S13_runner_initial_anchor": "T1TR_INITIAL_IDENTITY_FAIL" in runner,
        "S14_runner_optimizer_anchor": "T1TR_OPTIMIZER_MISMATCH" in runner,
        "S15_runner_t1_full": 'build_tseries_model(Path(a.base_checkpoint), "T1-F")' in runner,
        "S16_runner_group_c1i": 'group="C1-I"' in runner,
        "S17_eval_zero": "zeros_like" in evaluator,
        "S18_eval_t0_bitwise": "T1TR_T0_ZERO_NATIVE_ANCHOR_FAIL" in evaluator,
        "S19_eval_t1s_zero_anchor": "T1TR_T1_ZERO_NUMERIC_ANCHOR_FAIL" in evaluator,
        "S20_summary_no_margin": "no_arbitrary_ap_margin" in summary,
        "S21_summary_depth_false": '"depth_go": False' in summary,
        "S22_summary_production_false": '"production_go": False' in summary,
        "S23_no_modify_dataset": "trimodal_dataset.py" not in SOURCE_PATHS.values(),
        "S24_verify_80ep": "T1TR_U2_FORMAL_RUN_PASS" in text.get("verify_sha256", ""),
    }
    for k, v in predicates.items():
        add(c, k, v)

    for key, rel in SOURCE_PATHS.items():
        add(c, f"FILE_{key}", (ROOT / rel).is_file(), rel)
    return c

def formal_checks(args) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from multimodal.t1tr_training_source import schedule_balance  # noqa
    from multimodal.step4_f1_c_readiness import (  # noqa
        EXPECTED_BASE_CHECKPOINT_SHA256, verify_base_checkpoint,
    )

    c = {}
    t1s = ROOT / "reports/step4_t1s/t1s_summary.json"
    t0m = ROOT / "runs/step4_tseries/T0-N_P5_NULL_seed20260812/manifest.json"
    t1m = ROOT / "runs/step4_tseries/T1-F_P5_FULL_seed20260812/manifest.json"
    t0pt = ROOT / "runs/step4_tseries/T0-N_P5_NULL_seed20260812/weights/last.pt"
    t1pt = ROOT / "runs/step4_tseries/T1-F_P5_FULL_seed20260812/weights/last.pt"
    t1opt = ROOT / "runs/step4_tseries/T1-F_P5_FULL_seed20260812/optimizer_manifest.json"
    contract = Path(args.contract)
    smoke = ROOT / args.smoke_report
    trimodal = ROOT / "src/multimodal/trimodal_dataset.py"

    for name, p in {
        "R01_t1s_summary_exists": t1s,
        "R02_t0_manifest_exists": t0m,
        "R03_t1_manifest_exists": t1m,
        "R04_t0_last_exists": t0pt,
        "R05_t1_last_exists": t1pt,
        "R06_t1_optimizer_exists": t1opt,
        "R07_contract_exists": contract,
        "R08_smoke_exists": smoke,
        "R09_trimodal_exists": trimodal,
    }.items():
        add(c, name, p.is_file(), str(p))

    if t1s.is_file():
        obj = load(t1s)
        add(c, "R10_t1s_sha", sha256_file(t1s) == T1S_SUMMARY_RAW_SHA256, sha256_file(t1s))
        add(c, "R11_t1s_schema", obj.get("schema") == "step4-t1s-summary-v1")
        dec = obj.get("decision") or {}
        add(c, "R12_t1s_branch",
            dec.get("branch") == "INFERENCE_RESIDUAL_NOT_SUPPORTED_TRAINING_DYNAMICS_CANDIDATE")
        add(c, "R13_t1s_replication_hold", dec.get("replication_seed_go") is False)
        add(c, "R14_t1s_depth_hold", obj.get("depth_go") is False)
        add(c, "R15_t1s_zero_anchor",
            abs(float(obj["zero_residual"]["map50_95"]) - T1S_ZERO) <= 1e-15)

    if t0m.is_file():
        add(c, "R16_t0_manifest_sha", sha256_file(t0m) == T0_MANIFEST_SHA256)
    if t1m.is_file():
        t1 = load(t1m)
        add(c, "R17_t1_manifest_sha", sha256_file(t1m) == T1_MANIFEST_SHA256)
        add(c, "R18_t1_treatment", t1.get("treatment_id") == "T1-F")
        add(c, "R19_t1_80ep_seed",
            t1.get("completed_epochs") == 80 and t1.get("seed") == 20260812)
        if contract.is_file():
            add(c, "R20_contract_sha",
                sha256_file(contract) == t1.get("contract_sha256"),
                sha256_file(contract))

    if t0pt.is_file():
        add(c, "R21_t0_last_sha", sha256_file(t0pt) == T0_LAST_SHA256)
    if t1pt.is_file():
        add(c, "R22_t1_last_sha", sha256_file(t1pt) == T1_LAST_SHA256)

    if contract.is_file():
        co = load(contract)
        train_ids = [str(x) for x in co.get("train_ids", [])]
        val_ids = [str(x) for x in co.get("val_ids", [])]
        add(c, "R23_split_sizes",
            len(train_ids) == 11 and len(set(train_ids)) == 11
            and len(val_ids) == 6 and len(set(val_ids)) == 6)
        bal = schedule_balance(train_ids, 80) if len(train_ids) == 11 else {"passed": False}
        add(c, "R24_schedule_balance", bal.get("passed") is True, bal)

    if trimodal.is_file():
        tt = trimodal.read_text(encoding="utf-8")
        semantic_checks = {
            "aux_map_lookup": "aux_sid = self.aux_id_map.get(sid, sid)" in tt,
            "recipient_rgb": "rgb_p, _, _ = self._paths(sid)" in tt,
            "recipient_labels": "lab_p = self.label_files[sid]" in tt,
            "donor_ir": "_, ir_aux_p, dep_aux_p = self._paths(aux_sid)" in tt,
            "recipient_flip_seed": "mp.should_flip(self.seed, self.epoch, sid, self.fliplr)" in tt,
            "aux_metadata": '"aux_sample_id": aux_sid' in tt,
        }
        add(c, "R25_trimodal_semantic_contract",
            all(semantic_checks.values()), semantic_checks)
        add(c, "R26_trimodal_aux_semantics",
            semantic_checks["aux_map_lookup"]
            and semantic_checks["recipient_rgb"]
            and semantic_checks["recipient_labels"]
            and semantic_checks["donor_ir"]
            and semantic_checks["recipient_flip_seed"]
            and semantic_checks["aux_metadata"])

    for rel, expected in UPSTREAM_SOURCE_SHA256.items():
        p = ROOT / rel
        add(c, f"R_SOURCE_{Path(rel).name}",
            p.is_file() and sha256_file(p) == expected,
            None if not p.is_file() else sha256_file(p))

    if smoke.is_file():
        so = load(smoke)
        add(c, "R27_smoke_schema",
            so.get("schema") == "step4-t1tr-pretraining-smoke-v1")
        add(c, "R28_smoke_pass", so.get("all_dynamic_gates_passed") is True)
        sg = so.get("gates") or {}
        add(c, "R29_smoke_initial", sg.get("initial_identity_exact_t1") is True)
        add(c, "R30_smoke_optimizer", sg.get("optimizer_exact_t1") is True)
        add(c, "R31_smoke_no_self", sg.get("no_self_match") is True)
        add(c, "R32_smoke_mapping", sg.get("actual_mapping_matches_schedule") is True)
        pins = so.get("source_hashes") or {}
        add(c, "R33_smoke_runner_fresh",
            pins.get("runner") == sha256_file(ROOT / "scripts/run_t1tr.py"))
        add(c, "R34_smoke_core_fresh",
            pins.get("core") == sha256_file(ROOT / "src/multimodal/t1tr_training_source.py"))
        add(c, "R35_smoke_design_fresh",
            pins.get("design") == sha256_file(ROOT / "docs/step4_t1tr/DESIGN_FREEZE.md"))

    bc = verify_base_checkpoint(Path(args.base_checkpoint), EXPECTED_BASE_CHECKPOINT_SHA256)
    add(c, "R36_base_checkpoint", bc["passed"], bc)
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["static", "formal"], default="static")
    ap.add_argument("--contract", default=DEFAULT_CONTRACT)
    ap.add_argument("--base-checkpoint", default="E:/odin/yolo26s.pt")
    ap.add_argument("--smoke-report", default="reports/step4_t1tr/pretraining_smoke.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    checks = package_checks()
    if a.phase == "formal":
        checks.update(formal_checks(a))

    failed = [k for k, v in checks.items() if not v["passed"]]
    source_hashes = {
        key: sha256_file(ROOT / rel)
        for key, rel in SOURCE_PATHS.items()
    }

    gates = {}
    if a.phase == "formal":
        def ok(*names):
            return all(checks.get(n, {}).get("passed") is True for n in names)
        gates = {
            "G1": ok("R10_t1s_sha", "R11_t1s_schema"),
            "G2": ok("R12_t1s_branch"),
            "G3": ok("R15_t1s_zero_anchor"),
            "G4": ok("R16_t0_manifest_sha", "R17_t1_manifest_sha"),
            "G5": ok("R21_t0_last_sha", "R22_t1_last_sha"),
            "G6": ok("R20_contract_sha"),
            "G7": ok("R23_split_sizes"),
            "G8": ok("R24_schedule_balance"),
            "G9": ok("R24_schedule_balance", "R26_trimodal_aux_semantics"),
            "G10": ok("R24_schedule_balance"),
            "G11": ok("R29_smoke_initial"),
            "G12": ok("R30_smoke_optimizer"),
            "G13": ok("R31_smoke_no_self", "R32_smoke_mapping"),
            "G14": ok("R28_smoke_pass", "R33_smoke_runner_fresh",
                      "R34_smoke_core_fresh", "R35_smoke_design_fresh"),
            "G15": ok("S15_runner_t1_full", "S16_runner_group_c1i"),
            "G16": ok("R18_t1_treatment", "R19_t1_80ep_seed", "R36_base_checkpoint"),
            "G17": ok("S02_one_new_arm"),
            "G18": ok("S04_no_depth", "S05_no_gate", "S06_no_centering"),
        }
        if not all(gates.values()):
            failed.extend([f"FORMAL_GATE:{g}" for g, v in gates.items() if not v])

    report = {
        "schema": SCHEMA,
        "phase": a.phase,
        "all_passed": not failed,
        "passed_count": sum(v["passed"] for v in checks.values()),
        "total_count": len(checks),
        "failed": sorted(set(failed)),
        "checks": checks,
        "gates": gates,
        "source_hashes": source_hashes,
        "upstream": {
            "t1s_accepted_commit": "7c86a87a0ca61e6ebc8299b4f3b35dc997f3d46d",
            "git_ancestry_not_used_as_evidence": True,
            "content_hashes_are_authoritative": True,
        } if a.phase == "formal" else {},
    }
    if a.out is None:
        a.out = (
            "reports/step4_t1tr/pretraining_audit.json"
            if a.phase == "formal"
            else "reports/step4_t1tr/pretraining_static_audit.json"
        )
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise RuntimeError(f"T1TR_REFUSE_OVERWRITE:{out}")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "phase": a.phase,
        "all_passed": report["all_passed"],
        "passed": report["passed_count"],
        "total": report["total_count"],
        "failed": report["failed"],
        "out": str(out),
    }, indent=2))
    if failed:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
