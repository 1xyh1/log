#!/usr/bin/env python3
"""Audit all nine smoke runs and, only on full PASS, authorize formal execution."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_e5_core import FROZEN_E5_SECURITY_POLICY_SHA256, payload_ok as e5_payload_ok  # noqa: E402
from multimodal.t1gr_g_core import ARMS, SEEDS  # noqa: E402
from multimodal.t1gr_g_impl_core import SCHEMA_PREFLIGHT, SCHEMA_RUN, SCHEMA_SMOKE_AUDIT  # noqa: E402
from multimodal.t1gr_g_runtime import (  # noqa: E402
    frozen_launch_rows,
    implementation_source_hashes,
    read_json,
    run_report_rel,
)
from multimodal.t1gr_secure_io import (  # noqa: E402
    assert_public_safe,
    atomic_json_write,
    ensure_public_output,
    ensure_repo_input,
    fail,
    file_lock,
    read_json_bounded,
    safe_error_message,
    sha256_file,
    sha256_json,
)

SCRIPT_VERSION = "t1gr-g-smoke-audit-v1"


def run() -> dict:
    repo = ROOT.resolve(strict=True)
    security_path = ensure_repo_input(repo, "config/t1gr_e5_security_policy.json", "config")
    if sha256_file(security_path) != FROZEN_E5_SECURITY_POLICY_SHA256:
        fail("E5_SECURITY_POLICY_SHA_DRIFT")
    security = read_json_bounded(security_path, 1 << 20, "t1gr-e5-security-policy-v2")
    design_path = ensure_repo_input(repo, "config/t1gr_g_design.frozen.json", "config")
    preflight_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_g_implementation_preflight_public.json", "reports/step4_t1gr")
    output = ensure_public_output(
        repo, "reports/step4_t1gr/t1gr_g_smoke_audit_public.json", security["public_output_prefix"]
    )
    with file_lock(output.with_suffix(output.suffix + ".lock"), 5.0, 900.0):
        design = read_json(design_path)
        preflight = read_json_bounded(preflight_path, int(security["max_public_json_bytes"]), SCHEMA_PREFLIGHT)
        if not e5_payload_ok(preflight) or preflight.get("smoke_training_authorized") is not True:
            fail("T1GR_G_SMOKE_AUDIT_PREFLIGHT_FAIL")
        rows = frozen_launch_rows(design)
        report_rows = []
        reports = {}
        for row in rows:
            path = ensure_repo_input(repo, run_report_rel("smoke", row["seed"], row["arm"]), "reports/step4_t1gr")
            report = read_json_bounded(path, int(security["max_public_json_bytes"]), SCHEMA_RUN)
            if not e5_payload_ok(report):
                fail("T1GR_G_SMOKE_REPORT_INTEGRITY_FAIL")
            key = (int(row["seed"]), str(row["arm"]))
            reports[key] = report
            report_rows.append({
                "position_zero_based": int(row["position"]),
                "seed": key[0],
                "arm": key[1],
                "report_sha256": sha256_file(path),
            })
        source_hashes = implementation_source_hashes(repo)
        preflight_sha = sha256_file(preflight_path)
        row_checks = {}
        for row in rows:
            key = (int(row["seed"]), str(row["arm"]))
            report = reports[key]
            traces = report.get("epoch_trace_summaries") or []
            row_checks[f"s{key[0]}_{key[1]}"] = bool(
                report.get("mode") == "smoke"
                and report.get("run_gate_passed") is True
                and int(report.get("suite_position_zero_based", -1)) == int(row["position"])
                and int(report.get("epochs_expected", -1)) == 1
                and int(report.get("epochs_completed", -1)) == 1
                and len(traces) == 1
                and traces[0].get("source_condition_passed") is True
                and report.get("mechanism_runtime_passed") is True
                and "musgd" in str((report.get("optimizer") or {}).get("class_name", "")).lower()
                and report.get("actual_batch_size") == 4
                and report.get("actual_train_workers") == 8
                and report.get("actual_validation_workers") == 16
                and report.get("actual_train_sampler") == "RecipientEpochSampler"
                and report.get("validation_ir") == "ZERO_IR"
                and report.get("primary_checkpoint") == "last.pt"
                and report.get("preflight_public_sha256") == preflight_sha
                and report.get("implementation_source_hashes") == source_hashes
                and report.get("final_holdout_ids_available_to_runner") is False
                and report.get("final_holdout_open_authorized") is False
            )
        seed_checks = []
        for seed in SEEDS:
            values = [reports[(seed, arm)] for arm in ARMS]
            initial = {row.get("complete_initial_state_sha256") for row in values}
            start = {row.get("training_start_state_sha256") for row in values}
            optimizer = {(row.get("optimizer") or {}).get("contract_sha256") for row in values}
            effective = {row.get("effective_training_args_sha256") for row in values}
            identity_pass = len(initial) == len(start) == len(optimizer) == len(effective) == 1 and initial == start
            seed_checks.append({
                "seed": seed,
                "same_seed_complete_initial_state": len(initial) == 1,
                "same_seed_training_start_state": len(start) == 1,
                "preflight_equals_training_start": initial == start,
                "same_seed_optimizer_contract": len(optimizer) == 1,
                "same_seed_effective_args": len(effective) == 1,
                "passed": identity_pass,
            })
        treatment = {arm: reports[(SEEDS[0], arm)].get("model_treatment_id") for arm in ARMS}
        checks = {
            "nine_reports_complete": len(reports) == 9,
            "frozen_order_exact": all(row_checks.values()),
            "all_runtime_gates": all(row_checks.values()),
            "same_seed_identity_and_optimizer": all(row["passed"] for row in seed_checks),
            "g0_null_g1_g2_full": treatment == {"G0-N": "T0-N", "G1-P": "T1-F", "G2-S": "T1-F"},
            "implementation_unchanged": source_hashes == preflight.get("implementation_source_hashes"),
            "zero_ir_dev_validation": all(report.get("validation_ir") == "ZERO_IR" for report in reports.values()),
            "holdout_unavailable": all(report.get("final_holdout_open_authorized") is False for report in reports.values()),
        }
        passed = all(checks.values())
        request_fingerprint = sha256_json({
            "script": SCRIPT_VERSION,
            "design": sha256_file(design_path),
            "preflight": preflight_sha,
            "reports": report_rows,
            "sources": source_hashes,
        })
        report = {
            "schema": SCHEMA_SMOKE_AUDIT,
            "script_version": SCRIPT_VERSION,
            "design_file_sha256": sha256_file(design_path),
            "preflight_public_sha256": preflight_sha,
            "run_count": len(reports),
            "run_evidence": report_rows,
            "row_checks": row_checks,
            "same_seed_checks": seed_checks,
            "treatment_mapping": treatment,
            "implementation_source_hashes": source_hashes,
            "checks": checks,
            "passed_count": sum(bool(value) for value in checks.values()),
            "total_count": len(checks),
            "smoke_gate_passed": passed,
            "multiseed_training_authorized": passed,
            "execution_authorized": passed,
            "selective_rerun_authorized": False,
            "final_holdout_open_authorized": False,
            "depth_go": False,
            "production_go": False,
            "next_action": "run the frozen formal nine-run suite in Latin-square order",
        }
        assert_public_safe(report)
        digest, reused = atomic_json_write(output, report, private=False, request_fingerprint=request_fingerprint)
        if not passed:
            fail("T1GR_G_SMOKE_AUDIT_FAIL")
        return {
            "status": "PASS",
            "idempotent_reuse": reused,
            "public_output_sha256": digest,
            "multiseed_training_authorized": True,
            "final_holdout_open_authorized": False,
        }


def main() -> None:
    try:
        print(json.dumps(run(), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": safe_error_message(exc)}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
