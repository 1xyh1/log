#!/usr/bin/env python3
"""Audit all twelve U6 smokes before authorizing formal training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_u6_core import (  # noqa: E402
    ARMS,
    SCHEMA_PREFLIGHT,
    SCHEMA_RUN,
    SCHEMA_SMOKE_AUDIT,
    implementation_source_hashes,
    launch_rows,
    payload_ok,
    run_report_rel,
)
from multimodal.t1gr_e5_core import FROZEN_E5_SECURITY_POLICY_SHA256  # noqa: E402
from multimodal.t1gr_g_runtime import implementation_source_hashes as primary_source_hashes  # noqa: E402
from multimodal.t1gr_secure_io import (  # noqa: E402
    assert_public_safe,
    atomic_json_write,
    check_existing_output,
    ensure_public_output,
    ensure_repo_input,
    fail,
    file_lock,
    read_json_bounded,
    safe_error_message,
    sha256_file,
    sha256_json,
)

SCRIPT_VERSION = "t1gr-u6-server-smoke-audit-v1"


def _stem_contract(reports: dict[tuple[int, str], dict], seed: int) -> dict:
    rows = {arm: reports[(seed, arm)] for arm in ARMS}
    counts = {arm: rows[arm]["final_stem"]["channel_nonzero_counts"] for arm in ARMS}
    same_initial = len({rows[arm]["complete_initial_state_sha256"] for arm in ARMS}) == 1
    check = {
        "seed": seed,
        "same_complete_initial_state": same_initial,
        "same_initial_stem": all(rows[arm]["initial_stem"] == rows[ARMS[0]]["initial_stem"] for arm in ARMS[1:]),
        "same_server_environment": all(rows[arm]["environment"] == rows[ARMS[0]]["environment"] for arm in ARMS[1:]),
        "same_u6_view": all(
            rows[arm]["u6_view_manifest_private_sha256"] == rows[ARMS[0]]["u6_view_manifest_private_sha256"]
            for arm in ARMS[1:]
        ),
        "g0_aux_stem_remained_zero": counts["G0-N"][3:] == [0, 0, 0],
        "g1_ir_only_stem_learned": counts["G1-P"][3] > 0 and counts["G1-P"][4:] == [0, 0],
        "g2_ir_only_stem_learned": counts["G2-S"][3] > 0 and counts["G2-S"][4:] == [0, 0],
        "g3_all_aux_stem_slices_learned": all(value > 0 for value in counts["G3-D"][3:]),
    }
    check["passed"] = all(value is True for key, value in check.items() if key != "seed")
    return check


def run() -> dict:
    repo = ROOT.resolve(strict=True)
    security_path = ensure_repo_input(repo, "config/t1gr_e5_security_policy.json", "config")
    if sha256_file(security_path) != FROZEN_E5_SECURITY_POLICY_SHA256:
        fail("E5_SECURITY_POLICY_SHA_DRIFT")
    security = read_json_bounded(security_path, 1 << 20, "t1gr-e5-security-policy-v2")
    preflight_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_u6_server_preflight_public.json", "reports/step4_t1gr")
    preflight = read_json_bounded(preflight_path, int(security["max_public_json_bytes"]), SCHEMA_PREFLIGHT)
    if not payload_ok(preflight) or preflight.get("preflight_gate_passed") is not True:
        fail("T1GR_U6_SMOKE_AUDIT_PREFLIGHT_FAIL")
    output = ensure_public_output(repo, "reports/step4_t1gr/t1gr_u6_server_smoke_audit_public.json", security["public_output_prefix"])
    with file_lock(output.with_suffix(output.suffix + ".lock"), 5.0, 900.0):
        sources = implementation_source_hashes(repo)
        upstream = primary_source_hashes(repo)
        if sources != preflight.get("implementation_source_hashes") or upstream != preflight.get("legacy_primary_suite_source_hashes"):
            fail("T1GR_U6_SMOKE_AUDIT_SOURCE_DRIFT")
        evidence, reports = [], {}
        for row in launch_rows():
            path = ensure_repo_input(repo, run_report_rel("smoke", row["seed"], row["arm"]), "reports/step4_t1gr")
            report = read_json_bounded(path, int(security["max_public_json_bytes"]), SCHEMA_RUN)
            if (
                not payload_ok(report)
                or report.get("run_gate_passed") is not True
                or report.get("mode") != "smoke"
                or report.get("arm") != row["arm"]
                or int(report.get("seed", -1)) != row["seed"]
                or int(report.get("suite_position_zero_based", -1)) != row["position"]
                or int(report.get("lane_position_zero_based", -1)) != row["lane_position"]
                or int(report.get("epochs_completed", -1)) != 1
                or report.get("physical_first_conv_in_channels") != 6
                or report.get("epoch_modality_contract_passed") is not True
                or report.get("final_holdout_open_authorized") is not False
            ):
                fail("T1GR_U6_SMOKE_REPORT_FAIL")
            if (
                report.get("implementation_source_hashes") != sources
                or report.get("legacy_primary_suite_source_hashes") != upstream
            ):
                fail("T1GR_U6_SMOKE_REPORT_SOURCE_DRIFT")
            key = (row["seed"], row["arm"])
            reports[key] = report
            evidence.append({"seed": row["seed"], "arm": row["arm"], "report_sha256": sha256_file(path)})
        seed_checks = [_stem_contract(reports, int(seed)) for seed in sorted({row["seed"] for row in launch_rows()})]
        passed = len(evidence) == 12 and all(row["passed"] for row in seed_checks)
        request = sha256_json({
            "script": SCRIPT_VERSION,
            "preflight": sha256_file(preflight_path),
            "evidence": evidence,
            "sources": sources,
        })
        existing = check_existing_output(output, request)
        if existing is not None:
            if existing[0].get("smoke_gate_passed") is not True:
                fail("T1GR_U6_EXISTING_SMOKE_AUDIT_NOT_PASS")
            return {"status": "PASS", "idempotent_reuse": True, "public_output_sha256": existing[1]}
        report = {
            "schema": SCHEMA_SMOKE_AUDIT,
            "script_version": SCRIPT_VERSION,
            "preflight_public_sha256": sha256_file(preflight_path),
            "smoke_run_evidence": evidence,
            "same_seed_four_arm_checks": seed_checks,
            "implementation_source_hashes": sources,
            "legacy_primary_suite_source_hashes": upstream,
            "smoke_run_count": len(evidence),
            "smoke_gate_passed": passed,
            "formal_training_authorized": passed,
            "parallel_seed_lanes_authorized": passed,
            "legacy_primary_g_suite_mutation_authorized": False,
            "final_holdout_open_authorized": False,
            "production_go": False,
            "next_action": "run twelve 80-epoch U6 formal jobs, then evaluate native, common-input, LOFO, and Depth-domain views",
        }
        assert_public_safe(report)
        digest, _ = atomic_json_write(output, report, private=False, request_fingerprint=request)
        if not passed:
            fail("T1GR_U6_SMOKE_AUDIT_FAIL")
        return {
            "status": "PASS",
            "idempotent_reuse": False,
            "public_output_sha256": digest,
            "formal_training_authorized": True,
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
