#!/usr/bin/env python3
"""Apply the frozen IR/Depth selectors to completed T1-U6 DEV evidence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_u6_core import (  # noqa: E402
    SCHEMA_EVAL_AUDIT,
    SCHEMA_RESULTS,
    implementation_source_hashes,
    payload_sha256,
    payload_ok,
    summarize_results,
)
from multimodal.t1gr_e5_core import FROZEN_E5_SECURITY_POLICY_SHA256  # noqa: E402
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

SCRIPT_VERSION = "t1gr-u6-server-summarize-v1"


def run() -> dict:
    repo = ROOT.resolve(strict=True)
    security_path = ensure_repo_input(repo, "config/t1gr_e5_security_policy.json", "config")
    if sha256_file(security_path) != FROZEN_E5_SECURITY_POLICY_SHA256:
        fail("E5_SECURITY_POLICY_SHA_DRIFT")
    security = read_json_bounded(security_path, 1 << 20, "t1gr-e5-security-policy-v2")
    results_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_u6_server_results_public.json", "reports/step4_t1gr")
    audit_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_u6_server_eval_public.json", "reports/step4_t1gr")
    results = read_json_bounded(results_path, int(security["max_public_json_bytes"]), SCHEMA_RESULTS)
    audit = read_json_bounded(audit_path, int(security["max_public_json_bytes"]), SCHEMA_EVAL_AUDIT)
    if not payload_ok(results) or not payload_ok(audit) or audit.get("eval_gate_passed") is not True:
        fail("T1GR_U6_SUMMARY_INPUT_FAIL")
    if audit.get("implementation_source_hashes") != implementation_source_hashes(repo):
        fail("T1GR_U6_SUMMARY_SOURCE_DRIFT")
    if results.get("final_holdout_accessed") is not False or audit.get("final_holdout_open_authorized") is not False:
        fail("T1GR_U6_SUMMARY_HOLDOUT_BOUNDARY_FAIL")
    cross, summary = summarize_results(results)
    cross["script_version"] = SCRIPT_VERSION
    cross["results_public_sha256"] = sha256_file(results_path)
    cross["eval_audit_public_sha256"] = sha256_file(audit_path)
    summary["script_version"] = SCRIPT_VERSION
    summary["results_public_sha256"] = sha256_file(results_path)
    summary["eval_audit_public_sha256"] = sha256_file(audit_path)
    cross["payload_sha256"] = payload_sha256(cross)
    summary["cross_seed_payload_sha256"] = cross["payload_sha256"]
    summary["payload_sha256"] = payload_sha256(summary)
    cross_output = ensure_public_output(repo, "reports/step4_t1gr/t1gr_u6_server_cross_seed_public.json", security["public_output_prefix"])
    summary_output = ensure_public_output(repo, "reports/step4_t1gr/t1gr_u6_server_summary_public.json", security["public_output_prefix"])
    request = sha256_json({"script": SCRIPT_VERSION, "results": sha256_file(results_path), "audit": sha256_file(audit_path)})
    with file_lock(summary_output.with_suffix(summary_output.suffix + ".lock"), 5.0, 900.0):
        existing_cross, existing_summary = check_existing_output(cross_output, request), check_existing_output(summary_output, request)
        if existing_cross is not None and existing_summary is not None:
            return {"status": "PASS", "idempotent_reuse": True, "cross_seed_sha256": existing_cross[1], "summary_sha256": existing_summary[1]}
        if (existing_cross is None) != (existing_summary is None):
            fail("T1GR_U6_SUMMARY_PARTIAL_OUTPUT")
        assert_public_safe(cross)
        assert_public_safe(summary)
        cross_sha, _ = atomic_json_write(cross_output, cross, private=False, request_fingerprint=request)
        summary_sha, _ = atomic_json_write(summary_output, summary, private=False, request_fingerprint=request)
        return {
            "status": "PASS",
            "idempotent_reuse": False,
            "competition_recommendation": summary["competition_recommendation"],
            "ir_eligible": summary["ir_eligible"],
            "depth_eligible": summary["depth_eligible"],
            "manual_review_required": summary["manual_review_required"],
            "cross_seed_sha256": cross_sha,
            "summary_sha256": summary_sha,
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
