"""Sequential, resumable, non-selective T1-GR suite orchestration."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from .t1gr_g_impl_core import SCHEMA_RUN, SCHEMA_SUITE_STATE, payload_ok, payload_sha256
from .t1gr_g_runtime import frozen_launch_rows, read_json, run_report_rel
from .t1gr_secure_io import fail, is_within, sha256_file, sha256_json


def _write_state(path: Path, state: dict) -> None:
    final = dict(state)
    final["payload_sha256"] = payload_sha256(final)
    raw = json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            os.chmod(temp, 0o600)
        os.replace(temp, path)
    except Exception:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _validate_completed(repo: Path, mode: str, rows: list[dict], completed: list[dict]) -> None:
    for index, item in enumerate(completed):
        row = rows[index]
        if int(item.get("position", -1)) != index:
            fail("T1GR_G_SUITE_COMPLETED_POSITION_FAIL")
        report_path = repo / run_report_rel(mode, row["seed"], row["arm"])
        if not report_path.is_file() or sha256_file(report_path) != item.get("report_sha256"):
            fail("T1GR_G_SUITE_COMPLETED_REPORT_DRIFT")
        report = read_json(report_path)
        if (
            report.get("schema") != SCHEMA_RUN
            or not payload_ok(report)
            or report.get("run_gate_passed") is not True
            or report.get("mode") != mode
            or report.get("arm") != row["arm"]
            or int(report.get("seed", -1)) != int(row["seed"])
        ):
            fail("T1GR_G_SUITE_COMPLETED_REPORT_INVALID")


def run_suite(
    *,
    repo: Path,
    mode: str,
    design_path: Path,
    preflight_path: Path,
    view_manifest: Path,
    base_checkpoint: Path,
    run_root: Path,
    suite_state_path: Path,
    smoke_audit_path: Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo).resolve(strict=True)
    if mode not in {"smoke", "formal"}:
        fail("T1GR_G_SUITE_MODE_FAIL")
    for private in (view_manifest, suite_state_path):
        if is_within(Path(private).expanduser().resolve(strict=False), repo):
            fail("T1GR_G_SUITE_PRIVATE_PATH_INSIDE_REPO")
    view_manifest = Path(view_manifest).expanduser().resolve(strict=False)
    base_checkpoint = Path(base_checkpoint).expanduser().resolve(strict=False)
    run_root = Path(run_root).expanduser().resolve(strict=False)
    suite_state_path = Path(suite_state_path).expanduser().resolve(strict=False)
    if not view_manifest.is_file() or not base_checkpoint.is_file():
        fail("T1GR_G_SUITE_PRIVATE_INPUT_MISSING")
    design = read_json(design_path)
    rows = frozen_launch_rows(design)
    preflight = read_json(preflight_path)
    if preflight.get("preflight_gate_passed") is not True or preflight.get("smoke_training_authorized") is not True:
        fail("T1GR_G_SUITE_PREFLIGHT_NOT_PASS")
    smoke_audit_sha = None
    if mode == "formal":
        if smoke_audit_path is None or not smoke_audit_path.is_file():
            fail("T1GR_G_SUITE_SMOKE_AUDIT_MISSING")
        smoke_audit = read_json(smoke_audit_path)
        if smoke_audit.get("smoke_gate_passed") is not True or smoke_audit.get("multiseed_training_authorized") is not True:
            fail("T1GR_G_SUITE_FORMAL_NOT_AUTHORIZED")
        smoke_audit_sha = sha256_file(smoke_audit_path)
    request = sha256_json({
        "mode": mode,
        "design": sha256_file(design_path),
        "preflight": sha256_file(preflight_path),
        "view_manifest": sha256_file(view_manifest),
        "base_checkpoint": sha256_file(base_checkpoint),
        "run_root_binding": sha256_json(str(run_root).casefold() if os.name == "nt" else str(run_root)),
        "smoke_audit": smoke_audit_sha,
        "rows": rows,
    })
    if suite_state_path.exists():
        state = read_json(suite_state_path)
        if state.get("schema") != SCHEMA_SUITE_STATE or not payload_ok(state):
            fail("T1GR_G_SUITE_STATE_INTEGRITY_FAIL")
        if state.get("request_fingerprint") != request or state.get("rows") != rows:
            fail("T1GR_G_SUITE_STATE_REQUEST_CONFLICT")
    else:
        state = {
            "schema": SCHEMA_SUITE_STATE,
            "mode": mode,
            "status": "IN_PROGRESS",
            "request_fingerprint": request,
            "run_root_binding": sha256_json(str(run_root).casefold() if os.name == "nt" else str(run_root)),
            "rows": rows,
            "current_position": 0,
            "completed": [],
            "final_holdout_ids_present": False,
        }
        _write_state(suite_state_path, state)
        state = read_json(suite_state_path)
    completed = list(state.get("completed") or [])
    _validate_completed(repo, mode, rows, completed)
    if state.get("status") == "COMPLETE":
        if len(completed) != 9 or int(state.get("current_position", -1)) != 9:
            fail("T1GR_G_SUITE_COMPLETE_STATE_FAIL")
        return {"status": "PASS", "mode": mode, "idempotent_reuse": True, "completed_runs": 9}
    if state.get("status") != "IN_PROGRESS" or int(state.get("current_position", -1)) != len(completed):
        fail("T1GR_G_SUITE_PROGRESS_STATE_FAIL")
    while len(completed) < len(rows):
        position = len(completed)
        row = rows[position]
        command = [
            sys.executable,
            str(repo / "scripts" / "t1gr_g_run_one.py"),
            "--mode", mode,
            "--arm", row["arm"],
            "--seed", str(row["seed"]),
            "--view-manifest", str(view_manifest),
            "--base-checkpoint", str(base_checkpoint),
            "--run-root", str(run_root),
            "--suite-state", str(suite_state_path),
        ]
        result = subprocess.run(command, cwd=repo, check=False)
        if result.returncode != 0:
            fail("T1GR_G_SUITE_RUN_FAILED", f"position={position}")
        report_path = repo / run_report_rel(mode, row["seed"], row["arm"])
        report = read_json(report_path)
        if report.get("run_gate_passed") is not True:
            fail("T1GR_G_SUITE_RUN_REPORT_NOT_PASS")
        completed.append({
            "position": position,
            "seed": int(row["seed"]),
            "arm": row["arm"],
            "report_sha256": sha256_file(report_path),
        })
        state["completed"] = completed
        state["current_position"] = len(completed)
        state["status"] = "COMPLETE" if len(completed) == 9 else "IN_PROGRESS"
        _write_state(suite_state_path, state)
        state = read_json(suite_state_path)
    _validate_completed(repo, mode, rows, completed)
    return {
        "status": "PASS",
        "mode": mode,
        "idempotent_reuse": False,
        "completed_runs": len(completed),
        "final_holdout_open_authorized": False,
    }
