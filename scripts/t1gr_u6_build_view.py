#!/usr/bin/env python3
"""Build the T1-U6 TRAIN/DEV view from the frozen RGB/IR split plus Depth."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import uuid
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.t1gr_u6_core import (  # noqa: E402
    SCHEMA_SPEC,
    SCHEMA_VIEW,
    SCHEMA_VIEW_PUBLIC,
    encode_depth_array,
    payload_sha256,
    validate_spec,
)
from multimodal.t1gr_u6_runtime import verify_u6_view  # noqa: E402
from multimodal.t1gr_e5_core import (  # noqa: E402
    FROZEN_E5_SECURITY_POLICY_SHA256,
    SCHEMA_RECIPE,
    payload_ok as e5_payload_ok,
    scan_formal_zip,
)
from multimodal.t1gr_g_runtime import read_json, verify_multimodal_view  # noqa: E402
from multimodal.t1gr_secure_io import (  # noqa: E402
    Deadline,
    assert_public_safe,
    atomic_json_write,
    check_existing_output,
    ensure_private_input,
    ensure_public_output,
    ensure_repo_input,
    fail,
    file_lock,
    is_within,
    read_json_bounded,
    require_unchanged,
    safe_error_message,
    sha256_file,
    sha256_json,
    stat_token,
)

SCRIPT_VERSION = "t1gr-u6-depth-view-v1"


def _write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if os.name != "nt":
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def _decode_and_classify(raw: bytes, suffix: str) -> str:
    try:
        import cv2
        import numpy as np
    except Exception:
        fail("T1GR_U6_DEPTH_DECODE_IMPORT_FAIL")
    value = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if value is None:
        fail("T1GR_U6_DEPTH_DECODE_FAIL")
    if suffix == ".png":
        if value.dtype != np.uint16 or value.ndim != 2:
            fail("T1GR_U6_METRIC_PNG_STORAGE_FAIL")
    elif suffix in {".jpg", ".jpeg"}:
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
            fail("T1GR_U6_UNKNOWN_JPG_STORAGE_FAIL")
    else:
        fail("T1GR_U6_DEPTH_EXTENSION_FORBIDDEN")
    _, _, kind = encode_depth_array(value, suffix)
    return kind


def _public_report(view: dict, manifest_path: Path, spec_path: Path, recipe_path: Path) -> dict:
    manifest = view["u6_manifest"]
    report = {
        "schema": SCHEMA_VIEW_PUBLIC,
        "script_version": SCRIPT_VERSION,
        "spec_file_sha256": sha256_file(spec_path),
        "recipe_public_sha256": sha256_file(recipe_path),
        "u6_view_manifest_private_sha256": sha256_file(manifest_path),
        "base_g_view_manifest_private_sha256": manifest["base_g_view_manifest_sha256"],
        "train_count": view["train_count"],
        "dev_count": view["dev_count"],
        "depth_kind_counts": manifest["depth_kind_counts"],
        "depth_mapping_commitment": manifest["mapping_commitment"],
        "unknown_scale_jpg_treatment": "QUARANTINE_AS_MISSING",
        "millimeter_reconstruction_performed": False,
        "any_raw_sample_id_present": False,
        "view_gate_passed": True,
        "legacy_primary_g_suite_unchanged": True,
        "final_holdout_ids_available_to_view": False,
        "final_holdout_open_authorized": False,
    }
    assert_public_safe(report)
    return report


def run(args) -> dict:
    repo = ROOT.resolve(strict=True)
    security_path = ensure_repo_input(repo, "config/t1gr_e5_security_policy.json", "config")
    if sha256_file(security_path) != FROZEN_E5_SECURITY_POLICY_SHA256:
        fail("E5_SECURITY_POLICY_SHA_DRIFT")
    security = read_json_bounded(security_path, 1 << 20, "t1gr-e5-security-policy-v2")
    spec_path = ensure_repo_input(repo, "config/t1gr_u6_design.frozen.json", "config")
    recipe_path = ensure_repo_input(repo, "reports/step4_t1gr/e5_v2_step1_recipe_public.json", "reports/step4_t1gr")
    base_manifest_path = ensure_private_input(Path(args.g_view_manifest), repo)
    zip_path = Path(args.formal_zip).expanduser().resolve(strict=False)
    if not zip_path.is_file():
        fail("FORMAL_ZIP_NOT_FOUND")
    out_root = Path(args.out_root).expanduser().resolve(strict=False)
    if is_within(out_root, repo) or not out_root.parent.is_dir() or not os.access(out_root.parent, os.W_OK):
        fail("T1GR_U6_VIEW_ROOT_INVALID")
    public_path = ensure_public_output(
        repo, "reports/step4_t1gr/t1gr_u6_view_public.json", security["public_output_prefix"]
    )
    deadline = Deadline(float(args.timeout_seconds))
    with file_lock(out_root.parent / f".{out_root.name}.t1gru6.lock", 5.0, 900.0):
        spec = read_json_bounded(spec_path, int(security["max_public_json_bytes"]), SCHEMA_SPEC)
        recipe = read_json_bounded(recipe_path, int(security["max_public_json_bytes"]), SCHEMA_RECIPE)
        validate_spec(spec)
        if not e5_payload_ok(recipe):
            fail("T1GR_U6_RECIPE_INTEGRITY_FAIL")
        base = verify_multimodal_view(base_manifest_path, recipe, deadline=deadline)
        zip_token = stat_token(zip_path)
        zip_sha = sha256_file(zip_path, deadline)
        if zip_sha != recipe["formal_zip_sha256"]:
            fail("T1GR_U6_FORMAL_ZIP_SHA_DRIFT")
        scan = scan_formal_zip(
            zip_path,
            deadline,
            max_members=int(security["max_zip_members"]),
            max_label_member_bytes=int(security["max_label_member_bytes"]),
            max_total_label_bytes=int(security["max_total_label_bytes"]),
        )
        if scan["metadata_commitment"] != recipe["formal_zip_metadata_commitment"] or scan["labels_commitment"] != recipe["labels_commitment"]:
            fail("T1GR_U6_FORMAL_ZIP_COMMITMENT_DRIFT")
        require_unchanged(zip_path, zip_token, "T1GR_U6_ZIP_CHANGED")
        request = sha256_json({
            "script": SCRIPT_VERSION,
            "spec": sha256_file(spec_path, deadline),
            "recipe": sha256_file(recipe_path, deadline),
            "base_g_view": sha256_file(base_manifest_path, deadline),
            "formal_zip": zip_sha,
            "out_root_binding": sha256_json(str(out_root).casefold() if os.name == "nt" else str(out_root)),
        })
        existing = check_existing_output(public_path, request)
        manifest_path = out_root / "u6_view_manifest.private.json"
        if out_root.exists():
            if not manifest_path.is_file():
                fail("T1GR_U6_VIEW_ROOT_UNMANAGED")
            view = verify_u6_view(manifest_path, recipe, deadline=deadline)
            if view["u6_manifest"].get("request_fingerprint") != request:
                fail("OUTPUT_CONFLICT_DIFFERENT_REQUEST")
            if existing is not None:
                return {"status": "PASS", "idempotent_reuse": True, "public_output_sha256": existing[1]}
            digest, _ = atomic_json_write(public_path, _public_report(view, manifest_path, spec_path, recipe_path), private=False, request_fingerprint=request)
            return {"status": "PASS", "idempotent_reuse": True, "public_output_sha256": digest}
        temp = out_root.parent / f".{out_root.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        temp.mkdir(mode=0o700)
        try:
            rows, kind_counts = [], {"METRIC_UINT16_PNG": 0, "UNKNOWN_SCALE_JPG_QUARANTINED": 0}
            with zipfile.ZipFile(zip_path) as archive:
                for split, ids, folder in (("train", base["ids"]["train"], "train"), ("dev", base["ids"]["dev"], "val")):
                    for sid in ids:
                        deadline.check("T1GR_U6_VIEW_BUILD_TIMEOUT")
                        info = scan["maps"]["depth"].get(sid)
                        if info is None:
                            fail("T1GR_U6_DEPTH_ID_MISSING")
                        suffix = Path(info.filename).suffix.lower()
                        raw = archive.read(info)
                        kind = _decode_and_classify(raw, suffix)
                        rel = f"depth/{folder}/{sid}{suffix}"
                        _write_private(temp / rel, raw)
                        digest = hashlib.sha256(raw).hexdigest()
                        rows.append({
                            "sample_id": sid,
                            "split": split,
                            "depth_rel": rel,
                            "depth_sha256": digest,
                            "depth_bytes": len(raw),
                            "depth_kind": kind,
                        })
                        kind_counts[kind] += 1
            if not all(value > 0 for value in kind_counts.values()):
                fail("T1GR_U6_BOTH_DEPTH_DOMAINS_REQUIRED")
            material = sorted((row["split"], row["sample_id"], row["depth_rel"], row["depth_sha256"], row["depth_kind"]) for row in rows)
            manifest = {
                "schema": SCHEMA_VIEW,
                "script_version": SCRIPT_VERSION,
                "spec_file_sha256": sha256_file(spec_path, deadline),
                "recipe_public_sha256": sha256_file(recipe_path, deadline),
                "formal_zip_sha256": zip_sha,
                "base_g_view_manifest": str(base_manifest_path),
                "base_g_view_root": str(base_manifest_path.parent),
                "base_g_view_manifest_sha256": sha256_file(base_manifest_path, deadline),
                "mappings": rows,
                "mapping_count": len(rows),
                "mapping_commitment": sha256_json(material),
                "depth_kind_counts": kind_counts,
                "train_count": len(base["ids"]["train"]),
                "dev_count": len(base["ids"]["dev"]),
                "unknown_scale_jpg_treatment": "QUARANTINE_AS_MISSING",
                "final_holdout_ids_present": False,
            }
            manifest["payload_sha256"] = payload_sha256(manifest)
            atomic_json_write(temp / "u6_view_manifest.private.json", manifest, private=True, request_fingerprint=request)
            if out_root.exists():
                fail("T1GR_U6_VIEW_OUTPUT_APPEARED_CONCURRENTLY")
            os.replace(temp, out_root)
            if os.name != "nt":
                os.chmod(out_root, 0o700)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        view = verify_u6_view(manifest_path, recipe, deadline=deadline)
        digest, _ = atomic_json_write(public_path, _public_report(view, manifest_path, spec_path, recipe_path), private=False, request_fingerprint=request)
        return {
            "status": "PASS",
            "idempotent_reuse": False,
            "public_output_sha256": digest,
            "train_count": view["train_count"],
            "dev_count": view["dev_count"],
            "depth_kind_counts": view["u6_manifest"]["depth_kind_counts"],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g-view-manifest", required=True)
    parser.add_argument("--formal-zip", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": safe_error_message(exc)}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
