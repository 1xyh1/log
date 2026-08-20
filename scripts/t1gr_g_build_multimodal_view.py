#!/usr/bin/env python3
"""Build a private TRAIN/DEV-only RGB+IR view from the pinned formal ZIP."""
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

from multimodal.t1gr_e5_core import (  # noqa: E402
    FROZEN_E5_SECURITY_POLICY_SHA256,
    SCHEMA_RECIPE,
    canonical_ids_sha,
    payload_ok,
    scan_formal_zip,
    validate_e4_evidence,
)
from multimodal.t1gr_g_impl_core import (  # noqa: E402
    SCHEMA_VIEW_PRIVATE,
    SCHEMA_VIEW_PUBLIC,
    payload_sha256,
    validate_impl_spec,
)
from multimodal.t1gr_g_runtime import read_json, validate_frozen_chain, verify_multimodal_view  # noqa: E402
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

SCRIPT_VERSION = "t1gr-g-build-multimodal-view-v1"


def _raw_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except FileExistsError:
        fail("T1GR_G_VIEW_TEMP_COLLISION")
    except PermissionError:
        fail("WRITE_PERMISSION_DENIED")
    except OSError:
        fail("WRITE_IO_ERROR")


def run(args) -> dict:
    repo = ROOT.resolve(strict=True)
    security_path = ensure_repo_input(repo, "config/t1gr_e5_security_policy.json", "config")
    if sha256_file(security_path) != FROZEN_E5_SECURITY_POLICY_SHA256:
        fail("E5_SECURITY_POLICY_SHA_DRIFT")
    security = read_json_bounded(security_path, 1 << 20, "t1gr-e5-security-policy-v2")
    recipe_path = ensure_repo_input(repo, "reports/step4_t1gr/e5_v2_step1_recipe_public.json", "reports/step4_t1gr")
    design_path = ensure_repo_input(repo, "config/t1gr_g_design.frozen.json", "config")
    spec_path = ensure_repo_input(repo, "config/t1gr_g_implementation_spec.frozen.json", "config")
    freeze_path = ensure_repo_input(repo, "reports/step4_t1gr/e4_split_freeze_public.json", "reports/step4_t1gr")
    verify_path = ensure_repo_input(repo, "reports/step4_t1gr/e4_seal_verification_public.json", "reports/step4_t1gr")
    design_audit_path = ensure_repo_input(repo, "reports/step4_t1gr/t1gr_g_design_audit_public.json", "reports/step4_t1gr")
    train_dev_path = ensure_private_input(Path(args.train_dev_access), repo)
    out_root = Path(args.out_root).expanduser().resolve(strict=False)
    if is_within(out_root, repo):
        fail("T1GR_G_VIEW_ROOT_INSIDE_REPO")
    if not out_root.parent.is_dir() or not os.access(out_root.parent, os.W_OK):
        fail("T1GR_G_VIEW_PARENT_NOT_WRITABLE")
    public_path = ensure_public_output(
        repo, "reports/step4_t1gr/t1gr_g_multimodal_view_public.json", security["public_output_prefix"]
    )
    zip_path = Path(args.formal_zip).expanduser().resolve(strict=False)
    if not zip_path.is_file():
        fail("FORMAL_ZIP_NOT_FOUND")
    deadline = Deadline(float(args.timeout_seconds or security["view_build_timeout_seconds"]))
    lock = out_root.parent / f".{out_root.name}.t1grgview.lock"
    with file_lock(lock, 5.0, 900.0):
        recipe = read_json_bounded(recipe_path, int(security["max_public_json_bytes"]), SCHEMA_RECIPE)
        design = read_json(design_path)
        spec = read_json(spec_path)
        validate_impl_spec(spec)
        validate_frozen_chain(design, recipe, spec)
        if sha256_file(design_path, deadline) != spec["upstream"]["design_file_sha256"]:
            fail("T1GR_G_DESIGN_FILE_SHA_DRIFT")
        design_audit = read_json(design_audit_path)
        if design_audit.get("design_freeze_passed") is not True or design_audit.get("implementation_entry_authorized") is not True:
            fail("T1GR_G_DESIGN_AUDIT_NOT_PASS")
        if not payload_ok(recipe):
            fail("T1GR_G_RECIPE_INTEGRITY_FAIL")
        freeze = read_json_bounded(freeze_path, int(security["max_public_json_bytes"]))
        verification = read_json_bounded(verify_path, int(security["max_public_json_bytes"]))
        train_dev = read_json_bounded(train_dev_path, int(security["max_private_json_bytes"]))
        e4 = validate_e4_evidence(freeze, verification, train_dev, sha256_file(train_dev_path, deadline))
        if e4["commits"] != recipe["ids_commitments"]:
            fail("T1GR_G_E4_RECIPE_COMMITMENT_DRIFT")
        zip_token = stat_token(zip_path)
        zip_sha = sha256_file(zip_path, deadline)
        if zip_sha != recipe["formal_zip_sha256"]:
            fail("T1GR_G_FORMAL_ZIP_SHA_DRIFT")
        scan = scan_formal_zip(
            zip_path,
            deadline,
            max_members=int(security["max_zip_members"]),
            max_label_member_bytes=int(security["max_label_member_bytes"]),
            max_total_label_bytes=int(security["max_total_label_bytes"]),
        )
        if scan["metadata_commitment"] != recipe["formal_zip_metadata_commitment"]:
            fail("T1GR_G_ZIP_METADATA_DRIFT")
        if scan["labels_commitment"] != recipe["labels_commitment"]:
            fail("T1GR_G_ZIP_LABEL_DRIFT")
        require_unchanged(zip_path, zip_token, "T1GR_G_ZIP_CHANGED_DURING_BUILD")
        request_fingerprint = sha256_json({
            "script": SCRIPT_VERSION,
            "recipe": sha256_file(recipe_path, deadline),
            "design": sha256_file(design_path, deadline),
            "implementation": sha256_file(spec_path, deadline),
            "train_dev": sha256_file(train_dev_path, deadline),
            "formal_zip": zip_sha,
            "out_root_binding": sha256_json(str(out_root).casefold() if os.name == "nt" else str(out_root)),
        })
        existing_public = check_existing_output(public_path, request_fingerprint)
        manifest_path = out_root / "view_manifest.json"
        if out_root.exists():
            if not manifest_path.is_file():
                fail("T1GR_G_VIEW_ROOT_UNMANAGED")
            view = verify_multimodal_view(manifest_path, recipe, deadline=deadline)
            manifest = view["manifest"]
            if manifest.get("request_fingerprint") != request_fingerprint:
                fail("OUTPUT_CONFLICT_DIFFERENT_REQUEST")
            if existing_public is not None:
                return {"status": "PASS", "idempotent_reuse": True, "public_output_sha256": existing_public[1]}
            report = _public_report(view, manifest_path, recipe, recipe_path, spec_path, zip_sha)
            digest, _ = atomic_json_write(
                public_path, report, private=False, request_fingerprint=request_fingerprint
            )
            return {"status": "PASS", "idempotent_reuse": True, "public_output_sha256": digest}

        temp = out_root.parent / f".{out_root.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        temp.mkdir(mode=0o700)
        try:
            rows = []
            extension_counts: dict[str, int] = {}
            with zipfile.ZipFile(zip_path) as archive:
                for split, ids, folder in (("train", e4["train"], "train"), ("dev", e4["dev"], "val")):
                    for sid in ids:
                        deadline.check("T1GR_G_VIEW_BUILD_TIMEOUT")
                        infos = [scan["maps"][mod].get(sid) for mod in ("visible", "infrared", "labels")]
                        if any(info is None for info in infos):
                            fail("T1GR_G_VIEW_SOURCE_ID_MISSING")
                        visible_info, ir_info, label_info = infos
                        visible_ext = Path(visible_info.filename).suffix.lower()
                        ir_ext = Path(ir_info.filename).suffix.lower()
                        if visible_ext not in {".jpg", ".jpeg", ".png"} or ir_ext not in {".jpg", ".jpeg", ".png"}:
                            fail("T1GR_G_VIEW_IMAGE_EXTENSION_BAD")
                        visible_bytes = archive.read(visible_info)
                        ir_bytes = archive.read(ir_info)
                        label_bytes = archive.read(label_info)
                        image_rel = f"images/{folder}/{sid}{visible_ext}"
                        infrared_rel = f"infrared/{folder}/{sid}{ir_ext}"
                        label_rel = f"labels/{folder}/{sid}.txt"
                        _write_private(temp / image_rel, visible_bytes)
                        _write_private(temp / infrared_rel, ir_bytes)
                        _write_private(temp / label_rel, label_bytes)
                        rows.append({
                            "sample_id": sid,
                            "split": split,
                            "image_rel": image_rel,
                            "infrared_rel": infrared_rel,
                            "label_rel": label_rel,
                            "image_sha256": _raw_sha(visible_bytes),
                            "infrared_sha256": _raw_sha(ir_bytes),
                            "label_sha256": _raw_sha(label_bytes),
                            "image_bytes": len(visible_bytes),
                            "infrared_bytes": len(ir_bytes),
                            "label_bytes": len(label_bytes),
                        })
                        key = f"{split}_{visible_ext.lstrip('.')}_{ir_ext.lstrip('.')}"
                        extension_counts[key] = extension_counts.get(key, 0) + 1
            final_abs = out_root.as_posix()
            yaml_text = (
                f"path: {json.dumps(final_abs, ensure_ascii=False)}\n"
                "train: images/train\nval: images/val\nchannels: 4\nnc: 12\nnames:\n"
                + "".join(
                    f"  {i}: {json.dumps(recipe['class_names'][str(i)], ensure_ascii=False)}\n"
                    for i in range(12)
                )
            )
            _write_private(temp / "dataset.yaml", yaml_text.encode("utf-8"))
            material = sorted(
                (
                    row["split"], row["sample_id"], row["image_rel"], row["image_sha256"],
                    row["infrared_rel"], row["infrared_sha256"], row["label_rel"], row["label_sha256"],
                )
                for row in rows
            )
            manifest = {
                "schema": SCHEMA_VIEW_PRIVATE,
                "script_version": SCRIPT_VERSION,
                "recipe_public_file_sha256": sha256_file(recipe_path, deadline),
                "design_file_sha256": sha256_file(design_path, deadline),
                "implementation_spec_file_sha256": sha256_file(spec_path, deadline),
                "train_dev_access_private_sha256": sha256_file(train_dev_path, deadline),
                "formal_zip_sha256": zip_sha,
                "train_ids": e4["train"],
                "dev_ids": e4["dev"],
                "train_ids_sha256": canonical_ids_sha(e4["train"]),
                "dev_ids_sha256": canonical_ids_sha(e4["dev"]),
                "dataset_yaml_rel": "dataset.yaml",
                "dataset_yaml_sha256": _raw_sha(yaml_text.encode("utf-8")),
                "mappings": rows,
                "mapping_count": len(rows),
                "mapping_commitment": sha256_json(material),
                "extension_counts": extension_counts,
                "final_holdout_count": e4["holdout_count"],
                "final_holdout_ids_sha256": e4["commits"]["final_holdout"],
                "final_holdout_ids_present": False,
            }
            manifest["payload_sha256"] = payload_sha256(manifest)
            atomic_json_write(
                temp / "view_manifest.json", manifest, private=True, request_fingerprint=request_fingerprint
            )
            if out_root.exists():
                fail("T1GR_G_VIEW_OUTPUT_APPEARED_CONCURRENTLY")
            os.replace(temp, out_root)
            if os.name != "nt":
                os.chmod(out_root, 0o700)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        view = verify_multimodal_view(out_root / "view_manifest.json", recipe, deadline=deadline)
        report = _public_report(view, out_root / "view_manifest.json", recipe, recipe_path, spec_path, zip_sha)
        assert_public_safe(report)
        digest, _ = atomic_json_write(public_path, report, private=False, request_fingerprint=request_fingerprint)
        return {
            "status": "PASS",
            "idempotent_reuse": False,
            "public_output_sha256": digest,
            "train_count": view["train_count"],
            "dev_count": view["dev_count"],
        }


def _public_report(view, manifest_path: Path, recipe, recipe_path: Path, spec_path: Path, zip_sha: str) -> dict:
    report = {
        "schema": SCHEMA_VIEW_PUBLIC,
        "script_version": SCRIPT_VERSION,
        "recipe_public_sha256": sha256_file(recipe_path),
        "implementation_spec_sha256": sha256_file(spec_path),
        "view_manifest_private_sha256": sha256_file(manifest_path),
        "dataset_yaml_sha256": view["manifest"]["dataset_yaml_sha256"],
        "formal_zip_sha256": zip_sha,
        "train_count": view["train_count"],
        "dev_count": view["dev_count"],
        "final_holdout_count": recipe["sample_counts"]["final_holdout"],
        "ids_commitments": recipe["ids_commitments"],
        "mapping_commitment": view["mapping_commitment"],
        "modalities": {"visible": True, "infrared": True, "labels": True},
        "depth_present": False,
        "final_holdout_ids_available_to_view": False,
        "any_raw_sample_id_present": False,
        "view_gate_passed": True,
        "final_holdout_open_authorized": False,
    }
    assert_public_safe(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dev-access", required=True)
    parser.add_argument("--formal-zip", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print(json.dumps({"status": "FAIL", "error": "USER_INTERRUPT"}), file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": safe_error_message(exc)}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
