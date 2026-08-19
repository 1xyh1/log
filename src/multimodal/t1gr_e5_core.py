"""Pure helpers for T1-GR E5 hardened Step1 RGB baseline chain."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import zipfile
import contextlib
import threading
import _thread
from pathlib import Path
from typing import Any

from .t1gr_secure_io import (
    Deadline, fail, require_dict, require_keys, require_list, require_nonempty_string,
    sha256_file, sha256_json, validate_identifier, validate_zip_name,
)

CLASS_NAMES = [
    "person", "boat", "animal", "seat", "sign", "bicycle",
    "car", "ball", "light", "garbage can", "uav", "tricycle",
]
CLASS_NAME_MAP = {str(i): name for i, name in enumerate(CLASS_NAMES)}

SCHEMA_RECIPE = "t1gr-e5-step1-recipe-public-v1"
SCHEMA_VIEW_PRIVATE = "t1gr-e5-step1-view-private-v1"
SCHEMA_VIEW_PUBLIC = "t1gr-e5-step1-view-public-v1"
SCHEMA_PREFLIGHT = "t1gr-e5-step1-preflight-public-v1"
SCHEMA_RUN = "t1gr-e5-step1-run-public-v1"
SCHEMA_EVAL = "t1gr-e5-step1-eval-public-v1"
SCHEMA_FINAL = "t1gr-e5-final-audit-public-v1"

E4_FREEZE_SCHEMA = "t1gr-e4-split-freeze-public-v1"
E4_VERIFY_SCHEMA = "t1gr-e4-seal-verification-public-v1"
E4_TRAIN_DEV_SCHEMA = "t1gr-e4-train-dev-access-private-v1"

FORENSIC_SCHEMA = "t1gr-zip-forensic-public-v1"
TAXONOMY_SCHEMA = "t1gr-label-error-taxonomy-public-v1"

FROZEN_E5_TRAINING_SPEC_SHA256 = "01be2a9443d068fca13ce7b4fdaee481fb16de9ca4bc12ad2ef756d64cdfd32e"
FROZEN_E5_SECURITY_POLICY_SHA256 = "656fcacb191aa7e85d463c1abb46203cd3a2eb2347cfec72dd09bb7ae4a18c52"

REQUIRED_TRAIN_ARGS = (
    "epochs", "batch", "imgsz", "patience", "optimizer", "lr0", "lrf", "momentum",
    "weight_decay", "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "nbs",
    "box", "cls", "cls_pw", "dfl", "cos_lr", "amp", "workers", "deterministic",
    "cache", "rect", "fraction", "multi_scale", "compile", "close_mosaic",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix",
    "copy_paste", "copy_paste_mode", "erasing", "single_cls", "val", "save",
    "save_period", "plots", "end2end", "seed",
)
REQUIRED_EVAL_ARGS = ("split", "conf", "iou", "max_det", "half", "dnn", "plots", "save_json")
REQUIRED_RUNTIME = (
    "device", "smoke_epochs", "smoke_timeout_seconds", "formal_timeout_seconds",
    "eval_timeout_seconds", "lock_wait_seconds", "lock_stale_seconds",
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(x: str) -> dt.datetime:
    try:
        v = dt.datetime.fromisoformat(require_nonempty_string(x, "BAD_TIMESTAMP").replace("Z", "+00:00"))
    except ValueError:
        fail("BAD_TIMESTAMP")
    if v.tzinfo is None:
        fail("TIMESTAMP_NOT_AWARE")
    return v.astimezone(dt.timezone.utc)


def payload_ok(obj: dict) -> bool:
    claimed = obj.get("payload_sha256")
    if not isinstance(claimed, str) or not HEX64.match(claimed):
        return False
    base = dict(obj)
    base.pop("payload_sha256", None)
    base.pop("request_fingerprint", None)
    return sha256_json(base) == claimed


def canonical_ids_sha(ids: list[str]) -> str:
    return sha256_json(sorted(ids))


def validate_id_list(obj: Any, code: str) -> list[str]:
    xs = require_list(obj, code)
    out = [validate_identifier(x) for x in xs]
    if len(out) != len(set(out)):
        fail("DUPLICATE_ID")
    return sorted(out)


def validate_e2_evidence(forensic: dict, taxonomy: dict) -> None:
    if forensic.get("schema") != FORENSIC_SCHEMA:
        fail("E2_FORENSIC_SCHEMA_FAIL")
    if taxonomy.get("schema") != TAXONOMY_SCHEMA:
        fail("E2_TAXONOMY_SCHEMA_FAIL")
    if int(forensic.get("common_id_count", -1)) != 2000 or forensic.get("id_sets_equal") is not True:
        fail("E2_PAIRING_FAIL")
    mc = forensic.get("member_counts") or {}
    if any(int(mc.get(k, -1)) != 2000 for k in ("visible", "infrared", "depth", "labels")):
        fail("E2_MEMBER_COUNTS_FAIL")
    if any(int(v) != 0 for v in (forensic.get("duplicate_id_counts") or {}).values()):
        fail("E2_DUPLICATE_ID_FAIL")
    if int(forensic.get("triplet_extension_mismatch_count", -1)) != 0:
        fail("E2_EXTENSION_ALIGNMENT_FAIL")
    if int(forensic.get("triplet_header_dimension_mismatch_count", -1)) != 0:
        fail("E2_DIMENSION_ALIGNMENT_FAIL")
    if int(forensic.get("header_error_count", -1)) != 0:
        fail("E2_HEADER_FAIL")
    ext=forensic.get("extension_counts") or {}
    for mod in ("visible","infrared","depth"):
        if (ext.get(mod) or {}).get(".jpg")!=149 or (ext.get(mod) or {}).get(".png")!=1851:
            fail("E2_EXTENSION_DOMAIN_DRIFT")
    names = (taxonomy.get("class_names") or forensic.get("labels", {}).get("class_names"))
    if list(names or []) != CLASS_NAMES:
        fail("E2_CLASS_MAP_DRIFT")
    a = taxonomy.get("adjudication_counts") or {}
    if int(a.get("hard_schema_or_class_samples", -1)) != 0:
        fail("E2_HARD_LABEL_FAIL")
    if int(a.get("ultralytics_8_4_56_reject_samples", -1)) != 0:
        fail("E2_ULTRALYTICS_LABEL_FAIL")
    if int(taxonomy.get("n_label_visible_pairs", -1)) != 2000:
        fail("E2_TAXONOMY_COUNT_FAIL")
    cats=taxonomy.get("sample_category_counts") or {}
    expected_cats={"CLEAN":1667,"DERIVED_CORNER_OVERFLOW":332,"DUPLICATE_ROWS":2,"STRICT_[0,1]_ONLY":3}
    if any(int(cats.get(k,-1))!=v for k,v in expected_cats.items()):
        fail("E2_LABEL_TAXONOMY_DRIFT")
    if int(a.get("corner_overflow_only_samples",-1))!=328 or int(a.get("duplicate_row_samples",-1))!=2:
        fail("E2_LABEL_ADJUDICATION_DRIFT")


def validate_e4_evidence(freeze: dict, verify: dict, train_dev: dict, train_dev_sha: str) -> dict:
    if freeze.get("schema") != E4_FREEZE_SCHEMA or freeze.get("seal_gate_passed") is not True:
        fail("E4_FREEZE_GATE_FAIL")
    if verify.get("schema") != E4_VERIFY_SCHEMA or verify.get("seal_verification_passed") is not True:
        fail("E4_VERIFY_GATE_FAIL")
    if verify.get("e5_entry_authorized") is not True:
        fail("E4_E5_ENTRY_NOT_AUTHORIZED")
    if freeze.get("step1_training_authorized") is not False or verify.get("step1_training_authorized") is not False:
        fail("E4_AUTHORIZATION_PROVENANCE_DRIFT")
    if freeze.get("final_holdout_open_authorized") is not False or verify.get("final_holdout_open_authorized") is not False:
        fail("E4_HOLDOUT_OPEN_DRIFT")
    if train_dev.get("schema") != E4_TRAIN_DEV_SCHEMA or not payload_ok(train_dev):
        fail("E4_TRAIN_DEV_PRIVATE_INTEGRITY_FAIL")
    if "final_holdout_ids" in train_dev:
        fail("E4_TRAIN_DEV_EXPOSES_HOLDOUT_IDS")
    if train_dev_sha != freeze.get("train_dev_access_private_sha256"):
        fail("E4_TRAIN_DEV_SHA_DRIFT")
    if train_dev.get("freeze_timestamp_utc") != freeze.get("freeze_timestamp_utc"):
        fail("E4_FREEZE_TIMESTAMP_DRIFT")
    train = validate_id_list(train_dev.get("train_ids"), "E4_TRAIN_IDS_MISSING")
    dev = validate_id_list(train_dev.get("dev_ids"), "E4_DEV_IDS_MISSING")
    if set(train) & set(dev):
        fail("E4_TRAIN_DEV_OVERLAP")
    counts = freeze.get("sample_counts") or {}
    if len(train) != int(counts.get("train", -1)) or len(dev) != int(counts.get("dev", -1)):
        fail("E4_TRAIN_DEV_COUNT_DRIFT")
    hold_count = int(train_dev.get("final_holdout_count", -1))
    if hold_count != int(counts.get("final_holdout", -1)):
        fail("E4_HOLDOUT_COUNT_DRIFT")
    if len(train) + len(dev) + hold_count != 2000:
        fail("E4_TOTAL_COUNT_DRIFT")
    commits = freeze.get("ids_commitments") or {}
    got_train = canonical_ids_sha(train)
    got_dev = canonical_ids_sha(dev)
    if got_train != commits.get("train") or got_train != train_dev.get("train_ids_sha256"):
        fail("E4_TRAIN_COMMITMENT_DRIFT")
    if got_dev != commits.get("dev") or got_dev != train_dev.get("dev_ids_sha256"):
        fail("E4_DEV_COMMITMENT_DRIFT")
    hold_commit = train_dev.get("final_holdout_ids_sha256")
    if hold_commit != commits.get("final_holdout") or not isinstance(hold_commit, str) or not HEX64.match(hold_commit):
        fail("E4_HOLDOUT_COMMITMENT_DRIFT")
    if verify.get("ids_commitments") != commits or verify.get("sample_counts") != counts:
        fail("E4_VERIFY_PUBLIC_DRIFT")
    return {
        "train": train,
        "dev": dev,
        "holdout_count": hold_count,
        "commits": commits,
        "counts": counts,
        "freeze_timestamp_utc": freeze["freeze_timestamp_utc"],
    }


def _need_numeric(d: dict, key: str, *, minv=None, maxv=None, integer=False) -> None:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        fail("TRAINING_SPEC_TYPE_FAIL", key)
    if integer and int(v) != v:
        fail("TRAINING_SPEC_TYPE_FAIL", key)
    if not math.isfinite(float(v)):
        fail("TRAINING_SPEC_NONFINITE", key)
    if minv is not None and float(v) < minv:
        fail("TRAINING_SPEC_RANGE_FAIL", key)
    if maxv is not None and float(v) > maxv:
        fail("TRAINING_SPEC_RANGE_FAIL", key)


def validate_training_spec(spec: dict) -> None:
    if spec.get("schema") != "t1gr-e5-training-spec-v1":
        fail("E5_TRAINING_SPEC_SCHEMA_FAIL")
    if spec.get("status") != "REVIEWED_FROZEN":
        fail("E5_TRAINING_SPEC_NOT_REVIEWED")
    train = require_dict(spec.get("train_args"), "E5_TRAIN_ARGS_MISSING")
    eva = require_dict(spec.get("eval_args"), "E5_EVAL_ARGS_MISSING")
    runtime = require_dict(spec.get("runtime"), "E5_RUNTIME_MISSING")
    require_keys(train, REQUIRED_TRAIN_ARGS, "E5_TRAIN_ARGS_UNRESOLVED")
    require_keys(eva, REQUIRED_EVAL_ARGS, "E5_EVAL_ARGS_UNRESOLVED")
    require_keys(runtime, REQUIRED_RUNTIME, "E5_RUNTIME_UNRESOLVED")

    allowed_opt={"sgd","musgd","adam","adamax","adamw","nadam","radam","rmsprop","auto"}
    if not isinstance(train["optimizer"],str) or train["optimizer"].lower() not in allowed_opt:
        fail("E5_OPTIMIZER_INVALID")
    if str(train["optimizer"]).lower() == "auto":
        fail("E5_OPTIMIZER_AUTO_FORBIDDEN")
    if train["deterministic"] is not True:
        fail("E5_DETERMINISTIC_REQUIRED")
    if train["end2end"] is not True:
        fail("E5_YOLO26_END2END_REQUIRED")
    if train["single_cls"] is not False or train["val"] is not True or train["save"] is not True:
        fail("E5_CORE_TRAIN_POLICY_FAIL")
    if eva["split"] != "val":
        fail("E5_EVAL_MUST_BE_DEV_VAL")
    if spec.get("architecture") != "yolo26s" or spec.get("model_yaml") != "yolo26s.yaml":
        fail("E5_MODEL_ARCH_DRIFT")
    if int(spec.get("num_classes", -1)) != 12:
        fail("E5_NUM_CLASSES_DRIFT")
    if str(spec.get("expected_ultralytics_version")) != "8.4.56":
        fail("E5_ULTRALYTICS_EXPECTED_VERSION_DRIFT")

    for k in ("epochs", "batch", "imgsz", "patience", "workers", "nbs", "close_mosaic", "save_period", "seed"):
        _need_numeric(train, k, integer=True)
    if int(train["epochs"]) <= 0 or int(train["batch"]) <= 0 or int(train["imgsz"]) <= 0:
        fail("E5_TRAIN_SIZE_RANGE_FAIL")
    if int(train["workers"]) < 0 or int(train["close_mosaic"]) < 0:
        fail("E5_WORKER_OR_MOSAIC_RANGE_FAIL")
    for k in ("lr0","lrf","momentum","weight_decay","warmup_epochs","warmup_momentum","warmup_bias_lr",
              "box","cls","cls_pw","dfl","hsv_h","hsv_s","hsv_v","degrees","translate","scale",
              "shear","perspective","flipud","fliplr","bgr","mosaic","mixup","cutmix","copy_paste","erasing",
              "fraction","multi_scale"):
        _need_numeric(train, k)
    for k in ("conf","iou"):
        _need_numeric(eva, k, minv=0.0, maxv=1.0)
    _need_numeric(eva, "max_det", minv=1, integer=True)
    for k in ("smoke_epochs","smoke_timeout_seconds","formal_timeout_seconds","eval_timeout_seconds",
              "lock_wait_seconds","lock_stale_seconds"):
        _need_numeric(runtime, k, minv=1.0)
    if int(runtime["smoke_epochs"]) != 1:
        fail("E5_SMOKE_EPOCHS_MUST_BE_ONE")
    if not isinstance(runtime["device"], (str, int)) or str(runtime["device"]).strip() == "":
        fail("E5_DEVICE_INVALID")
    ds=str(runtime["device"]).strip().lower()
    if "," in ds or ds.startswith("["):
        fail("E5_MULTI_GPU_FORBIDDEN")
    bool_train=("amp","deterministic","rect","cos_lr","compile","single_cls","val","save","plots")
    if any(not isinstance(train[k],bool) for k in bool_train): fail("E5_TRAINING_SPEC_BOOL_TYPE_FAIL")
    bool_eval=("half","dnn","plots","save_json")
    if any(not isinstance(eva[k],bool) for k in bool_eval): fail("E5_EVAL_SPEC_BOOL_TYPE_FAIL")
    if train["copy_paste_mode"] not in {"flip","mixup"}: fail("E5_COPY_PASTE_MODE_INVALID")


def zip_modality(name: str) -> str | None:
    parts = [x for x in name.replace("\\", "/").split("/") if x]
    if len(parts) < 2:
        return None
    return parts[-2].lower()


def zip_sid(name: str) -> str:
    return Path(name).stem


def scan_formal_zip(zp: Path, deadline: Deadline, *, max_members: int = 12000,
                    max_label_member_bytes: int = 1_000_000, max_total_label_bytes: int = 100_000_000) -> dict:
    try:
        z = zipfile.ZipFile(zp)
    except FileNotFoundError:
        fail("FORMAL_ZIP_NOT_FOUND")
    except PermissionError:
        fail("READ_PERMISSION_DENIED")
    except zipfile.BadZipFile:
        fail("FORMAL_ZIP_BAD")
    maps = {m: {} for m in ("visible","infrared","depth","labels")}
    metadata = []
    label_hashes = []
    total_label = 0
    with z:
        infos = z.infolist()
        if len(infos) > int(max_members):
            fail("ZIP_MEMBER_COUNT_EXCEEDED")
        seen_names = set()
        for info in infos:
            deadline.check("ZIP_SCAN_TIMEOUT")
            if info.is_dir():
                continue
            validate_zip_name(info.filename)
            norm = info.filename.replace("\\", "/")
            if norm in seen_names:
                fail("ZIP_DUPLICATE_MEMBER_NAME")
            seen_names.add(norm)
            if info.flag_bits & 0x1:
                fail("ZIP_ENCRYPTED_MEMBER_FORBIDDEN")
            metadata.append((norm, int(info.CRC), int(info.file_size), int(info.compress_size)))
            mod = zip_modality(norm)
            if mod not in maps:
                continue
            sid = validate_identifier(zip_sid(norm))
            if sid in maps[mod]:
                fail("ZIP_DUPLICATE_SAMPLE_ID")
            maps[mod][sid] = info
        sets = {m:set(v) for m,v in maps.items()}
        if any(len(v) != 2000 for v in sets.values()) or len(set.intersection(*sets.values())) != 2000:
            fail("ZIP_FORMAL_PAIRING_DRIFT")
        for sid in sorted(maps["labels"]):
            deadline.check("LABEL_READ_TIMEOUT")
            info = maps["labels"][sid]
            if info.file_size < 0 or info.file_size > int(max_label_member_bytes):
                fail("LABEL_MEMBER_SIZE_EXCEEDED")
            total_label += int(info.file_size)
            if total_label > int(max_total_label_bytes):
                fail("TOTAL_LABEL_BYTES_EXCEEDED")
            try:
                raw = z.read(info)
            except Exception:
                fail("ZIP_MEMBER_READ_ERROR")
            label_hashes.append((sid, sha256_json([raw.hex()])))
    return {
        "maps": maps,
        "metadata_commitment": sha256_json(sorted(metadata)),
        "labels_commitment": sha256_json(label_hashes),
        "member_counts": {m: len(maps[m]) for m in maps},
    }


def environment_probe() -> dict:
    try:
        import platform
        import torch
        import ultralytics
    except Exception:
        fail("E5_RUNTIME_IMPORT_FAIL")
    try:
        from ultralytics.utils.events import events as ul_events
        ul_events.enabled = False
        analytics_disabled = True
    except Exception:
        analytics_disabled = False
    ultra_root = Path(ultralytics.__file__).resolve().parent
    source_files = {
        "default_yaml": ultra_root / "cfg" / "default.yaml",
        "trainer_py": ultra_root / "engine" / "trainer.py",
        "yolo26_yaml": ultra_root / "cfg" / "models" / "26" / "yolo26.yaml",
        "tasks_py": ultra_root / "nn" / "tasks.py",
        "events_py": ultra_root / "utils" / "events.py",
    }
    source_sha = {}
    for key, path in source_files.items():
        if not path.is_file():
            fail("E5_ULTRALYTICS_SOURCE_FILE_MISSING", key)
        source_sha[key] = sha256_file(path)
    out = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "ultralytics_version": str(ultralytics.__version__),
        "ultralytics_analytics_disabled_for_process": analytics_disabled,
        "ultralytics_source_sha256": source_sha,
        "cuda_runtime_version": str(getattr(torch.version, "cuda", None)),
        "cudnn_version": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability(0)
            out["device_compute_capability"] = f"{cap[0]}.{cap[1]}"
        except Exception:
            out["device_compute_capability"] = None
    else:
        out["device_compute_capability"] = None
    return out


def compare_environment(runtime: dict, expected: dict) -> None:
    for k in ("python_version","torch_version","ultralytics_version","cuda_runtime_version","cudnn_version","cuda_available",
              "ultralytics_analytics_disabled_for_process","ultralytics_source_sha256",
              "cuda_device_count","cuda_device_name"):
        if runtime.get(k) != expected.get(k):
            fail("E5_RUNTIME_ENV_DRIFT", k)
    if expected.get("device_compute_capability") is not None and runtime.get("device_compute_capability") != expected.get("device_compute_capability"):
        fail("E5_RUNTIME_ENV_DRIFT", "device_compute_capability")


def effective_args_mismatch(obj: Any, expected: dict) -> dict:
    out = {}
    for k, exp in expected.items():
        got = getattr(obj, k, None) if not isinstance(obj, dict) else obj.get(k)
        if isinstance(got, Path):
            got = str(got)
        if isinstance(exp, Path):
            exp = str(exp)
        if got != exp:
            out[k] = {"expected": exp, "effective": got}
    return out


def results_csv_epoch_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        fail("RESULTS_CSV_READ_FAIL")
    if not rows:
        fail("RESULTS_CSV_EMPTY")
    return len(rows)


def optimizer_fingerprint(opt: Any) -> dict:
    if opt is None:
        fail("OPTIMIZER_NOT_BUILT")
    groups = []
    for g in getattr(opt, "param_groups", []):
        groups.append({
            "lr": float(g.get("lr", 0.0)),
            "initial_lr": float(g.get("initial_lr", g.get("lr", 0.0))),
            "momentum": float(g["momentum"]) if "momentum" in g else None,
            "betas": [float(x) for x in g["betas"]] if "betas" in g else None,
            "weight_decay": float(g.get("weight_decay", 0.0)),
            "parameter_count": len(g.get("params", [])),
        })
    return {
        "class_name": type(opt).__name__,
        "module": type(opt).__module__,
        "param_group_count": len(groups),
        "groups": groups,
    }


def verify_view_tree(manifest_path: Path, recipe: dict, train_dev: dict, deadline: Deadline) -> dict:
    from .t1gr_secure_io import read_json_bounded, stat_token, require_unchanged

    token = stat_token(manifest_path)
    m = read_json_bounded(manifest_path, 64 * 1024 * 1024, SCHEMA_VIEW_PRIVATE)
    if not payload_ok(m):
        fail("E5_VIEW_PRIVATE_INTEGRITY_FAIL")
    root = manifest_path.parent.resolve(strict=True)
    if m.get("recipe_sha256") != recipe.get("recipe_sha256_self"):
        fail("E5_VIEW_RECIPE_PIN_FAIL")
    if m.get("train_dev_access_private_sha256") != recipe.get("train_dev_access_private_sha256"):
        fail("E5_VIEW_TRAIN_DEV_PIN_FAIL")
    if m.get("formal_zip_sha256") != recipe.get("formal_zip_sha256"):
        fail("E5_VIEW_ZIP_PIN_FAIL")

    train = validate_id_list(m.get("train_ids"), "E5_VIEW_TRAIN_IDS_MISSING")
    dev = validate_id_list(m.get("dev_ids"), "E5_VIEW_DEV_IDS_MISSING")
    td_train = validate_id_list(train_dev.get("train_ids"), "E5_TD_TRAIN_IDS_MISSING")
    td_dev = validate_id_list(train_dev.get("dev_ids"), "E5_TD_DEV_IDS_MISSING")
    if train != td_train or dev != td_dev:
        fail("E5_VIEW_ID_LIST_DRIFT")
    if canonical_ids_sha(train) != recipe["ids_commitments"]["train"]:
        fail("E5_VIEW_TRAIN_COMMITMENT_DRIFT")
    if canonical_ids_sha(dev) != recipe["ids_commitments"]["dev"]:
        fail("E5_VIEW_DEV_COMMITMENT_DRIFT")
    if set(train) & set(dev):
        fail("E5_VIEW_TRAIN_DEV_OVERLAP")
    if len(train) + len(dev) + int(recipe["sample_counts"]["final_holdout"]) != 2000:
        fail("E5_VIEW_TOTAL_COUNT_DRIFT")

    yaml_rel = require_nonempty_string(m.get("dataset_yaml_rel"), "E5_VIEW_YAML_REL_MISSING")
    yaml_path = (root / yaml_rel).resolve(strict=False)
    if root not in yaml_path.parents:
        fail("E5_VIEW_PATH_ESCAPE")
    if not yaml_path.is_file():
        fail("E5_VIEW_DATASET_YAML_MISSING")
    if sha256_file(yaml_path, deadline) != m.get("dataset_yaml_sha256"):
        fail("E5_VIEW_YAML_SHA_DRIFT")

    rows = require_list(m.get("mappings"), "E5_VIEW_MAPPINGS_MISSING")
    if len(rows) != len(train) + len(dev):
        fail("E5_VIEW_MAPPING_COUNT_DRIFT")
    seen = set()
    actual = {"train": set(), "dev": set()}
    mapping_commit_rows = []
    for row in rows:
        deadline.check("E5_VIEW_VERIFY_TIMEOUT")
        row = require_dict(row, "E5_VIEW_MAPPING_ROW_BAD")
        require_keys(row, ("sample_id","split","image_rel","label_rel","image_sha256","label_sha256"),
                     "E5_VIEW_MAPPING_ROW_MISSING")
        sid = validate_identifier(row["sample_id"])
        sp = row["split"]
        if sp not in ("train","dev"):
            fail("E5_VIEW_MAPPING_SPLIT_BAD")
        key = (sp, sid)
        if key in seen:
            fail("E5_VIEW_MAPPING_DUPLICATE")
        seen.add(key)
        actual[sp].add(sid)
        ir = require_nonempty_string(row["image_rel"], "E5_VIEW_IMAGE_REL_BAD")
        lr = require_nonempty_string(row["label_rel"], "E5_VIEW_LABEL_REL_BAD")
        ip = (root / ir).resolve(strict=False)
        lp = (root / lr).resolve(strict=False)
        if root not in ip.parents or root not in lp.parents:
            fail("E5_VIEW_PATH_ESCAPE")
        if not ip.is_file() or not lp.is_file():
            fail("E5_VIEW_FILE_MISSING")
        ish = sha256_file(ip, deadline)
        lsh = sha256_file(lp, deadline)
        if ish != row["image_sha256"] or lsh != row["label_sha256"]:
            fail("E5_VIEW_FILE_SHA_DRIFT")
        mapping_commit_rows.append((sp,sid,ir,ish,lr,lsh))
    if actual["train"] != set(train) or actual["dev"] != set(dev):
        fail("E5_VIEW_MAPPING_ID_COVERAGE_DRIFT")

    # Independent filesystem enumeration catches injected/extra files not listed in mappings.
    def stems(kind: str, sp: str) -> set[str]:
        d = root / kind / ("train" if sp == "train" else "val")
        if not d.is_dir():
            fail("E5_VIEW_SPLIT_DIR_MISSING")
        files = [p for p in d.iterdir() if p.is_file()]
        ids = [p.stem for p in files]
        if len(ids) != len(set(ids)):
            fail("E5_VIEW_DUPLICATE_STEM")
        return set(ids)
    for sp, expected in (("train",set(train)),("dev",set(dev))):
        if stems("images",sp) != expected or stems("labels",sp) != expected:
            fail("E5_VIEW_EXTRA_OR_MISSING_FILE")

    if sha256_json(sorted(mapping_commit_rows)) != m.get("mapping_commitment"):
        fail("E5_VIEW_MAPPING_COMMITMENT_DRIFT")
    require_unchanged(manifest_path, token, "E5_VIEW_MANIFEST_CHANGED_DURING_VERIFY")
    return {
        "manifest": m, "root": root, "dataset_yaml": yaml_path,
        "train_count": len(train), "dev_count": len(dev),
        "mapping_commitment": m["mapping_commitment"],
    }


@contextlib.contextmanager
def wall_clock_watchdog(seconds: float, code: str):
    if not isinstance(seconds,(int,float)) or not math.isfinite(float(seconds)) or float(seconds)<=0:
        fail("BAD_TIMEOUT")
    done=threading.Event();state={"expired":False}
    def worker():
        if not done.wait(float(seconds)):
            state["expired"]=True
            _thread.interrupt_main()
    t=threading.Thread(target=worker,name="t1gr-e5-watchdog",daemon=True);t.start()
    try:
        yield
    except KeyboardInterrupt:
        if state["expired"]:
            fail(code)
        raise
    finally:
        done.set()


@contextlib.contextmanager
def ultralytics_offline_guard(*, bypass_amp_download_check: bool):
    """Disable Ultralytics analytics/integration callbacks for this process.

    When bypass_amp_download_check=True, also replace the trainer's network-capable
    AMP probe with an explicit True result. The mandatory 1-epoch smoke is then the
    AMP qualification gate before formal training.
    """
    try:
        from ultralytics.utils import callbacks as ul_callbacks
        from ultralytics.utils.events import events as ul_events
        import ultralytics.engine.trainer as trainer_mod
    except Exception:
        fail("E5_ULTRALYTICS_OFFLINE_GUARD_IMPORT_FAIL")
    old_add = ul_callbacks.add_integration_callbacks
    old_enabled = bool(getattr(ul_events, "enabled", False))
    old_check_amp = getattr(trainer_mod, "check_amp", None)
    ul_callbacks.add_integration_callbacks = lambda instance: None
    ul_events.enabled = False
    if bypass_amp_download_check:
        if old_check_amp is None:
            fail("E5_AMP_CHECK_PATCH_TARGET_MISSING")
        trainer_mod.check_amp = lambda model: True
    try:
        yield {
            "analytics_disabled": True,
            "integration_callbacks_disabled": True,
            "amp_download_probe_bypassed": bool(bypass_amp_download_check),
        }
    finally:
        ul_callbacks.add_integration_callbacks = old_add
        ul_events.enabled = old_enabled
        if bypass_amp_download_check and old_check_amp is not None:
            trainer_mod.check_amp = old_check_amp


@contextlib.contextmanager
def private_umask():
    """Restrict POSIX-created run artifacts to current user; no NTFS ACL claim."""
    if os.name == "nt":
        yield {"posix_umask_applied": False, "windows_acl_proven": False}
        return
    old = os.umask(0o077)
    try:
        yield {"posix_umask_applied": True, "windows_acl_proven": None}
    finally:
        os.umask(old)
