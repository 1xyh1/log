"""View verification and training runtime for the isolated T1-U6 suite."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

from .t1gr_u6_core import ARMS, SCHEMA_VIEW, ZERO_IR, payload_ok, sha256_json
from .t1gr_u6_dataset import T1GRU6Dataset
from .t1gr_g_core import DEV_COUNT, SEEDS, TRAIN_COUNT, balanced_wrong_map
from .t1gr_g_runtime import EpochFreshInfiniteDataLoader, build_epoch_fresh_dataloader, verify_multimodal_view
from .t1gr_secure_io import Deadline, sha256_file

SOFTWARE_ENV_KEYS = (
    "python_version",
    "torch_version",
    "ultralytics_version",
    "ultralytics_analytics_disabled_for_process",
    "ultralytics_source_sha256",
    "cuda_runtime_version",
    "cudnn_version",
    "cuda_available",
)
SERVER_ENV_KEYS = SOFTWARE_ENV_KEYS + (
    "cuda_device_count",
    "cuda_device_name",
    "device_compute_capability",
)


def server_environment_preflight(runtime: Mapping[str, Any], e5_expected: Mapping[str, Any]) -> dict:
    mismatch = {
        key: {"expected": e5_expected.get(key), "actual": runtime.get(key)}
        for key in SOFTWARE_ENV_KEYS
        if runtime.get(key) != e5_expected.get(key)
    }
    if mismatch:
        raise RuntimeError(f"T1GR_U6_SERVER_SOFTWARE_DRIFT:{sorted(mismatch)}")
    if runtime.get("cuda_available") is not True or int(runtime.get("cuda_device_count", -1)) != 1:
        raise RuntimeError("T1GR_U6_ONE_VISIBLE_GPU_REQUIRED")
    if not runtime.get("cuda_device_name"):
        raise RuntimeError("T1GR_U6_GPU_NAME_MISSING")
    hardware_drift = {
        key: {"e5": e5_expected.get(key), "server": runtime.get(key)}
        for key in ("cuda_device_count", "cuda_device_name", "device_compute_capability")
        if runtime.get(key) != e5_expected.get(key)
    }
    return {"software_match": True, "hardware_drift_from_e5": hardware_drift}


def assert_same_server_environment(runtime: Mapping[str, Any], pinned: Mapping[str, Any]) -> None:
    drift = {
        key: (pinned.get(key), runtime.get(key))
        for key in SERVER_ENV_KEYS
        if pinned.get(key) != runtime.get(key)
    }
    if drift:
        raise RuntimeError(f"T1GR_U6_SERVER_ENV_CHANGED:{sorted(drift)}")


def ensure_u6_dataset_yaml(
    view_manifest_path: Path,
    recipe: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> Path:
    manifest_path = Path(view_manifest_path).expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_root = Path(str(manifest.get("base_g_view_root", ""))).expanduser().resolve(strict=True)
    target = manifest_path.parent / "dataset.u6_server.private.yaml"
    names = recipe.get("class_names") or {}
    if set(names) != {str(index) for index in range(12)}:
        raise RuntimeError("T1GR_U6_CLASS_MAP_FAIL")
    text = (
        f"path: {json.dumps(base_root.as_posix(), ensure_ascii=False)}\n"
        "train: images/train\nval: images/val\nchannels: 6\nnc: 12\nnames:\n"
        + "".join(f"  {index}: {json.dumps(names[str(index)], ensure_ascii=False)}\n" for index in range(12))
    )
    raw = text.encode("utf-8")
    if target.exists():
        if not target.is_file() or target.read_bytes() != raw:
            raise RuntimeError("T1GR_U6_DATASET_YAML_DRIFT")
    else:
        with target.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            os.chmod(target, 0o600)
    digest = sha256_file(target)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError("T1GR_U6_DATASET_YAML_SHA_DRIFT")
    return target


def verify_u6_view(
    view_manifest_path: Path,
    recipe: Mapping[str, Any],
    *,
    deadline: Deadline | None = None,
    verify_bytes: bool = True,
) -> dict:
    view_manifest_path = Path(view_manifest_path).expanduser().resolve(strict=True)
    manifest = json.loads(view_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA_VIEW or not payload_ok(manifest):
        raise RuntimeError("T1GR_U6_VIEW_INTEGRITY_FAIL")
    if manifest.get("final_holdout_ids_present") is not False:
        raise RuntimeError("T1GR_U6_VIEW_EXPOSES_HOLDOUT")
    base_manifest = Path(str(manifest.get("base_g_view_manifest", ""))).expanduser().resolve(strict=True)
    if sha256_file(base_manifest, deadline) != manifest.get("base_g_view_manifest_sha256"):
        raise RuntimeError("T1GR_U6_BASE_G_VIEW_SHA_DRIFT")
    base = verify_multimodal_view(base_manifest, recipe, deadline=deadline, verify_bytes=verify_bytes)
    expected_root = Path(str(manifest.get("base_g_view_root", ""))).expanduser().resolve(strict=True)
    if base_manifest.parent != expected_root:
        raise RuntimeError("T1GR_U6_BASE_G_VIEW_ROOT_DRIFT")
    rows = manifest.get("mappings")
    if not isinstance(rows, list) or len(rows) != TRAIN_COUNT + DEV_COUNT:
        raise RuntimeError("T1GR_U6_VIEW_MAPPING_COUNT_FAIL")
    root = view_manifest_path.parent
    by_split: dict[str, list[dict]] = {"train": [], "dev": []}
    seen: set[str] = set()
    material = []
    for raw in rows:
        if not isinstance(raw, dict) or raw.get("split") not in by_split:
            raise RuntimeError("T1GR_U6_VIEW_ROW_FAIL")
        sid = str(raw.get("sample_id", ""))
        if not sid or sid in seen:
            raise RuntimeError("T1GR_U6_VIEW_ID_FAIL")
        seen.add(sid)
        path = root / str(raw.get("depth_rel", ""))
        if not path.is_file() or root not in path.resolve().parents:
            raise RuntimeError("T1GR_U6_VIEW_PATH_FAIL")
        suffix = path.suffix.lower()
        expected_kind = (
            "METRIC_UINT16_PNG"
            if suffix == ".png"
            else "UNKNOWN_SCALE_JPG_QUARANTINED"
            if suffix in {".jpg", ".jpeg"}
            else ""
        )
        kind = str(raw.get("depth_kind", ""))
        if kind != expected_kind:
            raise RuntimeError("T1GR_U6_VIEW_KIND_FAIL")
        if verify_bytes and sha256_file(path, deadline) != raw.get("depth_sha256"):
            raise RuntimeError("T1GR_U6_VIEW_CONTENT_SHA_DRIFT")
        row = dict(raw)
        by_split[str(row["split"])].append(row)
        material.append((row["split"], sid, row["depth_rel"], row["depth_sha256"], kind))
    if len(by_split["train"]) != TRAIN_COUNT or len(by_split["dev"]) != DEV_COUNT:
        raise RuntimeError("T1GR_U6_VIEW_SPLIT_COUNT_FAIL")
    ids = {split: sorted(row["sample_id"] for row in values) for split, values in by_split.items()}
    if ids != base["ids"]:
        raise RuntimeError("T1GR_U6_BASE_ID_DRIFT")
    if sha256_json(sorted(material)) != manifest.get("mapping_commitment"):
        raise RuntimeError("T1GR_U6_VIEW_MAPPING_COMMITMENT_FAIL")
    depth_maps = {
        split: {str(row["sample_id"]): str(root / str(row["depth_rel"])) for row in values}
        for split, values in by_split.items()
    }
    kind_maps = {
        split: {str(row["sample_id"]): str(row["depth_kind"]) for row in values}
        for split, values in by_split.items()
    }
    dataset_yaml = ensure_u6_dataset_yaml(view_manifest_path, recipe)
    return {
        **base,
        "u6_manifest": manifest,
        "depth_rows": by_split,
        "depth_maps": depth_maps,
        "depth_kind_maps": kind_maps,
        "dataset_yaml": dataset_yaml,
        "depth_mapping_commitment": manifest["mapping_commitment"],
    }


def _source_summary(rows: list[dict], expected_ids: list[str], *, arm: str, seed: int, epoch: int) -> dict:
    anchors = [row for row in rows if row.get("role") == "anchor"]
    expected = tuple(str(value) for value in expected_ids)
    recipients = [str(row.get("recipient", "")) for row in anchors]
    donors = [str(row.get("donor", "")) for row in anchors]
    coverage = Counter(recipients) == Counter(expected) and len(anchors) == len(expected)
    if arm == "G0-N":
        expected_map = {sid: ZERO_IR for sid in expected}
    elif arm in {"G1-P", "G3-D"}:
        expected_map = {sid: sid for sid in expected}
    elif arm == "G2-S":
        expected_map = balanced_wrong_map(expected, int(seed), int(epoch))
    else:  # pragma: no cover - guarded by caller
        raise RuntimeError("T1GR_U6_SOURCE_ARM_INTERNAL_FAIL")
    mapping = coverage and all(str(row.get("donor", "")) == expected_map[str(row.get("recipient", ""))] for row in anchors)
    donor_counts = Counter(donors)
    bijection = arm != "G2-S" or (
        set(donor_counts) == set(expected)
        and all(donor_counts[sid] == 1 for sid in expected)
        and all(recipient != donor for recipient, donor in zip(recipients, donors))
    )
    epochs_exact = all(int(row.get("epoch", -1)) == int(epoch) for row in rows)
    normalized = [
        {
            "recipient": str(row.get("recipient", "")),
            "donor": str(row.get("donor", "")),
            "role": str(row.get("role", "")),
            "epoch": int(row.get("epoch", -1)),
        }
        for row in rows
    ]
    return {
        "anchor_count": len(anchors),
        "all_pair_count": len(rows),
        "anchor_coverage_exact": coverage,
        "anchor_mapping_exact": mapping,
        "wrong_ir_bijection_exact": bijection,
        "anchor_self_match_count": sum(left == right for left, right in zip(recipients, donors)),
        "all_rows_epoch_exact": epochs_exact,
        "source_condition_passed": bool(coverage and mapping and bijection and epochs_exact),
        "pair_trace_commitment": sha256_json(normalized),
    }


def _epoch_summary(
    pair_rows: list[dict],
    depth_rows: list[dict],
    batch_checks: list[dict],
    expected_ids: list[str],
    *,
    arm: str,
    seed: int,
    epoch: int,
) -> dict:
    source = _source_summary(pair_rows, expected_ids, arm=arm, seed=seed, epoch=epoch)
    if not source["source_condition_passed"]:
        raise RuntimeError("T1GR_U6_SOURCE_TRACE_FAIL")
    if not depth_rows or not batch_checks:
        raise RuntimeError("T1GR_U6_EMPTY_MODALITY_TRACE")
    kinds = {"METRIC_UINT16_PNG": 0, "UNKNOWN_SCALE_JPG_QUARANTINED": 0}
    for row in depth_rows:
        if row.get("arm") != arm or int(row.get("epoch", -1)) != epoch:
            raise RuntimeError("T1GR_U6_DEPTH_TRACE_IDENTITY_FAIL")
        kind = str(row.get("depth_kind", ""))
        if kind not in kinds:
            raise RuntimeError("T1GR_U6_DEPTH_TRACE_KIND_FAIL")
        kinds[kind] += 1
        if row.get("mask_binary") is not True or row.get("mask_zero_implies_depth_zero") is not True:
            raise RuntimeError("T1GR_U6_RAW_DEPTH_CONTRACT_FAIL")
        emitted_valid = int(row.get("emitted_valid_pixels", -1))
        emitted_depth = int(row.get("emitted_nonzero_depth_pixels", -1))
        if arm != "G3-D" and (
            row.get("depth_enabled") is not False
            or row.get("depth_source_decoded") is not False
            or emitted_valid != 0
            or emitted_depth != 0
        ):
            raise RuntimeError("T1GR_U6_CONTROL_ARM_EMITTED_DEPTH")
        if arm == "G3-D":
            if row.get("depth_enabled") is not True:
                raise RuntimeError("T1GR_U6_G3_DEPTH_DISABLED")
            if kind == "METRIC_UINT16_PNG" and row.get("depth_source_decoded") is not True:
                raise RuntimeError("T1GR_U6_G3_METRIC_DEPTH_NOT_DECODED")
            if kind == "UNKNOWN_SCALE_JPG_QUARANTINED" and (
                row.get("depth_source_decoded") is not False or emitted_valid != 0 or emitted_depth != 0
            ):
                raise RuntimeError("T1GR_U6_JPG_QUARANTINE_FAIL")
    if not all(value > 0 for value in kinds.values()):
        raise RuntimeError("T1GR_U6_DEPTH_DOMAIN_NOT_OBSERVED")
    anchors = [str(row.get("sample_id", "")) for row in depth_rows if row.get("role") == "anchor"]
    if Counter(anchors) != Counter(expected_ids) or len(anchors) != len(expected_ids):
        raise RuntimeError("T1GR_U6_DEPTH_ANCHOR_COVERAGE_FAIL")
    if not all(row.get("mask_binary") and row.get("mask_zero_implies_depth_zero") for row in batch_checks):
        raise RuntimeError("T1GR_U6_POST_TRANSFORM_DEPTH_CONTRACT_FAIL")
    post_depth = sum(int(row["depth_nonzero"]) for row in batch_checks)
    post_mask = sum(int(row["mask_nonzero"]) for row in batch_checks)
    post_ir = sum(int(row["ir_nonzero"]) for row in batch_checks)
    if arm == "G3-D" and (post_depth <= 0 or post_mask <= 0 or post_ir <= 0):
        raise RuntimeError("T1GR_U6_G3_POST_TRANSFORM_MODALITY_MISSING")
    if arm in {"G1-P", "G2-S"} and (post_ir <= 0 or post_depth != 0 or post_mask != 0):
        raise RuntimeError("T1GR_U6_IR_ARM_POST_TRANSFORM_FAIL")
    if arm == "G0-N" and (post_ir != 0 or post_depth != 0 or post_mask != 0):
        raise RuntimeError("T1GR_U6_G0_POST_TRANSFORM_AUX_FAIL")
    return {
        "arm": arm,
        "seed": int(seed),
        "epoch_zero_based": int(epoch),
        **source,
        "depth_record_count": len(depth_rows),
        "depth_anchor_count": len(anchors),
        "depth_kind_record_counts": kinds,
        "post_transform_batch_count": len(batch_checks),
        "post_transform_nonzero_ir": post_ir,
        "post_transform_nonzero_depth": post_depth,
        "post_transform_nonzero_mask": post_mask,
        "depth_contract_passed": True,
        "modality_contract_passed": True,
    }


class T1GRU6Trainer(DetectionTrainer):
    def __init__(self, *args, arm: str, seed: int, view: Mapping[str, Any], trace_dir: Path, **kwargs):
        if arm not in ARMS or int(seed) not in SEEDS:
            raise ValueError("T1GR_U6_TRAINER_REQUEST_FAIL")
        self.t1gr_arm = arm
        self.t1gr_seed = int(seed)
        self.t1gr_view = dict(view)
        self.t1gr_trace_dir = Path(trace_dir)
        self.t1gr_pair_rows: list[dict] = []
        self.t1gr_depth_rows: list[dict] = []
        self.t1gr_batch_checks: list[dict] = []
        self.t1gr_epoch_summaries: list[dict] = []
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        split = "train" if mode == "train" else "dev"
        dataset = T1GRU6Dataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            ir_by_sid=self.t1gr_view["ir_maps"][split],
            depth_by_sid=self.t1gr_view["depth_maps"][split],
            depth_kind_by_sid=self.t1gr_view["depth_kind_maps"][split],
            arm=self.t1gr_arm,
            seed=self.t1gr_seed,
            split=split,
            ir_condition="ARM_NATIVE",
            depth_condition="NATIVE",
        )
        if set(dataset.ids) != set(self.t1gr_view["ids"][split]):
            raise RuntimeError("T1GR_U6_DATASET_ID_DRIFT")
        return dataset

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = -1, mode: str = "train"):
        if mode not in {"train", "val"}:
            raise ValueError("T1GR_U6_DATALOADER_MODE_FAIL")
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            shuffle = False
        workers = int(self.args.workers if mode == "train" else self.args.workers * 2)
        return build_epoch_fresh_dataloader(
            dataset,
            batch=batch_size,
            workers=workers,
            shuffle=shuffle,
            rank=rank,
            drop_last=bool(self.args.compile and mode == "train"),
            pin_memory=mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        for group in batch.pop("source_pairs", ()):
            for row in group:
                self.t1gr_pair_rows.append(dict(row))
        for group in batch.pop("depth_records", ()):
            for row in group:
                self.t1gr_depth_rows.append(dict(row))
        image = batch.get("img")
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[1] != 6:
            raise RuntimeError("T1GR_U6_BATCH_CHANNEL_FAIL")
        ir, depth, mask = image[:, 3], image[:, 4], image[:, 5]
        binary = bool(torch.all((mask == 0) | (mask == 255)).item())
        zero_implies = bool(torch.count_nonzero(depth[mask == 0]).item() == 0)
        check = {
            "mask_binary": binary,
            "mask_zero_implies_depth_zero": zero_implies,
            "ir_nonzero": int(torch.count_nonzero(ir).item()),
            "depth_nonzero": int(torch.count_nonzero(depth).item()),
            "mask_nonzero": int(torch.count_nonzero(mask).item()),
        }
        if not binary or not zero_implies:
            raise RuntimeError("T1GR_U6_BATCH_DEPTH_CONTRACT_FAIL")
        self.t1gr_batch_checks.append(check)
        return super().preprocess_batch(batch)

    def begin_epoch(self) -> None:
        epoch = int(self.epoch)
        self.t1gr_pair_rows = []
        self.t1gr_depth_rows = []
        self.t1gr_batch_checks = []
        if not isinstance(self.train_loader, EpochFreshInfiniteDataLoader):
            raise RuntimeError("T1GR_U6_EPOCH_FRESH_LOADER_MISSING")
        self.train_loader.reset_for_epoch(epoch)

    def finish_epoch(self) -> None:
        epoch = int(self.epoch)
        summary = _epoch_summary(
            self.t1gr_pair_rows,
            self.t1gr_depth_rows,
            self.t1gr_batch_checks,
            self.t1gr_view["ids"]["train"],
            arm=self.t1gr_arm,
            seed=self.t1gr_seed,
            epoch=epoch,
        )
        self.t1gr_trace_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        for filename, rows in (
            ("source_pairs.private.jsonl", self.t1gr_pair_rows),
            ("depth_records.private.jsonl", self.t1gr_depth_rows),
        ):
            path = self.t1gr_trace_dir / filename
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        self.t1gr_epoch_summaries.append(summary)
        target = self.t1gr_trace_dir / "epoch_summaries.private.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.t1gr_epoch_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
