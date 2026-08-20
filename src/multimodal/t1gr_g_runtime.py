"""Runtime glue for the fail-closed T1-GR smoke and formal suites."""
from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import distributed
from torch.utils.data import Sampler

from ultralytics.data.build import (
    ContiguousDistributedSampler,
    InfiniteDataLoader,
    seed_worker,
)
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

from .t1gr_e5_core import canonical_ids_sha, payload_ok as e5_payload_ok
from .t1gr_secure_io import Deadline, fail, sha256_file, sha256_json
from .t1gr_g_core import ARMS, DEV_COUNT, SEEDS, TRAIN_COUNT, payload_ok as design_payload_ok
from .t1gr_g_dataset import T1GRDataset
from .t1gr_g_impl_core import payload_ok as impl_payload_ok, trace_epoch_summary, validate_impl_spec


SOURCE_FILES = (
    "config/t1gr_g_implementation_spec.frozen.json",
    "src/multimodal/t1gr_g_impl_core.py",
    "src/multimodal/t1gr_g_model.py",
    "src/multimodal/t1gr_g_dataset.py",
    "src/multimodal/t1gr_g_runtime.py",
    "src/multimodal/t1gr_g_suite.py",
    "scripts/t1gr_g_build_multimodal_view.py",
    "scripts/t1gr_g_implementation_preflight.py",
    "scripts/t1gr_g_run_one.py",
    "scripts/t1gr_g_train_entry.py",
    "scripts/t1gr_g_run_smoke_suite.py",
    "scripts/t1gr_g_smoke_audit.py",
    "scripts/t1gr_g_run_formal_suite.py",
    "scripts/t1gr_g_eval_suite.py",
)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"T1GR_G_JSON_READ_FAIL:{Path(path).name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"T1GR_G_JSON_OBJECT_REQUIRED:{Path(path).name}")
    return value


def validate_frozen_chain(design: Mapping[str, Any], recipe: Mapping[str, Any], spec: Mapping[str, Any]) -> dict:
    from .t1gr_g_core import validate_design

    design_info = validate_design(design)
    spec_info = validate_impl_spec(spec)
    if not e5_payload_ok(dict(recipe)):
        raise RuntimeError("T1GR_G_E5_RECIPE_PAYLOAD_FAIL")
    upstream = spec["upstream"]
    checks = {
        "design_payload_pin": design.get("payload_sha256") == upstream["design_payload_sha256"],
        "e5_recipe_schema_pin": recipe.get("schema") == upstream["e5_v2_recipe_schema"],
        "e5_recipe_payload_pin": recipe.get("payload_sha256") == upstream["e5_v2_recipe_payload_sha256"],
        "optimizer_musgd": (recipe.get("train_args") or {}).get("optimizer") == "MuSGD",
        "e5_epochs_80": (recipe.get("train_args") or {}).get("epochs") == 80,
        "e5_batch_4": (recipe.get("train_args") or {}).get("batch") == 4,
        "e5_imgsz_640": (recipe.get("train_args") or {}).get("imgsz") == 640,
        "e5_workers_8": (recipe.get("train_args") or {}).get("workers") == 8,
        "dev_max_det_100": (recipe.get("eval_args") or {}).get("max_det") == 100,
    }
    if not design_payload_ok(dict(design)) or not impl_payload_ok(dict(spec)) or not all(checks.values()):
        raise RuntimeError(f"T1GR_G_FROZEN_CHAIN_FAIL:{checks}")
    return {"design": design_info, "implementation": spec_info, "checks": checks}


def implementation_source_hashes(repo: Path, *, require_all: bool = True) -> dict[str, str]:
    values: dict[str, str] = {}
    for rel in SOURCE_FILES:
        path = Path(repo) / rel
        if not path.is_file():
            if require_all:
                raise RuntimeError(f"T1GR_G_SOURCE_FILE_MISSING:{rel}")
            continue
        values[rel] = sha256_file(path)
    return values


def _manifest_row_paths(root: Path, row: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    return (
        root / str(row["image_rel"]),
        root / str(row["infrared_rel"]),
        root / str(row["label_rel"]),
    )


def verify_multimodal_view(
    manifest_path: Path,
    recipe: Mapping[str, Any],
    *,
    deadline: Deadline | None = None,
    verify_bytes: bool = True,
) -> dict:
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "t1gr-g-multimodal-view-private-v1" or not impl_payload_ok(manifest):
        raise RuntimeError("T1GR_G_VIEW_MANIFEST_INTEGRITY_FAIL")
    if manifest.get("final_holdout_ids_present") is not False:
        raise RuntimeError("T1GR_G_VIEW_EXPOSES_HOLDOUT_IDS")
    rows = manifest.get("mappings")
    if not isinstance(rows, list) or len(rows) != TRAIN_COUNT + DEV_COUNT:
        raise RuntimeError("T1GR_G_VIEW_MAPPING_COUNT_FAIL")
    root = manifest_path.parent
    by_split: dict[str, list[dict]] = {"train": [], "dev": []}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("split") not in by_split:
            raise RuntimeError("T1GR_G_VIEW_MAPPING_ROW_FAIL")
        sid = str(row.get("sample_id", ""))
        if not sid or sid in seen:
            raise RuntimeError("T1GR_G_VIEW_DUPLICATE_OR_EMPTY_ID")
        seen.add(sid)
        image, infrared, label = _manifest_row_paths(root, row)
        for path in (image, infrared, label):
            if not path.is_file() or root not in path.resolve().parents:
                raise RuntimeError("T1GR_G_VIEW_PATH_FAIL")
        if verify_bytes:
            if deadline:
                deadline.check("T1GR_G_VIEW_VERIFY_TIMEOUT")
            expected = (row.get("image_sha256"), row.get("infrared_sha256"), row.get("label_sha256"))
            actual = tuple(sha256_file(path, deadline) for path in (image, infrared, label))
            if actual != expected:
                raise RuntimeError("T1GR_G_VIEW_CONTENT_SHA_DRIFT")
        by_split[str(row["split"])].append(dict(row))
    if len(by_split["train"]) != TRAIN_COUNT or len(by_split["dev"]) != DEV_COUNT:
        raise RuntimeError("T1GR_G_VIEW_SPLIT_COUNT_FAIL")
    ids = {split: sorted(str(row["sample_id"]) for row in values) for split, values in by_split.items()}
    if canonical_ids_sha(ids["train"]) != (recipe.get("ids_commitments") or {}).get("train"):
        raise RuntimeError("T1GR_G_VIEW_TRAIN_COMMITMENT_FAIL")
    if canonical_ids_sha(ids["dev"]) != (recipe.get("ids_commitments") or {}).get("dev"):
        raise RuntimeError("T1GR_G_VIEW_DEV_COMMITMENT_FAIL")
    yaml_path = root / str(manifest.get("dataset_yaml_rel", ""))
    if not yaml_path.is_file() or sha256_file(yaml_path, deadline) != manifest.get("dataset_yaml_sha256"):
        raise RuntimeError("T1GR_G_VIEW_DATASET_YAML_FAIL")
    ir_maps = {
        split: {str(row["sample_id"]): str(root / str(row["infrared_rel"])) for row in values}
        for split, values in by_split.items()
    }
    image_maps = {
        split: {str(row["sample_id"]): str(root / str(row["image_rel"])) for row in values}
        for split, values in by_split.items()
    }
    mapping_material = sorted(
        (
            str(row["split"]), str(row["sample_id"]), str(row["image_rel"]), row["image_sha256"],
            str(row["infrared_rel"]), row["infrared_sha256"], str(row["label_rel"]), row["label_sha256"],
        )
        for row in rows
    )
    if sha256_json(mapping_material) != manifest.get("mapping_commitment"):
        raise RuntimeError("T1GR_G_VIEW_MAPPING_COMMITMENT_FAIL")
    return {
        "manifest": manifest,
        "dataset_yaml": yaml_path,
        "rows": by_split,
        "ids": ids,
        "ir_maps": ir_maps,
        "image_maps": image_maps,
        "mapping_commitment": manifest["mapping_commitment"],
        "train_count": len(ids["train"]),
        "dev_count": len(ids["dev"]),
    }


class EpochFreshInfiniteDataLoader(InfiniteDataLoader):
    """Infinite loader whose old workers are synchronously destroyed per epoch."""

    def shutdown(self) -> None:
        old = getattr(self, "iterator", None)
        if old is not None and hasattr(old, "_shutdown_workers"):
            old._shutdown_workers()

    def reset(self) -> None:
        self.shutdown()
        self.iterator = self._get_iterator()

    def reset_for_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(int(epoch))
        sampler = getattr(self, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(epoch))
        self.reset()


class RecipientEpochSampler(Sampler[int]):
    """A complete seed/epoch-keyed permutation, independent of prefetch depth."""

    def __init__(self, dataset: T1GRDataset):
        self.dataset = dataset
        self.seed = int(dataset.t1gr_seed)
        self.epoch = int(dataset.t1gr_epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        keyed = []
        for index, sid in enumerate(self.dataset.ids):
            raw = f"T1GR_ORDER_V1\0{self.seed}\0{self.epoch}\0{sid}".encode("utf-8")
            keyed.append((hashlib.sha256(raw).digest(), sid, index))
        yield from (index for _, _, index in sorted(keyed))

    def __len__(self) -> int:
        return len(self.dataset)


def build_epoch_fresh_dataloader(
    dataset: T1GRDataset,
    *,
    batch: int,
    workers: int,
    shuffle: bool,
    rank: int = -1,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> EpochFreshInfiniteDataLoader:
    if int(rank) != -1:
        raise RuntimeError("T1GR_G_SINGLE_GPU_ONLY")
    batch = min(int(batch), len(dataset))
    workers = int(workers)
    if workers < 0:
        raise ValueError("T1GR_G_WORKER_COUNT_INVALID")
    sampler = RecipientEpochSampler(dataset) if shuffle and rank == -1 else (
        None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
        if shuffle else ContiguousDistributedSampler(dataset)
    )
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    loader = EpochFreshInfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        sampler=sampler,
        prefetch_factor=4 if workers > 0 else None,
        pin_memory=torch.cuda.device_count() > 0 and pin_memory,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last and len(dataset) % batch != 0,
    )
    if int(loader.num_workers) != workers:
        raise RuntimeError("T1GR_G_EFFECTIVE_WORKER_DRIFT")
    return loader


class T1GRDetectionTrainer(DetectionTrainer):
    """Detection trainer with arm-aware data and private runtime source tracing."""

    def __init__(
        self,
        *args,
        arm: str,
        seed: int,
        view: Mapping[str, Any],
        trace_dir: Path,
        **kwargs,
    ):
        if arm not in ARMS or int(seed) not in SEEDS:
            raise ValueError("T1GR_G_TRAINER_ARM_OR_SEED_FAIL")
        self.t1gr_arm = str(arm)
        self.t1gr_seed = int(seed)
        self.t1gr_view = dict(view)
        self.t1gr_trace_dir = Path(trace_dir)
        self.t1gr_trace_rows: list[dict] = []
        self.t1gr_epoch_summaries: list[dict] = []
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        split = "train" if mode == "train" else "dev"
        dataset = T1GRDataset(
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
            arm=self.t1gr_arm,
            seed=self.t1gr_seed,
            split=split,
        )
        if set(dataset.ids) != set(self.t1gr_view["ids"][split]):
            raise RuntimeError("T1GR_G_DATASET_ID_DRIFT")
        return dataset

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = -1, mode: str = "train"):
        if mode not in {"train", "val"}:
            raise ValueError("T1GR_G_DATALOADER_MODE_FAIL")
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            shuffle = False
        expected_workers = int(self.args.workers if mode == "train" else self.args.workers * 2)
        return build_epoch_fresh_dataloader(
            dataset,
            batch=batch_size,
            workers=expected_workers,
            shuffle=shuffle,
            rank=rank,
            drop_last=bool(self.args.compile and mode == "train"),
            pin_memory=mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        pair_groups = batch.pop("source_pairs", ())
        for group in pair_groups:
            for row in group:
                self.t1gr_trace_rows.append(dict(row))
        return super().preprocess_batch(batch)

    def begin_epoch(self) -> None:
        epoch = int(self.epoch)
        self.t1gr_trace_rows = []
        loader = self.train_loader
        if not isinstance(loader, EpochFreshInfiniteDataLoader):
            raise RuntimeError("T1GR_G_EPOCH_FRESH_LOADER_MISSING")
        loader.reset_for_epoch(epoch)

    def finish_epoch(self) -> None:
        epoch = int(self.epoch)
        summary = trace_epoch_summary(
            self.t1gr_trace_rows,
            self.t1gr_view["ids"]["train"],
            arm=self.t1gr_arm,
            seed=self.t1gr_seed,
            epoch=epoch,
        )
        if not summary["source_condition_passed"]:
            raise RuntimeError(f"T1GR_G_RUNTIME_SOURCE_TRACE_FAIL:epoch={epoch}")
        self.t1gr_trace_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        raw_path = self.t1gr_trace_dir / "source_pairs.private.jsonl"
        with raw_path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in self.t1gr_trace_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.t1gr_epoch_summaries.append(summary)
        summary_path = self.t1gr_trace_dir / "epoch_summaries.private.json"
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.t1gr_epoch_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, summary_path)


def run_name(mode: str, seed: int, arm: str) -> str:
    return f"T1GR_G_{mode.upper()}_S{int(seed)}_{arm.replace('-', '_')}"


def run_report_rel(mode: str, seed: int, arm: str) -> str:
    safe_arm = arm.lower().replace("-", "_")
    return f"reports/step4_t1gr/t1gr_g_{mode}_s{int(seed)}_{safe_arm}_public.json"


def frozen_launch_rows(design: Mapping[str, Any]) -> list[dict]:
    rows = []
    position = 0
    for block in design.get("launch_order") or []:
        seed = int(block["seed"])
        for within_seed, arm in enumerate(block["arms"]):
            rows.append({"position": position, "within_seed_position": within_seed, "seed": seed, "arm": str(arm)})
            position += 1
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if len(rows) != 9 or {(r["seed"], r["arm"]) for r in rows} != expected:
        raise RuntimeError("T1GR_G_LAUNCH_MATRIX_FAIL")
    return rows
