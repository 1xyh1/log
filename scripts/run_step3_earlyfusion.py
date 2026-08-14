#!/usr/bin/env python3
"""Step 3-A early-fusion runner with immutable formal runs and actual-yield G8.

Frozen Step-3 scientific design is unchanged:
    C0-N [RGB,0,0,0]
    C1-I [RGB,I,0,0]
    C2-D [RGB,0,D,M]

This patch only hardens execution/provenance:
- formal/recovery run directories may not already contain artifacts;
- float32 [0,1] inputs are never divided by 255 again;
- epoch sampler is advanced explicitly for single-GPU training;
- InfiniteDataLoader is reset after changing epoch state;
- G8 hashes sample IDs/flip flags actually yielded by DataLoader;
- expected and actual order must match or training aborts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import modality_preprocess as mp  # noqa: E402
from multimodal.early_fusion_yolo26 import SNAPSHOT_DEFAULT, load_snapshot, sha256_file  # noqa: E402
from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402
from ultralytics.data.build import InfiniteDataLoader  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402

R3_KW = dict(
    epochs=80,
    batch=4,
    nbs=4,
    warmup_epochs=0,
    workers=0,
    cache=False,
    imgsz=640,
    max_det=100,
    patience=100,
    close_mosaic=0,
    mosaic=0.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    scale=0.0,
    translate=0.0,
    degrees=0.0,
    shear=0.0,
    perspective=0.0,
    multi_scale=0.0,
    fliplr=0.30393,
    flipud=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    bgr=0.0,
    auto_augment=None,
    erasing=0.0,
    seed=20260812,
    deterministic=True,
    end2end=False,
    plots=False,
    cls_pw=0.0,
    optimizer="MuSGD",
    lr0=0.00038,
    lrf=0.88219,
    momentum=0.94751,
    weight_decay=0.00027,
    box=9.83241,
    cls=0.64896,
    dfl=0.95824,
)


def sha256_json(obj) -> str:
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def _guard_run_dir(run_dir: Path, run_kind: str) -> None:
    if run_kind in {"formal", "recovery"} and run_dir.exists():
        existing = [p for p in run_dir.rglob("*") if p.is_file()]
        if existing:
            preview = ", ".join(str(p.relative_to(run_dir)) for p in existing[:8])
            raise RuntimeError(
                f"FORMAL_RUN_DIR_EXISTS: {run_dir} already contains {len(existing)} files: {preview}. "
                "Use a new --name. Formal/recovery runs are immutable by design."
            )


class Step3Validator(DetectionValidator):
    """Float 6ch validator: stock detection semantics, no extra `/255`."""

    def preprocess(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["C0-N", "C1-I", "C2-D"], required=True)
    parser.add_argument("--snapshot", default=SNAPSHOT_DEFAULT)
    parser.add_argument("--contract", default=OUT_DEFAULT)
    parser.add_argument("--project", default="runs/step3_earlyfusion")
    parser.add_argument("--name", default=None, help="Physical run directory name. Formal recovery must use a new name.")
    parser.add_argument("--run-kind", choices=["smoke", "formal", "recovery"], default="formal")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--data-yaml",
        default="D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml",
    )
    args = parser.parse_args()

    if args.run_kind == "smoke" and args.name is None:
        args.name = f"smoke-{args.group}-e{args.epochs}-{datetime.now():%Y%m%d-%H%M%S}"
    elif args.name is None:
        args.name = args.group

    project = Path(args.project).resolve()
    run_dir = project / args.name
    _guard_run_dir(run_dir, args.run_kind)

    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    model = load_snapshot(args.snapshot)
    model.nc = 12

    # Keep a self-describing class mapping if the contract contains one; otherwise the
    # snapshot's real 12-class names remain untouched.
    if "_names" in contract:
        model.names = {int(k): v for k, v in contract["_names"].items()}

    requested_batch = int(args.batch)
    trace_path = run_dir / "step3_g8_trace.jsonl"
    growth_path = run_dir / "step3_kernel_growth.jsonl"

    class Step3Trainer(DetectionTrainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.step3_group = args.group
            self.step3_contract = contract
            self.step3_requested_batch = requested_batch
            self._epoch_dataset = None
            self._g8_actual_ids: list[str] = []
            self._g8_actual_flips: list[bool] = []

        def build_dataset(self, img_path, mode="train", batch=None):
            split = "train" if mode == "train" else "val"
            return TriModalDataset(
                self.step3_contract,
                split=split,
                group=self.step3_group,
                imgsz=self.args.imgsz,
                seed=self.args.seed,
                fliplr=self.args.fliplr,
                augment=(mode == "train"),
            )

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "train":
                dataset = self.build_dataset(dataset_path, mode, batch_size)
                self._epoch_dataset = dataset
                return InfiniteDataLoader(
                    dataset,
                    batch_size=batch_size,
                    sampler=dataset.sampler,
                    shuffle=False,
                    num_workers=self.args.workers,
                    collate_fn=dataset.collate_fn,
                    pin_memory=True,
                    drop_last=False,
                )
            # superclass calls our build_dataset(), so validation still uses TriModalDataset.
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        def preprocess_batch(self, batch):
            if int(self.args.batch) != self.step3_requested_batch:
                raise RuntimeError(
                    "ABORT_RESOURCE_PROFILE_CHANGED: "
                    f"requested batch={self.step3_requested_batch}, actual args.batch={self.args.batch}. "
                    "If OOM occurs, rerun ALL groups uniformly with batch=2/nbs=4."
                )

            # This is the *actual batch yielded by DataLoader*, before any device copy.
            ids = batch.get("sample_id")
            flips = batch.get("flip_applied")
            if ids is not None:
                self._g8_actual_ids.extend(str(x) for x in ids)
            if flips is not None:
                self._g8_actual_flips.extend(bool(x) for x in flips)

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
            batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
            # NO /255: TriModalDataset already emits float32 [0,1].
            return batch

        def get_validator(self):
            v = Step3Validator(self.test_loader, save_dir=self.save_dir, args=self.args)
            return v

        def final_eval(self):
            # AutoBackend's generic final path is not our 6ch provenance path.  The
            # post-hoc evaluator in this patch is authoritative and records hashes.
            print(f"[{args.group}] stock final_eval skipped; use eval_step3_causality.py")

    overrides = dict(R3_KW)
    overrides.update(
        epochs=args.epochs,
        batch=args.batch,
        seed=args.seed,
        project=str(project),
        name=args.name,
        device=args.device,
        data=args.data_yaml,
        exist_ok=True,  # safe because _guard_run_dir already enforces immutability
        model="yolo26s.yaml",
    )

    trainer = Step3Trainer(overrides=overrides)
    trainer.model = model
    trainer.model.nc = 12
    trainer.model.args = trainer.args

    # The trainer may create the directory during __init__.  From this point onward it
    # is the authoritative location for all evidence.
    run_dir = Path(trainer.save_dir)
    trace_path = run_dir / "step3_g8_trace.jsonl"
    growth_path = run_dir / "step3_kernel_growth.jsonl"
    for fp in (trace_path, growth_path):
        if fp.exists():
            # New formal/recovery runs are empty by guard; smoke names are timestamped.
            fp.unlink()

    def on_epoch_start(tr):
        ds = getattr(tr, "_epoch_dataset", None)
        if ds is None:
            raise RuntimeError("G8_NO_EPOCH_DATASET")
        ds.set_epoch(tr.epoch)
        # InfiniteDataLoader owns a persistent iterator. Reset after sampler epoch
        # changes so this epoch starts from the intended permutation even if workers or
        # prefetch settings change in a later experiment.
        if hasattr(tr.train_loader, "reset"):
            tr.train_loader.reset()
        tr._g8_actual_ids = []
        tr._g8_actual_flips = []

    def on_epoch_end(tr):
        ds = tr._epoch_dataset
        expected_ids = [ds.ids[i] for i in ds.sampler.perm]
        expected_flips = [mp.should_flip(ds.seed, tr.epoch, sid, ds.fliplr) for sid in expected_ids]
        actual_ids = list(tr._g8_actual_ids)
        actual_flips = list(tr._g8_actual_flips)

        if actual_ids != expected_ids:
            raise RuntimeError(
                f"G8_ACTUAL_ORDER_MISMATCH epoch={tr.epoch}: expected={expected_ids}, actual={actual_ids}"
            )
        if actual_flips != expected_flips:
            raise RuntimeError(
                f"G8_ACTUAL_FLIP_MISMATCH epoch={tr.epoch}: expected={expected_flips}, actual={actual_flips}"
            )

        # Keep legacy-compatible planned hashes so a recovered C0 can still be compared
        # with preserved C1/C2 traces without pretending those old traces were actual-yield evidence.
        legacy_flip_bits = "".join("1" if flag else "0" for flag in expected_flips)
        g8 = {
            "epoch": tr.epoch,
            "n_samples": len(actual_ids),
            "sample_order_sha256": ds.sampler.order_sha256(),
            "flip_schedule_sha256": sha256_text(legacy_flip_bits),
            "expected_order_sha256": sha256_json(expected_ids),
            "actual_order_sha256": sha256_json(actual_ids),
            "actual_flip_sha256": sha256_json(
                [f"{sid}:{int(flag)}" for sid, flag in zip(actual_ids, actual_flips)]
            ),
            "actual_matches_expected": True,
            "batch": int(tr.args.batch),
        }
        _append_jsonl(trace_path, g8)

        w = tr.model.model[0].conv.weight.detach().cpu().float()
        rgb_mean = sum(float(w[:, c].norm()) for c in range(3)) / 3.0
        growth = {
            "epoch": tr.epoch + 1,
            "wI_norm": float(w[:, 3].norm()),
            "wD_norm": float(w[:, 4].norm()),
            "wM_norm": float(w[:, 5].norm()),
            "qI": float(w[:, 3].norm()) / (rgb_mean + 1e-12),
            "qD": float(w[:, 4].norm()) / (rgb_mean + 1e-12),
            "qM": float(w[:, 5].norm()) / (rgb_mean + 1e-12),
            "batch": int(tr.args.batch),
        }
        _append_jsonl(growth_path, growth)

        if int(tr.args.batch) != tr.step3_requested_batch:
            raise RuntimeError(
                "ABORT_RESOURCE_PROFILE_CHANGED: "
                f"requested={tr.step3_requested_batch}, actual={tr.args.batch}"
            )

    trainer.add_callback("on_train_epoch_start", on_epoch_start)
    trainer.add_callback("on_train_epoch_end", on_epoch_end)

    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        import ultralytics
        ul_version = ultralytics.__version__
    except Exception:
        ul_version = "unknown"

    manifest = {
        "schema": "step3-run-manifest-v2",
        "group": args.group,
        "physical_run_name": args.name,
        "run_kind": args.run_kind,
        "expected_epochs": int(args.epochs),
        "snapshot": str(args.snapshot),
        "initial_checkpoint_sha256": sha256_file(Path(args.snapshot)),
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "recipe": "R3-causal-earlyfusion-sample",
        "requested_batch": requested_batch,
        "seed": int(args.seed),
        "channel_semantics": ["R", "G", "B", "IR_scalar", "log_depth", "valid_mask"],
        "g8_evidence": "actual_dataloader_yield_v2",
        "ultralytics_version": ul_version,
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    trainer.train()
    print(f"[{args.group}] DONE -> {run_dir}")


if __name__ == "__main__":
    main()
