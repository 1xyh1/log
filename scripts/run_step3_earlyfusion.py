#!/usr/bin/env python3
"""Step 3-A early-fusion runner: C0-N / C1-I / C2-D x 80 epochs (seed 20260812).

Frozen R3-causal-earlyfusion-sample recipe; float32 [0,1] 6ch direct pipeline:
    Step3Trainer.preprocess_batch / Step3Validator.preprocess move tensors WITHOUT /255.
    get_dataloader injects DeterministicEpochSampler (RANK=-1: stock trainer never
    calls set_epoch) and on_train_epoch_start advances dataset.set_epoch explicitly.
    plots=False / cls_pw=0.0 (stock plot_training_labels would access dataset.labels).
    batch auto-reduction fail-fast: requested batch != trainer batch -> ABORT.
Per-epoch records (G8 + diagnostics):
    step3_g8_trace.jsonl: epoch, sample_order_sha256, flip_schedule_sha256, batch
    step3_kernel_growth.jsonl: ||W_I||/||W_D||/||W_M|| + q relative to RGB stem
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import modality_preprocess as mp  # noqa: E402
from multimodal.early_fusion_yolo26 import (  # noqa: E402
    SNAPSHOT_DEFAULT, load_snapshot, sha256_file)
from multimodal.raw_sample_index import build_contract, OUT_DEFAULT  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402
from ultralytics.data.build import InfiniteDataLoader  # noqa: E402

# R3-causal-earlyfusion-sample (frozen): R2-core optimizer/loss + augmentation off
R3_KW = dict(
    epochs=80, batch=4, nbs=4, warmup_epochs=0, workers=0, cache=False,
    imgsz=640, max_det=100, patience=100, close_mosaic=0,
    mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
    scale=0.0, translate=0.0, degrees=0.0, shear=0.0, perspective=0.0,
    multi_scale=0.0, fliplr=0.30393, flipud=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0,
    auto_augment=None, erasing=0.0,
    seed=20260812, deterministic=True, end2end=False,
    plots=False, cls_pw=0.0,
    optimizer="MuSGD", lr0=0.00038, lrf=0.88219, momentum=0.94751,
    weight_decay=0.00027, box=9.83241, cls=0.64896, dfl=0.95824,
)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class Step3ValidatorMixin:
    """DetectionValidator overrides for float32 6ch (no /255) + trimodal dataset."""

    def build_dataset(self, img_path, mode="val", batch=None):
        return TriModalDataset(self.step3_contract, split="val", group=self.step3_group,
                               imgsz=self.args.imgsz, seed=self.args.seed,
                               fliplr=self.args.fliplr if hasattr(self.args, "fliplr") else 0.0,
                               augment=False)

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        # NO /255: input is already float32 [0,1]
        for k in ("bboxes", "cls", "batch_idx"):
            batch[k] = batch[k].to(self.device)
        return batch


class Step3Validator(Step3ValidatorMixin, DetectionValidator):
    """Mixin FIRST in MRO so our build_dataset/get_dataloader/preprocess overrides
    actually win over DetectionValidator's (observed: reversed order silently shadowed
    them and stock /255 + stock dataset ran on val)."""
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=["C0-N", "C1-I", "C2-D"], required=True)
    p.add_argument("--snapshot", default=SNAPSHOT_DEFAULT)
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--project", default="runs/step3_earlyfusion")
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--data-yaml", default="D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml")
    a = p.parse_args()

    contract = json.loads(Path(a.contract).read_text(encoding="utf-8"))
    model = load_snapshot(a.snapshot)
    model.nc = 12
    model.names = {int(k): v for k, v in contract["_names"].items()} if "_names" in contract \
        else model.names

    class Step3Trainer(DetectionTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step3_group = a.group
            self.step3_contract = contract
            self.step3_requested_batch = a.batch

        def build_dataset(self, img_path, mode="train", batch=None):
            split = "train" if mode == "train" else "val"
            return TriModalDataset(self.step3_contract, split=split, group=self.step3_group,
                                   imgsz=self.args.imgsz, seed=self.args.seed,
                                   fliplr=self.args.fliplr, augment=(mode == "train"))

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "train":
                dataset = self.build_dataset(dataset_path, mode, batch_size)
                self._epoch_dataset = dataset
                loader = InfiniteDataLoader(
                    dataset, batch_size=batch_size, sampler=dataset.sampler,
                    shuffle=False, num_workers=self.args.workers,
                    collate_fn=dataset.collate_fn, pin_memory=True, drop_last=False)
                return loader
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        def preprocess_batch(self, batch):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
            batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
            # NO /255: input is already float32 [0,1]
            return batch

        def get_validator(self):
            # training-time validation: BaseValidator.__call__ in the training branch does
            # NOT rebuild the dataloader; the loader must be passed at construction
            # (stock passes self.test_loader, built from our val TriModalDataset).
            v = Step3Validator(self.test_loader, save_dir=self.save_dir, args=self.args)
            v.step3_group = self.step3_group
            v.step3_contract = self.step3_contract
            return v

        def final_eval(self):
            # stock final_eval runs AutoBackend on a 3-channel data contract (data["channels"]
            # defaults to 3) -> warmup crash on the 6ch model, and its stock RGB val dataset
            # would be wrong anyway. Our post-hoc eval (eval_step3_causality) replaces it;
            # keep optimizer states in checkpoints (torch.load weights_only=False handles it).
            print(f"[{a.group}] final_eval skipped (post-hoc eval_step3_causality replaces it)")

    kw = dict(R3_KW)
    kw.update(epochs=a.epochs, batch=a.batch, seed=a.seed,
              project=a.project, name=a.group, device=a.device,
              data=a.data_yaml, exist_ok=True)

    # NOTE: bypass the YOLO wrapper entirely — YOLO.train() would build a STOCK
    # DetectionTrainer and silently drop our Step3Trainer overrides (observed:
    # stock preprocess /255 ran, 3ch model was rebuilt). Instantiate directly.
    # args.model must be a path string for BaseTrainer.__init__ (check_model_file_from_stem);
    # the real 6ch model is injected right after construction (setup_model early-returns
    # for nn.Module, so no yaml rebuild happens).
    kw["model"] = "yolo26s.yaml"
    trainer = Step3Trainer(overrides=kw)
    trainer.model = model
    trainer.model.nc = 12
    trainer.model.args = trainer.args  # v8DetectionLoss lazy-inits from model.args

    trace = []
    growth = []

    def on_epoch_start(tr):
        ds = getattr(tr, "_epoch_dataset", None)
        if ds is not None:
            ds.set_epoch(tr.epoch)  # RANK=-1: stock never calls set_epoch
            ordered_ids = [ds.ids[i] for i in ds.sampler.perm]
            flip_bits = "".join("1" if mp.should_flip(ds.seed, tr.epoch, sid, ds.fliplr)
                                else "0" for sid in ordered_ids)
            trace.append({"epoch": tr.epoch,
                          "sample_order_sha256": ds.sampler.order_sha256(),
                          "flip_schedule_sha256": sha256_text(flip_bits),
                          "batch": int(tr.args.batch)})

    def on_epoch_end(tr):
        w = tr.model.model[0].conv.weight.detach().cpu().float()
        rgb_mean = sum(float(w[:, c].norm()) for c in range(3)) / 3.0
        growth.append({"epoch": tr.epoch + 1,
                       "wI_norm": float(w[:, 3].norm()),
                       "wD_norm": float(w[:, 4].norm()),
                       "wM_norm": float(w[:, 5].norm()),
                       "qI": float(w[:, 3].norm()) / (rgb_mean + 1e-12),
                       "qD": float(w[:, 4].norm()) / (rgb_mean + 1e-12),
                       "qM": float(w[:, 5].norm()) / (rgb_mean + 1e-12),
                       "batch": int(tr.args.batch)})
        if tr.args.batch != tr.step3_requested_batch:
            raise RuntimeError(
                f"ABORT_RESOURCE_PROFILE_CHANGED: requested batch={tr.step3_requested_batch} "
                f"but trainer batch={tr.args.batch} (auto-reduction). All groups must "
                f"retrain uniformly at batch=2/nbs=4.")

    trainer.add_callback("on_train_epoch_start", on_epoch_start)
    trainer.add_callback("on_train_epoch_end", on_epoch_end)

    run_dir = Path(a.project) / a.group
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "group": a.group, "snapshot": a.snapshot,
        "initial_checkpoint_sha256": sha256_file(Path(a.snapshot)),
        "recipe": "R3-causal-earlyfusion-sample",
        "contract_sha256": sha256_file(Path(a.contract)),
        "requested_batch": a.batch,
        "channel_semantics": ["R", "G", "B", "IR_scalar", "log_depth", "valid_mask"],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    trainer.train()

    (run_dir / "step3_g8_trace.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trace) + "\n", encoding="utf-8")
    (run_dir / "step3_kernel_growth.jsonl").write_text(
        "\n".join(json.dumps(g) for g in growth) + "\n", encoding="utf-8")
    print(f"[{a.group}] DONE. epochs={len(trace)} batch_final={trace[-1]['batch']}")


if __name__ == "__main__":
    main()
