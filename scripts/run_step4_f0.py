#!/usr/bin/env python3
"""Step 4-F0 runner: F0-C0 / F0-I / F0-D x 80 epochs (seed 20260812).

Model = Step4F0Model (RGB frozen anchor + 2ch aux pyramid encoder + zero-init 1x1
residual injection at P3/P4/P5 + original neck/head). The 6ch TriModalDataset and
the float32 no-/255 trainer/validator contract are REUSED from the fixed Step-3
runner; the model splits the 6ch batch internally (aux_mode zero/ir/depth).

Freeze profile (F0): RGB backbone frozen (BN eval enforced on model.train());
trainable = aux encoder + fusion projections + neck/head.
Recipe: R3-causal-earlyfusion-sample (unchanged). Growth log records per-scale
fusion projection norms + aux encoder norm (F0-C0 must stay exactly 0).
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
from multimodal.early_fusion_yolo26 import MODEL_INIT_SEED, build_reference_3ch  # noqa: E402
from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.step4_f0_model import Step4F0Model  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402
from ultralytics.data.build import InfiniteDataLoader  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402

GROUPS = {"F0-C0": "zero", "F0-I": "ir", "F0-D": "depth"}
DATASET_GROUP = {"F0-C0": "C0-N", "F0-I": "C1-I", "F0-D": "C2-D"}  # same 6ch content contract

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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj) -> str:
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Step4Validator(DetectionValidator):
    """Float 6ch validator (no /255); the F0 model splits the 6ch batch internally."""

    def preprocess(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=["F0-C0", "F0-I", "F0-D"], required=True)
    p.add_argument("--project", default="runs/step4_f0")
    p.add_argument("--name", default=None)
    p.add_argument("--run-kind", choices=["smoke", "formal", "recovery"], default="formal")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=4)
    a = p.parse_args()

    if a.run_kind == "smoke" and a.name is None:
        a.name = f"smoke-{a.group}-e{a.epochs}"
    elif a.name is None:
        a.name = a.group
    project = Path(a.project).resolve()
    run_dir = project / a.name
    if a.run_kind in {"formal", "recovery"} and run_dir.exists():
        raise RuntimeError(f"FORMAL_RUN_DIR_EXISTS: {run_dir}")

    contract = json.loads(Path(a.contract).read_text(encoding="utf-8"))

    # reproducible aux-encoder init (RGB path is pretrained; aux encoder is new)
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(MODEL_INIT_SEED)
    model = Step4F0Model(build_reference_3ch(), aux_mode=GROUPS[a.group])
    torch.random.set_rng_state(rng_state)
    model.nc = 12

    requested_batch = int(a.batch)
    trace_path = run_dir / "step4_g8_trace.jsonl"
    growth_path = run_dir / "step4_growth.jsonl"

    class Step4Trainer(DetectionTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step4_group = a.group
            self.step4_contract = contract
            self.step4_requested_batch = requested_batch
            self._epoch_dataset = None
            self._g8_actual_ids: list[str] = []
            self._g8_actual_flips: list[bool] = []

        def build_dataset(self, img_path, mode="train", batch=None):
            split = "train" if mode == "train" else "val"
            return TriModalDataset(self.step4_contract, split=split,
                                   group=DATASET_GROUP[self.step4_group],
                                   imgsz=self.args.imgsz, seed=self.args.seed,
                                   fliplr=self.args.fliplr, augment=(mode == "train"))

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "train":
                dataset = self.build_dataset(dataset_path, mode, batch_size)
                self._epoch_dataset = dataset
                return InfiniteDataLoader(
                    dataset, batch_size=batch_size, sampler=dataset.sampler,
                    shuffle=False, num_workers=self.args.workers,
                    collate_fn=dataset.collate_fn, pin_memory=True, drop_last=False)
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        def preprocess_batch(self, batch):
            if int(self.args.batch) != self.step4_requested_batch:
                raise RuntimeError("ABORT_RESOURCE_PROFILE_CHANGED")
            # ACTUAL DataLoader yield evidence (before any device copy), same as the
            # fixed Step-3 runner — no planned-hash-only regression (reviewer P0-3).
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
            return batch

        def _build_train_pipeline(self):
            # Reviewer P0: stock _setup_train re-enables requires_grad on non-stock
            # freeze-pattern params (our names are rgb_backbone.*, freeze: null).
            # Re-freeze AFTER the unfreeze loop and BEFORE the optimizer is built.
            from multimodal.trainability import freeze_module
            freeze_module(self.model.rgb_backbone, freeze_bn_stats=True)
            super()._build_train_pipeline()
            # G5 hard gate: optimizer membership + frozen anchor (reviewer-mandated)
            opt_ids = {id(p) for grp in self.optimizer.param_groups for p in grp["params"]}
            missing = []
            for name, module in (("fusions", self.model.fusions),
                                 ("aux_encoder", self.model.aux_encoder),
                                 ("tail", self.model.tail)):
                for p in module.parameters():
                    if not p.requires_grad or id(p) not in opt_ids:
                        missing.append(name)
            frozen_violation = any(p.requires_grad for p in
                                   self.model.rgb_backbone.parameters())
            if missing or frozen_violation:
                raise RuntimeError(
                    f"G5_OPTIMIZER_MEMBERSHIP_FAIL: missing={sorted(set(missing))} "
                    f"rgb_trainable={frozen_violation}")
            print(f"[{a.group}] G5 optimizer membership PASS: "
                  f"fusions+aux+tail in optimizer, RGB frozen")

        def get_validator(self):
            v = Step4Validator(self.test_loader, save_dir=self.save_dir, args=self.args)
            return v

        def final_eval(self):
            print(f"[{a.group}] final_eval skipped (post-hoc eval_step4_causality)")

    kw = dict(R3_KW)
    kw.update(epochs=a.epochs, batch=a.batch, seed=a.seed, project=str(project),
              name=a.name, device=a.device,
              data="D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml",
              exist_ok=True, model="yolo26s.yaml")
    trainer = Step4Trainer(overrides=kw)
    trainer.model = model
    trainer.model.args = trainer.args

    trace = []
    growth = []

    def on_epoch_start(tr):
        ds = getattr(tr, "_epoch_dataset", None)
        if ds is None:
            raise RuntimeError("G8_NO_EPOCH_DATASET")
        ds.set_epoch(tr.epoch)
        if hasattr(tr.train_loader, "reset"):
            tr.train_loader.reset()  # InfiniteDataLoader owns a persistent iterator
        tr._g8_actual_ids = []
        tr._g8_actual_flips = []

    def on_epoch_end(tr):
        ds = tr._epoch_dataset
        expected_ids = [ds.ids[i] for i in ds.sampler.perm]
        expected_flips = [mp.should_flip(ds.seed, tr.epoch, sid, ds.fliplr)
                          for sid in expected_ids]
        actual_ids = list(tr._g8_actual_ids)
        actual_flips = list(tr._g8_actual_flips)
        if actual_ids != expected_ids:
            raise RuntimeError(f"G8_ACTUAL_ORDER_MISMATCH epoch={tr.epoch}")
        if actual_flips != expected_flips:
            raise RuntimeError(f"G8_ACTUAL_FLIP_MISMATCH epoch={tr.epoch}")
        legacy_flip_bits = "".join("1" if f else "0" for f in expected_flips)
        trace.append({"epoch": tr.epoch,
                      "n_samples": len(actual_ids),
                      "sample_order_sha256": ds.sampler.order_sha256(),
                      "flip_schedule_sha256": sha256_text(legacy_flip_bits),
                      "expected_order_sha256": sha256_json(expected_ids),
                      "actual_order_sha256": sha256_json(actual_ids),
                      "expected_flip_sha256": sha256_json(expected_flips),
                      "actual_flip_sha256": sha256_json(actual_flips),
                      "actual_matches_expected": True,
                      "batch": int(tr.args.batch)})
        w = tr.model.fusions
        growth.append({"epoch": tr.epoch + 1,
                       "projP3_norm": float(w["4"].proj.weight.norm()),
                       "projP4_norm": float(w["6"].proj.weight.norm()),
                       "projP5_norm": float(w["10"].proj.weight.norm()),
                       "aux_encoder_norm": float(sum(p.norm() for p in
                                                      tr.model.aux_encoder.parameters())),
                       "batch": int(tr.args.batch)})

    trainer.add_callback("on_train_epoch_start", on_epoch_start)
    trainer.add_callback("on_train_epoch_end", on_epoch_end)

    run_dir.mkdir(parents=True, exist_ok=True)

    def _state_sha(module) -> str:
        from multimodal.early_fusion_yolo26 import tensor_sha256
        h = hashlib.sha256()
        for n, p in sorted(module.state_dict().items()):
            h.update(n.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    manifest = {
        "schema": "step4-f0-manifest-v1",
        "group": a.group,
        "physical_run_name": a.name,
        "run_kind": a.run_kind,
        "model": "Step4F0Model (RGB anchor + zero-init residual)",
        "aux_mode": GROUPS[a.group],
        "freeze": "RGB backbone frozen, BN eval",
        "recipe": "R3-causal-earlyfusion-sample",
        "expected_epochs": a.epochs,
        "requested_batch": a.batch,
        "seed": a.seed,
        "aux_encoder_seed": MODEL_INIT_SEED,
        "contract_sha256": hashlib.sha256(Path(a.contract).read_bytes()).hexdigest(),
        "initial_rgb_backbone_sha256": _state_sha(model.rgb_backbone),
        "initial_aux_encoder_sha256": _state_sha(model.aux_encoder),
        "initial_fusion_sha256": _state_sha(model.fusions),
        "initial_model_state_sha256": _state_sha(model),
        "g8_evidence": "actual_dataloader_yield_v1",
        "ultralytics_version": __import__("ultralytics").__version__,
        "created_at": __import__("datetime").datetime.now().astimezone().isoformat(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                           encoding="utf-8")

    trainer.train()

    (run_dir / "step4_g8_trace.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trace) + "\n", encoding="utf-8")
    (run_dir / "step4_growth.jsonl").write_text(
        "\n".join(json.dumps(g) for g in growth) + "\n", encoding="utf-8")
    print(f"[{a.group}] DONE -> {run_dir}")


if __name__ == "__main__":
    main()
