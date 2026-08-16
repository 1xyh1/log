#!/usr/bin/env python3
"""Train Step 4-F1 RGB-primary IR reliability-gate experiments.

Formal groups share the same model state and data contract:
  F1-C0      : zero auxiliary input + learned gate (matched null control)
  F1-I-fixed : paired IR + q forced to 1 (implementation-matched F0-like control)
  F1-I-soft  : paired IR + learned scalar q (treatment)

No Depth is accepted in this stage.  The frozen Step-3/Step-4 data loader,
preprocessing, actual-yield G8, and no-/255 semantics are reused unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import modality_preprocess as mp  # noqa: E402
from multimodal.early_fusion_yolo26 import MODEL_INIT_SEED, build_reference_3ch  # noqa: E402
from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.step4_f1_ir_gate_model import Step4F1IRGateModel  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402
from ultralytics.data.build import InfiniteDataLoader  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402

GROUP_SPECS = {
    "F1-C0": {"aux_mode": "zero", "gate_mode": "learned", "dataset": "C0-N"},
    "F1-I-fixed": {"aux_mode": "ir", "gate_mode": "fixed_one", "dataset": "C1-I"},
    "F1-I-soft": {"aux_mode": "ir", "gate_mode": "learned", "dataset": "C1-I"},
}

R3_KW = dict(
    epochs=80, batch=4, nbs=4, warmup_epochs=0, workers=0, cache=False,
    imgsz=640, max_det=100, patience=100, close_mosaic=0,
    mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
    scale=0.0, translate=0.0, degrees=0.0, shear=0.0, perspective=0.0,
    multi_scale=0.0, amp=False, fliplr=0.30393, flipud=0.0,
    # F1 inherits the frozen F0 AMP-path incompatibility boundary.
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0,
    auto_augment=None, erasing=0.0,
    seed=20260812, deterministic=True, end2end=False,
    plots=False, cls_pw=0.0,
    optimizer="MuSGD", lr0=0.00038, lrf=0.88219, momentum=0.94751,
    weight_decay=0.00027, box=9.83241, cls=0.64896, dfl=0.95824,
)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_json(obj) -> str:
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state_sha(module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _param_diffs(current, initial) -> dict:
    l2_num, l2_den, max_abs = 0.0, 0.0, 0.0
    current_items = sorted(current.named_parameters())
    initial_items = sorted(initial.named_parameters())
    if [n for n, _ in current_items] != [n for n, _ in initial_items]:
        raise RuntimeError("parameter-name mismatch during G6")
    for (_, p), (_, q) in zip(current_items, initial_items):
        p = p.detach().cpu().float()
        q = q.detach().cpu().float()
        l2_num += float((p - q).pow(2).sum())
        l2_den += float(q.pow(2).sum())
        max_abs = max(max_abs, float((p - q).abs().max()))
    return {
        "global_rel_l2": (l2_num ** 0.5) / (l2_den ** 0.5 + 1e-12),
        "max_abs_change": max_abs,
    }


class Step4F1Validator(DetectionValidator):
    """Existing float 6ch validator contract, with no stock /255."""

    def preprocess(self, batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        return batch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=sorted(GROUP_SPECS), required=True)
    p.add_argument("--project", default="runs/step4_f1_ir_gate")
    p.add_argument("--name", default=None)
    p.add_argument("--run-kind", choices=["smoke", "formal", "recovery"],
                   default="formal")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--f0-project", default="runs/step4_f0")
    p.add_argument(
        "--audit-report",
        default=str(ROOT / "reports" / "step4_f1_ir_gate" / "pretrain_audit.json"),
    )
    p.add_argument("--data", default=(
        "D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml"
    ))
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

    contract_path = Path(a.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    audit_path = Path(a.audit_report)
    if not audit_path.exists():
        raise RuntimeError(f"F1_PRETRAIN_AUDIT_MISSING: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("schema") != "step4-f1-ir-gate-audit-v1"
            or audit.get("all_passed") is not True):
        raise RuntimeError("F1_PRETRAIN_AUDIT_NOT_PASSED")
    audit_prov = audit.get("provenance") or {}
    audit_targets = {
        "contract_sha256": contract_path,
        "audit_source_sha256": ROOT / "scripts" / "audit_step4_f1_ir_gate.py",
        "model_source_sha256": ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py",
        "gate_source_sha256": ROOT / "src" / "multimodal" / "reliability_gate.py",
        "f0_model_source_sha256": ROOT / "src" / "multimodal" / "step4_f0_model.py",
        "aux_encoder_source_sha256": ROOT / "src" / "multimodal" / "aux_encoder.py",
        "feature_fusion_source_sha256": ROOT / "src" / "multimodal" / "feature_fusion.py",
        "dataset_source_sha256": ROOT / "src" / "multimodal" / "trimodal_dataset.py",
        "trainability_source_sha256": ROOT / "src" / "multimodal" / "trainability.py",
        "f0_summary_sha256": Path(a.f0_project) / "_summary_step4.json",
        "f0_loo_sha256": Path(a.f0_project) / "step4_loo.json",
        "f0_summarizer_source_sha256": ROOT / "scripts" / "summarize_step4.py",
        "f0_closeout_source_sha256": ROOT / "src" / "multimodal" / "step4_closeout.py",
    }
    stale_audit = {
        key: {"recorded": audit_prov.get(key), "current": _sha_file(path)}
        for key, path in audit_targets.items()
        if audit_prov.get(key) != _sha_file(path)
    }
    for key, current in (("torch_version", torch.__version__),
                         ("ultralytics_version", __import__("ultralytics").__version__)):
        if audit_prov.get(key) != current:
            stale_audit[key] = {"recorded": audit_prov.get(key), "current": current}
    if stale_audit:
        raise RuntimeError(f"F1_PRETRAIN_AUDIT_STALE: {stale_audit}")
    spec = GROUP_SPECS[a.group]

    rng_state = torch.random.get_rng_state()
    torch.manual_seed(MODEL_INIT_SEED)
    model = Step4F1IRGateModel(
        build_reference_3ch(), aux_mode=spec["aux_mode"],
        gate_mode=spec["gate_mode"],
    )
    torch.random.set_rng_state(rng_state)
    model.nc = 12

    requested_batch = int(a.batch)

    class Step4F1Trainer(DetectionTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.f1_group = a.group
            self.f1_contract = contract
            self.f1_requested_batch = requested_batch
            self._epoch_dataset = None
            self._g8_actual_ids: list[str] = []
            self._g8_actual_flips: list[bool] = []

        def build_dataset(self, img_path, mode="train", batch=None):
            split = "train" if mode == "train" else "val"
            return TriModalDataset(
                self.f1_contract, split=split, group=spec["dataset"],
                imgsz=self.args.imgsz, seed=self.args.seed,
                fliplr=self.args.fliplr, augment=(mode == "train"),
            )

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "train":
                dataset = self.build_dataset(dataset_path, mode, batch_size)
                self._epoch_dataset = dataset
                return InfiniteDataLoader(
                    dataset, batch_size=batch_size, sampler=dataset.sampler,
                    shuffle=False, num_workers=self.args.workers,
                    collate_fn=dataset.collate_fn, pin_memory=True, drop_last=False,
                )
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        def preprocess_batch(self, batch):
            if int(self.args.batch) != self.f1_requested_batch:
                raise RuntimeError("ABORT_RESOURCE_PROFILE_CHANGED")
            ids = batch.get("sample_id")
            flips = batch.get("flip_applied")
            if ids is not None:
                self._g8_actual_ids.extend(str(x) for x in ids)
            if flips is not None:
                self._g8_actual_flips.extend(bool(x) for x in flips)
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(
                        self.device, non_blocking=self.device.type == "cuda"
                    )
            batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
            return batch

        def _build_train_pipeline(self):
            # Ultralytics can re-enable custom frozen parameters.  Re-freeze after
            # its unfreeze phase and before optimizer construction, as in F0.
            from multimodal.trainability import freeze_module
            freeze_module(self.model.rgb_backbone, freeze_bn_stats=True)
            super()._build_train_pipeline()
            opt_ids = {
                id(param) for group in self.optimizer.param_groups
                for param in group["params"]
            }
            missing = []
            for name, module in (
                ("fusions", self.model.fusions),
                ("aux_encoder", self.model.aux_encoder),
                ("reliability_gate", self.model.reliability_gate),
                ("tail", self.model.tail),
            ):
                for param in module.parameters():
                    if not param.requires_grad or id(param) not in opt_ids:
                        missing.append(name)
            rgb_trainable = any(
                param.requires_grad for param in self.model.rgb_backbone.parameters()
            )
            if missing or rgb_trainable:
                raise RuntimeError(
                    "G5_OPTIMIZER_MEMBERSHIP_FAIL: "
                    f"missing={sorted(set(missing))} rgb_trainable={rgb_trainable}"
                )
            print(f"[{a.group}] G5 PASS: aux+projection+gate+tail in optimizer; RGB frozen")

        def get_validator(self):
            return Step4F1Validator(
                self.test_loader, save_dir=self.save_dir, args=self.args
            )

        def final_eval(self):
            print(f"[{a.group}] final_eval skipped; run eval_step4_f1_causality.py")

    kw = dict(R3_KW)
    kw.update(
        epochs=a.epochs, batch=a.batch, seed=a.seed, project=str(project),
        name=a.name, device=a.device, data=a.data, exist_ok=True,
        model="yolo26s.yaml",
    )
    trainer = Step4F1Trainer(overrides=kw)
    trainer.model = model
    trainer.model.args = trainer.args

    trace: list[dict] = []
    growth: list[dict] = []

    def on_epoch_start(tr):
        ds = getattr(tr, "_epoch_dataset", None)
        if ds is None:
            raise RuntimeError("G8_NO_EPOCH_DATASET")
        ds.set_epoch(tr.epoch)
        if hasattr(tr.train_loader, "reset"):
            tr.train_loader.reset()
        tr._g8_actual_ids = []
        tr._g8_actual_flips = []
        tr.model.reset_gate_stats()

    def on_epoch_end(tr):
        ds = tr._epoch_dataset
        expected_ids = [ds.ids[i] for i in ds.sampler.perm]
        expected_flips = [
            mp.should_flip(ds.seed, tr.epoch, sid, ds.fliplr) for sid in expected_ids
        ]
        actual_ids = list(tr._g8_actual_ids)
        actual_flips = list(tr._g8_actual_flips)
        if actual_ids != expected_ids:
            raise RuntimeError(f"G8_ACTUAL_ORDER_MISMATCH epoch={tr.epoch}")
        if actual_flips != expected_flips:
            raise RuntimeError(f"G8_ACTUAL_FLIP_MISMATCH epoch={tr.epoch}")
        legacy_bits = "".join("1" if value else "0" for value in expected_flips)
        trace.append({
            "epoch": tr.epoch,
            "n_samples": len(actual_ids),
            "sample_order_sha256": ds.sampler.order_sha256(),
            "flip_schedule_sha256": _sha_text(legacy_bits),
            "expected_order_sha256": _sha_json(expected_ids),
            "actual_order_sha256": _sha_json(actual_ids),
            "expected_flip_sha256": _sha_json(expected_flips),
            "actual_flip_sha256": _sha_json(actual_flips),
            "actual_matches_expected": True,
            "batch": int(tr.args.batch),
        })
        gate_stats = tr.model.gate_stats()
        if gate_stats["count"] == 0:
            raise RuntimeError(f"G6_GATE_NOT_OBSERVED epoch={tr.epoch}")
        growth.append({
            "epoch": tr.epoch + 1,
            "projP3_norm": float(tr.model.fusions["4"].proj.weight.norm()),
            "projP4_norm": float(tr.model.fusions["6"].proj.weight.norm()),
            "projP5_norm": float(tr.model.fusions["10"].proj.weight.norm()),
            "projP3_bias_norm": float(tr.model.fusions["4"].proj.bias.norm()),
            "projP4_bias_norm": float(tr.model.fusions["6"].proj.bias.norm()),
            "projP5_bias_norm": float(tr.model.fusions["10"].proj.bias.norm()),
            "aux_encoder_norm": float(sum(
                param.norm() for param in tr.model.aux_encoder.parameters()
            )),
            "gate_param_norm": float(sum(
                param.norm() for param in tr.model.reliability_gate.parameters()
            )),
            "effective_q": gate_stats,
            "batch": int(tr.args.batch),
        })

    trainer.add_callback("on_train_epoch_start", on_epoch_start)
    trainer.add_callback("on_train_epoch_end", on_epoch_end)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "step4-f1-ir-gate-manifest-v1",
        "group": a.group,
        "physical_run_name": a.name,
        "run_kind": a.run_kind,
        "model": "Step4F1IRGateModel (RGB anchor + q*zero-init IR residual)",
        "aux_mode": spec["aux_mode"],
        "gate_mode": spec["gate_mode"],
        "dataset_group": spec["dataset"],
        "rgb_policy": "frozen and unscaled; BN eval",
        "recipe": "R3-causal-earlyfusion-sample",
        "expected_epochs": a.epochs,
        "requested_batch": a.batch,
        "seed": a.seed,
        "model_init_seed": MODEL_INIT_SEED,
        "contract_sha256": _sha_file(contract_path),
        "pretrain_audit_sha256": _sha_file(audit_path),
        "runner_source_sha256": _sha_file(Path(__file__)),
        "model_source_sha256": _sha_file(
            ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py"
        ),
        "gate_source_sha256": _sha_file(
            ROOT / "src" / "multimodal" / "reliability_gate.py"
        ),
        "f0_model_source_sha256": _sha_file(
            ROOT / "src" / "multimodal" / "step4_f0_model.py"
        ),
        "aux_encoder_source_sha256": _sha_file(
            ROOT / "src" / "multimodal" / "aux_encoder.py"
        ),
        "feature_fusion_source_sha256": _sha_file(
            ROOT / "src" / "multimodal" / "feature_fusion.py"
        ),
        "trainability_source_sha256": _sha_file(
            ROOT / "src" / "multimodal" / "trainability.py"
        ),
        "dataset_source_sha256": _sha_file(
            ROOT / "src" / "multimodal" / "trimodal_dataset.py"
        ),
        "preprocess_source_sha256": _sha_file(
            ROOT / "src" / "multimodal" / "modality_preprocess.py"
        ),
        "initial_rgb_backbone_sha256": _state_sha(model.rgb_backbone),
        "initial_aux_encoder_sha256": _state_sha(model.aux_encoder),
        "initial_fusion_sha256": _state_sha(model.fusions),
        "initial_gate_sha256": _state_sha(model.reliability_gate),
        "initial_model_state_sha256": _state_sha(model),
        "g8_evidence": "actual_dataloader_yield_v1",
        "ultralytics_version": __import__("ultralytics").__version__,
        "torch_version": torch.__version__,
        "created_at": __import__("datetime").datetime.now().astimezone().isoformat(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    trainer.train()

    (run_dir / "step4_g8_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in trace) + "\n", encoding="utf-8"
    )
    (run_dir / "step4_growth.jsonl").write_text(
        "\n".join(json.dumps(row) for row in growth) + "\n", encoding="utf-8"
    )

    trained = trainer.model
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(MODEL_INIT_SEED)
    initial = Step4F1IRGateModel(
        build_reference_3ch(), aux_mode=spec["aux_mode"],
        gate_mode=spec["gate_mode"],
    )
    torch.random.set_rng_state(rng_state)
    aux_diff = _param_diffs(trained.aux_encoder, initial.aux_encoder)
    gate_diff = _param_diffs(trained.reliability_gate, initial.reliability_gate)
    proj_norms = [
        float(trained.fusions[key].proj.weight.norm()) for key in ("4", "6", "10")
    ]
    proj_bias_norms = [
        float(trained.fusions[key].proj.bias.norm()) for key in ("4", "6", "10")
    ]
    last_q = growth[-1]["effective_q"] if growth else {
        "count": 0, "mean": None, "min": None, "max": None
    }
    q_finite = all(
        value is not None and math.isfinite(float(value))
        for value in (last_q["mean"], last_q["min"], last_q["max"])
    )
    gate = {
        "rgb_backbone_unchanged": (
            _state_sha(trained.rgb_backbone) == manifest["initial_rgb_backbone_sha256"]
        ),
        "aux_encoder_global_rel_l2": aux_diff["global_rel_l2"],
        "aux_encoder_max_abs_change": aux_diff["max_abs_change"],
        "gate_global_rel_l2": gate_diff["global_rel_l2"],
        "gate_max_abs_change": gate_diff["max_abs_change"],
        "proj_weight_norms": proj_norms,
        "proj_bias_norms": proj_bias_norms,
        "last_epoch_effective_q": last_q,
        "q_finite_and_bounded": bool(
            q_finite and 0.0 <= float(last_q["min"]) <= float(last_q["max"]) <= 1.0
        ),
    }
    if a.group == "F1-C0":
        passed = bool(
            gate["rgb_backbone_unchanged"]
            and gate["aux_encoder_global_rel_l2"] < 1e-3
            and max(proj_norms) == 0.0
            and gate["q_finite_and_bounded"]
        )
        gate["expected"] = "null aux stays below decay threshold; proj weights stay zero"
    elif a.group == "F1-I-fixed":
        passed = bool(
            gate["rgb_backbone_unchanged"]
            and gate["aux_encoder_global_rel_l2"] > 1e-3
            and min(proj_norms) > 0.0
            and gate["q_finite_and_bounded"]
            and last_q["min"] == 1.0 and last_q["max"] == 1.0
        )
        gate["expected"] = "active IR/projections learn; effective q remains exactly one"
    else:
        passed = bool(
            gate["rgb_backbone_unchanged"]
            and gate["aux_encoder_global_rel_l2"] > 1e-3
            and min(proj_norms) > 0.0
            and gate["gate_max_abs_change"] > 0.0
            and gate["q_finite_and_bounded"]
        )
        gate["expected"] = "active IR/projections/gate learn; q remains finite in [0,1]"
    gate["passed"] = passed
    (run_dir / "step4_update_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(f"G6_REAL_UPDATE_GATE_FAIL: {gate}")
    print(f"[{a.group}] G6 PASS -> {run_dir}")


if __name__ == "__main__":
    main()
