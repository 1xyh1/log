#!/usr/bin/env python3
"""Step 4-F1-B runner: F1 architecture + training-time deterministic IR
corruption/dropout.  B1-C0 / B1-I-fixed / B1-I-soft x 80 epochs.

Architecture, gate detach, recipe, amp=False, data contract, validator and G8
are inherited unchanged from F1.  The ONLY new variable is the frozen
training-time IR corruption schedule (see src/multimodal/step4_f1_b_corruption.py):
    clean 0.50 | zero 0.125 | noise/blur/contrast 0.125 each
    severity uniform {0.25, 0.50, 0.75, 1.00}; zero fixed 1.0; shift excluded.

G9 (new): per-epoch actual-yield corruption trace — sample_id/kind/severity,
expected vs actual schedule SHA, IR plane SHA before/after, RGB/Depth/label/
bbox untouched assertions, and per-epoch kind counts.  Any mismatch aborts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import modality_preprocess as mp  # noqa: E402
from multimodal.early_fusion_yolo26 import (  # noqa: E402
    MODEL_INIT_SEED, WEIGHTS_DEFAULT, build_reference_3ch)
from multimodal.modality_quality import content_mask_from_sample  # noqa: E402
from multimodal.raw_sample_index import CLASS_NAMES, OUT_DEFAULT  # noqa: E402
from multimodal.step4_f1_b_corruption import (  # noqa: E402
    KIND_PROBS, apply_schedule_to_plane, sample_schedule, schedule_for_epoch,
    schedule_sha256, sha256_plane)
from multimodal.step4_f1_c_readiness import (  # noqa: E402
    EXPECTED_BASE_CHECKPOINT_SHA256, check_initial_state_equality,
    verify_base_checkpoint, verify_data_yaml, verify_raw_data_freshness)
from multimodal.step4_f1_ir_gate_model import Step4F1IRGateModel  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402
from ultralytics.data.build import InfiniteDataLoader  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402

GROUP_SPECS = {
    "F1C-C0": {"aux_mode": "zero", "gate_mode": "learned",
               "gate_module": "magnitude", "dataset": "C0-N"},
    "F1C-I-fixed": {"aux_mode": "ir", "gate_mode": "fixed_one",
                    "gate_module": "magnitude", "dataset": "C1-I"},
    "F1C-I-magsoft": {"aux_mode": "ir", "gate_mode": "learned",
                      "gate_module": "magnitude", "dataset": "C1-I"},
    "F1C-I-soft": {"aux_mode": "ir", "gate_mode": "learned",
                   "gate_module": "original", "dataset": "C1-I"},
}

R3_KW = dict(
    epochs=80, batch=4, nbs=4, warmup_epochs=0, workers=0, cache=False,
    imgsz=640, max_det=100, patience=100, close_mosaic=0,
    mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
    scale=0.0, translate=0.0, degrees=0.0, shear=0.0, perspective=0.0,
    multi_scale=0.0, amp=False, fliplr=0.30393, flipud=0.0,
    # amp=False is the frozen F0/F1 boundary (AMP-path incompatibility).
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0,
    auto_augment=None, erasing=0.0,
    seed=20260812, deterministic=True, end2end=False,
    plots=False, cls_pw=0.0,
    optimizer="MuSGD", lr0=0.00038, lrf=0.88219, momentum=0.94751,
    weight_decay=0.00027, box=9.83241, cls=0.64896, dfl=0.95824,
)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_json(obj) -> str:
    # sort_keys=True matches the canonical schedule_sha256 serialization so
    # expected/actual schedule anchors compare byte-identically (G9).
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"),
                         sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _param_diffs(module_a, module_b) -> dict:
    l2_num, l2_den, max_abs = 0.0, 0.0, 0.0
    for (_, p), (_, q) in zip(sorted(module_a.named_parameters()),
                              sorted(module_b.named_parameters())):
        p = p.detach().cpu().float()
        q = q.detach().cpu().float()
        l2_num += float((p - q).pow(2).sum())
        l2_den += float(q.pow(2).sum())
        max_abs = max(max_abs, float((p - q).abs().max()))
    return {"global_rel_l2": (l2_num ** 0.5) / (l2_den ** 0.5 + 1e-12),
            "max_abs_change": max_abs}


def _state_sha(module) -> str:
    h = hashlib.sha256()
    for n, p in sorted(module.state_dict().items()):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class B1CorruptionDatasetView:
    """Wraps TriModalDataset: after the frozen preprocessing/letterbox output,
    corrupts ONLY channel 3 (IR) per the deterministic schedule and records
    the G9 evidence for the epoch.

    apply_corruption=False is used for B1-C0: the C0 aux input is already
    zero, so the corruption is intentionally NOT applied (any noise would
    only touch an unused channel); the schedule evidence is still recorded so
    the control shares the identical training-time schedule.
    """

    def __init__(self, dataset, seed: int, apply_corruption: bool = True):
        self.dataset = dataset
        self.seed = int(seed)
        self.apply_corruption = bool(apply_corruption)
        self.epoch = 0
        self.records: list[dict] = []

    def set_epoch(self, epoch: int):
        self.dataset.set_epoch(epoch)
        self.epoch = int(epoch)
        self.records = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        base = self.dataset[index]
        sample = dict(base)
        sid = str(sample["sample_id"])
        sched = sample_schedule(self.seed, self.epoch, sid)
        img = np.asarray(base["img"], dtype=np.float32).copy()
        rgb_before = img[:3].copy()
        dep_before = img[4:6].copy()
        ir_before_sha = sha256_plane(img[3])
        if self.apply_corruption:
            content = content_mask_from_sample(sample, imgsz=img.shape[-1])
            img[3] = apply_schedule_to_plane(
                img[3], sched, seed=self.seed, epoch=self.epoch,
                content_mask=content)
        ir_after_sha = sha256_plane(img[3])
        sample["img"] = np.ascontiguousarray(img)
        self.records.append({
            "sample_id": sid,
            "kind": sched["kind"],
            "severity": sched["severity"],
            "ir_sha_before": ir_before_sha,
            "ir_sha_after": ir_after_sha,
            "rgb_unchanged": bool((img[:3] == rgb_before).all()),
            "depth_unchanged": bool((img[4:6] == dep_before).all()),
            "labels_bboxes_same_object": sample["cls"] is base["cls"]
                                       and sample["bboxes"] is base["bboxes"],
        })
        return sample

    @property
    def collate_fn(self):
        return self.dataset.collate_fn


class Step4F1BValidator(DetectionValidator):
    """Float 6ch validator (no /255); the F1 model splits the 6ch batch."""

    def preprocess(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        return batch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=sorted(GROUP_SPECS), required=True)
    p.add_argument("--project", default="runs/step4_f1_c")
    p.add_argument("--name", default=None)
    p.add_argument("--run-kind", choices=["smoke", "formal", "recovery"],
                   default="formal")
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument(
        "--audit-report",
        default=str(ROOT / "reports" / "step4_f1_c" / "pretrain_audit.json"),
    )
    p.add_argument(
        "--readiness-report",
        default=str(ROOT / "reports" / "step4_f1_c" / "smoke_readiness.json"),
        help="formal only: fresh raw-smoke readiness report",
    )
    p.add_argument(
        "--data", default=(
            "D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml"))
    p.add_argument(
        "--base-checkpoint", default=WEIGHTS_DEFAULT,
        help="frozen external RGB anchor weights; SHA must equal "
             "EXPECTED_BASE_CHECKPOINT_SHA256 (smoke+formal, checked before "
             "model construction)")
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=4)
    a = p.parse_args()

    if a.run_kind == "formal":
        formal_drift = {
            "epochs": a.epochs if a.epochs != 80 else None,
            "batch": a.batch if a.batch != 4 else None,
            "seed": a.seed if a.seed != 20260812 else None,
        }
        formal_drift = {k: v for k, v in formal_drift.items() if v is not None}
        if formal_drift:
            raise RuntimeError(
                f"FORMAL_PROTOCOL_DRIFT: {formal_drift}; "
                "F1-C formal is frozen to epochs=80,batch=4,seed=20260812"
            )

    project = Path(a.project).resolve()
    if a.run_kind == "smoke" and a.name is None:
        # unique smoke directories: revision suffix so reruns never overwrite
        # historical smoke evidence (reviewer 2026-08-16)
        base = f"smoke-{a.group}-e{a.epochs}"
        a.name = base
        rev = 2
        while (project / a.name).exists():
            a.name = f"{base}-r{rev}"
            rev += 1
    elif a.name is None:
        a.name = a.group
    run_dir = project / a.name
    if run_dir.exists():
        raise RuntimeError(
            f"RUN_DIR_EXISTS: {run_dir} — formal/smoke directories are never "
            "reused; pass a fresh --name or let smoke auto-revision")

    contract_path = Path(a.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    spec = GROUP_SPECS[a.group]

    # ---- external runtime dependency closure (reviewer 2026-08-17 P0) ----
    # dataset.yaml 语义锁 (nc=12 + names == CLASS_NAMES) + 原始数据 17x4 重 hash。
    # smoke 与 formal 都执行:smoke 用错权重/数据等于整条链作废,提前失败优于
    # readiness 事后兜底。
    data_yaml_state = verify_data_yaml(Path(a.data), CLASS_NAMES)
    if "DATA_YAML_MISSING" in data_yaml_state["errors"]:
        raise RuntimeError(f"ABORT_DATA_YAML_MISSING: {a.data}")
    if not data_yaml_state["passed"]:
        raise RuntimeError(
            f"ABORT_DATA_YAML_SEMANTICS: {data_yaml_state['errors']}")
    raw_data_state = verify_raw_data_freshness(contract)
    if not raw_data_state["passed"]:
        raise RuntimeError(
            f"ABORT_RAW_DATA_STALE: {raw_data_state['errors']}")

    # ---- hard gate: B1 pretrain audit must exist, pass, and be FRESH ----
    audit_path = Path(a.audit_report)
    if not audit_path.exists():
        raise RuntimeError(f"F1C_PRETRAIN_AUDIT_MISSING: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("schema") != "step4-f1-c-audit-v3"
            or audit.get("all_passed") is not True):
        raise RuntimeError("F1C_PRETRAIN_AUDIT_NOT_PASSED")
    audit_prov = audit.get("provenance") or {}
    audit_targets = {
        "corruption_source_sha256": ROOT / "src" / "multimodal" / "step4_f1_b_corruption.py",
        "runner_source_sha256": ROOT / "scripts" / "run_step4_f1_c.py",
        "audit_source_sha256": ROOT / "scripts" / "audit_step4_f1_c.py",
        "gate_module_sha256": ROOT / "src" / "multimodal" / "reliability_gate.py",
        "model_source_sha256": ROOT / "src" / "multimodal" / "step4_f1_ir_gate_model.py",
        "builder_source_sha256": ROOT / "src" / "multimodal" / "early_fusion_yolo26.py",
        "f1c_design_freeze_sha256": ROOT / "docs" / "step4_f1_c" / "DESIGN_FREEZE.md",
        "a1_v2_last_sha256": ROOT / "reports" / "step4_f1_c_agreement" / "descriptor_audit_v2_last.json",
        "a1_v2_best_sha256": ROOT / "reports" / "step4_f1_c_agreement" / "descriptor_audit_v2_best.json",
        "b1_v22_summary_sha256": ROOT / "runs" / "step4_f1_b_corruption" / "_summary_step4_f1_b.json",
    }
    stale_audit = {
        key: {"recorded": audit_prov.get(key), "current": _sha_file(path)}
        for key, path in audit_targets.items()
        if audit_prov.get(key) != _sha_file(path)
    }
    if stale_audit:
        raise RuntimeError(f"F1C_PRETRAIN_AUDIT_STALE: {stale_audit}")

    readiness_path = Path(a.readiness_report)
    if not readiness_path.is_absolute():
        readiness_path = (ROOT / readiness_path).resolve()
    if a.run_kind == "formal":
        from multimodal.step4_f1_c_readiness import verify_readiness_report
        readiness = verify_readiness_report(
            ROOT, readiness_path, contract_path, requested_group=a.group,
            data_yaml_path=Path(a.data),
            base_checkpoint_path=Path(a.base_checkpoint),
            class_names=CLASS_NAMES,
        )
        if not readiness["passed"]:
            raise RuntimeError(
                f"F1C_FORMAL_READINESS_FAIL: {readiness['errors']}"
            )
        # dataset.yaml SHA 必须与 smoke 时一致 (ABORT_DATA_YAML_STALE)
        evidence_data_yaml = (
            (readiness.get("evidence") or {}).get("data_yaml") or {})
        if evidence_data_yaml.get("sha256") != data_yaml_state["sha256"]:
            raise RuntimeError(
                "ABORT_DATA_YAML_STALE: "
                f"smoke={evidence_data_yaml.get('sha256')} "
                f"current={data_yaml_state['sha256']}"
            )
        version_rows = list((readiness.get("evidence") or {}).get(
            "versions", {}).values())
        current_versions = {
            "torch": torch.__version__,
            "ultralytics": __import__("ultralytics").__version__,
        }
        if not version_rows or any(row != current_versions for row in version_rows):
            raise RuntimeError(
                "F1C_FORMAL_ENV_VERSION_MISMATCH: "
                f"smoke={version_rows} current={current_versions}"
            )

    # ---- base checkpoint lock: must run BEFORE build_reference_3ch() ----
    ckpt_state = verify_base_checkpoint(
        Path(a.base_checkpoint), EXPECTED_BASE_CHECKPOINT_SHA256)
    if "BASE_CHECKPOINT_MISSING" in ckpt_state["errors"]:
        raise RuntimeError(f"ABORT_BASE_CHECKPOINT_MISSING: {a.base_checkpoint}")
    if not ckpt_state["passed"]:
        raise RuntimeError(
            f"ABORT_BASE_CHECKPOINT_STALE: "
            f"actual={ckpt_state['sha256']} "
            f"expected={EXPECTED_BASE_CHECKPOINT_SHA256}"
        )

    rng_state = torch.random.get_rng_state()
    torch.manual_seed(MODEL_INIT_SEED)
    model = Step4F1IRGateModel(
        build_reference_3ch(), aux_mode=spec["aux_mode"],
        gate_mode=spec["gate_mode"],
        gate_module=spec.get("gate_module"),
    )
    torch.random.set_rng_state(rng_state)
    model.nc = 12

    # ---- initial-state equality (formal only): this instant's model must be
    # bitwise identical to the r3 smoke-frozen initial state (reviewer P0).
    # Computed once here and reused for the manifest (no duplicate SHA work).
    initial_shas = {
        "initial_rgb_backbone_sha256": _state_sha(model.rgb_backbone),
        "initial_aux_encoder_sha256": _state_sha(model.aux_encoder),
        "initial_fusion_sha256": _state_sha(model.fusions),
        "initial_gate_sha256": _state_sha(model.reliability_gate),
        "initial_model_state_sha256": _state_sha(model),
    }
    if a.run_kind == "formal":
        frozen = (
            (readiness.get("evidence") or {}).get("initial_state_frozen") or {})
        state_eq = check_initial_state_equality(initial_shas, frozen)
        if not state_eq["passed"]:
            raise RuntimeError(
                f"ABORT_INITIAL_STATE_MISMATCH: {state_eq['mismatches']}")

    requested_batch = int(a.batch)

    class Step4F1BTrainer(DetectionTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.b1_group = a.group
            self.b1_contract = contract
            self.b1_requested_batch = requested_batch
            self._epoch_dataset = None
            self._b1_view = None
            self._g8_actual_ids: list[str] = []
            self._g8_actual_flips: list[bool] = []

        def build_dataset(self, img_path, mode="train", batch=None):
            split = "train" if mode == "train" else "val"
            return TriModalDataset(self.b1_contract, split=split,
                                   group=spec["dataset"],
                                   imgsz=self.args.imgsz, seed=self.args.seed,
                                   fliplr=self.args.fliplr,
                                   augment=(mode == "train"))

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "train":
                dataset = self.build_dataset(dataset_path, mode, batch_size)
                self._epoch_dataset = dataset
                self._b1_view = B1CorruptionDatasetView(
                    dataset, seed=a.seed,
                    apply_corruption=(spec["aux_mode"] != "zero"))
                return InfiniteDataLoader(
                    self._b1_view, batch_size=batch_size,
                    sampler=dataset.sampler, shuffle=False,
                    num_workers=self.args.workers,
                    collate_fn=dataset.collate_fn, pin_memory=True,
                    drop_last=False)
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        def preprocess_batch(self, batch):
            if int(self.args.batch) != self.b1_requested_batch:
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
                        self.device, non_blocking=self.device.type == "cuda")
            batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
            return batch

        def _build_train_pipeline(self):
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
            return Step4F1BValidator(
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
    trainer = Step4F1BTrainer(overrides=kw)
    trainer.model = model
    trainer.model.args = trainer.args

    trace: list[dict] = []
    g9_trace: list[dict] = []
    g9_records: list[dict] = []
    growth: list[dict] = []

    def on_epoch_start(tr):
        ds = getattr(tr, "_epoch_dataset", None)
        view = getattr(tr, "_b1_view", None)
        if ds is None or view is None:
            raise RuntimeError("G9_NO_EPOCH_DATASET")
        ds.set_epoch(tr.epoch)
        view.set_epoch(tr.epoch)
        if hasattr(tr.train_loader, "reset"):
            tr.train_loader.reset()
        tr._g8_actual_ids = []
        tr._g8_actual_flips = []
        tr.model.reset_gate_stats()

    def on_epoch_end(tr):
        ds = tr._epoch_dataset
        view = tr._b1_view
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
        # ---- G9: actual-yield corruption trace ----
        records = list(view.records)
        if len(records) != len(actual_ids):
            raise RuntimeError(f"G9_RECORD_COUNT_MISMATCH epoch={tr.epoch}")
        expected_sched = schedule_for_epoch(a.seed, tr.epoch, ds.ids)
        actual_rows = [
            {"sample_id": r["sample_id"], "kind": r["kind"],
             "severity": r["severity"]} for r in records
        ]
        # align expected (sorted by id) to the actual yield order for the
        # sequence-level check
        expected_by_id = {r["sample_id"]: r for r in expected_sched}
        aligned = [{"sample_id": r["sample_id"],
                    "kind": expected_by_id[r["sample_id"]]["kind"],
                    "severity": expected_by_id[r["sample_id"]]["severity"]}
                   for r in records]
        if actual_rows != aligned:
            raise RuntimeError(f"G9_SCHEDULE_MISMATCH epoch={tr.epoch}")
        expected_sha = schedule_sha256(a.seed, tr.epoch, ds.ids)
        actual_sha = _sha_json(sorted(actual_rows, key=lambda r: r["sample_id"]))
        kind_counts = {}
        for r in records:
            kind_counts[r["kind"]] = kind_counts.get(r["kind"], 0) + 1
        rgb_ok = all(r["rgb_unchanged"] for r in records)
        dep_ok = all(r["depth_unchanged"] for r in records)
        labels_ok = all(r["labels_bboxes_same_object"] for r in records)
        if spec["aux_mode"] == "zero":
            # C0: the aux channel is already zero; corruption is a no-op but
            # the schedule must still be recorded identically.
            ir_changed_ok = all(r["ir_sha_before"] == r["ir_sha_after"]
                                for r in records)
        else:
            ir_changed_ok = all(
                (r["ir_sha_before"] == r["ir_sha_after"]) == (r["kind"] == "clean")
                for r in records)
        g9_ok = bool(rgb_ok and dep_ok and labels_ok and ir_changed_ok)
        # Per-sample evidence is persisted (reviewer: the epoch summary alone
        # cannot be independently re-judged from artifacts).
        epoch_records = [
            {"epoch": tr.epoch, **{k: r[k] for k in (
                "sample_id", "kind", "severity", "ir_sha_before", "ir_sha_after",
                "rgb_unchanged", "depth_unchanged",
                "labels_bboxes_same_object")}}
            for r in records
        ]
        g9_records.extend(epoch_records)
        records_sha = _sha_json(sorted(epoch_records,
                                       key=lambda r: r["sample_id"]))
        g9_trace.append({
            "epoch": tr.epoch,
            "n_samples": len(records),
            "expected_schedule_sha256": expected_sha,
            "actual_schedule_sha256": actual_sha,
            "expected_matches_actual": expected_sha == actual_sha,
            "records_sha256": records_sha,
            "ir_changed_for_corrupted_only": ir_changed_ok,
            "rgb_depth_labels_bboxes_unchanged": bool(rgb_ok and dep_ok and labels_ok),
            "kind_counts": kind_counts,
            "batch": int(tr.args.batch),
        })
        if not g9_ok or expected_sha != actual_sha:
            raise RuntimeError(f"G9_CORRUPTION_TRACE_FAIL epoch={tr.epoch}: {g9_trace[-1]}")
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

    def on_train_end(tr):
        # G10.7: record the final fp32 RGB backbone SHA from the trainer's
        # still-float32 model, BEFORE any downstream half-precision
        # serialization of checkpoints, and assert the match immediately.
        actual_final = _state_sha(tr.model.rgb_backbone)
        expected_initial = manifest["initial_rgb_backbone_sha256"]
        record = {
            "schema": "step4-f1-c-fp32-rgb-v1",
            "group": a.group,
            "expected_initial_sha256": expected_initial,
            "actual_final_sha256": actual_final,
            "match": actual_final == expected_initial,
            "note": "recorded at on_train_end from the fp32 trainer model "
                    "(pre-half checkpoint serialization)",
        }
        (run_dir / "step4_fp32_rgb_sha.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        if record["match"] is not True:
            raise RuntimeError(f"G10_7_FP32_RGB_SHA_MISMATCH: {record}")
        print(f"[{a.group}] G10.7 fp32 RGB SHA recorded and asserted")



    trainer.add_callback("on_train_epoch_start", on_epoch_start)
    trainer.add_callback("on_train_epoch_end", on_epoch_end)
    trainer.add_callback("on_train_end", on_train_end)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "step4-f1-c-manifest-v2",
        "group": a.group,
        "physical_run_name": a.name,
        "run_kind": a.run_kind,
        "model": "Step4F1IRGateModel (RGB anchor + q*zero-init IR residual)",
        "aux_mode": spec["aux_mode"],
        "gate_mode": spec["gate_mode"],
        "gate_module": model.gate_module_kind,
        "gate_module_kind_from_model": model.gate_module_kind,
        "dataset_group": spec["dataset"],
        "corruption_schedule": {
            "kind_probs": dict(KIND_PROBS),
            "severities": [0.25, 0.50, 0.75, 1.00],
            "zero_severity": 1.0,
            "shift": "evaluation-only",
            "randomness": "SHA256(seed|epoch|sample_id|field); noise field includes epoch",
        },
        "corruption_source_sha256": _sha_file(
            ROOT / "src" / "multimodal" / "step4_f1_b_corruption.py"
        ),
        "reliability_gate_source_sha256": _sha_file(
            ROOT / "src" / "multimodal" / "reliability_gate.py"
        ),
        "pretrain_audit_sha256": _sha_file(audit_path),
        "smoke_readiness_sha256": (
            _sha_file(readiness_path) if a.run_kind == "formal" else None
        ),
        "f1_v4_summary_sha256": _sha_file(
            ROOT / "runs" / "step4_f1_ir_gate" / "_summary_step4_f1.json"
        ),
        "design_freeze_sha256": _sha_file(
            ROOT / "docs" / "step4_f1_c" / "DESIGN_FREEZE.md"
        ),
        "rgb_policy": "frozen and unscaled; BN eval",
        "recipe": "R3-causal-earlyfusion-sample",
        "expected_epochs": a.epochs,
        "requested_batch": a.batch,
        "seed": a.seed,
        "model_init_seed": MODEL_INIT_SEED,
        "contract_sha256": _sha_file(contract_path),
        # v2 external runtime dependency closure (reviewer 2026-08-17 P0)
        "base_checkpoint_sha256": ckpt_state["sha256"],
        "builder_source_sha256": _sha_file(
            ROOT / "src" / "multimodal" / "early_fusion_yolo26.py"
        ),
        "data_yaml_sha256": data_yaml_state["sha256"],
        "data_yaml_names_sha256": data_yaml_state["names_sha256"],
        "data_yaml_n_classes": data_yaml_state["n_classes"],
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
        "quality_mask_source_sha256": _sha_file(
            ROOT / "src" / "multimodal" / "modality_quality.py"
        ),
        **initial_shas,
        "g8_evidence": "actual_dataloader_yield_v1",
        "g9_evidence": "actual_corruption_yield_v1",
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
    (run_dir / "step4_f1c_g9_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in g9_trace) + "\n", encoding="utf-8"
    )
    (run_dir / "step4_f1c_g9_records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in g9_records) + "\n", encoding="utf-8"
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
        gate_module=spec.get("gate_module"),
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
    decay_threshold = 1e-3 * (a.epochs / 80.0)
    if a.group == "F1C-C0":
        passed = bool(
            gate["rgb_backbone_unchanged"]
            and gate["aux_encoder_global_rel_l2"] < decay_threshold
            and max(proj_norms) == 0.0
            and gate["q_finite_and_bounded"]
        )
        gate["expected"] = (f"null aux stays below decay threshold "
                            f"(<{decay_threshold:.3g} scaled to {a.epochs}ep); "
                            "proj weights stay zero")
    elif a.group == "F1C-I-fixed":
        passed = bool(
            gate["rgb_backbone_unchanged"]
            and gate["aux_encoder_global_rel_l2"] > decay_threshold
            and min(proj_norms) > 0.0
            and gate["q_finite_and_bounded"]
            and last_q["min"] == 1.0 and last_q["max"] == 1.0
        )
        gate["expected"] = (f"active IR/projections learn beyond decay scale "
                            f"(>{decay_threshold:.3g} scaled to {a.epochs}ep); "
                            "effective q remains exactly one")
    else:
        passed = bool(
            gate["rgb_backbone_unchanged"]
            and gate["aux_encoder_global_rel_l2"] > decay_threshold
            and min(proj_norms) > 0.0
            and gate["gate_max_abs_change"] > 0.0
            and gate["q_finite_and_bounded"]
        )
        gate["expected"] = (f"active IR/projections/gate learn beyond decay scale "
                            f"(>{decay_threshold:.3g} scaled to {a.epochs}ep); "
                            "q remains finite in [0,1]")
    gate["decay_threshold_scaled_to_epochs"] = decay_threshold
    gate["passed"] = passed
    (run_dir / "step4_update_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(f"G6_REAL_UPDATE_GATE_FAIL: {gate}")
    print(f"[{a.group}] G6 PASS -> {run_dir}")


if __name__ == "__main__":
    main()
