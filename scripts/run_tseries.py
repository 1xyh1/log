#!/usr/bin/env python3
"""Train one T-series arm with the frozen Step4 recipe."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import modality_preprocess as mp  # noqa: E402
from multimodal.early_fusion_yolo26 import r3_hyp  # noqa: E402
from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402
from multimodal.trainability import freeze_module  # noqa: E402
from multimodal.tseries_core import (  # noqa: E402
    A4_ORIGINAL_FEEDBACK_SHA256, A5_ACCEPTED_COMMIT,
    A5_SUMMARY_CANONICAL_LF_SHA256, A5_SUMMARY_RAW_SHA256,
    FORMAL_BATCH, FORMAL_EPOCHS, FORMAL_SEED, RUN_NAMES, TREATMENTS,
    grad_abs_max, grad_norm, optimizer_group_snapshot, parameter_sha256,
    sha256_file, sha256_json, state_sha256, tensor_sha256,
)
from multimodal.tseries_runtime import (  # noqa: E402
    R3_KW, build_tseries_model, initial_identity, protocol_hash, verify_external_inputs,
)
from ultralytics.data.build import InfiniteDataLoader  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402

AUDIT_SCHEMA = "step4-tseries-pretraining-audit-v1"

class TSeriesValidator(DetectionValidator):
    def preprocess(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        return batch

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def _verify_formal_audit(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"T_SERIES_FORMAL_AUDIT_MISSING:{path}")
    obj = _read_json(path)
    if obj.get("schema") != AUDIT_SCHEMA or obj.get("all_passed") is not True:
        raise RuntimeError("T_SERIES_FORMAL_AUDIT_NOT_PASSING")
    if obj.get("phase") != "formal":
        raise RuntimeError("T_SERIES_FORMAL_AUDIT_WRONG_PHASE")
    gates = obj.get("gates") or {}
    expected_gates = {f"G{i}" for i in range(1, 19)}
    if set(gates) != expected_gates or not all(bool(v) for v in gates.values()):
        raise RuntimeError(f"T_SERIES_FORMAL_GATES_NOT_ALL_PASS:{gates}")
    pins = obj.get("source_hashes") or {}
    current = {
        "design_sha256": sha256_file(ROOT / "docs/step4_tseries/TRAINING_DESIGN_FREEZE.md"),
        "model_sha256": sha256_file(ROOT / "src/multimodal/tseries_p5_model.py"),
        "core_sha256": sha256_file(ROOT / "src/multimodal/tseries_core.py"),
        "runtime_sha256": sha256_file(ROOT / "src/multimodal/tseries_runtime.py"),
        "runner_sha256": sha256_file(ROOT / "scripts/run_tseries.py"),
        "suite_sha256": sha256_file(ROOT / "scripts/smoke_tseries_suite.py"),
        "audit_sha256": sha256_file(ROOT / "scripts/audit_tseries.py"),
        "posttrain_eval_sha256": sha256_file(ROOT / "scripts/eval_tseries_posttrain.py"),
        "paired_eval_sha256": sha256_file(ROOT / "scripts/eval_tseries_paired.py"),
        "formal_suite_sha256": sha256_file(ROOT / "scripts/run_tseries_formal_suite.py"),
        "summary_sha256": sha256_file(ROOT / "scripts/summarize_tseries.py"),
        "implementation_adjudication_sha256": sha256_file(
            ROOT / "docs/step4_tseries/IMPLEMENTATION_ADJUDICATION.md"
        ),
        "tests_sha256": sha256_file(ROOT / "tests/test_tseries.py"),
        "readme_sha256": sha256_file(ROOT / "T_SERIES_README.md"),
    }
    stale = {k: {"recorded": pins.get(k), "current": v} for k, v in current.items()
             if pins.get(k) != v}
    if stale:
        raise RuntimeError(f"T_SERIES_FORMAL_AUDIT_STALE:{stale}")
    return obj

def _scalar_loss(loss):
    if isinstance(loss, torch.Tensor):
        return loss.sum()
    if isinstance(loss, (list, tuple)):
        vals = [x.sum() for x in loss if torch.is_tensor(x)]
        if not vals:
            raise RuntimeError("T_SERIES_GRAD_PROBE_NO_TENSOR_LOSS")
        return sum(vals)
    raise RuntimeError(f"T_SERIES_GRAD_PROBE_BAD_LOSS:{type(loss).__name__}")

def _module_grad_norm(module) -> float | None:
    vals = [p.grad.detach().float().pow(2).sum() for p in module.parameters() if p.grad is not None]
    if not vals:
        return None
    return float(torch.sqrt(sum(vals)).item())

def gradient_probe(model, contract: dict) -> dict:
    """FP32 copy-only backward. Never mutates the formal/smoke training model."""
    probe = deepcopy(model).cpu().float()
    probe.args = r3_hyp()
    probe.criterion = None
    probe.train()
    ds = TriModalDataset(contract, split="train", group="C1-I", imgsz=640, augment=False)
    batch = TriModalDataset.collate_fn([ds[0]])
    batch["img"] = batch["img"].float()
    probe.zero_grad(set_to_none=True)
    preds = probe._predict_once(batch["img"])
    loss = _scalar_loss(probe.loss(batch, preds))
    loss.backward()
    pw = probe.p5_fusion.proj.weight
    pb = probe.p5_fusion.proj.bias
    row = {
        "loss": float(loss.detach().cpu()),
        "aux_encoder_grad_norm": _module_grad_norm(probe.aux_encoder),
        "p5_proj_weight_grad_norm": grad_norm(pw),
        "p5_proj_weight_grad_abs_max": grad_abs_max(pw),
        "p5_proj_bias_grad_norm": grad_norm(pb),
        "p5_proj_bias_grad_abs_max": grad_abs_max(pb),
        "tail_grad_norm": _module_grad_norm(probe.tail),
        "t0_aux_proj_disconnected": bool(
            model.treatment_id != "T0-N"
            or (
                _module_grad_norm(probe.aux_encoder) is None
                and pw.grad is None
                and pb.grad is None
            )
        ),
        "t2_bias_raw_grad_diagnostic_only": bool(model.treatment_id == "T2-A"),
        "t2_bias_numerical_guard_required": bool(model.treatment_id == "T2-A"),
        "fp32": True,
    }
    return row

def prediction_probe(model, contract: dict) -> dict:
    from multimodal import step3_eval_utils as evu
    ds = TriModalDataset(contract, split="val", group="C1-I", imgsz=640, augment=False)
    sample = ds[0]
    sid = str(sample["sample_id"])
    batch = ds.collate_fn([sample])
    img = batch["img"].float()
    model.eval()
    with torch.no_grad():
        output = model._predict_once(img)
        raw = evu.extract_detection_tensor(output).detach()
    trace = model.last_forward_trace
    return {
        "sample_id": sid,
        "raw_prediction_sha256": tensor_sha256(raw),
        "forward_trace": trace,
        "zero_init": bool(model.assert_zero_init()),
        "no_reliability_gate_attribute": not hasattr(model, "reliability_gate"),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--treatment", choices=sorted(TREATMENTS), required=True)
    p.add_argument("--run-kind", choices=["smoke", "formal", "recovery"], default="formal")
    p.add_argument("--project", default="runs/step4_tseries")
    p.add_argument("--name", default=None)
    p.add_argument("--contract", default=OUT_DEFAULT)
    p.add_argument("--data", default="D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml")
    p.add_argument("--base-checkpoint", default="E:/odin/yolo26s.pt")
    p.add_argument("--formal-audit", default="reports/step4_tseries/pretraining_audit.json")
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=FORMAL_SEED)
    p.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    p.add_argument("--batch", type=int, default=FORMAL_BATCH)
    a = p.parse_args()

    if a.run_kind == "formal":
        drift = {}
        if a.seed != FORMAL_SEED: drift["seed"] = a.seed
        if a.epochs != FORMAL_EPOCHS: drift["epochs"] = a.epochs
        if a.batch != FORMAL_BATCH: drift["batch"] = a.batch
        if drift:
            raise RuntimeError(f"T_SERIES_FORMAL_PROTOCOL_DRIFT:{drift}")
        _verify_formal_audit(ROOT / a.formal_audit)

    project = Path(a.project).resolve()
    if a.name is None:
        if a.run_kind == "formal":
            a.name = RUN_NAMES[a.treatment]
        else:
            base = f"smoke-{RUN_NAMES[a.treatment]}-e{a.epochs}"
            a.name = base
            rev = 2
            while (project / a.name).exists():
                a.name = f"{base}-r{rev}"
                rev += 1
    run_dir = project / a.name
    if run_dir.exists():
        raise RuntimeError(f"T_SERIES_RUN_DIR_EXISTS:{run_dir}")

    contract_path = Path(a.contract)
    contract = _read_json(contract_path)
    external = verify_external_inputs(contract, Path(a.data), Path(a.base_checkpoint))

    model = build_tseries_model(Path(a.base_checkpoint), a.treatment)
    init = initial_identity(model)
    grad_probe = gradient_probe(model, contract)
    pred_probe = prediction_probe(model, contract)
    if not pred_probe["zero_init"]:
        raise RuntimeError("T_SERIES_ZERO_INIT_FAIL")
    if not pred_probe["no_reliability_gate_attribute"]:
        raise RuntimeError("T_SERIES_GATE_PRESENT")
    if a.treatment == "T0-N" and not grad_probe["t0_aux_proj_disconnected"]:
        raise RuntimeError("T_SERIES_T0_LOSS_GRAPH_FAIL")
    # Raw FP32 T2 bias-gradient dust is diagnostic only. The formal guarantee is
    # enforced at the real optimizer step and verified by the one-epoch smoke.

    requested_batch = int(a.batch)
    trace_rows = []
    mechanism_rows = []
    optimizer_evidence = {}

    class TSeriesTrainer(DetectionTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ts_contract = contract
            self.ts_requested_batch = requested_batch
            self._epoch_dataset = None
            self._actual_ids = []
            self._actual_flips = []
            self._bias_epoch_start = None
            self._t2_bias_pre_zero_abs_max = 0.0
            self._t2_bias_zero_guard_calls = 0
            self._p5_bias_grad_abs_max = 0.0
            self._p5_weight_grad_norm_max = 0.0

        def build_dataset(self, img_path, mode="train", batch=None):
            split = "train" if mode == "train" else "val"
            return TriModalDataset(
                self.ts_contract,
                split=split,
                group="C1-I",
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
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        def preprocess_batch(self, batch):
            if int(self.args.batch) != self.ts_requested_batch:
                raise RuntimeError("T_SERIES_RESOURCE_PROFILE_CHANGED")
            ids = batch.get("sample_id")
            flips = batch.get("flip_applied")
            if ids is not None:
                self._actual_ids.extend(str(x) for x in ids)
            if flips is not None:
                self._actual_flips.extend(bool(x) for x in flips)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
            batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
            return batch

        def _build_train_pipeline(self):
            freeze_module(self.model.rgb_backbone, freeze_bn_stats=True)
            super()._build_train_pipeline()
            if any(p.requires_grad for p in self.model.rgb_backbone.parameters()):
                raise RuntimeError("T_SERIES_RGB_BACKBONE_NOT_FROZEN")
            opt_ids = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
            missing = []
            for label, module in (
                ("aux_encoder", self.model.aux_encoder),
                ("p5_fusion", self.model.p5_fusion),
                ("tail", self.model.tail),
            ):
                for p0 in module.parameters():
                    if not p0.requires_grad or id(p0) not in opt_ids:
                        missing.append(label)
            if missing:
                raise RuntimeError(f"T_SERIES_OPTIMIZER_MEMBERSHIP_FAIL:{sorted(set(missing))}")
            optimizer_evidence.update(optimizer_group_snapshot(self.model, self.optimizer))
            optimizer_evidence["proj_bias_name"] = next(
                n for n, p0 in self.model.named_parameters() if p0 is self.model.p5_fusion.proj.bias
            )
            bias_group = optimizer_evidence["assignment"][optimizer_evidence["proj_bias_name"]]
            optimizer_evidence["proj_bias_group_index"] = int(bias_group)
            optimizer_evidence["proj_bias_weight_decay"] = float(
                self.optimizer.param_groups[bias_group].get("weight_decay", 0.0)
            )

        def optimizer_step(self):
            weight = self.model.p5_fusion.proj.weight
            bias = self.model.p5_fusion.proj.bias
            if weight.grad is not None:
                self._p5_weight_grad_norm_max = max(
                    self._p5_weight_grad_norm_max,
                    float(weight.grad.detach().float().norm().item()),
                )
            if bias.grad is not None:
                raw_bias = float(bias.grad.detach().abs().max().item())
                self._p5_bias_grad_abs_max = max(self._p5_bias_grad_abs_max, raw_bias)
            # P0 numerical-exactness guard. Forward remains post-projection AC_ALL;
            # only the theoretically-zero T2 projection-bias gradient is made exact
            # before MuSGD sees it.
            if a.treatment == "T2-A" and bias.grad is not None:
                self._t2_bias_pre_zero_abs_max = max(
                    self._t2_bias_pre_zero_abs_max,
                    float(bias.grad.detach().abs().max().item()),
                )
                bias.grad.zero_()
                self._t2_bias_zero_guard_calls += 1
            return super().optimizer_step()

        def get_validator(self):
            return TSeriesValidator(self.test_loader, save_dir=self.save_dir, args=self.args)

        def final_eval(self):
            print(f"[{a.treatment}] final_eval skipped; use eval_tseries_posttrain.py")

    kw = dict(R3_KW)
    kw.update(
        epochs=a.epochs,
        batch=a.batch,
        seed=a.seed,
        project=str(project),
        name=a.name,
        device=a.device,
        data=str(a.data),
        exist_ok=True,
        model="yolo26s.yaml",
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema": "step4-tseries-run-manifest-v1",
        "treatment_id": a.treatment,
        "treatment_mode": TREATMENTS[a.treatment],
        "physical_run_name": a.name,
        "run_kind": a.run_kind,
        "model_class": "TSeriesP5Model",
        "topology": "P5-only direct IR injection; no P3/P4 direct IR; no gate",
        "seed": a.seed,
        "expected_epochs": a.epochs,
        "requested_batch": a.batch,
        "contract_sha256": sha256_file(contract_path),
        "requested_base_checkpoint": str(Path(a.base_checkpoint)),
        "requested_base_checkpoint_sha256": external["base_checkpoint"]["sha256"],
        "consumed_base_checkpoint": str(Path(a.base_checkpoint)),
        "consumed_base_checkpoint_sha256": external["base_checkpoint"]["sha256"],
        "initial_identity": init,
        "gradient_probe": grad_probe,
        "prediction_probe": pred_probe,
        "aux_bn_stats_policy": "train-mode buffers for all arms; T0 aux forward is no_grad",
        "rgb_bn_stats_policy": "frozen eval",
        "tail_bn_stats_policy": "stock trainer",
        "t2_bias_numerical_guard": (
            "zero proj.bias.grad immediately before optimizer_step"
            if a.treatment == "T2-A" else "not_applicable"
        ),
        "protocol_hash": protocol_hash(kw),
        "upstream_provenance": {
            "a5_accepted_commit": A5_ACCEPTED_COMMIT,
            "a5_summary_raw_sha256": A5_SUMMARY_RAW_SHA256,
            "a5_summary_canonical_lf_sha256": A5_SUMMARY_CANONICAL_LF_SHA256,
            "a4_original_feedback_sha256": A4_ORIGINAL_FEEDBACK_SHA256,
            "a4_erratum_sha256": sha256_file(
                ROOT / "docs/step4_a4/feedback/2026-08-19_erratum.md"
            ),
            "formal_audit_sha256": (
                sha256_file(ROOT / a.formal_audit) if a.run_kind == "formal" else None
            ),
        },
        "source_hashes": {
            "model": sha256_file(ROOT / "src/multimodal/tseries_p5_model.py"),
            "core": sha256_file(ROOT / "src/multimodal/tseries_core.py"),
            "runtime": sha256_file(ROOT / "src/multimodal/tseries_runtime.py"),
            "runner": sha256_file(ROOT / "scripts/run_tseries.py"),
            "design": sha256_file(ROOT / "docs/step4_tseries/TRAINING_DESIGN_FREEZE.md"),
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    trainer = TSeriesTrainer(overrides=kw)
    if Path(trainer.save_dir).resolve() != run_dir:
        raise RuntimeError(
            f"T_SERIES_SAVE_DIR_MISMATCH:{Path(trainer.save_dir).resolve()}!={run_dir}"
        )
    trainer.model = model
    trainer.model.args = trainer.args

    def on_epoch_start(tr):
        ds = tr._epoch_dataset
        if ds is None:
            raise RuntimeError("T_SERIES_NO_EPOCH_DATASET")
        ds.set_epoch(tr.epoch)
        if hasattr(tr.train_loader, "reset"):
            tr.train_loader.reset()
        tr._actual_ids = []
        tr._actual_flips = []
        tr.model.reset_mechanism_stats()
        tr._bias_epoch_start = tr.model.p5_fusion.proj.bias.detach().cpu().clone()
        tr._t2_bias_pre_zero_abs_max = 0.0
        tr._t2_bias_zero_guard_calls = 0
        tr._p5_bias_grad_abs_max = 0.0
        tr._p5_weight_grad_norm_max = 0.0

    def on_epoch_end(tr):
        ds = tr._epoch_dataset
        expected_ids = [ds.ids[i] for i in ds.sampler.perm]
        expected_flips = [mp.should_flip(ds.seed, tr.epoch, sid, ds.fliplr) for sid in expected_ids]
        if tr._actual_ids != expected_ids:
            raise RuntimeError(f"T_SERIES_ACTUAL_ORDER_MISMATCH:epoch={tr.epoch}")
        if tr._actual_flips != expected_flips:
            raise RuntimeError(f"T_SERIES_ACTUAL_FLIP_MISMATCH:epoch={tr.epoch}")
        trace_rows.append({
            "epoch": int(tr.epoch),
            "sample_order_sha256": sha256_json(tr._actual_ids),
            "flip_schedule_sha256": sha256_json(tr._actual_flips),
            "n_samples": len(tr._actual_ids),
            "actual_matches_expected": True,
        })
        bias_now = tr.model.p5_fusion.proj.bias.detach().cpu()
        bias_delta = float((bias_now - tr._bias_epoch_start).abs().max().item())
        row = {
            "epoch": int(tr.epoch) + 1,
            **tr.model.mechanism_stats(),
            "p5_proj_weight_norm": float(tr.model.p5_fusion.proj.weight.detach().float().norm().item()),
            "p5_proj_bias_norm": float(tr.model.p5_fusion.proj.bias.detach().float().norm().item()),
            "p5_proj_bias_epoch_max_abs_delta": bias_delta,
            "p5_proj_weight_grad_norm_max": float(getattr(tr, "_p5_weight_grad_norm_max", 0.0)),
            "p5_proj_bias_raw_grad_abs_max": float(getattr(tr, "_p5_bias_grad_abs_max", 0.0)),
            "t2_bias_pre_zero_grad_abs_max": float(getattr(tr, "_t2_bias_pre_zero_abs_max", 0.0)),
            "t2_bias_zero_guard_calls": int(getattr(tr, "_t2_bias_zero_guard_calls", 0)),
        }
        mechanism_rows.append(row)

    trainer.add_callback("on_train_epoch_start", on_epoch_start)
    trainer.add_callback("on_train_epoch_end", on_epoch_end)
    trainer.train()

    trained = trainer.model
    final = {
        "aux_encoder_state_sha256": state_sha256(trained.aux_encoder),
        "aux_encoder_param_sha256": parameter_sha256(trained.aux_encoder),
        "p5_fusion_state_sha256": state_sha256(trained.p5_fusion),
        "p5_fusion_param_sha256": parameter_sha256(trained.p5_fusion),
        "p5_bias_sha256": tensor_sha256(trained.p5_fusion.proj.bias),
        "tail_state_sha256": state_sha256(trained.tail),
        "complete_model_state_sha256": state_sha256(trained),
        "p5_bias_max_abs": float(trained.p5_fusion.proj.bias.detach().abs().max().item()),
    }
    manifest["optimizer"] = optimizer_evidence
    manifest["final_identity"] = final
    manifest["completed_epochs"] = len(mechanism_rows)
    manifest["t0_aux_params_unchanged"] = (
        a.treatment != "T0-N"
        or init["aux_encoder_param_sha256"] == final["aux_encoder_param_sha256"]
    )
    manifest["t0_p5_params_unchanged"] = (
        a.treatment != "T0-N"
        or init["p5_fusion_param_sha256"] == final["p5_fusion_param_sha256"]
    )
    manifest["t2_bias_unchanged"] = (
        a.treatment != "T2-A"
        or init["p5_bias_sha256"] == final["p5_bias_sha256"]
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "tseries_data_order.jsonl").write_text(
        "\n".join(json.dumps(x) for x in trace_rows) + "\n", encoding="utf-8"
    )
    (run_dir / "tseries_mechanism.jsonl").write_text(
        "\n".join(json.dumps(x) for x in mechanism_rows) + "\n", encoding="utf-8"
    )
    (run_dir / "optimizer_manifest.json").write_text(
        json.dumps(optimizer_evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if a.run_kind == "smoke":
        if a.treatment == "T0-N" and not (
            manifest["t0_aux_params_unchanged"] and manifest["t0_p5_params_unchanged"]
        ):
            raise RuntimeError("T_SERIES_T0_SILENT_OPTIMIZER_UPDATE")
        if a.treatment == "T2-A" and not manifest["t2_bias_unchanged"]:
            raise RuntimeError("T_SERIES_T2_BIAS_OPTIMIZER_UPDATE")

    print(json.dumps({
        "status": "COMPLETE",
        "treatment": a.treatment,
        "run_dir": str(run_dir),
        "run_kind": a.run_kind,
        "completed_epochs": len(mechanism_rows),
    }, indent=2))

if __name__ == "__main__":
    main()
