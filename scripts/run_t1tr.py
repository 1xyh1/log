#!/usr/bin/env python3
"""Train the single new T1-TR arm U2-S with balanced fully-wrong IR sources."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal import modality_preprocess as mp  # noqa: E402
from multimodal.raw_sample_index import OUT_DEFAULT  # noqa: E402
from multimodal.trimodal_dataset import TriModalDataset  # noqa: E402
from multimodal.trainability import freeze_module  # noqa: E402
from multimodal.tseries_core import (  # noqa: E402
    optimizer_group_snapshot, sha256_file, sha256_json,
)
from multimodal.tseries_runtime import (  # noqa: E402
    R3_KW, build_tseries_model, initial_identity, verify_external_inputs,
)
from multimodal.t1tr_training_source import (  # noqa: E402
    FORMAL_BATCH, FORMAL_EPOCHS, FORMAL_SEED, U2_RUN_NAME,
    T1_MANIFEST_SHA256, balanced_derangement_map, schedule_balance,
    verify_epoch_mapping,
)
from ultralytics.data.build import InfiniteDataLoader  # noqa: E402
from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.models.yolo.detect.val import DetectionValidator  # noqa: E402

T1_RUN = ROOT / "runs/step4_tseries/T1-F_P5_FULL_seed20260812"
T1_MANIFEST = T1_RUN / "manifest.json"
T1_OPTIMIZER = T1_RUN / "optimizer_manifest.json"

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def fail(code: str, detail=None):
    raise RuntimeError(code if detail is None else f"{code}:{detail}")

def normalize_optimizer_snapshot(obj: dict) -> dict:
    return {
        "groups": obj["groups"],
        "assignment": obj["assignment"],
        "all_names": obj["all_names"],
    }

class T1TRValidator(DetectionValidator):
    def preprocess(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        return batch

def verify_formal_audit(path: Path) -> dict:
    obj = load_json(path)
    if (
        obj.get("schema") != "step4-t1tr-pretraining-audit-v1"
        or obj.get("phase") != "formal"
        or obj.get("all_passed") is not True
    ):
        fail("T1TR_FORMAL_AUDIT_NOT_PASSING")
    gates = obj.get("gates") or {}
    if set(gates) != {f"G{i}" for i in range(1, 19)} or not all(gates.values()):
        fail("T1TR_FORMAL_GATES_NOT_ALL_PASS", gates)

    current = {
        "design_sha256": sha256_file(ROOT / "docs/step4_t1tr/DESIGN_FREEZE.md"),
        "core_sha256": sha256_file(ROOT / "src/multimodal/t1tr_training_source.py"),
        "runner_sha256": sha256_file(ROOT / "scripts/run_t1tr.py"),
        "smoke_sha256": sha256_file(ROOT / "scripts/smoke_t1tr.py"),
        "audit_sha256": sha256_file(ROOT / "scripts/audit_t1tr.py"),
        "eval_sha256": sha256_file(ROOT / "scripts/eval_t1tr.py"),
        "summary_sha256": sha256_file(ROOT / "scripts/summarize_t1tr.py"),
        "verify_sha256": sha256_file(ROOT / "scripts/verify_t1tr_run.py"),
        "tests_sha256": sha256_file(ROOT / "tests/test_t1tr.py"),
        "readme_sha256": sha256_file(ROOT / "T1TR_README.md"),
    }
    pins = obj.get("source_hashes") or {}
    stale = {
        k: {"recorded": pins.get(k), "current": v}
        for k, v in current.items() if pins.get(k) != v
    }
    if stale:
        fail("T1TR_FORMAL_AUDIT_STALE", stale)
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-kind", choices=["smoke", "formal"], required=True)
    ap.add_argument("--project", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--contract", default=OUT_DEFAULT)
    ap.add_argument("--data", default="D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml")
    ap.add_argument("--base-checkpoint", default="E:/odin/yolo26s.pt")
    ap.add_argument("--formal-audit", default="reports/step4_t1tr/pretraining_audit.json")
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=FORMAL_SEED)
    ap.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    ap.add_argument("--batch", type=int, default=FORMAL_BATCH)
    a = ap.parse_args()

    if not T1_MANIFEST.is_file() or not T1_OPTIMIZER.is_file():
        fail("T1TR_FROZEN_T1_CONTROL_MISSING")
    if sha256_file(T1_MANIFEST) != T1_MANIFEST_SHA256:
        fail("T1TR_T1_MANIFEST_SHA_DRIFT")
    t1_manifest = load_json(T1_MANIFEST)
    t1_optimizer = load_json(T1_OPTIMIZER)

    if a.run_kind == "formal":
        if (a.seed, a.epochs, a.batch) != (FORMAL_SEED, FORMAL_EPOCHS, FORMAL_BATCH):
            fail("T1TR_FORMAL_PROTOCOL_DRIFT", {
                "seed": a.seed, "epochs": a.epochs, "batch": a.batch,
            })
        verify_formal_audit(ROOT / a.formal_audit)

    if a.project is None:
        a.project = "runs/step4_t1tr" if a.run_kind == "formal" else "runs/step4_t1tr_smoke"
    project = (ROOT / a.project).resolve()

    if a.name is None:
        if a.run_kind == "formal":
            a.name = U2_RUN_NAME
        else:
            base = "smoke-U2-S_P5_FULL_BALANCED_SHUFFLED_seed20260812-e1"
            a.name = base
            n = 2
            while (project / a.name).exists():
                a.name = f"{base}-r{n}"
                n += 1

    run_dir = project / a.name
    if run_dir.exists():
        fail("T1TR_RUN_DIR_EXISTS", str(run_dir))

    contract_path = Path(a.contract)
    contract = load_json(contract_path)
    external = verify_external_inputs(contract, Path(a.data), Path(a.base_checkpoint))
    train_ids = [str(x) for x in contract["train_ids"]]
    val_ids = [str(x) for x in contract["val_ids"]]
    if len(train_ids) != 11 or len(set(train_ids)) != 11 or len(val_ids) != 6:
        fail("T1TR_CONTRACT_SPLIT_SIZE_DRIFT")
    if sha256_file(contract_path) != t1_manifest["contract_sha256"]:
        fail("T1TR_CONTRACT_SHA_DRIFT")

    balance = schedule_balance(train_ids, FORMAL_EPOCHS)
    if not balance["passed"] or balance["expected_each_nonself_pair"] != 8:
        fail("T1TR_SCHEDULE_BALANCE_FAIL", balance)

    model = build_tseries_model(Path(a.base_checkpoint), "T1-F")
    init = initial_identity(model)
    if init != t1_manifest["initial_identity"]:
        # Report only top-level differing fields; never relax.
        diff = {
            k: {"u2": init.get(k), "t1": t1_manifest["initial_identity"].get(k)}
            for k in set(init) | set(t1_manifest["initial_identity"])
            if init.get(k) != t1_manifest["initial_identity"].get(k)
        }
        fail("T1TR_INITIAL_IDENTITY_FAIL", diff)

    trace_rows = []
    mechanism_rows = []
    optimizer_evidence = {}

    class T1TRTrainer(DetectionTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ts_contract = contract
            self._epoch_dataset = None
            self._actual_ids = []
            self._actual_aux_ids = []
            self._actual_flips = []

        def build_dataset(self, img_path, mode="train", batch=None):
            split = "train" if mode == "train" else "val"
            aux_map = (
                balanced_derangement_map(train_ids, 0)
                if mode == "train" else None
            )
            return TriModalDataset(
                self.ts_contract,
                split=split,
                group="C1-I",
                imgsz=self.args.imgsz,
                seed=self.args.seed,
                fliplr=self.args.fliplr,
                augment=(mode == "train"),
                aux_id_map=aux_map,
            )

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "train":
                ds = self.build_dataset(dataset_path, mode, batch_size)
                self._epoch_dataset = ds
                return InfiniteDataLoader(
                    ds,
                    batch_size=batch_size,
                    sampler=ds.sampler,
                    shuffle=False,
                    num_workers=self.args.workers,
                    collate_fn=ds.collate_fn,
                    pin_memory=True,
                    drop_last=False,
                )
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        def preprocess_batch(self, batch):
            ids = batch.get("sample_id")
            aux = batch.get("aux_sample_id")
            flips = batch.get("flip_applied")
            if ids is None or aux is None or flips is None:
                fail("T1TR_BATCH_METADATA_MISSING")
            ids = [str(x) for x in ids]
            aux = [str(x) for x in aux]
            if any(r == d for r, d in zip(ids, aux)):
                fail("T1TR_RUNTIME_SELF_DONOR", list(zip(ids, aux)))
            self._actual_ids.extend(ids)
            self._actual_aux_ids.extend(aux)
            self._actual_flips.extend(bool(x) for x in flips)

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
            batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
            return batch

        def _build_train_pipeline(self):
            freeze_module(self.model.rgb_backbone, freeze_bn_stats=True)
            super()._build_train_pipeline()
            snap = optimizer_group_snapshot(self.model, self.optimizer)
            expected = normalize_optimizer_snapshot(t1_optimizer)
            actual = normalize_optimizer_snapshot(snap)
            if actual != expected:
                fail("T1TR_OPTIMIZER_MISMATCH")
            optimizer_evidence.update(snap)

        def get_validator(self):
            return T1TRValidator(self.test_loader, save_dir=self.save_dir, args=self.args)

        def final_eval(self):
            print("[U2-S] final_eval skipped; use eval_t1tr.py")

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
        "schema": "step4-t1tr-u2-run-manifest-v1",
        "experiment": "T1-TR",
        "arm": "U2-S",
        "run_kind": a.run_kind,
        "physical_run_name": a.name,
        "model_class": "TSeriesP5Model",
        "model_treatment_id": "T1-F",
        "training_source_mode": "BALANCED_FULLY_WRONG_IR",
        "schedule_formula": "shift=1+(epoch mod 10); donor[i]=train_ids[(i+shift) mod 11]",
        "seed": a.seed,
        "expected_epochs": a.epochs,
        "requested_batch": a.batch,
        "contract_sha256": sha256_file(contract_path),
        "base_checkpoint_sha256": external["base_checkpoint"]["sha256"],
        "initial_identity": init,
        "initial_identity_exact_t1": True,
        "schedule_balance_80ep": balance,
        "primary_inference_for_adjudication": "ZERO",
        "frozen_T1_manifest_sha256": T1_MANIFEST_SHA256,
        "frozen_T1_optimizer_manifest_sha256": sha256_file(T1_OPTIMIZER),
        "formal_audit_sha256": (
            sha256_file(ROOT / a.formal_audit) if a.run_kind == "formal" else None
        ),
        "native_validation_curve": "descriptive_only",
        "source_hashes": {
            "design": sha256_file(ROOT / "docs/step4_t1tr/DESIGN_FREEZE.md"),
            "core": sha256_file(ROOT / "src/multimodal/t1tr_training_source.py"),
            "runner": sha256_file(ROOT / "scripts/run_t1tr.py"),
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    trainer = T1TRTrainer(overrides=kw)
    if Path(trainer.save_dir).resolve() != run_dir:
        fail("T1TR_SAVE_DIR_MISMATCH")
    trainer.model = model
    trainer.model.args = trainer.args

    def on_epoch_start(tr):
        ds = tr._epoch_dataset
        if ds is None:
            fail("T1TR_NO_EPOCH_DATASET")
        mapping = balanced_derangement_map(train_ids, tr.epoch)
        chk = verify_epoch_mapping(train_ids, tr.epoch, mapping)
        if not chk["passed"]:
            fail("T1TR_EPOCH_MAPPING_FAIL", chk)
        ds.aux_id_map = mapping
        ds.set_epoch(tr.epoch)
        if hasattr(tr.train_loader, "reset"):
            tr.train_loader.reset()
        tr._actual_ids = []
        tr._actual_aux_ids = []
        tr._actual_flips = []
        tr.model.reset_mechanism_stats()

    def on_epoch_end(tr):
        ds = tr._epoch_dataset
        mapping = balanced_derangement_map(train_ids, tr.epoch)
        expected_ids = [ds.ids[i] for i in ds.sampler.perm]
        expected_aux = [mapping[sid] for sid in expected_ids]
        expected_flips = [
            mp.should_flip(ds.seed, tr.epoch, sid, ds.fliplr)
            for sid in expected_ids
        ]
        if tr._actual_ids != expected_ids:
            fail("T1TR_ACTUAL_RECIPIENT_ORDER_MISMATCH", tr.epoch)
        if tr._actual_aux_ids != expected_aux:
            fail("T1TR_ACTUAL_DONOR_SEQUENCE_MISMATCH", tr.epoch)
        if tr._actual_flips != expected_flips:
            fail("T1TR_ACTUAL_FLIP_MISMATCH", tr.epoch)

        row = {
            "epoch": int(tr.epoch),
            "shift": 1 + (int(tr.epoch) % 10),
            "recipient_order_sha256": sha256_json(tr._actual_ids),
            "aux_source_sequence_sha256": sha256_json(tr._actual_aux_ids),
            "flip_schedule_sha256": sha256_json(tr._actual_flips),
            "epoch_mapping_sha256": sha256_json(mapping),
            "n_samples": len(tr._actual_ids),
            "self_matches": sum(r == d for r, d in zip(tr._actual_ids, tr._actual_aux_ids)),
            "actual_mapping_matches_schedule": True,
        }
        trace_rows.append(row)
        mechanism_rows.append({
            "epoch": int(tr.epoch) + 1,
            **tr.model.mechanism_stats(),
            "p5_proj_weight_norm": float(
                tr.model.p5_fusion.proj.weight.detach().float().norm().item()
            ),
            "p5_proj_bias_norm": float(
                tr.model.p5_fusion.proj.bias.detach().float().norm().item()
            ),
        })

    trainer.add_callback("on_train_epoch_start", on_epoch_start)
    trainer.add_callback("on_train_epoch_end", on_epoch_end)
    trainer.train()

    manifest["completed_epochs"] = len(trace_rows)
    manifest["optimizer"] = optimizer_evidence
    manifest["optimizer_exact_t1"] = (
        normalize_optimizer_snapshot(optimizer_evidence)
        == normalize_optimizer_snapshot(t1_optimizer)
    )
    manifest["runtime_all_epochs_no_self"] = all(r["self_matches"] == 0 for r in trace_rows)
    manifest["runtime_schedule_exact"] = all(r["actual_mapping_matches_schedule"] for r in trace_rows)

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "optimizer_manifest.json").write_text(
        json.dumps(optimizer_evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "t1tr_source_schedule.jsonl").write_text(
        "\n".join(json.dumps(x) for x in trace_rows) + "\n", encoding="utf-8"
    )
    (run_dir / "t1tr_mechanism.jsonl").write_text(
        "\n".join(json.dumps(x) for x in mechanism_rows) + "\n", encoding="utf-8"
    )

    if len(trace_rows) != a.epochs:
        fail("T1TR_COMPLETED_EPOCH_COUNT_FAIL")
    if not manifest["optimizer_exact_t1"]:
        fail("T1TR_OPTIMIZER_POSTCHECK_FAIL")
    if not manifest["runtime_all_epochs_no_self"] or not manifest["runtime_schedule_exact"]:
        fail("T1TR_RUNTIME_SCHEDULE_POSTCHECK_FAIL")

    print(json.dumps({
        "status": "COMPLETE",
        "arm": "U2-S",
        "run_kind": a.run_kind,
        "run_dir": str(run_dir),
        "completed_epochs": len(trace_rows),
    }, indent=2))

if __name__ == "__main__":
    main()
