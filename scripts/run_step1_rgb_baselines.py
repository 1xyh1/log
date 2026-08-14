#!/usr/bin/env python3
"""Step 1-Sample Probe: YOLO26s RGB head/recipe probe (B0-A/B/C/D train + B0-E audit).

Standalone experiment controller for Ultralytics 8.4.56. Does NOT import the
multimodal package (src/); official Trainer only.

Groups:
    B0-A  E2E  R0-sample-resolved   (default.yaml resolved for 12-cls short training)
    B0-B  O2M  R0-sample-resolved
    B0-C  O2M  R1-sample-small      (R0 + lr0=0.001, mosaic=0.5)
    B0-D  O2M  R2-sample-ckpt-public(ckpt train_args via whitelist, public API only)
    B0-F  E2E  R2-sample-ckpt-public(Head x Recipe interaction check; = B0-D, end2end=True)
    B0-E       Audit (--audit-only) classified checkpoint keys vs R2 whitelist

CommonExperiment: epochs=80 imgsz=640 batch=4 nbs=4 warmup_epochs=0.0 workers=0
cache=ram seed=20260812 deterministic=True max_det=100 patience=100 close_mosaic=10

Fail fast: any semantic mismatch raises; no silent fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Frozen definitions
# ---------------------------------------------------------------------------

RELEASE_NAME = "v0.3.1"

RECIPE_R0 = {  # R0-sample-resolved: official default.yaml, auto resolved for nc=12 short train
    "optimizer": "AdamW",
    "lr0": 0.000625,  # 0.002*5/(4+12), auto's resolved lr
    "lrf": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "scale": 0.5,
    "translate": 0.1,
    "fliplr": 0.5,
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
}

RECIPE_R1_OVERRIDES = {"lr0": 0.001, "mosaic": 0.5}  # R1-sample-small

# R2-sample-ckpt-public: keys expressible via public 8.4.56 API
R2_WHITELIST = {
    "optimizer", "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
    "box", "cls", "dfl",
    "mosaic", "mixup", "copy_paste", "scale", "translate",
    "degrees", "shear", "perspective", "fliplr", "flipud",
    "hsv_h", "hsv_s", "hsv_v",
    "auto_augment", "erasing", "close_mosaic",
}

# Keys never taken from checkpoint: runtime (ignored) + common (overridden).
# warmup_epochs/nbs/patience/close_mosaic/max_det/... are COMMON_EXPERIMENT:
# overridden uniformly, so they must not leak from ckpt into R2.
COMMON_KEYS = {
    "epochs", "batch", "imgsz", "seed", "max_det", "nbs", "warmup_epochs",
    "patience", "close_mosaic", "deterministic", "cache",
}

# Fixed values actually executed by this probe: single source of truth for both
# training kwargs and the B0-E audit's effective-common bucket.
COMMON_PROBE = {"nbs": 4, "warmup_epochs": 0.0, "patience": 100, "close_mosaic": 10,
                "deterministic": True}


def common_kwargs(a: argparse.Namespace) -> dict:
    kw = dict(COMMON_PROBE)
    kw.update(epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers,
              cache=a.cache, seed=a.seed, max_det=a.max_det)
    return kw
RUNTIME_KEYS = {
    "model", "data", "mode", "task", "cfg", "session", "project", "name",
    "exist_ok", "device", "workers", "resume", "pretrained", "save_dir",
    "val", "plots", "save", "verbose", "amp", "rect", "cos_lr", "time",
    "fraction", "freeze", "multi_scale", "compile", "overlap_mask", "mask_ratio",
    "dropout", "save_period", "single_cls", "split", "save_json", "conf",
    "iou", "half", "dnn", "augment", "visualize", "agnostic_nms", "classes",
    "retina_masks", "embed", "show", "save_frames", "save_txt", "save_conf",
    "save_crop", "show_labels", "show_conf", "show_boxes", "line_width",
    "format", "keras", "optimize", "int8", "dynamic", "simplify", "opset",
    "workspace", "nms", "pose", "kobj", "bgr", "cutmix", "copy_paste_mode",
    "tracker", "source", "vid_stride", "stream_buffer",
}
UNSUPPORTED_INTERNAL = {"o2m", "topk", "detach_epoch"}  # hardcoded in 8.4.56 E2ELoss


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ids_sha256(image_dir: Path) -> str:
    files = sorted(p.name for p in image_dir.iterdir() if p.is_file())
    return sha256_text("\n".join(files))


# ---------------------------------------------------------------------------
# Recipe resolution
# ---------------------------------------------------------------------------

# R2-neutral (Step 2 frozen): R2 minus RGB photometric augmentation.
# hsv/bgr would corrupt [logD,logD,mask] physics; auto_augment/erasing are
# classify-only and set for protocol readability (None is safe in 8.4.56 check_cfg).
R2_NEUTRAL_OVERRIDES = {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "bgr": 0.0,
                        "auto_augment": None, "erasing": 0.0}


def extract_r2(ckpt_train_args: dict, default_cfg: dict) -> dict:
    """R2 whitelist extraction from checkpoint train_args (public API only)."""
    extracted = {}
    for k in sorted(R2_WHITELIST):
        if k in ckpt_train_args and ckpt_train_args[k] is not None:
            if k in default_cfg:  # must be a valid 8.4.56 arg
                extracted[k] = ckpt_train_args[k]
    return extracted


def recipe_for(a: argparse.Namespace, ckpt_train_args: dict, default_cfg: dict) -> tuple[dict, str]:
    """Return (recipe_overrides, source_desc). R2 extracted from ckpt via whitelist."""
    if a.recipe == "default":
        return dict(RECIPE_R0), "R0-sample-resolved (explicit)"
    if a.recipe == "small-data":
        r0 = dict(RECIPE_R0)
        r0.update(RECIPE_R1_OVERRIDES)
        return r0, "R1-sample-small (R0 + lr0=0.001, mosaic=0.5)"
    if a.recipe == "checkpoint":
        return extract_r2(ckpt_train_args, default_cfg), "R2-sample-ckpt-public (whitelist extraction)"
    if a.recipe == "checkpoint-neutral":
        r2 = extract_r2(ckpt_train_args, default_cfg)
        r2.update(R2_NEUTRAL_OVERRIDES)
        return r2, "R2-neutral-sample (R2 - RGB photometric: hsv/bgr/auto_augment/erasing)"
    raise RuntimeError(f"unknown recipe {a.recipe!r}")


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_data_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def check_dataset(data_yaml: Path) -> tuple[Path, list, list]:
    """Resolve dataset root (yaml dir, path: must be absent) and id lists."""
    d = load_data_yaml(data_yaml)
    if "path" in d:
        raise RuntimeError(
            f"dataset.yaml still has a 'path:' key ({d['path']!r}); remove it "
            f"so Ultralytics resolves {data_yaml.parent} as root"
        )
    root = data_yaml.parent
    train_dir = root / str(d["train"])
    val_dir = root / str(d["val"])
    if not train_dir.is_dir():
        raise RuntimeError(f"train dir missing: {train_dir}")
    if not val_dir.is_dir():
        raise RuntimeError(f"val dir missing: {val_dir}")
    return root, sorted(p.name for p in train_dir.iterdir() if p.is_file()), \
        sorted(p.name for p in val_dir.iterdir() if p.is_file())


def build_all17(data_yaml: Path, out_dir: Path) -> Path:
    """Copy all images+labels into a single dir; return all17 dataset yaml."""
    root = data_yaml.parent
    d = load_data_yaml(data_yaml)
    imgs = sorted((root / "images").glob("*/*.png")) + sorted((root / "images").glob("*/*.jpg"))
    if len(imgs) < 2:
        raise RuntimeError("all17 build: expected >=2 images")
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / "images"
    lab_out = out_dir / "labels"
    img_out.mkdir(exist_ok=True)
    lab_out.mkdir(exist_ok=True)
    for p in imgs:
        shutil.copy2(p, img_out / p.name)
        lab = root / "labels" / p.parent.name / (p.stem + ".txt")
        if lab.is_file():
            shutil.copy2(lab, lab_out / lab.name)
    all17_yaml = out_dir / "dataset_all17.yaml"
    names = "\n".join(f"  {k}: {v}" for k, v in d["names"].items())
    channels = f"channels: {d['channels']}\n" if "channels" in d else ""
    all17_yaml.write_text(
        f"path: {out_dir.as_posix()}\n"
        f"train: images\n"
        f"val: images\n"
        f"{channels}"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    return all17_yaml


# ---------------------------------------------------------------------------
# Transfer / head audits
# ---------------------------------------------------------------------------

def transfer_audit(model, weights_path: Path, backbone_n: int) -> dict:
    import torch
    ckpt_sd = torch.load(weights_path, map_location="cpu", weights_only=False)["model"].state_dict()
    dst_sd = model.model.state_dict()
    transferred, skipped, missing, unexpected = [], [], [], []
    for k, v in dst_sd.items():
        if k in ckpt_sd and ckpt_sd[k].shape == v.shape:
            transferred.append(k)
        elif k in ckpt_sd:
            skipped.append(k)
        else:
            missing.append(k)
    for k in ckpt_sd:
        if k not in dst_sd:
            unexpected.append(k)

    def part(key: str) -> str:
        try:
            i = int(key.split(".")[1])  # "model.<i>.<rest>"
        except (IndexError, ValueError):
            return "other"
        if i < backbone_n:
            return "backbone"
        if i == len(model.model.model) - 1:
            return "detect"
        return "neck"

    def part_params(keys) -> int:
        return sum(v.numel() for k, v in dst_sd.items() if k in keys)

    def part_hash(keys) -> str:
        h = hashlib.sha256()
        for k in sorted(keys):
            h.update(k.encode())
            h.update(dst_sd[k].cpu().numpy().tobytes())
        return h.hexdigest()

    backbone_keys = [k for k in transferred if part(k) == "backbone"]
    neck_keys = [k for k in transferred if part(k) == "neck"]
    detect_keys = [k for k in transferred if part(k) == "detect"]
    return {
        "transferred_tensors": len(transferred),
        "transferred_params": sum(v.numel() for k, v in dst_sd.items() if k in transferred),
        "skipped_shape_mismatch": len(skipped),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "backbone_transferred_params": part_params(backbone_keys),
        "neck_transferred_params": part_params(neck_keys),
        "detect_transferred_params": part_params(detect_keys),
        "backbone_sd_hash": part_hash(backbone_keys),
        "neck_sd_hash": part_hash(neck_keys),
    }


def _det_model(model):
    """Normalize to the DetectionModel: YOLO.model is a DetectionModel, trainer passes
    a DetectionModel directly (whose .model is the nn.Sequential)."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "yaml"):
        return inner  # YOLO wrapper
    return model  # DetectionModel itself


def head_five(model, requested: bool) -> dict:
    det = _det_model(model)
    return {
        "requested_head": requested,
        "yaml.end2end": bool(det.yaml["end2end"]),
        "model.end2end": bool(det.end2end),
        "trainer.args.end2end": None,  # filled on_train_start
        "detect_head.end2end": bool(det.model[-1].end2end),
    }


def assert_head_five(vals: dict, where: str):
    good = vals["requested_head"] == vals["yaml.end2end"] == \
        vals["model.end2end"] == vals["detect_head.end2end"]
    if vals.get("trainer.args.end2end") is not None:
        good = good and vals["trainer.args.end2end"] == vals["requested_head"]
    if not good:
        raise RuntimeError(f"head five-way mismatch at {where}: {vals}")


# ---------------------------------------------------------------------------
# Step 2: stem adaptation + two gate families
# ---------------------------------------------------------------------------

def tensor_sha256(t) -> str:
    h = hashlib.sha256()
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def gate_inputs(kind: str, ir_img: Path | None, depth_naive_img: Path | None,
                depth_adapt_img: Path | None):
    """CPU float32 eval tensors sharing the same physical content (naive vs adapted)."""
    import cv2
    import torch
    if kind == "ir":
        im3 = cv2.imread(str(ir_img), cv2.IMREAD_COLOR)
        im1 = cv2.imread(str(ir_img), cv2.IMREAD_GRAYSCALE)
        if im1 is not None and im1.ndim == 3:  # ultralytics-patched cv2: gray read may be (H,W,1)
            im1 = im1[..., 0]
        # im3 is channel-symmetric (repeat3) so the BGR->RGB flip is a no-op numerically
        return (torch.from_numpy(im3.transpose(2, 0, 1)[None]).float() / 255.0,
                torch.from_numpy(im1[None, None]).float() / 255.0)
    if kind == "depth":
        # disk BGR order -> model RGB order (same flip Ultralytics applies in its loader)
        im_n = cv2.imread(str(depth_naive_img), cv2.IMREAD_COLOR)[..., ::-1].copy()
        im_a = cv2.imread(str(depth_adapt_img), cv2.IMREAD_COLOR)[..., ::-1].copy()
        return (torch.from_numpy(im_n.transpose(2, 0, 1)[None]).float() / 255.0,
                torch.from_numpy(im_a.transpose(2, 0, 1)[None]).float() / 255.0)
    raise ValueError(kind)


def _compare(a, b) -> dict:
    import torch
    d = (a - b).abs()
    rel = d / torch.clamp(a.abs(), min=1e-6)
    return {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "max_rel_diff": float(rel.max())}


def stem_fp_gate(naive_block, adapt_block, x_naive, x_adapt, where: str,
                 threshold_conv: float = 1e-5, threshold_block: float = 1e-4) -> dict:
    """FUNCTION_PRESERVATION gate: raw Conv + Conv+BN+Act outputs, same input content.

    Valid ONLY at init / before the first optimizer.step(); after training the
    parameterizations diverge by design (no equivalence expected).
    CPU float32: different channel-summation order gives ~1e-7 raw-conv noise;
    BN/Act amplify it to ~1e-5..1e-4, hence the looser block threshold.
    """
    import torch
    naive_block.eval()
    adapt_block.eval()
    with torch.no_grad():
        g_conv = _compare(naive_block.conv(x_naive), adapt_block.conv(x_adapt))
        g_block = _compare(naive_block(x_naive), adapt_block(x_adapt))
    passed = g_conv["max_abs_diff"] <= threshold_conv and g_block["max_abs_diff"] <= threshold_block
    return {"where": where, "threshold_raw_conv": threshold_conv,
            "threshold_conv_bn_act": threshold_block, "raw_conv": g_conv,
            "conv_bn_act": g_block, "passed": bool(passed)}


def ckpt_reload_integrity(ckpt_path: Path, expected_stem_shape: list) -> dict:
    """CHECKPOINT_PERSISTENCE: reloaded stem matches checkpoint contents; no naive compare."""
    import torch
    from ultralytics import YOLO
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_model = ck.get("model")
    sd = ck_model.float().state_dict() if hasattr(ck_model, "float") else ck_model
    w_ckpt = sd["model.0.conv.weight"]
    m = YOLO(str(ckpt_path))
    w_model = m.model.model[0].conv.weight.detach().cpu().float()
    stem_shape_ok = list(w_model.shape) == expected_stem_shape
    tensor_ok = bool(torch.equal(w_ckpt, w_model))
    sha_ok = tensor_sha256(w_ckpt) == tensor_sha256(w_model)
    try:
        m.model.eval()
        with torch.no_grad():
            m.model(torch.zeros(1, expected_stem_shape[1], 320, 320))
        forward_ok = True
    except Exception as exc:  # noqa: BLE001
        forward_ok = f"FAILED: {exc}"
    return {"stem_shape_ok": stem_shape_ok, "stem_tensor_matches_checkpoint": tensor_ok,
            "stem_sha256_matches": sha_ok, "forward_ok": forward_ok,
            "reload_stem_shape": list(w_model.shape)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def metric_dict(box, names: dict) -> dict:
    out = {"map50": float(box.map50), "map50_95": float(box.map)}
    per = {}
    idxs = box.ap_class_index.tolist() if hasattr(box, "ap_class_index") else []
    aps = box.ap.tolist() if hasattr(box, "ap") else []
    ap50s = getattr(box, "ap50", None)
    ap50s = ap50s.tolist() if ap50s is not None else []
    for i, ci in enumerate(idxs):
        name = names.get(str(int(ci)), str(int(ci)))
        per[name] = {
            "ap50": round(ap50s[i], 4) if i < len(ap50s) else None,
            "ap50_95": round(aps[i], 4) if i < len(aps) else None,
        }
    out["per_class"] = per
    return out


def eval_set(model, data_yaml: str, device: str, names: dict, split: str = "val") -> dict:
    from ultralytics import YOLO
    m = model.val(data=data_yaml, split=split, imgsz=640, batch=4, device=device,
                  max_det=100, plots=False, verbose=False)
    return metric_dict(m.box, names)


def run_sets_eval(ckpt_path: Path, data_yaml: Path, all17_yaml: Path, device: str,
                  names: dict, run_dir: Path, out_name: str = "eval_sets.json") -> dict:
    """Three-set eval (train11 / val6 / all17) of one checkpoint; writes json to run_dir.

    train11 = "did the model learn the training samples"; val6 = held-out transfer;
    all17 = pipeline sanity only (mixed set, NOT a generalization metric).
    """
    from ultralytics import YOLO
    m = YOLO(str(ckpt_path))
    ckpt_args_e2e = None
    ckpt = getattr(m, "ckpt", None)
    if isinstance(ckpt, dict) and "train_args" in ckpt:
        ckpt_args_e2e = bool(ckpt["train_args"].get("end2end"))
    out = {
        "checkpoint": ckpt_path.name,
        "train": eval_set(m, str(data_yaml), device, names, split="train"),
        "val": eval_set(m, str(data_yaml), device, names, split="val"),
        "all17": eval_set(m, str(all17_yaml), device, names, split="val"),
        "head_on_ckpt": {
            "ckpt_train_args.end2end": ckpt_args_e2e,
            "detect_head.end2end": bool(m.model.model[-1].end2end),
        },
    }
    (run_dir / out_name).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def make_manifest(a, recipe: dict, source: str, data_meta: dict, transfer: dict,
                  head_before: dict, effective_args: dict) -> dict:
    import torch
    import ultralytics
    common = {k: effective_args.get(k) for k in sorted(COMMON_KEYS)}
    config_fingerprint = {
        "head": a.head, "recipe": a.recipe, "recipe_overrides": {k: recipe[k] for k in sorted(recipe)},
        "common": common, "seed": a.seed, "imgsz": a.imgsz, "batch": a.batch, "epochs": a.epochs,
    }
    return {
        "release_name": RELEASE_NAME,
        "group": a.group,
        "head": a.head,
        "recipe": a.recipe,
        "recipe_source": source,
        "seed": a.seed,
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "weights_sha256": data_meta["weights_sha256"],
        "dataset_yaml_sha256": data_meta["dataset_yaml_sha256"],
        "train_ids_sha256": data_meta["train_ids_sha256"],
        "val_ids_sha256": data_meta["val_ids_sha256"],
        "transfer_audit": transfer,
        "head_check_before_train": head_before,
        "config_sha256": sha256_text(json.dumps(config_fingerprint, sort_keys=True, ensure_ascii=False)),
    }


# ---------------------------------------------------------------------------
# Audit (B0-E)
# ---------------------------------------------------------------------------

def run_audit(a: argparse.Namespace, weights: Path) -> dict:
    import torch
    from ultralytics.cfg import DEFAULT_CFG_DICT
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    ta = dict(ckpt.get("train_args") or {})
    audit = {"recipe_used": {}, "checkpoint_common_metadata": {},
             "effective_common_experiment": common_kwargs(a),
             "ignored_runtime": {}, "unsupported_or_internal": {}}
    for k, v in sorted(ta.items()):
        if k in UNSUPPORTED_INTERNAL or k not in DEFAULT_CFG_DICT:
            audit["unsupported_or_internal"][k] = v
        elif k in COMMON_KEYS:
            # the checkpoint's ORIGINAL training metadata, NOT what this probe ran
            audit["checkpoint_common_metadata"][k] = v
        elif k in RUNTIME_KEYS:
            audit["ignored_runtime"][k] = v
        elif k in R2_WHITELIST:
            audit["recipe_used"][k] = v
        else:
            audit["ignored_runtime"][k] = v
    audit["note"] = (
        "R2-sample-ckpt-public = recipe_used values, expressible via public 8.4.56 API. "
        "checkpoint_common_metadata = ORIGINAL yolo26s.pt training metadata (batch128/"
        "epochs70/nbs64/seed0/...), overridden by effective_common_experiment (what this "
        "probe actually executed: batch4/epochs80/nbs4/warmup0/max_det100/...). "
        "o2m/topk/detach_epoch/cls_w/muon_w/sgd_w/stride_ratio are hardcoded internals "
        "of 8.4.56 and cannot be reproduced via public args."
    )
    return audit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GROUPS = {
    ("e2e", "default"): "B0-A",
    ("o2m", "default"): "B0-B",
    ("o2m", "small-data"): "B0-C",
    ("o2m", "checkpoint"): "B0-D",
    ("e2e", "checkpoint"): "B0-F",
}


def parse_args():
    p = argparse.ArgumentParser(description="YOLO26s RGB Step1-Sample Probe (ultralytics 8.4.56)")
    p.add_argument("--weights", default="E:/odin/yolo26s.pt")
    p.add_argument("--data", default="D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml")
    p.add_argument("--head", choices=["e2e", "o2m"], default="e2e")
    p.add_argument("--recipe", choices=["default", "small-data", "checkpoint", "checkpoint-neutral"],
                   default="default")
    p.add_argument("--stem-adapt", choices=["none", "ir", "depth"], default="none",
                   help="Step2: none | ir (native 1ch Conv2d, W=sum(source RGB stem)) | "
                        "depth (3ch [W0+W1,W2,0] for [D,M,0] input)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cache", default="ram")
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--project", default="runs/step1_rgb")
    p.add_argument("--name", default=None)
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--audit-only", action="store_true", help="B0-E: classified audit, no training")
    p.add_argument("--eval-only", default=None, metavar="RUN_DIR",
                   help="no training: re-evaluate best.pt + last.pt of an existing run dir")
    p.add_argument("--save-manifest", action="store_true", help="write manifest json to run dir")
    p.add_argument("--gate-ir-img", default=None, help="stem fp gate input: IR png (same file, color+gray read)")
    p.add_argument("--gate-depth-naive-img", default=None, help="stem fp gate naive input: [D,D,M] png")
    p.add_argument("--gate-depth-adapt-img", default=None, help="stem fp gate adapted input: [D,M,0] png")
    return p.parse_args()


def main():
    a = parse_args()
    if isinstance(a.cache, str) and a.cache.lower() in ("false", "none", ""):
        a.cache = False  # Step 2: deterministic=True + ram cache is warned against; 17 imgs
    a.group = a.name or GROUPS[(a.head, a.recipe)]
    # absolute project: ultralytics resolves relative project against settings.runs_dir
    # (runs/detect), which silently forks the run dir away from our pre-checks
    a.project = str(Path(a.project).resolve())
    data_yaml = Path(a.data).resolve()
    weights = Path(a.weights).resolve()

    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT

    if ultralytics.__version__ != "8.4.56":
        raise RuntimeError(f"ultralytics=={ultralytics.__version__}, design requires 8.4.56")
    if "end2end" not in DEFAULT_CFG_DICT:
        raise RuntimeError("'end2end' is not a valid 8.4.56 arg (wrong version?)")

    if a.audit_only:
        audit = run_audit(a, weights)
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        if a.save_manifest:
            out = Path(a.project) / "B0-E-audit"
            out.mkdir(parents=True, exist_ok=True)
            (out / "audit_r2.json").write_text(
                json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"audit written to {out / 'audit_r2.json'}")
        return

    if a.eval_only:
        # fixed-budget re-evaluation of an existing run dir; no training, no model build
        root, train_ids, val_ids = check_dataset(data_yaml)
        names = load_data_yaml(data_yaml)["names"]
        run_dir = Path(a.eval_only).resolve()
        best_path = run_dir / "weights" / "best.pt"
        last_path = run_dir / "weights" / "last.pt"
        if not best_path.is_file():
            raise RuntimeError(f"best.pt missing: {best_path}")
        all17_dir = Path(a.project) / "_probe_all17"
        all17_yaml = build_all17(data_yaml, all17_dir)
        ev_best = run_sets_eval(best_path, data_yaml, all17_yaml, a.device, names,
                                run_dir, "eval_sets.json")
        ev_last = None
        if last_path.is_file():
            ev_last = run_sets_eval(last_path, data_yaml, all17_yaml, a.device, names,
                                    run_dir, "eval_sets_last.json")
        man_path = run_dir / "experiment_manifest.json"
        if man_path.is_file():
            man = json.loads(man_path.read_text(encoding="utf-8"))
            man["eval_sets"] = ev_best
            if ev_last is not None:
                man["eval_sets_last"] = ev_last
            man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"manifest updated: {man_path}")
        print(json.dumps({
            "group": run_dir.name,
            "best_train_map50_95": ev_best["train"]["map50_95"],
            "best_val_map50_95": ev_best["val"]["map50_95"],
            "best_all17_map50_95": ev_best["all17"]["map50_95"],
            "last_train_map50_95": ev_last["train"]["map50_95"] if ev_last else None,
            "last_val_map50_95": ev_last["val"]["map50_95"] if ev_last else None,
            "last_all17_map50_95": ev_last["all17"]["map50_95"] if ev_last else None,
        }, indent=2))
        return

    # ---- dataset pre-check + fingerprints ----
    root, train_ids, val_ids = check_dataset(data_yaml)
    names = load_data_yaml(data_yaml)["names"]
    data_meta = {
        "weights_sha256": sha256_file(weights),
        "dataset_yaml_sha256": sha256_file(data_yaml),
        "train_ids_sha256": ids_sha256(root / "images" / "train"),
        "val_ids_sha256": ids_sha256(root / "images" / "val"),
    }

    # ---- checkpoint train_args (only used for R2 extraction + B0-E) ----
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    ckpt_args = dict(ckpt.get("train_args") or {})

    recipe, source = recipe_for(a, ckpt_args, DEFAULT_CFG_DICT)

    if a.recipe in ("checkpoint", "checkpoint-neutral"):
        try:
            from ultralytics.optim.muon import MuSGD  # fail fast, no silent fallback
        except ImportError as e:
            raise RuntimeError("checkpoint recipe aborted: MuSGD unavailable in this ultralytics build") from e

    # ---- build: yaml structure + pretrained weights only ----
    model = YOLO("yolo26s.yaml")
    model.load(str(weights))

    # ---- head: three-way consistent switch ----
    e2e = a.head == "e2e"
    model.model.yaml["end2end"] = e2e
    model.model.end2end = e2e
    head_before = head_five(model, e2e)
    head_before.pop("trainer.args.end2end")  # Trainer not built yet; recorded after build
    assert_head_five(head_before, "before-train")

    # ---- effective training args: recipe + common (common wins) ----
    kw = {**recipe, **common_kwargs(a),
          "data": str(data_yaml), "project": a.project, "name": a.group,
          "device": a.device, "exist_ok": a.exist_ok, "end2end": e2e}

    # ---- run dir conflict pre-check (keeps run_dir inference exact) ----
    run_dir = Path(a.project) / a.group
    if run_dir.exists() and not a.exist_ok:
        raise RuntimeError(f"run dir already exists: {run_dir}; use --exist-ok to reuse "
                           f"or --name for a new run")

    # ---- transfer audit (backbone/neck hashes guard the A/B single-variable claim) ----
    backbone_n = len(model.model.yaml["backbone"])
    transfer = transfer_audit(model, weights, backbone_n)

    # ---- Step 2: stem adaptation (source stem ALWAYS from the original checkpoint) ----
    stem_meta = None
    fp_gates = []
    naive = None
    x_n = x_a = None
    if a.stem_adapt != "none":
        import torch
        import torch.nn as nn
        src_model = YOLO(str(weights))  # Ultralytics official loader (P0: never partial-transferred target)
        src_w = src_model.model.state_dict()["model.0.conv.weight"].float().detach()
        tgt0 = model.model.model[0]
        stem_meta = {"stem_source": f"{weights.name}::model.0.conv.weight",
                     "source_stem_shape": list(src_w.shape),
                     "source_stem_sha256": tensor_sha256(src_w)}
        # naive reference = fresh build + load + same head switch (pre-adaptation equivalent)
        naive = YOLO("yolo26s.yaml")
        naive.load(str(weights))
        naive.model.yaml["end2end"] = e2e
        naive.model.end2end = e2e
        if a.stem_adapt == "ir":
            new_conv = nn.Conv2d(1, tgt0.conv.out_channels, tgt0.conv.kernel_size,
                                 tgt0.conv.stride, tgt0.conv.padding,
                                 groups=tgt0.conv.groups, bias=tgt0.conv.bias is not None)
            with torch.no_grad():
                new_conv.weight.copy_(src_w.sum(dim=1, keepdim=True))
            tgt0.conv = new_conv
            stem_meta["adapt_formula"] = "W_IR = sum(source RGB stem, dim=1, keepdim=True); true Conv2d(in_channels=1)"
        elif a.stem_adapt == "depth":
            with torch.no_grad():
                new_w = torch.stack([src_w[:, 0] + src_w[:, 1], src_w[:, 2],
                                     torch.zeros_like(src_w[:, 0])], dim=1)
                tgt0.conv.weight.copy_(new_w)
            stem_meta["adapt_formula"] = "W_new = [W0+W1, W2, 0] (3ch compatibility for [D,M,0] input)"
        stem_meta["pretrainer_target_stem_shape"] = list(tgt0.conv.weight.shape)
        stem_meta["interpretation_note"] = (
            "function-preserving holds ONLY at epoch-0 init; parameterization differs, so "
            "optimization trajectories are NOT equivalent (no 3x/2x LR compensation). "
            "Adapt groups are minimum-adapted representation candidates, not reparameterization controls.")
        if a.stem_adapt == "ir" and not a.gate_ir_img:
            a.gate_ir_img = str(sorted((root / "images" / "train").glob("*.png"))[0])
        x_n, x_a = gate_inputs(
            a.stem_adapt,
            Path(a.gate_ir_img).resolve() if a.gate_ir_img else None,
            Path(a.gate_depth_naive_img).resolve() if a.gate_depth_naive_img else None,
            Path(a.gate_depth_adapt_img).resolve() if a.gate_depth_adapt_img else None)
        g1 = stem_fp_gate(naive.model.model[0], model.model.model[0], x_n, x_a,
                          "adaptation-complete")
        fp_gates.append(g1)
        if not g1["passed"]:
            raise RuntimeError(f"[{a.group}] stem fp gate FAILED stage-1: {g1}")
        print(f"[{a.group}] stem fp gate stage-1 PASS: raw {g1['raw_conv']['max_abs_diff']:.2e} "
              f"block {g1['conv_bn_act']['max_abs_diff']:.2e}")
        del src_model

    # ---- on_train_start: head five-way + stage-2 stem fp gate (pre optimizer.step) ----
    head_after_trainer = {}
    def _on_train_start(trainer):
        vals = head_five(trainer.model, e2e)
        vals["trainer.args.end2end"] = bool(trainer.args.end2end) if trainer.args.end2end is not None else None
        assert_head_five(vals, "on-train-start")
        head_after_trainer.update(vals)
        print(f"[{a.group}] head five-way OK on_train_start: {vals}")
        if stem_meta is not None:
            dev = next(trainer.model.parameters()).device
            was_training = trainer.model.training
            trainer.model.eval()
            g2 = stem_fp_gate(naive.model.model[0].to(dev), trainer.model.model[0],
                              x_n.to(dev), x_a.to(dev), "trainer-built-pre-optimizer")
            if was_training:
                trainer.model.train()
            fp_gates.append(g2)
            stem_meta["trainer_target_stem_shape"] = list(trainer.model.model[0].conv.weight.shape)
            if not g2["passed"]:
                raise RuntimeError(f"[{a.group}] stem fp gate FAILED stage-2: {g2}")
            print(f"[{a.group}] stem fp gate stage-2 PASS: raw {g2['raw_conv']['max_abs_diff']:.2e} "
                  f"block {g2['conv_bn_act']['max_abs_diff']:.2e}")
    model.add_callback("on_train_start", _on_train_start)

    # ---- manifest (pre-train) + summary ----
    manifest = make_manifest(a, recipe, source, data_meta, transfer, head_before, kw)
    summary = {k: manifest[k] for k in
               ("group", "head", "recipe", "recipe_source", "seed", "ultralytics", "torch")}
    summary["effective_args"] = {k: kw[k] for k in
                                 ("optimizer", "lr0", "lrf", "momentum", "weight_decay", "box",
                                  "cls", "dfl", "mosaic", "mixup", "copy_paste", "scale",
                                  "warmup_epochs", "nbs", "patience", "close_mosaic", "max_det",
                                  "deterministic", "cache", "workers", "seed", "imgsz", "batch", "epochs")}
    summary["transfer"] = {k: transfer[k] for k in
                           ("transferred_tensors", "transferred_params", "skipped_shape_mismatch",
                            "backbone_transferred_params", "neck_transferred_params",
                            "detect_transferred_params", "backbone_sd_hash", "neck_sd_hash")}
    summary["head_before_train"] = head_before
    summary["config_sha256"] = manifest["config_sha256"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- train ----
    model.train(**kw)

    # ---- post-train: three-set eval (train11/val6/all17) on best.pt (aux) and last.pt (FIXED primary) ----
    # 6-val is too small for stable best.pt selection (observed best epoch 1 in several
    # runs); the fixed-epoch80 last.pt is the primary comparison axis for this probe.
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    for ck in (best_path, last_path):
        if not ck.is_file():
            raise RuntimeError(f"{ck.name} missing after train: {ck}")
    all17_dir = Path(a.project) / "_probe_all17"
    all17_yaml = build_all17(data_yaml, all17_dir)
    eval_best = run_sets_eval(best_path, data_yaml, all17_yaml, a.device, names,
                              run_dir, "eval_sets.json")
    eval_last = run_sets_eval(last_path, data_yaml, all17_yaml, a.device, names,
                              run_dir, "eval_sets_last.json")

    # ---- CHECKPOINT_PERSISTENCE (best/last reload integrity; no naive compare) ----
    expected_shape = stem_meta["pretrainer_target_stem_shape"] if stem_meta else \
        list(model.model.model[0].conv.weight.shape)
    integrity = {"expected_stem_shape": expected_shape,
                 "best.pt": ckpt_reload_integrity(best_path, expected_shape),
                 "last.pt": ckpt_reload_integrity(last_path, expected_shape)}
    for ck_name, integ in integrity.items():
        if isinstance(integ, dict) and not (integ.get("stem_shape_ok")
                                            and integ.get("stem_tensor_matches_checkpoint")
                                            and integ.get("stem_sha256_matches")):
            raise RuntimeError(f"[{a.group}] checkpoint persistence FAILED {ck_name}: {integ}")

    # ---- manifest (post-train, complete) ----
    if a.save_manifest:
        manifest["eval_sets"] = eval_best
        manifest["eval_sets_last"] = eval_last
        manifest["head_check_after_trainer"] = head_after_trainer or None
        manifest["head_check_after_checkpoint_reload"] = {
            "best.pt": eval_best["head_on_ckpt"],
            "last.pt": eval_last["head_on_ckpt"],
        }
        manifest["checkpoint_reload_integrity"] = integrity
        if stem_meta is not None:
            manifest["stem_meta"] = stem_meta
            manifest["stem_function_preservation"] = fp_gates
        manifest_path = run_dir / "experiment_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"manifest written to {manifest_path}")

    print(f"[{a.group}] DONE. best: train {eval_best['train']['map50_95']:.4f} "
          f"val {eval_best['val']['map50_95']:.4f} all17 {eval_best['all17']['map50_95']:.4f} | "
          f"last: train {eval_last['train']['map50_95']:.4f} "
          f"val {eval_last['val']['map50_95']:.4f} all17 {eval_last['all17']['map50_95']:.4f}")


if __name__ == "__main__":
    main()
