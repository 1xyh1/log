"""Step 3-A 6ch early-fusion YOLO26s (O2M) builder + init gates G4/G4b/G5/G6/G7.

Build path (frozen):
    original yolo26s.pt (E2E, COCO pretrain)
        -> 3ch YOLO26s built from yolo26s.yaml, end2end flipped False, weights
           transferred (same path as Step 1 B0-G) = 3ch O2M reference
        -> deepcopy reference; replace first layer with TRUE Conv2d(in_channels=6);
           weight = [W_R, W_G, W_B, 0, 0, 0] with source stem from the ORIGINAL
           checkpoint (never from Ultralytics partial transfer)
        -> save unique snapshot step3_6ch_rgb_equiv_init.pt (+ SHA256)
    C0-N / C1-I / C2-D all load from that single snapshot.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

WEIGHTS_DEFAULT = "E:/odin/yolo26s.pt"
SNAPSHOT_DEFAULT = "step3_6ch_rgb_equiv_init.pt"
CHANNEL_SEMANTICS = ["R", "G", "B", "IR_scalar", "log_depth", "valid_mask"]
MODEL_INIT_SEED = 2026081200  # frozen: 12-class head random init must be reproducible


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(t) -> str:
    h = hashlib.sha256()
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_reference_3ch(weights: str = WEIGHTS_DEFAULT, nc: int = 12):
    """3ch O2M reference with PHYSICAL nc=12 head (reviewer P0-A: the Trainer never
    rebuilds the head for a passed-in nn.Module - setup_model early-returns and
    set_model_attributes only sets .nc/.names/.args attributes).

    Note: Step 1 (B0-*) trained 80-class physical heads for this reason (its
    per-class APs are still internally valid); Step 3 builds nc=12 from the start.
    Class-head convs (80->12 shape mismatch) are random-initialized by partial
    transfer - standard COCO->12-class fine-tuning.
    """
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from multimodal.raw_sample_index import CLASS_NAMES
    d = yaml_model_load("yolo26s.yaml")
    d["nc"] = nc
    d["end2end"] = False
    m = DetectionModel(d, ch=3, nc=nc)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    m.load(ckpt)  # DetectionModel.load takes the full ckpt dict (reads weights["model"])
    # 8.4.56 DetectionModel does NOT set .nc/.args in __init__ (trainer sets them via
    # set_model_attributes); set explicitly so the snapshot is self-describing.
    m.nc = nc
    m.names = {i: CLASS_NAMES[i] for i in range(nc)}  # real 12-class names
    m.eval()
    return m


def build_6ch(reference, weights: str = WEIGHTS_DEFAULT, save_snapshot: str = SNAPSHOT_DEFAULT) -> dict:
    """6ch O2M model from the 3ch reference; source stem ALWAYS from the original ckpt."""
    src_model = YOLO(str(weights))
    src_stem = src_model.model.state_dict()["model.0.conv.weight"].float().detach().clone()
    m6 = deepcopy(reference)
    first = m6.model[0]
    new_conv = nn.Conv2d(6, first.conv.out_channels, first.conv.kernel_size,
                         first.conv.stride, first.conv.padding,
                         groups=first.conv.groups, bias=first.conv.bias is not None)
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, 0:3].copy_(src_stem)
    first.conv = new_conv
    m6.yaml["channels"] = 6  # keep YAML metadata in sync with the physical 6ch structure
    m6.eval()
    snapshot = Path(save_snapshot)
    torch.save({"model": m6, "meta": {
        "channel_semantics": CHANNEL_SEMANTICS,
        "first_conv_init": "[W_R,W_G,W_B,0,0,0]",
        "source": f"{Path(weights).name}::model.0.conv.weight",
        "source_stem_sha256": tensor_sha256(src_stem),
        "end2end": False,
        "nc": 12,
        "model_init_seed": MODEL_INIT_SEED,
    }}, snapshot)
    return {"model": m6, "src_stem": src_stem, "snapshot": snapshot,
            "snapshot_sha256": sha256_file(snapshot),
            "source_stem_sha256": tensor_sha256(src_stem)}


def load_snapshot(path: str = SNAPSHOT_DEFAULT):
    """The exact loading path the training runs will use (G4b)."""
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def _head_nc_physical(model) -> list:
    """Physical cls-head out channels of ALL THREE scales (cv3 branches)."""
    head = model.model[-1]
    out = []
    for branch in head.cv3:
        last = branch[-1]
        conv = last.conv if hasattr(last, "conv") else last
        out.append(int(conv.out_channels))
    return out


def gate_g4(m6, src_stem) -> dict:
    w = m6.model[0].conv.weight.detach().cpu().float()
    rgb_ok = bool(torch.equal(w[:, 0:3], src_stem.cpu()))
    aux_max = float(w[:, 3:6].abs().max())
    return {"shape": list(w.shape), "rgb_part_bitwise_equal_source": rgb_ok,
            "aux_part_max_abs": aux_max, "passed": bool(rgb_ok and aux_max == 0.0)}


def gate_g5(ref3, m6) -> dict:
    vals = {"ref3_yaml": bool(ref3.yaml["end2end"]),
            "ref3_model": bool(ref3.end2end),
            "m6_yaml": bool(m6.yaml["end2end"]),
            "m6_model": bool(m6.end2end),
            "m6_head": bool(m6.model[-1].end2end),
            "ref3_nc_attr": int(ref3.nc),
            "ref3_head_nc_attr": int(ref3.model[-1].nc),
            "ref3_head_physical": _head_nc_physical(ref3),
            "m6_nc_attr": int(m6.nc),
            "m6_head_nc_attr": int(m6.model[-1].nc),
            "m6_head_physical": _head_nc_physical(m6)}
    passed = (not any(bool(vals[k]) for k in
                      ("ref3_yaml", "ref3_model", "m6_yaml", "m6_model", "m6_head"))
              and vals["ref3_nc_attr"] == 12 and vals["ref3_head_nc_attr"] == 12
              and vals["ref3_head_physical"] == [12, 12, 12]
              and vals["m6_nc_attr"] == 12 and vals["m6_head_nc_attr"] == 12
              and vals["m6_head_physical"] == [12, 12, 12])
    vals["passed"] = passed
    return vals


def _compare(a, b) -> dict:
    d = (a - b).abs()
    return {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean())}


def gate_g6a(ref3, m6) -> dict:
    """Model equivalence: same constructed RGB tensor, aux=0. raw conv / conv+bn+act / final."""
    rng_state = torch.random.get_rng_state()  # gates must not pollute RNG (frozen protocol)
    torch.manual_seed(0)
    x3 = torch.rand(1, 3, 640, 640)
    x6 = torch.cat([x3, torch.zeros(1, 3, 640, 640)], dim=1)
    ref3.eval()
    m6.eval()
    with torch.no_grad():
        c3 = ref3.model[0].conv(x3)
        c6 = m6.model[0].conv(x6)
        g_conv = _compare(c3, c6)
        b3 = ref3.model[0](x3)
        b6 = m6.model[0](x6)
        g_block = _compare(b3, b6)
        def _tensors(o, acc):
            if torch.is_tensor(o):
                acc.append(o)
            elif isinstance(o, dict):
                for v in o.values():
                    _tensors(v, acc)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    _tensors(v, acc)
            return acc

        t3, t6 = _tensors(ref3._predict_once(x3), []), _tensors(m6._predict_once(x6), [])
        if len(t3) != len(t6):
            raise RuntimeError(f"final output tensor count mismatch: {len(t3)} vs {len(t6)}")
        g_final = max((_compare(a, b) for a, b in zip(t3, t6)),
                      key=lambda d: d["max_abs_diff"])
    torch.random.set_rng_state(rng_state)  # restore: gates must not pollute RNG
    thr = 1e-5
    passed = g_conv["max_abs_diff"] <= thr and g_block["max_abs_diff"] <= thr \
        and g_final["max_abs_diff"] <= thr
    return {"raw_conv": g_conv, "conv_bn_act": g_block, "final_detector": g_final,
            "threshold": thr, "passed": bool(passed)}


def gate_g6b(contract: dict) -> dict:
    """RGB pipeline equivalence: our first-3-channels vs the OFFICIAL LetterBox transform
    applied to the same raw image (no YOLODataset / label cache involved)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from multimodal import modality_preprocess as mp
    from ultralytics.data.augment import LetterBox

    lb = LetterBox(new_shape=(640, 640), scaleup=False)
    diffs = []
    for sid in contract["all17_ids"]:
        rgb_p = next((Path(contract["_raw_dir"]) / "visible").glob(f"{sid}.*"))
        rgb_u8 = mp.load_rgb_rgb(str(rgb_p))
        ours, _ = mp.letterbox_rgb(rgb_u8, 640)  # (640,640,3) float [0,1] RGB
        bgr = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)  # BGR uint8
        ref_bgr = lb(image=bgr)
        ref_rgb = ref_bgr[..., ::-1].astype(np.float32) / 255.0
        d = float(np.abs(ours - ref_rgb).max())
        diffs.append(d)
    max_d = max(diffs)
    return {"n_images": len(diffs), "max_abs_diff_vs_official_letterbox": max_d,
            "passed": bool(max_d <= 1e-6)}


def gate_g4b(snapshot_path: str, src_stem) -> dict:
    """Reload the snapshot via the exact training loading path; full identity re-check:
    G4 stem, input conv in_channels=6, physical nc [12,12,12], model/head nc=12,
    snapshot meta nc + source stem sha, O2M, forward."""
    ck = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    meta = ck.get("meta") or {}
    m = ck["model"]
    g4 = gate_g4(m, src_stem)
    try:
        m.eval()
        with torch.no_grad():
            m._predict_once(torch.zeros(1, 6, 320, 320))
        fwd_ok = True
    except Exception as exc:  # noqa: BLE001
        fwd_ok = f"FAILED: {exc}"
    vals = {
        "reload_g4": g4,
        "input_conv_in_channels": int(m.model[0].conv.in_channels),
        "physical_head_nc": _head_nc_physical(m),
        "model_nc": int(m.nc),
        "head_nc": int(m.model[-1].nc),
        "snapshot_meta_nc": meta.get("nc"),
        "snapshot_meta_init_seed": meta.get("model_init_seed"),
        "source_stem_sha_matches": bool(meta.get("source_stem_sha256") == tensor_sha256(src_stem)),
        "o2m": bool(not m.end2end and not m.model[-1].end2end),
        "forward_ok": fwd_ok,
    }
    vals["passed"] = bool(
        g4["passed"] and vals["input_conv_in_channels"] == 6
        and vals["physical_head_nc"] == [12, 12, 12]
        and vals["model_nc"] == 12 and vals["head_nc"] == 12
        and vals["snapshot_meta_nc"] == 12 and vals["snapshot_meta_init_seed"] == MODEL_INIT_SEED
        and vals["source_stem_sha_matches"] and vals["o2m"] and fwd_ok is True)
    return vals


def r3_hyp():
    """Actual R3-causal-earlyfusion-sample hyp namespace (R2-core loss weights from the
    frozen B0-E audit; augmentation per the frozen R3 recipe)."""
    from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
    d = {k: getattr(DEFAULT_CFG, k) for k in vars(DEFAULT_CFG)}
    d.update({"box": 9.83241, "cls": 0.64896, "dfl": 0.95824,  # R2-core (B0-E audit recipe_used)
              "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "bgr": 0.0,
              "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
              "scale": 0.0, "translate": 0.0, "degrees": 0.0, "shear": 0.0,
              "perspective": 0.0, "multi_scale": 0.0,
              "fliplr": 0.30393, "flipud": 0.0,
              "auto_augment": None, "erasing": 0.0, "close_mosaic": 0})
    return IterableSimpleNamespace(**d)


def gate_g7(contract: dict, group: str, m6, threshold_scale: float = 0.0) -> dict:
    """Gradient flow: CPU/float32/train/aug-off, one real batch, forward+loss+backward.

    Uses the REAL nc=12 snapshot and R3 hyp. Expects exact zeros for inactive channels
    (input is exactly 0 -> grad is exactly 0) and positive L2 for active ones;
    reports relative r_c vs mean RGB channel grad.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from multimodal.trimodal_dataset import GROUPS, TriModalDataset
    m = deepcopy(m6)
    m.args = r3_hyp()  # v8DetectionLoss uses model.args as hyp (IterableSimpleNamespace)
    m.criterion = m.init_criterion()
    m.train()
    ds = TriModalDataset(contract, split="train", group=group, augment=False)
    batch = TriModalDataset.collate_fn([ds[0], ds[1]])
    img = batch["img"]
    preds = m._predict_once(img)
    loss = m.loss(batch, preds)
    loss = loss.sum() if isinstance(loss, torch.Tensor) else sum(v.sum() for v in loss if torch.is_tensor(v))
    m.zero_grad()
    loss.backward()
    g = m.model[0].conv.weight.grad.detach().float()  # (32, 6, 3, 3)
    norms = [float(g[:, c].norm()) for c in range(6)]
    mean_rgb = float(np.mean(norms[0:3])) + 1e-12
    rel = [round(n / mean_rgb, 6) for n in norms]
    mask = GROUPS[group]
    expected = {"I": mask["I"], "D": mask["D"], "M": mask["M"]}
    ok = True
    checks = {}
    for ch, active in (("I", expected["I"]), ("D", expected["D"]), ("M", expected["M"])):
        idx = {"I": 3, "D": 4, "M": 5}[ch]
        if active:
            checks[ch] = norms[idx] > 0
        else:
            checks[ch] = norms[idx] == 0.0  # exact zero (zero input)
        ok = ok and checks[ch]
    checks["rgb"] = all(n > 0 for n in norms[0:3])
    ok = ok and checks["rgb"]
    return {"group": group, "grad_l2_per_channel": norms,
            "relative_to_mean_rgb": rel, "checks": checks,
            "note": "connectivity sanity only; NO cross-channel magnitude comparison",
            "passed": bool(ok)}


def run_all(weights: str = WEIGHTS_DEFAULT, contract_path: str = None) -> dict:
    import random as _random
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from multimodal.raw_sample_index import build_contract, OUT_DEFAULT
    contract = build_contract(out_path=contract_path or OUT_DEFAULT)
    # frozen model-init RNG discipline: snapshot all three RNG states, seed, build, restore
    rng_states = (_random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    torch.manual_seed(MODEL_INIT_SEED)
    ref3 = build_reference_3ch(weights)
    built = build_6ch(ref3, weights)
    _random.setstate(rng_states[0])
    np.random.set_state(rng_states[1])
    torch.random.set_rng_state(rng_states[2])
    m6, src_stem = built["model"], built["src_stem"]
    report = {
        "init": {"snapshot": str(built["snapshot"]),
                 "snapshot_sha256": built["snapshot_sha256"],
                 "source_stem_sha256": built["source_stem_sha256"],
                 "channel_semantics": CHANNEL_SEMANTICS},
        "G4": gate_g4(m6, src_stem),
        "G4b": gate_g4b(str(built["snapshot"]), src_stem),
        "G5": gate_g5(ref3, m6),
        "G6a_model_equiv": gate_g6a(ref3, m6),
        "G6b_rgb_pipeline_equiv": gate_g6b(contract),
        "G7": {g: gate_g7(contract, g, m6) for g in ("C0-N", "C1-I", "C2-D")},
    }
    report["all_passed"] = bool(
        report["G4"]["passed"] and report["G4b"]["passed"] and report["G5"]["passed"]
        and report["G6a_model_equiv"]["passed"] and report["G6b_rgb_pipeline_equiv"]["passed"]
        and all(v["passed"] for v in report["G7"].values()))
    out = Path("reports") / "step3_gates_g4_g7.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("G7", "init")}, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(report["init"], indent=2, ensure_ascii=False))
    for g, v in report["G7"].items():
        print(f"G7 {g}: passed={v['passed']} rel={v['relative_to_mean_rgb']}")
    print("ALL PASSED:", report["all_passed"], "->", out)
    return report


if __name__ == "__main__":
    run_all()
