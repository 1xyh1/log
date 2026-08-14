# Step 3-A 实现日志

约定：每改一板块代码，在本文件追加一节（时间 / 文件 / 改动后的完整代码段 / 验证结果）。
本文件由 git watcher（tools/auto_git_watch.py）自动同步到远端仓库 https://github.com/1xyh1/log。

---

## 2026-08-14 · 板块 1：数据层三文件 + 审计入口（初始创建）

### `src/multimodal/raw_sample_index.py`（新建）
18→17 数据契约、类别支持三分裂、group split（first_id_field_proxy_v1）、raw 文件 hash。
（初始版；板块 6 中升级为 G1a 全量格式审计 + G2 集合关系，见下。）

### `src/multimodal/modality_preprocess.py`（新建）
RGB 显式 BGR→RGB / IR median / Depth fixed-log float + valid-aware resize + NEAREST mask / pad 策略 / stateless flip。

### `src/multimodal/trimodal_dataset.py`（新建）
6ch 组装 + validator 元数据 + 确定性 sampler + collate。

### `scripts/audit_step3_data.py`（新建）
G1/G2/G3 入口。

（三文件初始全文见 git 历史首个 commit。）

---

## 2026-08-14 · 板块 2：P0-B 修正 — bboxes 契约改为 normalized xywh（letterbox 域）

原因（审阅者）：`v8DetectionLoss` 与 Validator 都对 batch["bboxes"] 内部做 `xywh2xyxy`，Dataset 输出 xyxy 会被二次解码。

`trimodal_dataset.py` 的 `_read_labels`（改为输出 letterbox 域 normalized xywh）：

```python
def _read_labels(path: Path, r: float, left: int, top: int, new_unpad: tuple,
                 imgsz: int) -> tuple[np.ndarray, np.ndarray]:
    """Label txt -> cls (N,) int64 + bboxes (N,4) normalized xywh [cx,cy,w,h] in the
    FINAL letterboxed 640x640 space (what v8DetectionLoss / validator expect: they
    apply xywh2xyxy internally). Flip later only maps cx -> 1 - cx."""
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    cls, boxes = [], []
    new_h, new_w = new_unpad
    for line in lines:
        if not line.strip():
            continue
        c, cx, cy, bw, bh = (float(x) for x in line.split()[:5])
        cx_lb = (cx * new_w + left) / imgsz
        cy_lb = (cy * new_h + top) / imgsz
        w_lb = bw * new_w / imgsz
        h_lb = bh * new_h / imgsz
        cls.append(int(c))
        boxes.append([cx_lb, cy_lb, w_lb, h_lb])
    if not cls:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    return np.asarray(cls, dtype=np.int64), np.asarray(boxes, dtype=np.float32)
```

`modality_preprocess.py` 的 `apply_flip`（xywh 语义，cx 取反）：

```python
def apply_flip(planes: list[np.ndarray], bboxes: np.ndarray) -> list[np.ndarray]:
    """Horizontal flip of all planes; bboxes are normalized xywh [cx,cy,w,h] -> cx = 1-cx."""
    out = [np.ascontiguousarray(p[:, ::-1]) for p in planes]
    bboxes[:, 0] = 1.0 - bboxes[:, 0]
    return out
```

---

## 2026-08-14 · 板块 3：P0-C 修正 — ratio_pad 改为 ((r,r),(left,top))

原因（审阅者）：`scale_boxes` 读 `ratio_pad[0][0]`（gain）与 `ratio_pad[1]`（(left,top)）。

```python
def letterbox_geometry(h: int, w: int, imgsz: int = 640) -> tuple[float, int, int, tuple[int, int]]:
    """Replicates ultralytics letterbox rounding.
    Returns (ratio, left, top, new_unpad (h,w)); right/bottom implied by imgsz-new_unpad."""
    shape = (imgsz, imgsz)
    r = min(shape[0] / h, shape[1] / w)
    new_unpad = (int(round(h * r)), int(round(w * r)))
    dh, dw = shape[0] - new_unpad[0], shape[1] - new_unpad[1]
    left, top = int(round(dw / 2 - 0.1)), int(round(dh / 2 - 0.1))
    return r, left, top, new_unpad


def letterbox_rgb(rgb_u8: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, tuple[tuple, tuple]]:
    """RGB uint8 -> letterboxed float32 [0,1] (pad 114/255).

    ratio_pad = ((r, r), (left, top)) — the exact format ultralytics scale_boxes
    expects (gain = ratio_pad[0][0], pad = ratio_pad[1] = (left, top)).
    """
    h, w = rgb_u8.shape[:2]
    r, left, top, new_unpad = letterbox_geometry(h, w, imgsz)
    resized = cv2.resize(rgb_u8, (new_unpad[1], new_unpad[0]), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), RGB_PAD_U8, dtype=np.uint8)
    canvas[top:top + new_unpad[0], left:left + new_unpad[1]] = resized
    ratio_pad = ((r, r), (left, top))
    return canvas.astype(np.float32) / 255.0, ratio_pad
```

---

## 2026-08-14 · 板块 4：P0-A 修正 — 6ch 快照本身为 nc=12 O2M（物理 head）

原因（审阅者）：`setup_model` 对传入 nn.Module 直接 return，`set_model_attributes` 只改 .nc/.names/.args 属性，**不会重建 12 类 Detect head**。Step 3 从构建起就是 nc=12。

`early_fusion_yolo26.py` 的 `build_reference_3ch`：

```python
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
    d = yaml_model_load("yolo26s.yaml")
    d["nc"] = nc
    d["end2end"] = False
    m = DetectionModel(d, ch=3, nc=nc)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    m.load(ckpt["model"].float().state_dict())  # DetectionModel.load takes a state_dict
    m.eval()
    return m
```

---

## 2026-08-14 · 板块 5：G1/G2/G3 强化（审阅第二轮 P0-1/2/3）

- **G1a**：18 个 raw group 全量真实格式审计（cv2.imread observed shape/dtype 进入 PASS 条件）：usable 17 需 RGB/IR 3ch uint8 1080×1920、Depth 2D uint16 1080×1920；00000008 需实测 360×640 且 depth 3ch uint8。`metric_format_passed` + `format_gate_passed` 入 contract。
- **G2**：新增 `split_relations`：sample_overlap_empty / group_proxy_overlap_empty / split_union_equals_usable / train_count=11 / val_count=6 / unique_count=17，全部满足才 PASS。
- **G3**：改为 **all17 全 17 张**、三组各一遍；新增 **aux padding band 严格为 0** 检查（按每张的 letterbox_geometry 计算 band，I/D/M 三通道 max_abs==0）。
- 修复 letterbox 解包顺序 bug（板块 3 改返回序 (r,left,top,new_unpad) 后，`letterbox_scalar`/`valid_aware_resize` 两处旧序 (r,top,left,...) 未同步 → 广播错误；已改为 `r, left, top, new_unpad`）。

`raw_sample_index.py` 新增（G1a 审计 + G2 集合关系）：

```python
METRIC_RGB_SHAPE = (1080, 1920)
EXCLUDED_EXPECTED = {"00000008": {"rgb": (360, 640, 3), "ir": (360, 640, 3),
                                  "depth": (360, 640, 3)}}


def audit_raw_format(raw: Path, sid: str) -> dict:
    """G1a: real cv2.imread format audit of one raw group (not a filename whitelist)."""
    def fmt(p):
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if a is None:
            return None
        return {"shape": list(a.shape), "dtype": str(a.dtype)}

    return {
        "rgb": fmt(next((raw / "visible").glob(f"{sid}.*"))),
        "ir": fmt(next((raw / "infrared").glob(f"{sid}.*"))),
        "depth": fmt(next((raw / "depth").glob(f"{sid}.*"))),
    }


def _metric_format_ok(f: dict) -> bool:
    rgb, ir, dep = f["rgb"], f["ir"], f["depth"]
    return bool(
        rgb is not None and ir is not None and dep is not None
        and rgb["dtype"] == "uint8" and len(rgb["shape"]) == 3 and rgb["shape"][2] == 3
        and ir["dtype"] == "uint8" and len(ir["shape"]) == 3 and ir["shape"][2] == 3
        and dep["dtype"] == "uint16" and len(dep["shape"]) == 2
        and rgb["shape"][:2] == list(METRIC_RGB_SHAPE)
        and ir["shape"][:2] == list(METRIC_RGB_SHAPE)
        and dep["shape"] == list(METRIC_RGB_SHAPE))
```

build_contract 中（audit + 排除组实测 + 集合关系）：

```python
    excluded_present = {sid: sid in raw_ids for sid in EXCLUDED}
    # G1a: full format audit of ALL raw groups (observed fact, not filename whitelist)
    raw_format_audit = {sid: audit_raw_format(raw, sid) for sid in raw_ids}
    metric_ok = {sid: _metric_format_ok(raw_format_audit[sid]) for sid in raw_ids}
    for sid in EXCLUDED:
        f = raw_format_audit[sid]
        expected = EXCLUDED_EXPECTED[sid]
        if not (f["rgb"]["shape"] == list(expected["rgb"])
                and f["ir"]["shape"] == list(expected["ir"])
                and f["depth"]["shape"] == list(expected["depth"])):
            raise RuntimeError(f"excluded {sid} observed format changed: {f} vs {expected}")
    bad_metric = sorted(sid for sid, ok in metric_ok.items()
                        if not ok and sid not in EXCLUDED)
    if bad_metric:
        raise RuntimeError(f"non-excluded ids failing metric-format audit: {bad_metric}")
    usable = sorted(sid for sid in ids_with_all_modalities if sid not in EXCLUDED)
```

```python
    # G2: sample-level AND group-proxy-level split checks + set relations
    train_groups = sorted(set(group_of(s) for s in split_ids["train"]))
    val_groups = sorted(set(group_of(s) for s in split_ids["val"]))
    intersection = sorted(set(train_groups) & set(val_groups))
    sample_overlap = sorted(set(split_ids["train"]) & set(split_ids["val"]))
    union_equals_usable = set(split_ids["train"]) | set(split_ids["val"]) == set(usable)
    missing_from_split = sorted(set(usable) - (set(split_ids["train"]) | set(split_ids["val"])))
    split_relations = {
        "sample_overlap_empty": len(sample_overlap) == 0,
        "group_proxy_overlap_empty": len(intersection) == 0,
        "split_union_equals_usable": bool(union_equals_usable),
        "missing_from_split": missing_from_split,
        "train_count": len(split_ids["train"]),
        "val_count": len(split_ids["val"]),
        "unique_count": len(set(split_ids["train"]) | set(split_ids["val"])),
    }
    split_gate_passed = bool(split_relations["sample_overlap_empty"]
                             and split_relations["group_proxy_overlap_empty"]
                             and split_relations["split_union_equals_usable"]
                             and split_relations["train_count"] == 11
                             and split_relations["val_count"] == 6
                             and split_relations["unique_count"] == 17)
```

contract 新增字段：`raw_format_audit` / `metric_format_passed` / `format_gate_passed` / `split_relations`。

`audit_step3_data.py` 的 G3（all17 + aux pad band）：

```python
def check_g3(group: str, contract: dict) -> dict:
    ds = TriModalDataset(contract, split="all17", group=group, augment=False)
    samples = [ds[i] for i in range(len(ds))]
    img = torch.stack([torch.as_tensor(s["img"]) for s in samples])  # (17, 6, 640, 640)
    out = {"group": group, "n_images": len(samples), "shape": list(img.shape),
           "finite": bool(torch.isfinite(img).all())}
    rgb, i, d, m = img[:, 0:3], img[:, 3], img[:, 4], img[:, 5]
    out["rgb_range"] = [float(rgb.min()), float(rgb.max())]
    out["rgb_pad_value_present"] = bool((rgb - 114 / 255).abs().min() < 1e-4)
    out["i_range"] = [float(i.min()), float(i.max())]
    out["d_range"] = [float(d.min()), float(d.max())]
    out["m_unique"] = sorted(float(v) for v in torch.unique(m))[:8]
    out["m_binary"] = bool(set(torch.unique(m).tolist()) <= {0.0, 1.0})
    out["m0_implies_d0"] = bool((d[m == 0].abs().max() < 1e-6))
    ok = (out["finite"] and out["rgb_range"][0] >= 0 and out["rgb_range"][1] <= 1
          and out["i_range"][0] >= 0 and out["i_range"][1] <= 1
          and out["d_range"][0] >= 0 and out["d_range"][1] <= 1
          and out["m_binary"] and out["m0_implies_d0"])
    # aux pad-band strictly zero (letterbox geometry from each sample's ori_shape;
    # all 17 share 1080x1920 - verified per sample)
    for idx, s in enumerate(samples):
        h, w = s["ori_shape"]
        r, left, top, new_unpad = mp.letterbox_geometry(h, w, 640)
        band = np.ones((640, 640), dtype=bool)
        band[top:top + new_unpad[0], left:left + new_unpad[1]] = False
        for ch, name in ((3, "I"), (4, "D"), (5, "M")):
            v = float(img[idx, ch][band].abs().max())
            out.setdefault(f"aux_pad_{name}_max_abs", 0.0)
            out[f"aux_pad_{name}_max_abs"] = max(out[f"aux_pad_{name}_max_abs"], v)
            ok = ok and v == 0.0
    if group == "C0-N":
        aux_zero = bool(i.abs().max() == 0 and d.abs().max() == 0 and m.abs().max() == 0)
        out["aux_strictly_zero"] = aux_zero
        ok = ok and aux_zero
    if group == "C1-I":
        out["i_nonzero"] = bool(i.abs().max() > 0)
        out["d_m_zero"] = bool(d.abs().max() == 0 and m.abs().max() == 0)
        ok = ok and out["i_nonzero"] and out["d_m_zero"]
    if group == "C2-D":
        out["i_zero"] = bool(i.abs().max() == 0)
        out["d_nonzero"] = bool(d.abs().max() > 0)
        out["m_nonzero"] = bool(m.abs().max() > 0)
        ok = ok and out["i_zero"] and out["d_nonzero"] and out["m_nonzero"]
    out["passed"] = bool(ok)
    return out
```

**G1/G2/G3 重跑结果（P0-1/2/3 修正后）**：

```
G3 C0-N: passed=True shape=[17, 6, 640, 640] m_unique=[0.0] rgb_pad114=True
G3 C1-I: passed=True shape=[17, 6, 640, 640] m_unique=[0.0] rgb_pad114=True
G3 C2-D: passed=True shape=[17, 6, 640, 640] m_unique=[0.0, 1.0] rgb_pad114=True
ALL PASSED: True -> reports\step3_gates_g1_g2_g3.json
```

---

## 2026-08-14 · 板块 6：G5 三尺度 / G4b 扩展 / model_init_seed / G6b 官方 LetterBox（审阅第二轮 P1）

`early_fusion_yolo26.py`：

```python
MODEL_INIT_SEED = 2026081200  # frozen: 12-class head random init must be reproducible


def _head_nc_physical(model) -> list:
    """Physical cls-head out channels of ALL THREE scales (cv3 branches)."""
    head = model.model[-1]
    out = []
    for branch in head.cv3:
        last = branch[-1]
        conv = last.conv if hasattr(last, "conv") else last
        out.append(int(conv.out_channels))
    return out


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
```

G4b 扩展（训练入口同路径 reload 的完整身份检查）：

```python
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
```

run_all 的 RNG 纪律（构建前 snapshot 三套 RNG → seed → 构建 → 恢复）：

```python
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
```

快照 meta 增加 `"nc": 12, "model_init_seed": MODEL_INIT_SEED`。

G6b 改为官方 LetterBox transform 直接参考（不经过 YOLODataset/label cache）：

```python
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
```

**修复**：letterbox 解包顺序 bug——`letterbox_scalar` 与 `valid_aware_resize` 两处由 `r, top, left, new_unpad` 改为 `r, left, top, new_unpad`（与 letterbox_geometry 返回序一致）；修复前 G3 广播错误 (360,500) vs (360,640)。

## 待办（下一步）
1. 重跑修正后 G1–G7（G5/G4b/G6b/seed 修正后；旧 nc80 结果全部作废）
2. Step 1/2 checkpoint head-nc 实测审计（6 个 last.pt 的物理 head nc）→ closing report
3. 三个 roundtrip 测试（head_nc12 / bbox overlay / ratio_pad roundtrip）
4. Runner（float 直通 Trainer/Validator、plots=False/cls_pw=0、batch fail-fast）+ G8 真实接管
5. 三组 1-epoch smoke → seed12 × 80

---

## 2026-08-14 · 板块 7：修正后 G1–G7 全部重跑 PASS（旧 nc80 结果作废）

修复链：`DetectionModel.load` 需完整 ckpt dict（非裸 state_dict）；8.4.56 的 DetectionModel 构建时**不设 .nc/.args 属性**（由训练时 set_model_attributes 设置）→ 构建 reference 时显式 `m.nc = nc; m.names = ...` 使快照自描述；G1a 格式审计兼容 ultralytics-patched cv2（IMREAD_UNCHANGED 可能返回 (H,W,1)，squeeze 处理）。

**G4-G7 最终结果（reports/step3_gates_g4_g7.json）**：
- G4：RGB 部分与 source stem 逐位相等、aux 部分 max_abs=0.0 ✓
- G4b：reload 后 input_conv_in_channels=6、physical_head_nc=[12,12,12]、model/head nc=12、meta nc=12、init_seed=2026081200、source stem sha 一致、O2M、forward OK ✓
- G5：ref3 与 m6 的 yaml/model/head end2end 全 False；nc 属性 12；**物理 cls-head 三尺度 out_channels = [12,12,12]** ✓
- G6a：raw conv / conv+BN+SiLU / 最终 detector 输出 max_abs_diff **全为 0.0**（≤1e-5）✓
- G6b：17 张全测，与官方 LetterBox transform 逐像素一致（max_abs_diff=0.0）✓
- G7：C0-N rel=[0.912, 1.036, 1.052, **0, 0, 0**]；C1-I rel=[..., **1.546**, 0, 0]；C2-D rel=[..., 0, **2.043, 2.080**] —— inactive 通道梯度精确 0、active 通道为正 ✓
- 快照：`step3_6ch_rgb_equiv_init.pt`，SHA256=594e1754...；source stem SHA256=25d9e6b8...
- G1/G2/G3（板块 5 强化后）：all17 全 17 张三组语义全过、aux pad band 严格 0、18→17 格式审计与集合关系全过 ✓

---

## 2026-08-14 · 板块 8：Step 1/2 checkpoint 物理 head 实测审计（纠正板块 4 的错误推断）

**实测结果（runs/step2_modality/step2_head_nc_audit.json）**：B0-D / B0-G / B1-A / B1-B / B2-A / B2-B 六个 last.pt 全部 **model_nc=12、head_nc=12、物理 cls-head 三尺度 out_channels=[12,12,12]**。

**纠正**：板块 4 中"Step 1 (B0-*) trained 80-class physical heads"是从 Trainer 源码推断的表述，**实测不成立**——Ultralytics 8.4.56 在 YOLO 对象训练流程中实际重建了 12 类物理 head。Step 3 的 nc=12 快照构建方式（DetectionModel yaml nc 覆盖 + 显式 m.nc）与实测的 Step-1/2 检查点状态一致，P0-A 修正仍然正确（防御性显式构建），但"Step 1 是 80 类"的说法作废。结论口径：**Step 1/2 与 Step 3 全部为 12 类物理 head**。
