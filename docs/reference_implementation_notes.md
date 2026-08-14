# Reference implementation notes — RDTTrack × YOLOv5 multispectral × YOLO26

> 目的：不是把参考仓库“搬进”当前工程，而是把已经验证过的多模态工程模式拆成可审计的设计决策。本文固定参考快照：
>
> - 本项目镜像：`1xyh1/log@82655b7ef78efa116533b53ea919ec6bbf9dbe57`
> - RDTTrack：`xuefeng-zhu5/RDTTrack@794de41ff3f52ed100e4f449cf22ceb4932e0d36`（MIT）
> - YOLOv5 multispectral：`DocF/multispectral-object-detection@fb591c9b163177c0e950db08e213e24ddc912d41`（AGPL-3.0）
> - Ultralytics YOLO26 architecture reference：`ultralytics/ultralytics@0449ea011cfd6c9a0d50a0bf1043aca5190cd476`
>
> 本补丁中的实现均为针对当前 YOLO26 工程重新编写的独立实现；没有复制 AGPL 仓库的大段实现代码。

---

## 0. 先定位当前 Step 3 的问题

### 0.1 “R3 不收敛”目前不是最合理的主诊断

保留下来的 C1-I / C2-D 80 epoch 曲线表明训练主链能正常学习：

- C1-I：`mAP50-95` 从 epoch1 的约 0.0386，上升并在后期稳定在约 0.24–0.25，epoch80 约 0.2539；train loss 从约 1.79 降到约 0.34。
- C2-D：epoch80 `mAP50-95` 约 0.2106，后期同样没有“归零式崩溃”。

因此现阶段不应因为 C0 的一条异常日志就改 R3 recipe。当前优先级应是：

1. 修复 post-hoc evaluator；
2. 判定 C0 正式产物是否被 smoke 覆盖；
3. 恢复一个不可覆盖的 C0 formal/recovery run；
4. 再判断三组的真实差异。

### 0.2 evaluator 有确定的 xywh→xyxy 几何 bug

当前 `scripts/eval_step3_causality.py` 先修改 `x1/y1`，随后又使用被修改后的值构造 `x2/y2`，导致 GT 框系统性缩小。训练时 stock validator 对 C1-I 报约 0.25，而 post-hoc evaluator 对同一模型给接近 0，这种量级的冲突与该 bug 完全一致。

修复原则：**不再复制 validator 的几何/NMS/匹配语义**。新的 evaluator 直接调用 Ultralytics `DetectionValidator` 的：

- `postprocess()`
- `_prepare_batch()`
- `_prepare_pred()`
- `_process_batch()`

我们只保留“6ch float 已归一化，所以不 `/255`”这一处项目特有逻辑。

### 0.3 C0 正式产物已经发生 provenance 混合

当前镜像中：

- `C0-N/results.csv` 只有 1 个 epoch；
- 但已有 `eval_step3_causality.json` 的 `late10` 却包含 10 epoch 统计；
- 说明目录中至少混合了“旧 formal 派生产物”和“新 1-epoch smoke 训练产物”。

这比“C0 训练崩了”更严重：**控制组现在不可作为控制组使用**。

本补丁增加 run integrity gate：formal 评估前必须检查 `args.yaml / results.csv / G8 / kernel growth / weights / eval provenance` 是否同源。

### 0.4 G8 当前记录的是计划 sampler，而不是实际 DataLoader yield

旧 runner 记录 `dataset.sampler.perm` 的 hash。它能证明“预期 schedule 一致”，但没有记录真正进入 `preprocess_batch` 的 `sample_id`。

在 `workers=0`、每 epoch 完整消费 sampler 的当前设置下，实际顺序大概率与计划一致；但正式证据应该记录真实 batch yield。本补丁把 `sample_id` / `flip_applied` 从 batch 中直接采集，并在 epoch 结束时：

- 与 expected permutation 对比；
- mismatch 立即报错；
- 保存 `actual_order_sha256` / `actual_flip_sha256`。

---

# 1. RDTTrack：逐文件拆解

## 1.1 `lib/models/rdtt/vit_ce_prompt.py`

### 源码在做什么

核心有两个模块：

1. `DepthIR_ort_block`
   - Depth、TIR 各自经过 1×1 projection；
   - hidden channel 很小（默认 8）；
   - 对两路辅助特征做“去相关”式处理；
   - concat 后 1×1 投影回 backbone embedding 维度。

2. `Prompt_block`
   - RGB feature 与 D/T 融合 feature 拼接；
   - 再做窄通道 1×1 bottleneck；
   - 经过类似 spatial softmax/Fovea 的重加权；
   - 输出 prompt，残差加回 RGB token。

`rdtt_deep` 还会在 transformer 的多个深度重复注入 prompt。

### 可以直接借什么

**可以直接借工程原则，不直接复制实现：**

- RGB 预训练路径作为 anchor；
- 辅助模态先在轻量 adapter 中处理；
- fusion 输出以 residual/prompt 方式注入 RGB，而不是一开始就把三路主干同权重训练；
- auxiliary bottleneck 很窄，适合小数据和显存受限场景。

### 只能借思想的部分

`DepthIR_ort_block` 名字叫 orthogonal，但源码并不是严格的向量投影：它使用逐元素乘法配合 channel L2 norm，而不是 `(x·y / ||y||²) y`。

因此 Step 4 不应该把它原样叫“正交投影”。本补丁提供两个**明确分开的**实验模块：

- `StrictOrthogonalDecorrelation`：数学上严格的 channel-vector projection；
- `RDTTrackStyleDecorrelation`：复现其运算思想，但名字明确写 `Style`。

两者必须做独立 ablation。

### 对应当前工程文件

| 参考点 | 当前/新增文件 | 阶段 |
|---|---|---|
| RGB anchor + aux prompt | `src/multimodal/reference_fusion_blocks.py` | Step4 |
| D/T 去相关 | `StrictOrthogonalDecorrelation` / `RDTTrackStyleDecorrelation` | Step4 |
| residual prompt | `ResidualPromptFusion` | Step4 |
| quality-conditioned prompt/gate | `SoftModalityGate` | QAF |

### 必须新增的 unit test

- identical inputs：strict projection 应显著压低相关分量；
- synthetic orthogonal inputs：strict projection 应基本保留原分量；
- zero input：无 NaN/Inf；
- RDTTrack-style zero input：无 NaN/Inf；
- shape preservation；
- backward finite；
- identity-safe fusion 初始化输出必须逐位等于 RGB。

---

## 1.2 `lib/models/rdtt/ostrack_prompt.py`

### 源码在做什么

- 先构建 prompt-capable backbone；
- 再用 pretrained OSTrack `strict=False` 加载；
- 新增 prompt/auxiliary 模块自然落入 missing keys；
- detector/tracker 主能力由预训练 RGB 模型提供。

### 可直接借

**“先构建目标架构，再显式 transfer 预训练 RGB 参数”** 的初始化纪律。

这与我们 Step3 已经采用的：

`3ch nc=12 O2M reference -> 6ch function-preserving stem`

是一致的。

### 不直接借

不能使用 `strict=False` 后仅打印 missing keys 就结束。我们现有 G4/G4b/G5/G6 的做法更严格：

- 物理 head shape；
- stem bitwise identity；
- reload integrity；
- final output equivalence。

Step4 也应沿用这种 gate，而不是退回“加载成功即可”。

---

## 1.3 `lib/train/base_functions.py`

### 源码在做什么

当 prompt type 是 RDTTrack 时，仅把名字包含 `prompt` 或 `DepthIR_ORT` 的参数放入 optimizer，其余参数 `requires_grad=False`。

### 可以直接借

这是 Step4 第一阶段最值得借的训练策略：

```text
Stage 4-A:
  freeze RGB backbone
  train IR adapter + Depth adapter + fusion only

Stage 4-B:
  若有稳定收益，再低 LR 解冻 RGB backbone 后半段
```

### 我们必须比参考实现多做一步

YOLO26 backbone 是 Conv + BatchNorm。只把参数 `requires_grad=False`，如果外层 `model.train()`，BN 的 running mean/var 仍可能变化。

因此新增 `src/multimodal/trainability.py`：

- 参数冻结；
- 冻结模块强制 `.eval()`；
- 冻结 BN running stats；
- unit test 验证一次 optimizer step 后 frozen RGB 参数和 BN stats 都完全不变。

---

## 1.4 `lib/train/dataset/rgbdt.py`

### 源码在做什么

- RGB、Depth、IR 同一个 frame index 对齐；
- Depth 被 per-image min-max normalize 到 8-bit；
- 再变为 JET colormap；
- 最终打包成 9ch `[RGB, depth_colormap, IR]`。

### 只能借什么

只借：**一个 sample/frame ID 必须唯一锚定所有模态**。

### 明确不能借什么

Depth 处理不能搬：

- per-image min-max 破坏毫米尺度的一致物理意义；
- JET colormap 引入人为色彩结构；
- invalid depth 没有显式 binary mask。

本项目当前 `log-depth + valid mask + validity-aware resize` 更适合赛题，不应回退。

---

## 1.5 `experiments/rdtt/baseline.yaml`

值得借的不是具体 LR，而是配置表达出的训练纪律：

- pretrained RGB model；
- prompt-only training；
- `FIX_BN: true`；
- adapter/fusion 先稳定，再谈全网 fine-tune。

Step4 设计文档应明确保留这三个 gate：

1. `rgb_frozen_parameters_unchanged`
2. `rgb_bn_running_stats_unchanged`
3. `only_declared_modules_in_optimizer`

---

# 2. YOLOv5 multispectral：逐文件拆解

## 2.1 `models/transformer/yolov5l_fusion_transformerx3_llvip.yaml`

### 源码在做什么

这是最值得参考的结构文件：

- RGB stream 与 IR stream 两套 backbone；
- 在 P3/8、P4/16、P5/32 三个尺度分别做 cross-modal transformer；
- transformer 输出以 residual 形式加回各自 stream；
- 最后每尺度再融合；
- 标准 YOLO neck/head 消费融合后的 P3/P4/P5。

### 可以直接借的架构边界

**融合放在 detector backbone 的 P3/P4/P5，而尽量不改 detector neck/head。**

YOLO26 官方 YAML 对应：

- backbone layer 4：P3/8
- backbone layer 6：P4/16
- backbone layer 10：P5/32
- Detect 使用 neck 输出 layer 16/19/22。

因此 Step4 首选插点就是 `[4, 6, 10]`。

### 不应直接复制

- 两个完整 backbone 对当前 4060/小样本过重；
- GPT block 计算量与数据量不匹配；
- 两模态固定逻辑不能自然扩展到 RGB/IR/Depth。

我们的第一版 Step4 应是：

```text
RGB pretrained YOLO26 backbone
        │ P3/P4/P5
        │
IR lightweight adapter ─┐
Depth+M lightweight ────┼─> fusion per scale -> existing YOLO26 neck/head
                        ┘
```

不是三套完整 YOLO26 backbone。

---

## 2.2 `models/common.py` — `GPT`, `Add`, `Add2`

### GPT 的关键实现

- 先把两路 feature adaptive-pool 到固定 8×8；
- 把 RGB/IR token concat；
- transformer 处理；
- 分成两路；
- resize 回原 feature map；
- `Add2` 把 cross-modal result residual 加回各 stream。

### 可以借

- **先降空间分辨率再做昂贵 cross-modal interaction**；
- residual injection；
- 保留原 backbone skip topology。

### 只能借思想

GPT 不能作为 Step4 第一基线。正确实验顺序：

1. `IdentityConcatFusion`（1×1，function-preserving）
2. `ResidualPromptFusion`
3. orthogonal/decorrelation ablation
4. soft gate
5. 只有前面证明有价值，才考虑更重的 token attention。

---

## 2.3 `models/yolo_test.py`

### 源码在做什么

为了双输入，它修改 Model API 为 `forward(x, x2)`，并在 YAML graph parser 中用特殊 `from=-4` 表示第二路输入。

### 明确不借

这个方案与旧版 YOLOv5 parser 强耦合，移植到 Ultralytics 8.4.56 会造成：

- parser magic number；
- export / trainer / validator 不透明；
- 很容易重复我们 Step3 已踩过的 wrapper/trainer 重建问题。

Step4 应写明确的 wrapper/feature-tap model，而不是修改 Ultralytics parser 的特殊索引语义。

---

## 2.4 `utils/datasets.py` — `LoadMultiModalImagesAndLabels`

### 可以直接借的 invariant

- RGB/IR 使用同一个 sample index；
- 同一个 letterbox geometry；
- flip 同步；
- collate 后仍保持 paired sample。

这与当前 `TriModalDataset` 的基本方向完全一致。

### 参考代码里值得警惕的地方

它对 RGB 和 IR 都调用 HSV augmentation。对热红外而言，这是 RGB 色彩空间假设，不适合我们的数据。

当前项目应该继续坚持：

- RGB photometric aug 与辅助模态 photometric aug 分开定义；
- 空间变换共享；
- modality-specific interpolation/pad。

---

## 2.5 `load_mosaic_RGB_IR` / `random_perspective_rgb_ir`

### 最有价值的实现思想

它显式断言 RGB/IR index 一致，并为两路使用同一个 mosaic tile placement / affine matrix。

这个原则可以直接借到后续增强，但三模态版本必须改写成“transform plan”：

```text
sample IDs
mosaic slots
crop rectangle
flip bit
affine matrix
```

只随机生成一次，然后分别让：

- RGB：linear + RGB pad
- IR：linear + aux pad0
- Depth：validity-aware warp
- Mask：nearest
- boxes：同一个矩阵

执行。

**Step3 当前不要重新开启 mosaic**；先修 evaluator 与 C0 provenance。若以后需要 small-data rescue，新增 matched-mosaic 是一个独立 recipe，不应悄悄改 R3。

---

# 3. 对照当前 `1xyh1/log`：哪些保留，哪些要改

| 当前文件 | 当前状态 | 参考实现给出的结论 | 本补丁动作 |
|---|---|---|---|
| `modality_preprocess.py` | 物理语义清晰 | 比两份参考的 Depth/IR preprocessing 更适合本赛题 | 保留 |
| `trimodal_dataset.py` | shared geometry + stateless flip | 与成熟 paired loader 原则一致 | 增加 `flip_applied` 真实证据 |
| `early_fusion_yolo26.py` | function-preserving 6ch init | 与 RDTTrack 的 pretrained anchor 思路一致 | Step3 保留 |
| `run_step3_earlyfusion.py` | float/no-/255 正确；目录可覆盖；G8 是计划 hash | 参考工程强调明确 train path；需要更强 provenance | formal immutable + actual-yield G8 |
| `eval_step3_causality.py` | 有手工 GT conversion bug | 不应复制 validator 逻辑 | 改为调用 stock validator primitives |
| `summarize_step3.py` | 可读 stale/mixed artifact | formal control 必须可信 | 加 run integrity gate |
| 新增 `run_integrity.py` | — | 实验工程必须可追溯 | Step3 |
| 新增 `reference_fusion_blocks.py` | — | RDTTrack + YOLOv5 的 Step4 候选 | Step4/QAF，仅隔离实现 |
| 新增 `trainability.py` | — | RDTTrack freeze discipline | Step4 |

---

# 4. Step3 / Step4 / QAF 的边界

## Step3：只修证据链，不换架构

现在应该完成：

1. C1-I / C2-D 用修正版 evaluator 重评；
2. C0 当前目录应被 run integrity 判 FAIL；
3. 重新跑一个独立 `C0-N-recovery-*`；
4. 使用真实 provenance 的 C0/C1/C2 做 NORMAL/ZERO/SHUFFLE；
5. 再决定 Step3 结论。

禁止在这一步加入：

- prompt；
- orthogonal block；
- P3/P4/P5 fusion；
- quality gate。

否则 Step3 的“输入级辅助信息是否有增益”问题被改变。

## Step4：reference-guided feature fusion

建议顺序：

### F0 — IdentityConcatFusion

`[RGB, IR_adapter, D_adapter] -> 1×1 Conv`

初始化为 `[I,0,0]`，因此输出逐位等于 RGB feature，但 auxiliary kernel 从第一步就有梯度。这是最干净的 mid-fusion baseline。

### F1 — ResidualPromptFusion

RDTTrack 思想：辅助特征产生 prompt，残差注入 RGB。

### F2a/F2b — decorrelation

- strict orthogonal；
- RDTTrack-style；

必须单独比较。

### F3 — SoftModalityGate

把 CSSA/EvaNet 类“模态可靠性”思想放进 soft gate；不做 hard switch。

## QAF：质量先验只作为 gate prior

质量指标不要直接替换视觉特征，建议变成 gate logits 的 prior：

```text
RGB quality: exposure / clipping / contrast / blur
IR quality : dynamic range / flatness / saturation
Depth      : valid ratio / hole ratio / clipping / discontinuity
              ↓
          qR qI qD
              ↓
softmax(feature logits + quality prior)
```

这样“质量”只调节信任度，不抢 detector 的语义建模权。

---

# 5. 新增 unit tests 清单

## Step3 必须新增

1. `test_eval_gt_conversion_matches_stock_prepare_batch`
   - synthetic normalized xywh；
   - 项目 evaluator 和 stock `_prepare_batch` 输出逐位一致。

2. `test_eval_perfect_prediction_scores_perfect_iou`
   - prediction = GT；
   - IoU50:95 应全部匹配。

3. `test_run_integrity_rejects_overwritten_formal`
   - args epochs=1 / expected80；
   - 必须 FAIL。

4. `test_run_integrity_rejects_stale_eval_hash`
   - 修改 results.csv 后旧 eval provenance 必须失效。

5. `test_actual_loader_order_is_recorded`
   - 不是仅比较 sampler.perm；
   - 从 batch `sample_id` 收集实际 yield。

6. `test_shuffle_map_is_bijection_and_no_self`
   - D/M donor 成对移动；
   - donor 不重复；
   - 无 self donor；
   - 能跨 proxy group 时必须跨组。

## Step4 必须新增

7. `test_identity_concat_is_exact_rgb_identity`
8. `test_identity_concat_aux_kernel_gets_gradient`
9. `test_strict_orthogonal_identical_input_reduces_component`
10. `test_strict_orthogonal_true_orthogonal_input_preserved`
11. `test_decorrelation_zero_input_finite`
12. `test_residual_prompt_identity_initialization`
13. `test_soft_gate_weights_sum_to_one`
14. `test_soft_gate_quality_prior_monotonic`
15. `test_missing_modality_zero_is_finite`
16. `test_frozen_rgb_parameters_unchanged_after_step`
17. `test_frozen_rgb_bn_stats_unchanged_after_train_call`
18. `test_yolo26_backbone_taps_are_p3_p4_p5`
   - layer 4/6/10；
   - stride 8/16/32；
   - 不硬编码 channel，运行时记录并校验。

---

# 6. 这版代码为什么不直接实现“三路完整 backbone”

参考源码能证明“多尺度 feature fusion”值得做，但不能证明“三个完整 YOLO26s backbone”是本项目的最优第一步。

当前约束更适合：

- 4060 8GB；
- sample probe 只有 17 个 usable group；
- RGB 有强预训练，IR/Depth 没有等价预训练；
- 比赛最终要求单模型单推理链。

所以先做 RGB anchor + lightweight auxiliary adapters，信息增益成立后再增加容量，是风险最小、归因最清楚的路径。

---

# 7. 立即执行顺序

```text
A. 应用本补丁
B. pytest Step3 evaluator/run-integrity tests
C. validate C0/C1/C2 current runs
   - C0 应 FAIL（被覆盖）
   - C1/C2 应 training-complete，legacy eval provenance warning
D. 修正版 evaluator 重评 C1/C2
E. 新目录恢复 C0 formal/recovery
F. 修正版 evaluator 重评 C0
G. summarize
H. 只有 Step3 结论冻结后，才启动 Step4 F0
```

这条顺序的核心是：**先把实验事实修正，再让参考论文影响下一阶段架构；不要用架构升级掩盖一个 evaluator/provenance bug。**
