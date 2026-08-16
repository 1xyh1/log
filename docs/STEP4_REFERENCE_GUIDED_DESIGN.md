# Step 4 — Reference-guided RGB-anchor feature fusion

> **历史探索文档，禁止作为当前执行协议。** 本文第 2～7 节记录了 F0 实现前的
> IdentityConcat/Prompt/QAF 候选，已被真实完成的 `Step4F0Model`（zero-init residual）
> 与冻结 F0 结论取代。当前下一阶段以
> `docs/step4_f1_ir_gate/` 为唯一设计和执行入口；不要按本文重写 F0 或堆叠模块。

## 1. 为什么不是三套完整 YOLO26s

RDTTrack 与成熟 YOLOv5 multispectral 都证明“模态专属处理 + feature interaction”有价值，但当前工程有三个约束：

- RGB 有强预训练；IR/Depth 没有等价预训练；
- 样例阶段数据极少；
- 4060 8GB，后续还要部署单模型。

第一版 Step4 因此采用：

```text
                         P3 ────────────────┐
RGB -> pretrained YOLO26 P4 ────────────────┼-> existing neck/head
                         P5 ────────────────┘
                         ↑      ↑      ↑
IR lightweight adapter ─┘      │      │
Depth+M adapter ────────────────┴──────┘
```

官方 YOLO26 backbone tap：layer 4 / 6 / 10，对应 P3/8、P4/16、P5/32。代码不只相信 YAML 注释，`inspect_yolo26_backbone_taps()` 会运行时确认 stride。

## 2. 历史 F0 候选：IdentityConcatFusion（未采用，禁止执行）

每个尺度先把 IR / Depth 适配到与 RGB 相同的 C，然后：

```text
cat(RGB, IR, D) -> 1x1 Conv(C*3 -> C)
```

卷积初始化为：

```text
[I, 0, 0]
```

所以 epoch0 输出严格等于 pretrained RGB feature；同时 aux kernel 从第一步就能拿到梯度。

这个比直接随机初始化一个 concat-conv 更适合做公平 baseline。

## 3. F1：ResidualPromptFusion

借 RDTTrack 的“auxiliary prompt 注入 RGB anchor”思想：

```text
RGB reduce ─┐
            ├-> spatial weighting -> prompt -> residual gate -> RGB + delta
Aux reduce ─┘
```

补丁把 residual gate 初始化为 0，保护 RGB 功能。注意：gate=0 时 prompt 子网第一步梯度受限，因此正式训练时应记录 gate 轨迹，并测试第一个 optimizer step 后 prompt path 是否开始得到梯度。

## 4. F2：两个 decorrelation 必须分开

### F2a StrictOrthogonalDecorrelation

严格按 channel-vector dot product 做 projection。

### F2b RDTTrackStyleDecorrelation

按 RDTTrack 源码的 elementwise-product + channel-norm 风格做。

不能把 F2b 在论文/答辩里写成“严格正交投影”。两者性能若不同，本身就是有价值的 ablation。

## 5. 历史 F3 / QAF 候选：SoftModalityGate（未进入当前路线）

不做 CSSA 式 hard switching。输出：

```text
w = softmax(learned_logits + quality_prior)
```

QAF 的 quality prior 只改变信任度：

```text
RGB: exposure/clipping/contrast/blur
IR : dynamic range/flatness
D  : valid ratio/hole/clipping/discontinuity
```

`SoftModalityGate(identity_start=True)` 在初始化时仍严格输出 RGB；质量门控只有经过训练、residual gate 打开后才参与实际融合。

## 6. 冻结纪律

Step4 Stage A：

```text
RGB backbone params frozen
RGB BN running stats frozen
IR adapter trainable
Depth adapter trainable
fusion trainable
neck/head：两种可分开的实验
  A1 freeze（只测 adapter/fusion 能否接入）
  A2 train（更接近 detector fine-tune）
```

RDTTrack 只需要冻结 ViT 参数；YOLO26 还有 BatchNorm，所以必须每次外层 `model.train()` 后重新把 frozen RGB 模块设为 eval。

## 7. 训练阶段建议

```text
Step4-F0-sample
  └─ identity concat P3/P4/P5

若 F0 > RGB control 且 N/Z/S 有一致方向
  ↓
F1 residual prompt
  ↓
F2a/F2b decorrelation
  ↓
F3 soft gate
  ↓
QAF quality prior
```

不要 F1+F2+F3 一次全堆，否则无法归因。

## 8. 正式集成前必须过的 tests

- P3/P4/P5 stride gate；
- identity output；
- aux gradient connectivity；
- zero/missing modality finite；
- frozen RGB params unchanged；
- frozen BN stats unchanged；
- gate weights sum1；
- quality prior monotonic；
- strict vs RDT-style decorrelation synthetic tests；
- export/inference shape contract（正式接 YOLO26 后再加）。
