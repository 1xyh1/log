# F1-C 设计冻结：magnitude-aware scalar gate（幅度门控）

## 1. 要回答的问题

F1-C-A1 诊断已证明：**显式 log-RMS 幅度轴与 q 抑制收益（AP(q0)−AP(q1)）稳定
相关**（last/best 的 tie-aware Spearman +0.70~+0.98，leave-one-family-out 全正）。
B1 的 learned gate 学到近似常数 q，未把输入条件化。

F1-C 只回答：**给 gate 显式加入逐样本 log-RMS 幅度输入后，gate 能否学到输入
条件化的可靠性并稳定转化为 matched-control 净收益？**

## 2. A1 结论的准确表述（审阅者修正，不得写错）

- **不能写**"当前 LayerNorm 输入几乎不携带幅度信息"。
- **正确结论**：joint-LN 的分尺度标量摘要存在信息，但**方向随尺度冲突**
  （P3 last −0.548/best −0.649；P4 −0.548/−0.393；P5 +0.792/+0.769）；
  现有 gate 没有把这些信息转化为输入条件化的可靠性控制。A1 证明的是显式
  log-RMS 幅度轴与 q 抑制收益稳定相关，不是证明完整 LN 向量"没有信息"。

## 3. 结构（等价形式，保护旧初始化）

```text
z = LayerNorm(concat(GAP(A3), GAP(A4), GAP(A5)))
m = [logRMS(A3), logRMS(A4), logRMS(A5)]
h = old_fc1(z) + magnitude_fc(m)      # magnitude_fc: 无 bias、zero-init
q = sigmoid(old_fc2(SiLU(h)))
F_i = R_i + q * P_i(A_i)              # 残差结构不变
```

硬约束：
- `logRMS` 逐样本计算，沿 (C,H,W)，不跨 batch；公式
  `log(sqrt(mean(A²)) + 1e-9)`（与 A1 audit 一致）。
- `magnitude_fc` 无 bias、权重 zero-init。
- 旧 `norm/fc1/fc2` 的初始化与 RNG 顺序与原 gate 完全一致 → 新分支为 0 时
  q 与旧 gate 初始输出逐位等价。
- gate 输入继续 detach（G10.5）；RGB anchor、projection、residual、P5 路由、
  B1 corruption schedule 全部不变。
- **不加入** relative energy、Depth、spatial cosine、Transformer/QAF。

## 4. 组（formal 四组；smoke 三组）

| 组 | aux | gate 模块 | effective q | 作用 |
|---|---|---|---|---|
| `F1C-C0` | `[0,0]` | magnitude | learned | null control（含 corruption schedule） |
| `F1C-I-fixed` | `[I,0]` | **magnitude** | fixed 1 | matched 未门控（同模块结构） |
| `F1C-I-magsoft` | `[I,0]` | magnitude | learned | treatment |
| `F1C-I-soft` | `[I,0]` | original | learned | **同代码链 original-gate matched control（formal 必跑）** |

历史 B1-soft 0.304028 仅作辅助外部基线，不能替代新链 matched base。

## 5. 门禁

G1–G9 沿用 F1-B（RGB 等价、zero-init、梯度解锁、P5 路由、optimizer 成员、
训练后更新证据（C0 proj 精确零）、G8 actual yield、G9 corruption trace），
C0/fixed 阈值 epoch 缩放仅 smoke。

**G10（新增，七项）**：
1. 单样本 log-RMS 与 batch 内同一样本完全一致；
2. batch permutation 不改变各样本 descriptor；
3. zero-init magnitude 分支下，新旧 gate q 初始逐位等价；
4. `magnitude_fc.weight` 梯度有限且非零，受控更新后离开 0；
5. gate→aux 梯度仍为 0，residual→aux 梯度仍非零；
6. q 始终有限的 B×1；RGB 与其他模态输入不被修改；
7. runner 在半精度 checkpoint 序列化前记录 final fp32 RGB SHA
   （`step4_fp32_rgb_sha.json`）。

**G11（reviewer 2026-08-17 裁决，外部运行依赖闭包）**：formal 构模的外部
运行依赖必须进入 readiness freshness 闭包（manifest v2 / audit v3 /
readiness v2）——base checkpoint 文件 SHA（`EXPECTED_BASE_CHECKPOINT_SHA256`，
`E:/odin/yolo26s.pt`，`646f8bc3…a1b`）、builder 源码
（`early_fusion_yolo26.py`，三处 pin 表）、17×4 原始数据重 hash 与
`contract["file_hashes"]` 比对、dataset.yaml 语义锁（nc=12 + names ==
CLASS_NAMES）；formal 构模后、Trainer 创建前，5 个 initial state SHA 与
smoke 冻结值逐位相同。**任一外部依赖变更强制完整重跑链
（pytest → audit → smoke → readiness），不单独豁免。**

## 6. 晋级要求（在 B1 条件之外）

- 新 magsoft 超过历史 B1-soft last **0.304028**（辅助外部基线）；
- 超新链 C0、fixed、ZERO、SHUFFLE；LOO 稳定；
- macro/worst-4 同时超过 own QCLEAN；learned−QCLEAN ≥ 9/17；
- 同时必须与同链 original-gate soft（F1C-I-soft）比较：**magsoft 相对
  original-soft 的增益才归因于幅度输入**（matched base）。
