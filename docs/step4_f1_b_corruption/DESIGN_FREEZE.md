# F1-B 设计冻结：训练期确定性 IR corruption/dropout

## 1. 要回答的问题

F1-A0 已证明：q≈常数 0.504，learned gate 与 FORCE-QCLEAN 等价（identity 差
−1.6e-5；退化集 6/17 正、mean +0.000553）——**恒定衰减有效，自适应未证明**。

F1-B 只回答：**在训练期加入确定性 IR 退化/丢弃后，learned gate 能否学到输入
条件化的可靠性，并同时超过 separately-trained B1-fixed 与 FORCE-QCLEAN？**

不回答 Depth、spatial gate、Transformer、双向主干、BDDS 形变 loss 或 RGB–IR
agreement（后者仅在 F1-B 后 q 仍近常数时作为下一候选输入）。

## 2. 结构

与 F1 完全相同（RGB anchor + q·zero-init IR residual + gate 输入 detach），
**不加载任何 F1 checkpoint**，三组从相同 pretrained 初始状态（MODEL_INIT_SEED）
重新训练。amp=False、数据、bbox、validator、G8、G6 阈值逻辑全部复用。

## 3. 冻结 corruption schedule（预注册）

| 条件 | 概率 |
|---|---|
| clean | 0.50 |
| zero（dropout） | 0.125（severity 固定 1.0） |
| noise | 0.125 |
| blur | 0.125 |
| contrast | 0.125 |

- noise/blur/contrast 的 severity 从 {0.25, 0.50, 0.75, 1.00} 等概率选取。
- **shift 不进训练**（A0 扫描：shift 四档最佳 q 均为 1.0，无证据它是"应被抑制的
  劣质 IR"；保留为 evaluation-only 配准诊断）。
- 随机条件一律由 SHA256(`seed|epoch|sample_id|field`) 的 digest 字节驱动，
  **禁用 Python 内置 hash()**（进程盐化不可复现）。noise 随机场必须包含 epoch
  （同一张噪声图不得跨 epoch 重复）；severity 与 noise 场取不相交的 digest 字节段。
- corruption 只作用于 IR 通道（6ch 的 channel 3），在 TriModalDataset 冻结输出
  之后施加；RGB/Depth/label/bbox 不变。
- B1-C0 的 aux 输入恒零，corruption 不实际施加（无模型影响），但 schedule 证据
  照常记录——三组共享完全相同的训练期调度。

## 4. 正式组

| 组 | aux | gate | 作用 |
|---|---|---|---|
| `B1-C0` | `[0,0]` | learned | 同结构 null control（含 corruption schedule） |
| `B1-I-fixed` | `[I,0]` | 固定 q=1 | 未门控 residual（corruption 训练） |
| `B1-I-soft` | `[I,0]` | learned | F1-B treatment |

## 5. 硬门禁（G1–G8 沿用 F1 + 新增 G9）

- G1 RGB 等价（≤1e-5）、G2 zero-init、G3 梯度解锁（gate detach 语义：step1
  gate/aux grad 为零、手动更新 proj 后 step2 解锁）、G4 P5 路由、G5 optimizer
  成员（aux/proj/gate/tail 在、RGB 冻结）、G6 训练后更新证据（C0 proj 精确零 +
  bias 衰减级中性；active 组 proj/gate 学；阈值 epoch 线性缩放仅限 smoke 活性检查，
  formal 用原始 1e-3）、G8 actual order/flip 逐 epoch 一致。
- **G9（新增）**：每个 epoch 从实际 dataloader yield 记录：
  - 每样本 sample_id / kind / severity；
  - expected 与 actual schedule SHA（按 sample_id 排序的规范化序列）逐 epoch 相等；
  - corruption 前后 IR 平面 SHA（clean 不变、非 clean 必变；B1-C0 全部不变）；
  - RGB/Depth/label/bbox 未变化断言；
  - per-epoch kind 计数分布；
  - 三组 expected schedule 逐 epoch 字节一致（同 seed 同 schedule）。
  - 任一不一致立即停止（fail-fast）。
- 任一硬门禁失败禁止启动 formal。

## 6. B1 晋级要求（在 F1 原有条件之外）

F1 原条件（clean 轴）：SOFT_N > C0、SOFT_N > SOFT_Z、SOFT_N > SOFT_S、
SOFT−C0 LOO median > 0 且 ≥4/6 正。

B1 新增：
1. clean SOFT > **separately-trained** B1-fixed，且 SOFT−FIXED LOO median > 0；
2. 17 个退化条件 macro AP：SOFT **同时**超过 B1-fixed 与 FORCE-QCLEAN；
3. worst-4 条件 mean AP：SOFT **同时**超过两者；
4. learned−QCLEAN 至少 **9/17** 条件为正；
5. **FORCE-QCLEAN 重新取 B1-soft clean identity 的 mean q**，不得沿用 F1 的 0.503991；
6. q–severity 单调性仅作诊断，不作为成功替代品。

soft 必须同时超过 fixed 与 FORCE-QCLEAN，才能说"自适应可靠性"成立；只超其一不算。

## 7. 后续分支

- F1-B 后 q 仍近常数：增加 RGB–IR agreement 描述输入，不上 Transformer。
- 只有 IR 路线稳定后，才允许 Depth 单独套相同门控；Depth 独立通过后才能联合。
- BDDS 双向融合/形变 loss 仅在小目标分层与固定 shift 诊断证明错位是主要误差后考虑。
