# F1 设计冻结

## 1. 要回答的问题

F0-I 已证明正确配对 IR 会影响模型，并且明显优于 ZERO，但还没有稳定超过 RGB
matched control。F1 只回答：**能否让模型按图像决定 IR residual 的强度，从而减少
负迁移并取得独立 val 泛化收益？**

F1 不回答 Depth、跨模态正交、Transformer、双向主干、额外检测头或图像融合质量。

## 2. 结构

保持 F0 的 RGB backbone、aux encoder、P3/P4/P5 projection、neck/head 与关键 P5
路由不变，只增加一个 IR 金字塔标量门：

```text
q = sigmoid(MLP(LayerNorm(GAP(A3) || GAP(A4) || GAP(A5))))
F_i = R_i + q * P_i(A_i),  i in {P3,P4,P5}
```

- `R_i` 永远不乘权重、不放大、不削弱；RGB 是 pretrained anchor。
- `q` 每张图一个标量，三个尺度共享；第一步不做 channel/spatial gate。
- `q` 从 projection 前的 `A_i` 预测，避免 zero-init projection 抹掉门控输入。
- gate 最后一层采用极小随机权重、零 bias，使 q 接近 0.5；projection 仍为精确零，
  所以初始 detector 与 RGB 严格等价，同时避免 gate 内部再多一层零初始化阻塞。
- `q` 是检测任务控制信号，不宣称是感知图像质量分数。

## 3. 对照组

| 组 | aux | gate | 作用 |
|---|---|---|---|
| `F1-C0` | `[0,0]` | learned | 同结构 null control |
| `F1-I-fixed` | `[I,0]` | effective q 强制为 1 | 同代码路径的未门控 residual |
| `F1-I-soft` | `[I,0]` | learned | F1 treatment |

三组必须从相同 pretrained RGB 和相同 `MODEL_INIT_SEED` 初始化，禁止从 F0 checkpoint
继续训练。`F1-I-fixed` 不是 ensemble，而是同一统一模型中的结构对照。

`F1-C0` 是 matched null control，不是“纯 RGB 复刻”：现有 F0 projection 保留 bias，
因此 zero aux 时 projection bias 及其 learned scalar 仍可能形成与输入无关的校准项。这种
自由度在 C0 和 treatment 中结构匹配，但 C0 不能冒充另一个纯 RGB baseline；训练后的
projection weight 必须仍为零，bias/gate 的变化作为可审计诊断保留。

## 4. 硬门禁

- 初始 detector 对 RGB reference 的 max abs diff `<=1e-5`。
- P3/P4/P5 projection weight+bias 精确为零。
- step1：projection grad > 0；aux/gate grad 因 projection 为零而等于零。
- 手动更新 projection 后 step2：aux 与 learned gate grad > 0。
- RGB 参数冻结、BN eval，训练后 RGB SHA 不变。
- optimizer 包含 aux/projection/gate/tail，排除 RGB。
- fused P5 通过 `x=y[10]` 进入 neck layer 11。
- G8 每 epoch actual order/flip 与 expected 逐项一致。
- `amp=False` 延续 F0 冻结边界。
- 任一门禁失败立即停止，禁止启动 formal。

## 5. 正式评估与晋级

主口径仍为 `last.pt + NORMAL/ZERO/SHUFFLE + matched C0 + val6 LOO`。`FORCE-Q0`、
`FORCE-Q1` 和确定性 IR 退化只用于机制诊断。

`FORCE-Q0` 只证明同一个已训练模型内的 IR residual 被关闭；由于 neck/head 在训练中
仍会共同适配，它不等价于重新获得 `F1-C0`。RGB matched baseline 只能看独立的
`F1-C0` formal run。

F1 gate 晋级要求同时满足：

1. `SOFT_NORMAL > C0`；
2. `SOFT_NORMAL > SOFT_ZERO`；
3. `SOFT_NORMAL > SOFT_SHUFFLE`；
4. `SOFT-C0` LOO median > 0 且至少 4/6 fold 为正；
5. `SOFT_NORMAL > FIXED_NORMAL` 且 `SOFT-FIXED` LOO median > 0。
6. 在预注册的 17 个 IR 退化条件中，至少 9 个条件的 mean q 比 clean 低 `>1e-4`，且
   至少一个退化条件下 learned gate 的 AP 高于 `FORCE-Q1`。

若前四项成立但第 5 项不成立，只能说 IR feature fusion 成为互补候选，不能说 gate
优于 q=1。若前五项成立但第 6 项不成立，只能说 soft scalar 有益，不能宣称它学到了
输入条件化的可靠性，进入 F1-B corruption/dropout。通过全部条件后只补一个确认 seed，
不做 seed sweep。

## 6. 后续分支

- IR 仍被使用但不超过 C0：F1-B 只增加确定性 IR corruption/dropout 训练，结构不变。
- q 接近常数：先增加 RGB–IR agreement 描述或轻量质量先验，不上 Transformer。
- F1 成功但局部遮挡/错位失败：再做每尺度 `1xHxW` spatial reliability gate。
- 只有 IR 路线稳定后，才让 Depth 单独经过相同门控；Depth 独立通过后才能联合。
- BDDS 的双向融合/形变敏感损失仅在小目标或配准敏感性被实验确认后考虑。
