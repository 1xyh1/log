# T-series TRAINING DESIGN FREEZE
## P5-only Direct IR Injection Training

状态：**TRAINING DESIGN FROZEN / GO TO IMPLEMENTATION**
日期：2026-08-19

## 0. 上游状态

```text
A5
  ACCEPTED / CLOSED
  DIAGNOSTIC COMPLETE
  accepted result commit:
    f154c1ff9af6d31e60bc2c9a2c4fd5baafc3d8b8

A5 evidence:
  G1–G15 PASS
  P3_FULL_SUFFICIENT_FLIP = true
  P4_FULL_SUFFICIENT_FLIP = true
  BOTH_FULL_INDIVIDUALLY_SUFFICIENT = true
  JOINT_FULL_CONTEXT_REQUIRED = false
  BOTH_CENTERED_RESTORE = false
  CENTERING_FAILS_TO_RESTORE = true
  training_go = false
```

A5 的 `training_go=false` 只表示 A5 是 evaluation-only、无权直接授权训练。
本文件是独立的 reviewer training design adjudication。

正式训练假设：

> 去掉 P3/P4 backbone lateral direct IR injection 是否足够改善训练后的模型；在此基础上，P5 post-projection spatial centering 是否提供额外收益？

---

# 1. 架构术语冻结

禁止写：

```text
P3 = pure RGB
P4 = pure RGB
P5 = RGB + IR
```

因为 P5 注入后的 IR 信息可经原 YOLO26 top-down / PAN-FPN neck 继续传播到最终多尺度 detection features。

正式表述：

> **P3/P4 backbone taps receive no direct IR residual. IR enters the detector only through the backbone P5 injection site and may propagate downstream through the unchanged YOLO26 neck/head.**

统一名称：

```text
P5-only direct IR injection
```

---

# 2. 唯一新模型类

三组必须使用同一模型类、同一模块树、同一参数命名、同一初始化顺序。

```text
RGB -> existing YOLO26 RGB backbone -> R3,R4,R5

IR -> existing AuxEncoder -> A3,A4,A5

ONLY A5 is consumed:
A5 -> P5 1x1 projection (bias=True) -> delta5

P3 direct IR injection: NONE
P4 direct IR injection: NONE

P5 treatment:
NULL / FULL / AC_ALL

then:
y[10] = fused5
x = y[10]
-> unchanged YOLO26 neck/head
```

硬门禁：

```text
y[10] = fused5
x = y[10]
```

必须动态证明 fused P5 真正进入原 YOLO26 top-down neck。

---

# 3. 不允许 reliability gate

新模型 forward graph 中：

```text
NO reliability_gate
NO q
NO q3/q4/q5
NO static learned scale weights
```

不是“gate 存在但 q=1”，而是 gate 不属于 T-series forward graph。

---

# 4. AuxEncoder

沿用当前 AuxEncoder。

允许内部产生：

```text
A3
A4
A5
```

但：

```text
A3: no projection / no direct injection
A4: no projection / no direct injection
A5: consumed by P5 projection
```

第一轮禁止新建 P5-specialized encoder。

---

# 5. 三个 matched training run

三组共享：

```text
same model class
same module tree
same parameter names
same initialization RNG order
same base checkpoint SHA
same dataset/split
same seed
same augmentation
same sampler semantics
same batch
same nbs
same imgsz
same optimizer family
same LR schedule
same epochs
same RGB freeze policy
same tail trainability
same loss/head
same evaluator
```

第一轮：

```text
seed = 20260812
epochs = 80
```

仅一个 seed。

## T0-N — architecture-matched NULL

正式 prediction path：

```text
F5 = R5
```

P5 aux/projection 模块仍存在，但不得影响 detection loss。

### P0：T0 loss-graph null semantics

禁止实现成：

```python
fused5 = r5 + 0.0 * delta5
```

因为 `delta5` 仍可能留在 autograd graph 中并获得 zero-gradient tensor；在某些 optimizer / weight-decay 语义下，这与“参数不更新”并不等价。

正式要求：

```text
prediction path:
  fused5 = R5

optional mechanism logging:
  delta5 may be computed
  but must be detached / no_grad with respect to detection loss
```

T0 必须动态验证 aux/proj 参数没有 detection-loss 更新。

## T1-F — P5 FULL

```text
delta5 = P5(A5)
F5 = R5 + delta5
```

## T2-A — P5 AC_ALL

```text
delta5 = P5(A5)
mu5 = mean_HW(delta5)
delta5_ac = delta5 - mu5
F5 = R5 + delta5_ac
```

第一版必须用 `AC_ALL`，禁止切到 AC_CONTENT。

---

# 6. T2 projection-bias 数学性质

若：

```text
delta = W A + b
```

则：

```text
delta - mean_HW(delta)
= W(A - mean_HW(A))
```

因此在 exact full-map centering 下：

```text
projection bias contribution == 0
dL/db == 0
```

前提：

```text
centering immediately follows the same biased 1x1 projection
full HxW mean
no alternate bias-dependent path into loss
```

T2 仍保留同样的 `Conv2d(..., bias=True)`，禁止删除 bias。

---

# 7. P0：optimizer 对 bias 的更新语义必须审计

只验证：

```text
T2 proj.bias grad ≈ 0
```

不够。

正式 smoke 必须记录 `proj.bias` 所在 optimizer param group，并验证：

```text
weight_decay_on_proj_bias == 0
```

或提供等价 no-update guarantee。

至少在 smoke optimizer step 前后验证：

```text
T2 proj.bias before == after
```

若 grad 为零但参数仍变化：

```text
T_SERIES_OPTIMIZER_BIAS_UPDATE_FAIL
```

禁止开始 formal 80ep。

---

# 8. BatchNorm / buffer 语义

T0 如果为了日志仍计算 AuxEncoder forward，而模型处于 train mode，BatchNorm buffer 可能更新。

因此三组必须冻结同一 BN policy，并在 manifest 记录：

```text
aux_bn_stats_policy
rgb_bn_stats_policy
tail_bn_stats_policy
```

不得让 T0 的 BN buffer policy 与 T1/T2 不同。

---

# 9. Optimizer parameter-set matched control

三组 optimizer 参数名字集合与 param-group assignment 必须一致。

记录：

```text
optimizer_parameter_names
optimizer_group_index_by_name
weight_decay_by_group
initial_lr_by_group
```

要求 T0 == T1 == T2。

参数集合一致，不要求实际梯度都非空；梯度差异本身就是 treatment 的结果。

---

# 10. 初始化与 epoch-0 equivalence

P5 projection 继续 zero-init：

```text
weight = 0
bias = 0
```

三组必须验证：

```text
initial RGB state SHA equal
initial AuxEncoder state SHA equal
initial P5 projection state SHA equal
initial tail state SHA equal
initial complete state_dict SHA equal
```

真实 frozen sample：

```text
T0 raw prediction == T1 raw prediction == T2 raw prediction
```

默认要求 bitwise equal。

同时验证：

```text
T0 fused5 == R5
T1 fused5 == R5
T2 fused5 == R5
T1 delta5 == 0
T2 centered delta5 == 0
```

---

# 11. 初始梯度 smoke

同一真实 training batch、同一 initial state，分别 forward/backward，不 optimizer step。

记录：

```text
AuxEncoder gradient norm
P5 projection weight gradient norm
P5 projection bias gradient norm
tail gradient norm
```

预期：

T0-N:
```text
aux/proj grad = None / loss-disconnected
```

T1-F:
```text
proj.weight grad may be nonzero
proj.bias grad may be nonzero
```

T2-A:
```text
proj.weight grad may be nonzero
proj.bias grad approximately zero
```

T2 bias identity gate 用 FP32 smoke，不用 AMP。

---

# 12. Zero-init 对 AuxEncoder 首步梯度

projection weight 初始为 0，因此 epoch-0 首个 backward 中 AuxEncoder gradient 可能为 0 或极小，这是预期现象，不作为 FAIL。

---

# 13. Trainability policy

继承上游正式 Step4 训练 policy，不得临时改变：

```text
RGB backbone freeze/train status
YOLO neck/head trainability
AuxEncoder trainability
P5 projection trainability
BN stats policy
```

三组 `requires_grad` map 必须完全一致。

唯一 treatment 差异：

```text
T0 aux/proj loss connectivity = OFF
T1 FULL
T2 AC_ALL
```

---

# 14. 训练协议

正式 runs：

```text
T0-N_P5_NULL_seed20260812
T1-F_P5_FULL_seed20260812
T2-A_P5_ACALL_seed20260812
```

共同：

```text
80 epochs
same split
same deterministic sampling
same augmentation
same batch
same nbs
same optimizer
same LR
same imgsz
same O2M head/loss
same base checkpoint
same RGB freeze policy
same tail trainability
```

禁止额外 seed、rescue run、treatment-specific hyperparameter。

---

# 15. RNG / data-order closure

manifest 必须记录：

```text
global seed
torch seed
numpy seed
python seed
dataloader worker seed policy
sampler configuration
deterministic flag
```

若 harness 能暴露 batch sample ids，保存 epoch0 前 N 个 batch sample-id sequence，三组必须一致。

---

# 16. Base checkpoint

三组必须从同一个 frozen base checkpoint SHA 构造。

禁止：

```text
T2 warm-start from T1
T1/T2 from old F1-C last.pt while T0 uses base
```

manifest 记录：

```text
requested_base_checkpoint
requested_base_checkpoint_sha256
consumed_base_checkpoint
consumed_base_checkpoint_sha256
```

必须闭合。

---

# 17. 每 epoch 机制日志

T1/T2 至少记录：

```text
RMS(delta5)
RMS(DC(delta5))
RMS(AC(delta5))
DC_energy / FULL_energy
AC_energy / FULL_energy
proj.weight norm
proj.bias norm
```

T2 额外：

```text
post_center_channel_mean_abs_max
proj.bias grad norm
proj.bias parameter delta
```

---

# 18. 正式性能 endpoints

Primary：

```text
last val6
late10 median val6
train11
```

Secondary：

```text
all17
LOO sensitivity
best epoch descriptive only
```

禁止 best-only decision。

---

# 19. Matched training contrasts

```text
T1 - T0:
  P5 FULL direct IR injection training benefit

T2 - T1:
  extra benefit from full-map centering

T2 - T0:
  net P5-centered IR treatment benefit
```

解释矩阵：

```text
T1 ≈ T2 > T0
  -> removing P3/P4 direct injection is likely the main gain

T2 > T1 ≈ T0
  -> P5 centering is the key treatment candidate

T2 > T1 > T0
  -> topology and centering both contribute

T1 > T2 > T0
  -> P5-only injection helps but forced centering may discard useful trainable content

T0 >= T1,T2
  -> old-checkpoint mechanism does not transfer into formal retraining
```

---

# 20. Retrained paired-causality audit

训练结束后必须重新做 recipient-vs-donor audit。

T2：

```text
Delta_pair_T2 =
AP(P5 AC recipient) - AP(P5 AC donor)
```

T1：

```text
Delta_pair_T1 =
AP(P5 FULL recipient) - AP(P5 FULL donor)
```

保持：

```text
same val6
same donor map
same evaluator
same P5-only direct-injection topology
```

训练性能提升与 paired causality 必须分开判。

---

# 21. Training success 两层证据

Layer 1：

```text
T2 > T0
最好 T2 > T1
```

结合 last / late10 / train11 / LOO。

Layer 2：

```text
T2 native AC > donor AC
LOO direction stable
```

两层同时成立，才允许写：

> P5-centered architecture improves detector performance while retaining dependence on correctly paired IR.

若 AP 上升但 native≈donor，只能解释为 architectural / regularization benefit。

---

# 22. 第一轮不加 Depth

T-series 只验证 RGB+IR。

Depth 暂不加入，T-series 收敛后另开 D-series，在获胜的 RGB+IR topology 上做 Depth scale / injection audit。

---

# 23. Erratum handling

不修改已被 A5 dependency closure pin 的：

```text
docs/step4_a4/feedback/2026-08-19_formal-review.md
```

新增：

```text
docs/step4_a4/feedback/2026-08-19_erratum.md
```

仅说明：

```text
"1/5 positive" -> "1/6 positive"
documentation-only typo
no result / label / route / A5 evidence change
```

T-series provenance pin：

```text
original A4 feedback SHA
erratum SHA
A5 summary SHA/blob
accepted A5 commit f154c1ff...
```

---

# 24. Pre-training hard gates

```text
G1  A5 upstream accepted/frozen
G2  identical model class/module/state_dict keys
G3  no reliability gate/q execution
G4  P3/P4 direct injection count=0; P5 injection count=1
G5  y[10]=fused5 and x=y[10] neck handoff
G6  matched initial state SHAs
G7  epoch-0 raw prediction bitwise equivalence
G8  zero-init P5 projection
G9  identical requires_grad map
G10 identical optimizer param groups
G11 T0 null loss graph
G12 T0 no silent optimizer update
G13 T2 bias cancellation
G14 T2 bias optimizer safety
G15 matched BN/buffer policy
G16 requested-vs-consumed base checkpoint closure
G17 RNG/data-order closure
G18 protocol equality except treatment
```

任一失败：

```text
T_SERIES_FORMAL_TRAINING_HOLD
```

不得开始三组 80ep formal run。

---

# 25. 禁止项

```text
NO P3 direct IR injection
NO P4 direct IR injection
NO reliability gate
NO q
NO q3/q4/q5
NO learned scale weights
NO new AuxEncoder
NO AC_CONTENT in T2
NO Depth
NO best.pt-only decision
NO T2 warm-start from T1
NO treatment-specific hyperparameters
NO post-result threshold changes
NO extra seed before first triad closes
```

---

# 26. 项目状态

```text
A5
  CLOSED / ACCEPTED
  DIAGNOSTIC COMPLETE

T-series
  TRAINING DESIGN FROZEN
  GO TO IMPLEMENTATION / PRETRAINING SMOKES

T0-N:
  architecture-matched null

T1-F:
  P5 FULL direct IR injection

T2-A:
  P5 AC_ALL direct IR injection

P3/P4:
  no direct IR residual injection

gate:
  absent

Depth:
  HOLD until T-series closes
```

核心命题：

```text
去掉 P3/P4 direct IR injection 是否足够，
以及 P5 spatial centering 是否提供额外训练收益。
```
