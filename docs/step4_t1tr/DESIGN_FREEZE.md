# T1-TR TRAINING DESIGN FREEZE
## Training-time IR Source-Specificity Ablation

状态：**TRAINING DESIGN FROZEN / ONE NEW ARM ONLY / GO TO IMPLEMENTATION**
日期：2026-08-19

---

# 0. 上游状态

```text
T-series
  ACCEPTED / CLOSED @ 1d318d0bfcbd9f6e2ebd88870c9ea571f984be2c

T1-S
  ACCEPTED / CLOSED @ 7c86a87a0ca61e6ebc8299b4f3b35dc997f3d46d
  DIAGNOSTIC COMPLETE

T1-S result:
  identity/native        = 0.27951034259857793
  ZERO residual          = 0.29596085371085373
  frozen A2 donor        = 0.2904392987964417
  derangement median     = 0.2762406530885447
  p(identity specificity)= 0.462406015037594

decision:
  INFERENCE_RESIDUAL_NOT_SUPPORTED_TRAINING_DYNAMICS_CANDIDATE
```

T1-TR 不重新训练 T0/T1。

---

# 1. 唯一问题

T1-TR 只回答：

> **T1 的训练收益是否依赖训练时正确的 RGB↔IR source identity，还是 fully-wrong IR 也能产生相同的训练动力学/正则化收益？**

禁止把结果提前解释为 Depth、production 或最终多模态方案。

---

# 2. 三臂定义

## U0-N — frozen existing control

复用：

```text
T0-N_P5_NULL_seed20260812
```

训练：

```text
P5 residual disconnected / NULL
```

推理 primary：

```text
ZERO / RGB
```

不重训。

## U1-P — frozen existing paired-training control

复用：

```text
T1-F_P5_FULL_seed20260812
```

训练：

```text
correctly paired recipient RGB + recipient IR
P5 FULL direct IR residual
```

推理 primary：

```text
ZERO residual
```

不重训。

## U2-S — only new training arm

新训练一次：

```text
U2-S_P5_FULL_BALANCED_SHUFFLED_seed20260812
```

模型：

```text
完全复用 TSeriesP5Model(treatment_id="T1-F")
P5 FULL
NO P3/P4 direct IR
NO gate/q
NO centering
NO Depth
```

训练数据：

```text
recipient RGB / labels / geometry unchanged
IR source = deterministic fully-wrong donor
```

推理 primary：

```text
ZERO residual
```

---

# 3. 为什么 primary 必须统一 ZERO inference

T1-S 已证明：

```text
T1 native IR < T1 ZERO
```

所以 T1-TR 不再让推理期 residual 混入“训练阶段 source identity”问题。

Primary comparison 统一为：

```text
U0 checkpoint -> ZERO/RGB
U1 checkpoint -> ZERO
U2 checkpoint -> ZERO
```

因此：

```text
U1 - U0
```

回答正确配对 IR 训练 treatment 是否留下收益。

```text
U2 - U0
```

回答 fully-wrong IR 训练 treatment 是否也留下收益。

```text
U1 - U2
```

回答正确 pairing 在训练阶段是否具有额外价值。

---

# 4. U2-S 错配 schedule

train11 frozen order直接来自 contract：

```text
train_ids = contract["train_ids"]
n = 11
```

epoch `e`：

```text
shift(e) = 1 + (e mod 10)
donor(train_ids[i], e)
    = train_ids[(i + shift(e)) mod 11]
```

性质：

```text
每 epoch:
  11 -> 11 bijection
  no self-match
  each donor used exactly once

80 epochs:
  shifts 1..10 each occur exactly 8 times

therefore:
  each recipient sees each of its 10 non-self donors exactly 8 times
  each ordered non-self recipient/donor pair occurs exactly 8 times
```

这比固定单一 donor permutation 更能回答 source identity。

---

# 5. Dataset semantics

必须复用现有 `TriModalDataset(aux_id_map=...)`。

其冻结语义：

```text
RGB path       = recipient
label/geometry = recipient
IR path        = aux_id_map[recipient]
flip decision  = recipient sid + epoch
same horizontal flip applied to recipient RGB and donor IR
```

不得修改 `trimodal_dataset.py`。

U2-S 使用 `group="C1-I"`，Depth/M 仍为零。

---

# 6. Sampler / augmentation matching

U2-S 保持 T1 的：

```text
seed = 20260812
epochs = 80
batch = 4
nbs = 4
workers = 0
imgsz = 640
fliplr = 0.30393
all other frozen R3_KW hyperparameters
```

每 epoch 必须记录并验证：

```text
recipient sample order SHA
flip schedule SHA
aux source sequence SHA
epoch donor-map SHA
no-self count = 11
bijection = true
shift = 1..10
```

recipient sampler/order 与 T1 recipe 不变。

---

# 7. Model initialization hard anchor

U2-S model 构造必须逐项等于 frozen T1 initial identity：

```text
RGB backbone state SHA
AuxEncoder state SHA
AuxEncoder parameter SHA
P5 fusion state SHA
P5 fusion parameter SHA
P5 bias SHA
tail state SHA
complete model state SHA
state_dict keys
requires_grad map
```

T1 manifest SHA：

```text
081afec392d96ee2d570a3424e5f015f05ee308297daed8900ece5584c707312
```

任何 mismatch：

```text
T1TR_INITIAL_IDENTITY_FAIL
```

---

# 8. Optimizer hard anchor

U2-S real trainer build 后：

```text
optimizer parameter names
group assignment
initial lr
weight_decay
momentum
```

必须与 frozen T1：

```text
runs/step4_tseries/T1-F_P5_FULL_seed20260812/optimizer_manifest.json
```

逐结构一致。

不得因 shuffled training 改 optimizer membership。

---

# 9. Epoch-0 data smoke

正式训练前至少 1 epoch real smoke。

必须证明：

```text
all 11 recipient ids appear once
all 11 donor ids appear once
recipient != donor for all 11
actual donor == scheduled donor
recipient order matches deterministic sampler
flip schedule matches frozen rule
model initial identity == T1
optimizer snapshot == T1
```

formal pretraining audit 必须读取 smoke evidence。

---

# 10. Formal training

只新增：

```text
U2-S × 80ep
seed 20260812
```

不新增：

```text
U0 seed
U1 seed
U2 extra seed
rescue run
Depth
```

U2 validation during training保持 stock/native paired input，只用于 descriptive training curve。

Primary T1-TR adjudication不使用该 native curve。

---

# 11. Post-training common ZERO evaluator

对三个 checkpoint 用同一 evaluator：

```text
U0 = T0 last.pt
U1 = T1 last.pt
U2 = U2 last.pt
```

每个 sample：

```text
zero = zeros_like(P5 residual)
prediction = model.predict_with_p5_residual(input, zero)
```

U0 额外 hard anchor：

```text
T0 normal prediction == T0 zero override
bitwise per sample
```

U1 val6 hard anchor：

```text
ZERO mAP50-95
==
T1-S ZERO
0.29596085371085373
```

避免 evaluator 语义漂移。

---

# 12. Primary endpoints

只用 final `last.pt` 的 ZERO inference：

```text
val6 mAP50-95
train11 mAP50-95
all17 mAP50-95
```

Secondary：

```text
val6 LOO sensitivity
mAP50
U2 native-paired last.pt
U2 native training curve
```

禁止 best.pt-only decision。

---

# 13. Matched contrasts

```text
P = U1_ZERO - U0_ZERO
S = U2_ZERO - U0_ZERO
Q = U1_ZERO - U2_ZERO
```

每个 contrast 在三个 primary endpoints 上分类：

```text
STABLE_POSITIVE
  all 3 > 0

STABLE_NEGATIVE
  all 3 < 0

EXACT_TIE
  all 3 == 0

MIXED
  otherwise
```

不设任意 AP margin。

---

# 14. Decision branches

优先级固定：

## A. SHUFFLED_TRAINING_OUTPERFORMS_PAIRED

```text
Q = STABLE_NEGATIVE
```

解释：

> fully-wrong source training 稳定优于 correctly paired training。

```text
paired training source specificity = rejected
replication = HOLD
Depth = HOLD
```

## B. PAIRED_TRAINING_SOURCE_SPECIFICITY_SUPPORTED_SINGLE_SEED

要求：

```text
P = STABLE_POSITIVE
Q = STABLE_POSITIVE
```

解释：

> 正确配对训练相对 NULL 有稳定收益，并稳定优于 balanced fully-wrong training。

```text
replication_seed_go = true
Depth = HOLD
Production = HOLD
```

U2 是否高于 U0 作为 effect-size/机制次级信息，不改变 B 的定义。

## C. GENERIC_TRAINING_REGULARIZATION_SOURCE_IDENTITY_UNPROVEN

要求：

```text
P = STABLE_POSITIVE
S = STABLE_POSITIVE
Q = MIXED or EXACT_TIE
```

解释：

> paired 与 fully-wrong IR 训练都稳定优于 NULL，但正确 source identity 没有稳定额外价值。

## D. SHUFFLED_TRAINING_HAS_GAIN_PAIRED_ADVANTAGE_INCONCLUSIVE

要求：

```text
S = STABLE_POSITIVE
Q = MIXED
```

且未触发 C。

解释：

> fully-wrong training 自身已有稳定收益，paired 是否额外更好不稳定。

## E. PAIRED_TRAINING_ADVANTAGE_INCONCLUSIVE

要求：

```text
P = STABLE_POSITIVE
```

且未触发 B/C/A。

解释：

> paired training 有收益，但对 shuffled 的优势未稳定闭合。

## F. TRAINING_TREATMENT_GAIN_NOT_STABLE

其余情况。

---

# 15. LOO sensitivity

val6 ZERO 条件下输出：

```text
U1-U0 LOO
U2-U0 LOO
U1-U2 LOO
```

仅 sensitivity，不覆盖三 endpoint primary decision。

---

# 16. Hard gates G1–G18

```text
G1  T1-S accepted result evidence pinned
G2  T1-S branch = inference residual not supported
G3  T1-S ZERO numeric anchor pinned
G4  T0/T1 manifest SHA pinned
G5  T0/T1 last.pt SHA pinned
G6  contract SHA matches frozen T1
G7  train11 count=11 / val6 count=6
G8  schedule exact: shift 1..10 cyclic
G9  no self-match every epoch
G10 80ep pair balance = each ordered nonself pair 8 times
G11 U2 model initial identity == T1 initial identity
G12 U2 optimizer groups == T1
G13 smoke actual recipient/donor mapping matches schedule
G14 smoke sampler/flip closure
G15 no P3/P4 direct IR / no gate / FULL P5
G16 protocol seed/epochs/batch/R3_KW matched
G17 U2 only new formal arm
G18 no Depth / no production / no extra seed
```

任一失败：

```text
T1TR_FORMAL_TRAINING_HOLD
```

---

# 17. Outputs

```text
reports/step4_t1tr/pretraining_static_audit.json
reports/step4_t1tr/pretraining_smoke.json
reports/step4_t1tr/pretraining_audit.json
reports/step4_t1tr/posttrain_zero_eval.json
reports/step4_t1tr/t1tr_summary.json

runs/step4_t1tr/
  U2-S_P5_FULL_BALANCED_SHUFFLED_seed20260812/
```

---

# 18. 禁止项

```text
NO modify T0/T1 checkpoints
NO retrain T0/T1
NO fixed one-off shuffle mapping
NO random unlogged donor mapping
NO self donor
NO donor sampling with replacement
NO P3/P4 direct IR
NO gate/q
NO centering
NO Depth
NO best-only decision
NO arbitrary AP margin
NO extra seed before T1-TR closes
NO production GO from this single seed
```

---

# 19. 当前项目状态

```text
T-series:
  CLOSED

T1-S:
  CLOSED

T1-TR:
  TRAINING DESIGN FROZEN
  GO TO IMPLEMENTATION/PREFLIGHT

NEW TRAINING:
  U2-S only

Replication:
  HOLD until T1-TR result

Depth:
  HOLD

Production:
  HOLD
```
