# T1-S DESIGN FREEZE
## P5 FULL Residual Source-Specificity Audit

状态：**PRE-EXECUTION / EVALUATION-ONLY / DESIGN FROZEN**
日期：2026-08-19

---

# 0. 上游正式状态

```text
T-series
  ACCEPTED / CLOSED @ 1d318d0bfcbd9f6e2ebd88870c9ea571f984be2c

T1-F
  architecture-level performance benefit vs matched NULL:
  SINGLE-SEED SUPPORTED

T1-F retrained paired causality:
  SEED20260812_NEGATIVE_PAIRED_EVIDENCE

T2-A:
  centering training benefit NOT ESTABLISHED

replication_seed_go = false
depth_go            = false
production_go       = false
```

T1-S 不训练模型，不改变 T1 checkpoint，不发 production/depth GO。

---

# 1. 唯一问题

T1-S 只回答：

> **T1-F 的性能收益是否依赖“正确 recipient ↔ IR residual source 身份”，还是任意/错误 source 的 P5 FULL residual 也能产生相同或更好的效果？**

禁止把 T1-S 扩展成新的 fusion architecture 搜索。

---

# 2. 冻结系统

唯一 checkpoint：

```text
T1-F_P5_FULL_seed20260812 / weights/last.pt
SHA256:
8380e21504fabd0d8c3715398739bbb0bed5aaafd9c822dfc14c9503af2daeee
```

唯一模型语义：

```text
P5-only direct IR injection
P5 residual = FULL post-projection residual
NO P3 direct IR
NO P4 direct IR
NO reliability gate/q
NO centering
NO Depth
```

---

# 3. val6 与 source 集合

使用 frozen val6，recipient IDs 与 residual source IDs 为同一 6 元集合：

```text
000003_013_00000085
000004_013_00000081
000004_014_00000001
000016
000016_001_00000001
000016_042_suppl_00000164
```

不得增删、重排语义或替换 donor map。

---

# 4. Residual cache

对每个 source `s`：

```text
delta_s = T1-F.p5_residual_from_input(source_sample_s)
```

T1-F 中该函数必须返回：

```text
P5 FULL post-projection residual
```

不是 AC，不是 DC，不是 normalized residual。

每个 cache entry 记录：

```text
source_id
tensor SHA256
shape
RMS
spatial channel mean abs max
```

且必须与已经接受的 `posttrain_paired.json` 中 T1 native residual SHA 逐 source 一致。

---

# 5. 6×6 recipient/source matrix

对每个 recipient `r`、source `s`：

```text
prediction(r <- s)
=
T1 RGB recipient r
+
P5 FULL residual delta_s
+
unchanged YOLO26 neck/head
```

总计：

```text
6 recipients × 6 sources = 36 matrix cells
```

必须缓存每个 cell 的：

```text
recipient_id
source_id
residual_sha256
raw detection sha256
validator stat
```

validator semantics 复用冻结的 Step3/T-series evaluator。

---

# 6. Native identity anchor

矩阵对角线：

```text
r <- r
```

必须逐样本满足：

```text
matrix diagonal raw detection SHA
==
normal T1-F checkpoint raw detection SHA
==
accepted posttrain_paired T1 native detection SHA
```

任一不等：

```text
T1S_NATIVE_ANCHOR_FAIL
```

整个 audit abort。

Identity mapping：

```text
I = {
  r1->r1,
  ...
  r6->r6
}
```

其 AP 必须闭合现有 T1 native last-val6 AP。

---

# 7. Frozen A2 donor-map anchor

A2 donor map SHA256：

```text
c5cd8e852663eae3243bc5e4c263c6f2c26c7b2faa977ae3b60cb5c1ea122af5
```

从 6×6 matrix 中按该 frozen mapping 抽取 6 个 cell。

必须逐 recipient 满足：

```text
matrix fixed-donor detection SHA
==
accepted posttrain_paired T1 donor detection SHA
```

并且 assembled AP 闭合 accepted donor AP。

任一不等：

```text
T1S_FIXED_DONOR_ANCHOR_FAIL
```

---

# 8. ZERO residual condition

对每个 recipient：

```text
delta_zero = zeros_like(delta_recipient)
prediction_zero = T1 RGB recipient + 0 residual
```

注意：

> ZERO 是 **T1 checkpoint 内的 inference-time residual ablation**，不是 T0-N checkpoint。

因此：

```text
ZERO AP != T0 AP
```

并不构成错误。

ZERO 回答：

> T1 训练完成后，推理时保留 residual 本身是否还有净效用？

---

# 9. 全 265 个 derangements

对 6 个 source 的全部排列，筛选：

```text
forall recipient i:
source_perm[i] != recipient[i]
```

必须精确得到：

```text
!6 = 265
```

不抽样，不随机。

36 个 matrix cell 计算完成后，不再重新 forward 265 次。

每个 derangement AP 由对应的 6 个已缓存 validator stats 离线组合得到。

---

# 10. 统计量

记录：

```text
I = identity/native AP
Z = ZERO AP
F = frozen A2 donor-map AP

D = 265 derangement AP distribution
```

Primary metric：

```text
mAP50-95
```

Secondary：

```text
mAP50
```

必须输出：

```text
I - Z
median(D) - Z
I - median(D)

min(D)
Q1(D)
median(D)
Q3(D)
max(D)
mean(D)

identity descending rank among {I}+D
count(D > I)
count(D == I)
count(D < I)

identity strict percentile vs D
fixed donor rank/percentile within D
ZERO percentile vs D
```

---

# 11. Exact randomization evidence

冻结单侧 exact randomization p：

```text
p =
(1 + count{d in D : AP(d) >= AP(identity)})
/
266
```

其中：

```text
266 = identity + 265 fully-wrong derangements
```

预注册：

```text
alpha = 0.05
```

禁止事后改变 alpha。

这个 p 只回答：

> identity pairing 是否显著优于 fully-wrong source assignments？

它不证明跨数据集/多 seed 泛化。

---

# 12. Primary decision branches

## A. PAIRED_SOURCE_SPECIFICITY_SUPPORTED_SINGLE_SEED

要求同时：

```text
I > Z
p <= 0.05
```

解释：

> T1 inference residual 有正效用，且正确 source identity 在全部 fully-wrong mappings 中具有显著优势。

路线：

```text
T1 replication seed candidate = GO
Depth = HOLD
Production = HOLD
```

## B. GENERIC_RESIDUAL_BENEFIT_SOURCE_IDENTITY_UNPROVEN

要求：

```text
I > Z
median(D) > Z
p > 0.05
```

解释：

> residual presence 本身普遍有益，但正确 source identity 没建立特殊性。

## C. WRONG_SOURCE_TYPICALLY_OUTPERFORMS_NATIVE

要求：

```text
median(D) > I
```

此分支优先于 B。

## D. INFERENCE_RESIDUAL_NOT_SUPPORTED_TRAINING_DYNAMICS_CANDIDATE

要求：

```text
Z >= I
```

且没有先触发 C。

## E. SOURCE_SPECIFICITY_INCONCLUSIVE

以上均不满足。

---

# 13. 判级优先级

固定：

```text
1. C WRONG_SOURCE_TYPICALLY_OUTPERFORMS_NATIVE
2. D INFERENCE_RESIDUAL_NOT_SUPPORTED_TRAINING_DYNAMICS_CANDIDATE
3. A PAIRED_SOURCE_SPECIFICITY_SUPPORTED_SINGLE_SEED
4. B GENERIC_RESIDUAL_BENEFIT_SOURCE_IDENTITY_UNPROVEN
5. E SOURCE_SPECIFICITY_INCONCLUSIVE
```

---

# 14. 固定 donor map 只是 anchor，不再是唯一 causal test

现有 T-series paired negative 由一个 frozen derangement 给出。
T1-S 不删除、不覆盖它。
T1-S 将它定位为：

```text
one member of the full 265-derangement distribution
```

必须输出该 mapping 在 265 个 derangements 中的 rank/percentile。

---

# 15. 输出

```text
reports/step4_t1s/preexecution_audit.json
reports/step4_t1s/source_matrix.json
reports/step4_t1s/derangements.json
reports/step4_t1s/t1s_summary.json
```

所有 evaluator 输出默认拒绝 overwrite。

---

# 16. Hard gates G1–G15

```text
G1  T-series accepted result evidence pinned
G2  T1 performance branch = architecture gain / paired complementarity unproven
G3  posttrain performance raw SHA pinned
G4  posttrain paired raw SHA pinned
G5  T1 manifest SHA pinned
G6  T1 last.pt SHA pinned
G7  val6 exact IDs
G8  A2 donor-map SHA and derangement semantics
G9  T-series source hashes fresh
G10 exact 6×6 matrix contract
G11 native diagonal bitwise anchor enabled
G12 frozen fixed-donor bitwise anchor enabled
G13 exact derangement count = 265
G14 ZERO condition is T1-checkpoint inference ablation
G15 evaluation-only / no training / no Depth / no production GO
```

任一失败：

```text
T1S_EXECUTION_HOLD
```

---

# 17. 禁止项

```text
NO model training
NO checkpoint modification
NO new seed
NO P3/P4 direct IR
NO gate/q
NO centering
NO Depth
NO random donor sampling
NO subset of derangements
NO threshold tuning after results
NO overwrite accepted T-series reports
NO claim that architecture gain == paired multimodal gain
```

---

# 18. 状态

```text
T-series:
  CLOSED / ACCEPTED

T1-S:
  DESIGN FROZEN
  EVALUATION ONLY
  GO TO IMPLEMENTATION / PREEXECUTION AUDIT

Replication:
  HOLD until T1-S result

Depth:
  HOLD

Production:
  HOLD
```
