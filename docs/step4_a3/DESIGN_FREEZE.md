# A3 设计冻结：RGB–IR Spatial / Semantic Agreement Audit

状态：**PRE-EXECUTION / EVALUATION-ONLY / DESIGN FROZEN**  
上游正式基线：

```text
F1-C:
  CLOSED / FAILED @ bf983c4
  F1C_GATE_FAILED_CAUSAL_PROTOCOL

A2:
  CLOSED / DIAGNOSTIC COMPLETE
  RESULT ACCEPTED @ 2188593bf9797f0097efe7e4a230694cc19502fb
  result_sha256 =
    756093358153c5e203f485dce96e0f2a5e91881fb6c6e4b49c036cbfdc6d1c6b
  all_gates_passed = true

A2 reviewer conclusion:
  NO_SCALE_MEETS_STABLE_PAIRED_POSITIVE_CRITERION
```

A2 已确认：

- P5 是最稳定的 paired-negative scale；
- P3/P4 的作用明显依赖训练系统与其他尺度上下文；
- 某些 static gain 能改善已有 checkpoint 的 operating point；
- 但 residual utility 不等于 recipient-specific IR semantic value；
- 因此不进入 static per-scale training，也不进入动态 `q3/q4/q5`。

A3 只做诊断，不训练，不修改 F1-C/A2 冻结证据。

---

## 1. A3 只回答什么

A3 只回答：

> **为什么当前已训练 RGB+IR residual 系统没有在 P3/P4/P5 上建立稳定、跨训练系统复现的 paired IR 正因果价值？**

A3 将四个可能机制拆开审计：

1. **H-R：Registration**
   - RGB 与 IR 的物理/几何对应是否存在系统性空间偏移；
   - 如果把由独立结构证据估计出的偏移进行 cross-fitted 修正，paired residual 是否得到因果性 rescue。

2. **H-S：Spatial correspondence**
   - 在 projection 之后，IR residual 的空间激活位置是否与同一 recipient 的 RGB feature 空间结构对应；
   - native residual 是否比 donor residual 更匹配 recipient RGB。

3. **H-M：Semantic / object-region agreement**
   - IR residual 是否真正把能量/结构放在当前 recipient 的目标区域；
   - native residual 是否比 donor residual 更贴合 recipient 的 GT object regions。

4. **H-B：Generic residual bias**
   - A2 中 residual 带来的 AP 变化是否主要来自 recipient-independent 的平均扰动、通道偏置或 generic feature prior；
   - 而不是来自当前样本正确配对的 IR 内容。

A3 **不强迫四个假设互斥**。  
最终允许同时出现：

```text
registration implicated
+ generic bias supported
+ semantic correspondence weak
```

不得为了得到一个“唯一根因”而把多条证据压成单标签。

---

## 2. A3 明确不回答什么

A3 不回答：

- 新 gate 应该怎么设计；
- `q3/q4/q5` 是否有效；
- static per-scale weights 训练后是否有效；
- Depth 如何融合；
- 新 AuxEncoder 是否优于旧 AuxEncoder；
- 是否应增加 alignment loss；
- 是否应改 detector neck/head；
- 是否应进行 RGB/IR 在线标定或图像 warp 作为正式生产方案。

这些都是 **A3 之后** 的设计问题。

A3 只负责建立诊断证据。

---

## 3. 冻结 checkpoint 与角色

### 3.1 Primary

```text
runs/step4_f1_c/F1C-I-fixed/weights/last.pt
```

身份必须保持：

```text
group = F1C-I-fixed
aux_mode = ir
gate_mode = fixed_one
gate_module = magnitude
q ≡ 1
epochs = 80
seed = 20260812
```

Primary 用于最干净地观察：

> RGB backbone + 已训练 IR encoder/projection residual 本身的 registration / spatial / semantic / generic-bias 性质。

### 3.2 Replication

```text
runs/step4_f1_c/F1C-I-soft/weights/last.pt
```

身份必须保持：

```text
group = F1C-I-soft
aux_mode = ir
gate_mode = learned
gate_module = original
epochs = 80
seed = 20260812
```

SOFT 仅作为独立训练系统上的 replication。

禁止把 FIXED vs SOFT 解释成：

```text
same weights + only q changed
```

它们是两个不同训练系统。

### 3.3 Checkpoint 选择

只允许：

```text
last.pt
```

`best.pt` 不参与 A3 正式结论。

---

## 4. 数据与 donor 身份继续冻结

A3 primary probe 继续使用 A2/F1-C 冻结的同一 val6：

```text
000003_013_00000085
000004_013_00000081
000004_014_00000001
000016
000016_001_00000001
000016_042_suppl_00000164
```

val6 顺序必须完全一致。

Donor map **直接继承 A2**，不得重新生成：

```text
donor_map_sha256 =
c5cd8e852663eae3243bc5e4c263c6f2c26c7b2faa977ae3b60cb5c1ea122af5
```

A3 必须读取并验证：

```text
reports/step4_a2/val_donor_map.json
```

且：

```text
A3 donor map == A2 donor map
```

逐项完全相同。

A3 不允许因为新诊断需要更“好”的 donor 而换 donor。

---

## 5. 上游 provenance 闭包

A3 启动前必须验证：

### 5.1 A2 结果冻结

```text
reports/step4_a2/scale_ir_residual_causality.json
```

必须满足：

```text
schema = step4-a2-scale-ir-residual-causality-v2
all_gates_passed = true
```

原文件 SHA256 必须等于：

```text
756093358153c5e203f485dce96e0f2a5e91881fb6c6e4b49c036cbfdc6d1c6b
```

### 5.2 F1-C / A2 dependency closure

A3 必须继承 A2 v2 已冻结的执行语义：

- F1-C summary SHA；
- FIXED / SOFT last.pt SHA；
- FIXED / SOFT manifest SHA；
- Step3 authoritative eval helper SHA；
- F1 model SHA；
- reliability gate SHA；
- trimodal dataset SHA；
- F0 model SHA；
- AuxEncoder SHA；
- feature fusion SHA；
- trainability SHA；
- causality intervention SHA；
- raw sample index SHA；
- contract SHA；
- torch version；
- ultralytics version；
- val6 identity/order；
- `modality_preprocess.py` 必须锁到 A3 冻结时的规范化 LF 后的精确 Git blob：`ed3a52150eedee18c60f163401dc64a198398662`，并同时记录运行时 SHA256。

任何依赖漂移：

```text
A3_FROZEN_DEPENDENCY_CLOSURE_FAIL
```

立即停止。

A3 不允许通过“checkpoint SHA 还一样”来容忍 Python class semantics 已改变。

---

# 6. A3 的因果边界仍然是：先算 native q，再做任何 residual intervention

SOFT 对 recipient `x`：

```text
A3^x, A4^x, A5^x
        ↓
q_x = q(A3^x, A4^x, A5^x)
        ↓
freeze q_x
```

然后：

```text
δ_i^x = P_i(A_i^x)
```

之后才允许 A3 intervention：

```text
SHIFT
MEAN
DC
AC
DONOR
```

禁止任何 A3 condition 反向改变 gate input。

FIXED 仍要求：

```text
q_native ≡ 1
```

SOFT 要求同一 recipient 在所有 A3 intervention 下：

```text
q_native(condition) == q_native(untouched recipient)
```

A3 不做 dataset-level IR shuffle。

---

# 7. A3-A：Registration Audit

## 7.1 目的

只回答：

> 是否存在跨样本一致的 RGB↔IR 几何平移误差，并且这种误差在不使用 held-out AP 选择 shift 的情况下，可以因果性 rescue IR residual？

A3 不使用 AP 最大化来寻找 shift。

---

## 7.2 原始结构图

对进入模型前、完成当前冻结 resize / geometric preprocessing 后的 RGB 与 IR。**必须先裁掉 letterbox padding，只在共同有效内容区估计 shift**；不得让 RGB=114/255 与 IR=0 的 pad 边界进入 phase-correlation 主证据：

### RGB

```text
G_rgb =
sqrt(SobelX(gray(RGB))^2 + SobelY(gray(RGB))^2)
```

### IR

```text
G_ir =
sqrt(SobelX(IR)^2 + SobelY(IR)^2)
```

每张图分别：

```text
G <- (G - mean(G)) / (std(G) + eps)
```

不做可学习变换。

---

## 7.3 每样本 raw translation estimate

使用固定算法：

```text
phase correlation on G_rgb vs G_ir
```

记录：

```text
dx_raw
dy_raw
phase_response
```

这里的 `(dx_raw, dy_raw)` 只作为结构性诊断量。

禁止：

- 用 GT box 选择 shift；
- 用 detector AP 选择 shift；
- 为每个尺度单独调 raw shift；
- 看结果后更换 registration estimator。

---

## 7.4 Cross-fitted global shift

为了避免同一样本上：

```text
estimate shift -> apply shift -> evaluate AP
```

形成 oracle circularity，A3 必须做 val6 leave-one-out cross-fitting。

对 held-out recipient `x`：

```text
S_train = raw shifts of the other 5 samples
s_x = componentwise median(S_train)
```

得到：

```text
(dx_cf_x, dy_cf_x)
```

然后把输入像素 shift 转换为每个 feature scale 的整数 cell shift：

```text
P3: round(dx_cf / stride3), round(dy_cf / stride3)
P4: round(dx_cf / stride4), round(dy_cf / stride4)
P5: round(dx_cf / stride5), round(dy_cf / stride5)
```

stride 必须从当前冻结 model route / feature shape 明确验证，禁止手写错误 stride。

---

## 7.5 Shift intervention 的位置

只允许：

```text
δ_i = P_i(A_i)
        ↓
integer translation with zero fill
        ↓
residual add
```

禁止 wrap-around。

不得 shift RGB。

不得 shift gate input。

不得重新跑 AuxEncoder 得到“对齐后”的 A_i。

这是 post-projection diagnostic shift，不是生产 registration pipeline。

---

## 7.6 两种 shift context

每尺度 `i ∈ {P3,P4,P5}` 都评估：

### Standalone

只有 target scale 开启：

```text
SHIFT_i_ONLY
```

其他 residual：

```text
0
```

对比 A2：

```text
KEEP_i paired native
```

定义：

\[
\Delta^{shift,standalone}_i
=
AP(SHIFT_i\_ONLY)-AP(KEEP_i)
\]

### Conditional

其他两尺度保持 native：

```text
SHIFT_i_COND
```

对比：

```text
M111
```

定义：

\[
\Delta^{shift,conditional}_i
=
AP(SHIFT_i\_COND)-AP(M111)
\]

所有 AP 使用同一 Step3 authoritative validator。

---

## 7.7 Registration rescue 标签

分别对 standalone / conditional，按与 A2 同风格的 sign criterion：

### STRONG_POSITIVE_RESCUE

```text
FIXED full > 0
FIXED LOO median > 0
FIXED positive folds >= 4/6
SOFT full > 0
```

### STRONG_NEGATIVE_RESCUE

```text
FIXED full < 0
FIXED LOO median < 0
FIXED negative folds >= 4/6
SOFT full < 0
```

### INCONCLUSIVE

其余。

不加入临时 AP margin。

### Registration implicated

只有当某尺度至少一个 context 获得：

```text
STRONG_POSITIVE_RESCUE
```

时，才允许写：

```text
registration implicated at scale i
```

仅仅观察到 non-zero phase-correlation shift，不足以单独声明 registration 是 causal root cause。

---

# 8. A3-B：Feature Spatial Correspondence Audit

## 8.1 不直接比较 raw channel cosine

A3 不把：

```text
cosine(R_i, δ_i)
```

作为主证据。

原因：

`δ_i` 是 additive residual，不要求与 RGB feature channel direction 同向；低 channel cosine 不等于空间错位。

---

## 8.2 空间能量图

对 RGB anchor feature：

\[
E^R_i(h,w)
=
\sqrt{
\frac{1}{C}
\sum_c R_i(c,h,w)^2
}
\]

对 projected IR residual：

\[
E^\delta_i(h,w)
=
\sqrt{
\frac{1}{C}
\sum_c \delta_i(c,h,w)^2
}
\]

每个 map 独立 z-normalize。

---

## 8.3 Native vs donor spatial correspondence

对 recipient `x`：

### Native

```text
Corr_native_i(x)
=
corr(
    E^R_i(x),
    E^δ_i(x)
)
```

### Donor

```text
Corr_donor_i(x)
=
corr(
    E^R_i(x),
    E^δ_i(donor(x))
)
```

主统计量：

\[
\Delta^{spatial}_i(x)
=
Corr_{native,i}(x)
-
Corr_{donor,i}(x)
\]

报告：

```text
per-sample values
median
mean
positive_count
negative_count
```

FIXED 与 SOFT 分开。

---

## 8.4 Shift surface

另外对：

```text
E^R_i(x)
vs
E^δ_i(x)
```

计算小范围整数 cell shift 下的 normalized correlation surface。

搜索窗口固定：

```text
dx,dy ∈ {-2,-1,0,1,2} feature cells
```

只用于描述：

```text
best_feature_shift
corr_at_zero
corr_at_best
best_minus_zero
```

**禁止使用这个 per-sample best shift 做 AP rescue。**

AP rescue 只能使用 §7 的 raw-space cross-fitted global shift。

这样避免：

```text
在同一个 feature map 上找最优 shift
→ 再用最优 shift 证明自己有用
```

的循环论证。

---

## 8.5 Spatial recipient-specific label

每尺度分别判断：

### STRONG_RECIPIENT_SPECIFIC

```text
FIXED median(Δspatial) > 0
FIXED positive samples >= 4/6
SOFT median(Δspatial) > 0
```

### STRONG_DONOR_FAVORED

```text
FIXED median(Δspatial) < 0
FIXED negative samples >= 4/6
SOFT median(Δspatial) < 0
```

其余：

```text
INCONCLUSIVE
```

不设置额外 cosine/correlation magnitude threshold。

---

# 9. A3-C：Semantic / Object-Region Agreement Audit

## 9.1 语义定义边界

A3 当前 val6 太小，不把“类别可分性”作为正式主结论。

A3 的 semantic 主问题限定为：

> residual 的空间激活是否对当前 recipient 的 GT object regions 有 recipient-specific localization value？

这里的 semantic 是 **object-region semantics**，不是完整 class representation theorem。

---

## 9.2 GT object mask

使用冻结 detection labels。

对每个 scale：

```text
GT boxes
→ project to feature map coordinates
→ union binary object mask M_i
```

需要固定且测试：

- box clipping；
- feature-map coordinate conversion；
- empty object cell handling；
- overlapping boxes union；
- 不读取 prediction 生成 GT mask。

---

## 9.3 Object-region energy AUROC

使用：

```text
Eδ_i(h,w)
```

作为 score，

```text
M_i(h,w)
```

作为 binary target。

得到：

```text
AUROC_native_i(x)
```

以及 donor residual 放到 recipient 坐标上：

```text
AUROC_donor_i(x)
```

定义：

\[
\Delta^{semantic}_i(x)
=
AUROC_{native,i}(x)
-
AUROC_{donor,i}(x)
\]

如果某样本/尺度 GT mask 退化到全 0 或全 1：

```text
SEMANTIC_MASK_DEGENERATE
```

该样本该 metric 标记 invalid，不能偷偷填 0.5。

如果任一尺度有效样本少于冻结 val6 的 5/6：

```text
A3_SEMANTIC_COVERAGE_FAIL
```

正式 semantic label 不输出。

---

## 9.4 Object/background enrichment

同时记录描述量：

\[
ER_i
=
\frac{
mean(E^\delta_i \mid object)
}{
mean(E^\delta_i \mid background)+\epsilon
}
\]

分别计算：

```text
ER_native
ER_donor
```

以及：

```text
log(ER_native) - log(ER_donor)
```

AUROC 是主 semantic metric；enrichment 仅作解释性 supporting metric。

---

## 9.5 Semantic recipient-specific label

按 AUROC difference：

### STRONG_RECIPIENT_SPECIFIC

```text
FIXED median(Δsemantic) > 0
FIXED positive samples >= 4/6
SOFT median(Δsemantic) > 0
```

### STRONG_DONOR_FAVORED

```text
FIXED median(Δsemantic) < 0
FIXED negative samples >= 4/6
SOFT median(Δsemantic) < 0
```

否则：

```text
INCONCLUSIVE
```

---

# 10. A3-D：Generic Residual Bias Audit

## 10.1 动机

A2 已出现：

```text
residual may improve AP
but paired native ≈ donor
```

尤其 P3。

因此 A3 必须区分：

```text
recipient-specific information
```

和：

```text
generic additive perturbation / learned prior / channel bias
```

---

## 10.2 只在 standalone scale context 做主分解

为避免 P3/P4/P5 interaction 再次污染解释，A3-D 主实验只做：

```text
target scale i ON
other two scales = 0
```

也就是 standalone context。

Full-context 只保留 A2 的 conditional evidence，不在 A3-D 重做复杂 decomposition。

---

## 10.3 residual 分解

对 recipient `x`、scale `i`：

\[
\delta_i^x
\]

定义 native per-channel DC：

\[
b_i^x(c)
=
mean_{h,w}\delta_i^x(c,h,w)
\]

broadcast：

\[
DC_i^x(c,h,w)=b_i^x(c)
\]

AC：

\[
AC_i^x
=
\delta_i^x-DC_i^x
\]

定义 leave-one-out residual mean tensor：

\[
MEAN_i^{-x}
=
\frac{1}{5}
\sum_{y\neq x}\delta_i^y
\]

以及 leave-one-out mean DC：

\[
MEAN\_DC_i^{-x}
=
broadcast\left(
\frac{1}{5}
\sum_{y\neq x}b_i^y
\right)
\]

所有 mean 都必须排除 recipient `x`。

禁止用全 6 张平均后再评 recipient。

---

## 10.4 A3-D 正式 conditions

每个 scale：

```text
ZERO_i
NATIVE_i
DONOR_i
LOO_MEAN_i
NATIVE_DC_i
NATIVE_AC_i
LOO_MEAN_DC_i
```

其中：

- `ZERO_i`、`NATIVE_i`、`DONOR_i` 可以直接复用 A2 standalone 结果；
- A3 新 forward 必须只为：
  - `LOO_MEAN_i`
  - `NATIVE_DC_i`
  - `NATIVE_AC_i`
  - `LOO_MEAN_DC_i`

不得因为实现方便而重新定义 A2 的 native/donor 条件。

---

## 10.5 Generic bias 核心效应

### Recipient-specific advantage over mean

\[
\Delta^{native-mean}_i
=
AP(NATIVE_i)-AP(LOO\_MEAN_i)
\]

### Mean utility

\[
U^{mean}_i
=
AP(LOO\_MEAN_i)-AP(ZERO_i)
\]

### Native DC utility

\[
U^{dc}_i
=
AP(NATIVE\_DC_i)-AP(ZERO_i)
\]

### Native AC utility

\[
U^{ac}_i
=
AP(NATIVE\_AC_i)-AP(ZERO_i)
\]

### Generic DC utility

\[
U^{meanDC}_i
=
AP(LOO\_MEAN\_DC_i)-AP(ZERO_i)
\]

全部同时给：

```text
full
LOO median
positive folds
negative folds
```

---

## 10.6 Generic-bias interpretation

A3 不设置一个武断的 AP margin。

### GENERIC_COMPONENT_SUPPORTED

某尺度满足：

```text
U_mean:
  FIXED full > 0
  FIXED LOO median > 0
  FIXED positive folds >= 4/6
  SOFT full > 0
```

同时：

```text
Δ_native-mean
不是 STRONG_POSITIVE
```

则可写：

> recipient-independent mean residual carries replicated utility, while native recipient identity has not established added value.

### GENERIC_DC_SUPPORTED

若 `U_meanDC` 或 `U_dc` 得到同类 replicated positive evidence，允许进一步写：

> a spatially constant/channel-bias component contributes measurable utility.

### SPATIAL_AC_SUPPORTED

若 `U_ac` replicated positive，而 DC/meanDC 不成立：

> utility depends more on spatially varying residual structure than on pure channel bias.

这些标签可以同时成立。

---

# 11. A3 不把“mean residual 有用”误写成“IR 有语义价值”

必须区分：

```text
LOO mean residual improves AP
```

和：

```text
paired IR contains recipient-specific information
```

前者只说明：

> learned auxiliary branch has produced a reusable additive prior.

它不能证明：

> detector is using the correct IR observation of the current scene.

A3-D 的目标正是把这两者拆开。

---

# 12. A3 核心输出矩阵

最终必须按 scale 输出：

| Scale | Registration rescue | Spatial recipient-specific | Semantic recipient-specific | Generic component | Generic DC | Spatial AC |
|---|---|---|---|---|---|---|
| P3 | ... | ... | ... | ... | ... | ... |
| P4 | ... | ... | ... | ... | ... | ... |
| P5 | ... | ... | ... | ... | ... | ... |

不得只写一句：

```text
P5 bad
```

或：

```text
registration problem
```

---

# 13. A3 决策逻辑

A3 不选择“唯一根因”，但规定后续路线。

## Branch R — Registration

如果某尺度：

```text
registration shift rescue = STRONG_POSITIVE_RESCUE
```

则下一步优先：

```text
R1 — RGB/IR Registration Correction Experiment
```

先修几何配准，再讨论 fusion architecture。

此时：

```text
HOLD q3/q4/q5
HOLD new gate
HOLD static per-scale training
```

---

## Branch S — Representation spatial mismatch

如果：

```text
registration rescue not positive
```

但：

```text
spatial recipient-specific = weak / donor-favored
```

尤其伴随：

```text
corr_at_best >> corr_at_zero
```

或不同尺度空间结构严重不一致，

则下一步优先审：

```text
AuxEncoder receptive field
projection semantics
feature stride / resize correspondence
P3/P4/P5 representation alignment
```

而不是 gate。

---

## Branch M — Semantic mismatch

如果：

```text
raw/spatial geometry mostly acceptable
```

但：

```text
semantic recipient-specific = weak / donor-favored
```

则优先进入：

```text
object-semantic representation audit / redesign
```

候选后续可以研究：

- object-aware alignment；
- region-level supervision；
- auxiliary feature distillation；
- late fusion / decision fusion；

但 A3 本身不训练这些。

---

## Branch B — Generic bias

如果：

```text
GENERIC_COMPONENT_SUPPORTED
```

且：

```text
native-vs-mean recipient-specific advantage not established
```

则正式结论应是：

> 当前 residual utility 至少有一部分来自 recipient-independent learned perturbation，而不是正确 paired IR 内容。

此时继续训练 dynamic reliability gate 缺少因果基础。

优先：

```text
redesign how IR information is represented/used
```

而不是：

```text
design a better q predictor
```

---

## Branch U — unresolved

如果四条证据都不形成稳定方向：

```text
A3_DIAGNOSIS_INCONCLUSIVE
```

下一步应扩大诊断样本/检查采集与数据契约，而不是根据 6 张 val 图继续加复杂 gate。

---

# 14. A3 统计纪律

A3 继续使用：

```text
full val6
+ leave-one-out
```

AP intervention 证据沿用：

```text
full sign
LOO median sign
positive/negative fold counts
SOFT replication full sign
```

空间/语义 per-sample metric 使用：

```text
median sign
positive/negative sample counts
SOFT replication median sign
```

不添加事后：

```text
+0.01
+0.02
5% relative gain
```

等阈值。

A3 是小样本诊断，不做 p-value significance theater。

---

# 15. A3 硬门禁

## A3-G1 UPSTREAM

必须验证：

```text
A2 result SHA
A2 all_gates_passed
F1-C frozen identity
A2 donor map SHA
```

全部正确。

失败：

```text
A3_UPSTREAM_FREEZE_FAIL
```

---

## A3-G2 CHECKPOINT / DEPENDENCY

FIXED/SOFT last.pt 与整个 frozen dependency closure 必须一致。

失败：

```text
A3_FROZEN_DEPENDENCY_CLOSURE_FAIL
```

---

## A3-G3 EVAL-ONLY

禁止：

```text
optimizer
backward
train()
parameter update
buffer mutation
```

每个系统：

```text
state_sha256_before == state_sha256_after
```

失败：

```text
A3_PARAMETER_MUTATION
```

---

## A3-G4 NATIVE EQUIVALENCE

A3 untouched native forward 必须与 A2 M111/native detection tensor 逐样本 bitwise 一致。

失败：

```text
A3_NATIVE_EQUIVALENCE_FAIL
```

---

## A3-G5 Q FREEZE

SOFT 所有 intervention：

```text
q_native == untouched recipient q_native
```

FIXED：

```text
q_native == 1
```

失败：

```text
A3_Q_FREEZE_FAIL
```

---

## A3-G6 DONOR FREEZE

A3 donor map 必须逐项等于 A2 donor map，且 SHA：

```text
c5cd8e852663eae3243bc5e4c263c6f2c26c7b2faa977ae3b60cb5c1ea122af5
```

失败：

```text
A3_DONOR_MAP_DRIFT
```

---

## A3-G7 REGISTRATION CROSS-FIT

必须证明每个 held-out sample 的 shift：

```text
只由另外 5 张样本估计
```

不得读取 held-out AP/GT 来选 shift。

需要落盘：

```text
train_ids_for_shift
raw_shift_rows
median_shift
feature_cell_shift
```

失败：

```text
A3_REGISTRATION_LEAKAGE
```

---

## A3-G8 POST-PROJECTION INTERVENTION

SHIFT / DONOR / MEAN / DC / AC 只能发生在：

```text
projection -> δ_i -> A3 intervention -> residual add
```

source trace 必须证明：

- target tensor 来自正确 source；
- 其他 scale 未被误换；
- gate 未被重新调用。

失败：

```text
A3_RESIDUAL_INTERVENTION_SEMANTICS_FAIL
```

---

## A3-G9 SEMANTIC MASK

GT mask 构造必须与 frozen labels 对齐。

任何尺度有效 semantic sample 少于 5/6：

```text
A3_SEMANTIC_COVERAGE_FAIL
```

---

## A3-G10 LOO MEAN NO-SELF

对 recipient `x`：

```text
LOO_MEAN_i(x)
```

必须严格只使用另外 5 张 residual。

失败：

```text
A3_LOO_MEAN_SELF_LEAKAGE
```

---

## A3-G11 STOCK EVAL

所有 AP 继续使用 Step3 authoritative validator semantics。

禁止 stock `/255` preprocess 漂移。

---

## A3-G12 PROVENANCE

必须记录：

- A3 DESIGN_FREEZE SHA；
- evaluator SHA；
- intervention/metric helper SHA；
- A2 result SHA；
- A2 donor map SHA；
- F1-C summary SHA；
- FIXED/SOFT last.pt SHA；
- FIXED/SOFT manifests；
- model/gate/dataset/F0/AuxEncoder/fusion/trainability/eval helper SHAs；
- contract SHA；
- torch / ultralytics versions；
- val6 ids/order；
- registration estimator config；
- semantic mask config；
- all condition identities。

---

## A3-G13 INTERPRETATION

不是 runtime boolean promotion gate，但正式报告必须遵守：

- FIXED primary / SOFT replication 分开；
- 不把 correlation 当因果；
- registration causal claim 必须依赖 cross-fitted shift rescue；
- residual utility 不等于 paired semantic value；
- mean/DC utility 不等于 multimodal information use；
- 不根据 A3 结果反改 label 规则。

任一 G1–G12 失败：

```text
A3_ABORT
```

不得输出正式机制标签。

---

# 16. A3 预执行测试最低集合

实现前必须至少覆盖：

### Dependency / provenance

1. A2 result SHA mismatch -> fail；
2. donor map SHA mismatch -> fail；
3. checkpoint/source/version drift -> fail；
4. val6 order drift -> fail；
4a. modality_preprocess Git blob drift -> fail。

### Registration

5. held-out id 不得出现在自己的 shift training ids；
6. median shift 仅由其他 5 个 raw shift 计算；
7. shift zero-fill，不得 wrap；
8. feature stride conversion 正确；
9. per-sample best feature shift 不能被 AP evaluator 使用。

### q / forward

10. FIXED q 恒 1；
11. SOFT intervention 不改变 q；
12. untouched native == A2 native bitwise；
13. state SHA before==after。

### Spatial / semantic

14. native/donor energy map shape 完全一致；
15. donor metric 使用 recipient RGB / recipient GT，而不是 donor GT；
16. GT box -> feature mask coordinate test；
17. degenerate semantic mask fail-fast；
18. semantic valid coverage <5/6 -> fail。

### Generic bias

19. LOO mean 排除 recipient；
20. `NATIVE_DC` 是 spatial broadcast；
21. `NATIVE_AC` 使用 `native - native_DC`；
22. `LOO_MEAN_DC` 不含 recipient；
23. A3 新 conditions 只改 target scale；
24. A2 ZERO/NATIVE/DONOR identity 复用时数值/condition 身份一致。

### Interpretation

25. fixed positive / soft negative -> 不得标 STRONG；
26. metric-only correlation 不得触发 registration causal label；
27. generic mean utility positive 但 native-mean 未正 -> GENERIC_COMPONENT_SUPPORTED；
28. no branch evidence -> A3_DIAGNOSIS_INCONCLUSIVE。

---

# 17. 正式产物

建议冻结输出：

```text
docs/step4_a3/DESIGN_FREEZE.md

reports/step4_a3/preexecution_audit.json
reports/step4_a3/raw_registration.json
reports/step4_a3/registration_rescue.json
reports/step4_a3/spatial_correspondence.json
reports/step4_a3/semantic_agreement.json
reports/step4_a3/generic_residual_bias.json
reports/step4_a3/a3_summary.json
```

其中：

```text
a3_summary.json
```

必须是唯一正式决策入口。

它至少包含：

```text
schema
all_gates_passed
G1...G12
primary / replication identities
upstream provenance
per-scale mechanism matrix
registration rescue labels
spatial labels
semantic labels
generic-bias labels
decision branches
interpretation_notes
```

---

# 18. A3 明确禁止

```text
NO training
NO optimizer
NO backward
NO new seed
NO new checkpoint
NO best.pt primary
NO Depth
NO q3/q4/q5
NO new learned gate
NO static per-scale training
NO online alignment training
NO dataset-level IR shuffle
NO intervention before gate
NO per-sample AP-selected shift
NO GT-selected registration shift
NO donor-map change
NO val6 change
NO F1-C/A2 evidence rewrite
NO post-result threshold edits
```

---

# 19. A3 完成后的合法结论形式

允许：

> On the frozen val6 and the two frozen trained systems, cross-fitted spatial correction produced replicated positive rescue at P5, implicating registration as a contributor to the negative paired P5 effect.

允许：

> P3 residual utility was largely reproduced by leave-one-out mean residual while native-vs-mean recipient-specific advantage was not established, supporting a generic residual component.

允许：

> Native P4 residual showed stronger object-region localization than donor residual in FIXED but not SOFT; semantic recipient specificity is therefore inconclusive.

禁止：

> IR registration is the root cause.

禁止：

> P5 is universally harmful.

禁止：

> The model does not understand IR.

禁止：

> Static weights will fix the problem.

禁止：

> Better q3/q4/q5 should solve it.

---

# 20. 当前正式路线状态

```text
F1-C
  CLOSED / FAILED

A2
  CLOSED / DIAGNOSTIC COMPLETE
  NO_SCALE_MEETS_STABLE_PAIRED_POSITIVE_CRITERION

A3
  PRE-EXECUTION
  RGB–IR Spatial / Semantic Agreement Audit
  EVALUATION-ONLY

HOLD
  static per-scale fusion training
  single-scale IR training
  q3/q4/q5
  new reliability gate
  Depth branch changes
```

A3 只有在 G1–G12 全部 PASS 后，才允许把任何机制判断写入正式工程状态。
