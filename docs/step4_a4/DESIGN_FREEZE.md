# A4 DESIGN_FREEZE — Residual DC/AC Paired Causality Audit

状态：**PRE-EXECUTION / EVALUATION-ONLY / DESIGN FROZEN**

---

## 0. 上游正式状态

```text
F1-C
  CLOSED / FAILED @ bf983c4
  F1C_GATE_FAILED_CAUSAL_PROTOCOL

A2
  CLOSED / DIAGNOSTIC COMPLETE
  RESULT ACCEPTED @ 2188593bf9797f0097efe7e4a230694cc19502fb
  no stable paired-positive scale

A3
  CLOSED / DIAGNOSTIC COMPLETE
  RESULT ACCEPTED @ 4e15c1ec2cd64af39031d3fcfde200f2d248b65a
  all_gates_passed = true
  G1–G12 PASS
```

A3 正式结果：

```text
P3:
  registration rescue         INCONCLUSIVE
  spatial specificity         INCONCLUSIVE
  semantic specificity        STRONG_RECIPIENT_SPECIFIC
  generic component           GENERIC_COMPONENT_SUPPORTED
  generic DC                  NOT_ESTABLISHED
  spatial AC                  SPATIAL_AC_SUPPORTED

P4:
  registration rescue         INCONCLUSIVE
  spatial specificity         INCONCLUSIVE
  semantic specificity        INCONCLUSIVE
  generic component           NOT_ESTABLISHED
  generic DC                  NOT_ESTABLISHED
  spatial AC                  NOT_ESTABLISHED

P5:
  registration rescue         INCONCLUSIVE
  spatial specificity         STRONG_RECIPIENT_SPECIFIC
  semantic specificity        INCONCLUSIVE
  generic component           NOT_ESTABLISHED
  generic DC                  NOT_ESTABLISHED
  spatial AC                  SPATIAL_AC_SUPPORTED
```

A2 同时冻结：

```text
P5 standalone pairedness  = STRONG_NEGATIVE
P5 conditional pairedness = STRONG_NEGATIVE
```

A3 进一步观察到：

```text
P5 U_ac   > 0 across FIXED/SOFT
P5 U_dc   < 0 across FIXED/SOFT
P5 U_mean < 0 across FIXED/SOFT
```

但 A3 的：

```text
U_ac = AP(native AC) - AP(ZERO)
```

只证明 **AC utility**，并没有证明：

```text
AP(native AC) > AP(donor AC)
```

因此 A4 的核心任务是把：

```text
AC utility
```

与：

```text
AC paired causality
```

严格拆开。

---

# 1. A4 唯一正式问题

A4 只回答：

> **对冻结的已训练 IR residual 系统，移除 post-projection residual 的空间 DC 分量后，AC-only residual 是否恢复 recipient-specific paired causal value，并且这种恢复是否同时改善或至少不损害 native detection performance？**

A4 的主假设是 P5。

次级问题：

- P3：centering 能否把已存在的 recipient-specific semantic information 转化成 paired AP value？
- P4：作为 diagnostic control，观察 FIXED/SOFT 不稳定性是否继续存在。

---

# 2. A4 明确不回答

A4 不回答：

- 新 gate 是否应该训练；
- `q3/q4/q5` 是否有效；
- static per-scale weights 是否值得训练；
- projection bias 参数是否有害；
- AuxEncoder 是否需要重训；
- object-conditioned residual 是否优于当前 residual；
- Depth 如何融合；
- 是否应做 online registration。

A4 只做 **evaluation-only causal decomposition**。

---

# 3. 冻结 checkpoint / data / donor / q

## 3.1 Primary

```text
runs/step4_f1_c/F1C-I-fixed/weights/last.pt
```

## 3.2 Replication

```text
runs/step4_f1_c/F1C-I-soft/weights/last.pt
```

仍然：

```text
FIXED = primary
SOFT  = replication
```

禁止解释成同权重 q ablation。

只允许：

```text
last.pt
```

不使用 `best.pt`。

---

## 3.3 val6

继续使用 A2/A3 同一 val6，身份和顺序完全冻结：

```text
000003_013_00000085
000004_013_00000081
000004_014_00000001
000016
000016_001_00000001
000016_042_suppl_00000164
```

---

## 3.4 donor map

直接复用 A2/A3 donor map：

```text
donor_map_sha256 =
c5cd8e852663eae3243bc5e4c263c6f2c26c7b2faa977ae3b60cb5c1ea122af5
```

不得重新生成。

---

## 3.5 q

所有 SOFT condition 必须：

```text
recipient untouched A3/A4/A5
        ↓
q_native
        ↓
freeze q_native
        ↓
post-projection residual DC/AC intervention
```

FIXED：

```text
q_native ≡ 1
```

禁止任何 intervention 反向改变 gate input。

---

# 4. 冻结 residual 定义

对 scale `i ∈ {P3,P4,P5}`：

```text
δ_i = P_i(A_i)
```

所有 A4 decomposition 都发生在：

```text
projection -> δ_i
```

之后。

---

# 5. Primary DC/AC 定义：A3-compatible full-map DC

A4 primary 必须保留 A3 的原始定义：

\[
DC^{all}_i(x,c)
=
mean_{h,w}\left(\delta_i(x,c,h,w)\right)
\]

broadcast：

\[
\widehat{DC}^{all}_i(x,c,h,w)
=
DC^{all}_i(x,c)
\]

定义：

\[
AC^{all}_i
=
\delta_i-\widehat{DC}^{all}_i
\]

代码语义：

```text
DC_ALL = residual.mean(dim=(-2,-1), keepdim=True).expand_as(residual)
AC_ALL = residual - DC_ALL
```

这一组是 **A4 primary causal definition**。

原因：

> A4 首先要验证 A3 已观测到的那个 full-map DC-negative / AC-positive decomposition 是否能恢复 pairedness。

A4 不允许因为 content-aware 定义更物理而事后替换 primary。

---

# 6. Diagnostic-only：padding-aware content DC

A3 的 `DC_all` 对整个 feature map 求平均。

当前冻结输入：

```text
RGB letterbox pad = 114/255
IR letterbox pad  = 0
```

AuxEncoder / BN / projection 后 padding 区域不保证 residual 为零。

因此：

\[
mean_{all\ HW}(\delta)
\]

可能混入：

```text
letterbox-padding-induced feature statistics
```

A4 增加 diagnostic-only content-aware DC。

---

## 6.1 Content mask 来源

对每个 sample，使用冻结 dataset sample 中：

```text
ori_shape
ratio_pad
```

恢复最终 640×640 letterbox 中真实内容 rectangle。

输入 content mask：

```text
M_input ∈ {0,1}^{H×W}
```

真实内容 rectangle 内 = 1，padding = 0。

不得读取：

```text
prediction
GT
AP
feature correlation
```

来构造 content mask。

---

## 6.2 Feature-scale content mask

对 P3/P4/P5：

```text
coverage_i = adaptive_avg_pool2d(M_input, feature_hw)
```

得到：

```text
coverage_i ∈ [0,1]
```

正式 content-mean 权重使用：

```text
coverage_i
```

而不是先硬二值化。

这是为了避免 feature cell 跨 content/padding 边界时，把 partial coverage 全算成完整内容。

---

## 6.3 Content DC

\[
DC^{content}_i(c)
=
\frac{
\sum_{h,w} coverage_i(h,w)\delta_i(c,h,w)
}{
\sum_{h,w} coverage_i(h,w)
}
\]

broadcast：

\[
\widehat{DC}^{content}_i
\]

定义：

\[
AC^{content}_i
=
\delta_i-\widehat{DC}^{content}_i
\]

注意：

```text
AC_CONTENT
```

仍然在整个 feature map 上减去同一个 content-estimated channel mean。

它不是把 padding 区域清零。

---

## 6.4 Donor content mean

对 donor AC-content，必须使用：

```text
donor residual
+
donor own ori_shape / ratio_pad
+
donor own feature content coverage
```

求 donor 自己的 `DC_content`。

严禁：

```text
donor residual - recipient content mean
```

---

## 6.5 Content diagnostic 解释边界

A4 primary route decision **不允许只由 AC_content 触发**。

它用于回答：

> full-map centering 的效果是否主要来自 padding/global feature statistics，还是内容区域 DC 本身。

解释：

```text
AC_all rescues, AC_content does not
  → padding/global feature statistics likely contribute

AC_all and AC_content both rescue
  → stronger support for harmful post-projection content/global DC

AC_content rescues but AC_all does not
  → full-map mean may mask a content-specific DC effect
```

这是 diagnostic interpretation，不改变 primary endpoint。

---

# 7. Donor AC 的硬定义

对 donor `d`：

\[
AC^{all}_{donor}
=
\delta^d
-
mean_{HW}(\delta^d)
\]

必须用 donor 自己的 mean。

严禁：

\[
\delta^d-mean(\delta^{recipient})
\]

同理：

\[
AC^{content}_{donor}
\]

必须使用 donor 自己的 residual 与 donor 自己的 content coverage。

然后 donor AC tensor 才可注入 recipient target scale。

recipient 的：

```text
RGB
GT
q_native
other-scale residuals
```

全部保持 recipient 身份。

---

# 8. A4 两个核心 endpoint

A4 不能只看一个 AP 差。

---

## 8.1 AC paired causality

对 scale `i`：

\[
\Delta^{AC-pair}_i
=
AP(AC^{recipient}_i)
-
AP(AC^{donor}_i)
\]

回答：

> AC-only utility 是否依赖正确 paired IR？

---

## 8.2 Centering rescue

\[
\Delta^{center}_i
=
AP(AC^{recipient}_i)
-
AP(FULL^{recipient}_i)
\]

回答：

> 去掉 DC 本身是否改善 native detector performance？

两者必须独立报告。

---

# 9. A4 正式 block matrix

## A4-S — Standalone AC pairedness

只开启 target scale `i`。

### Native

```text
AC_ALL_NATIVE_i_ONLY
```

### Donor

```text
AC_ALL_DONOR_i_ONLY
```

其他尺度 residual：

```text
0
```

正式 paired effect：

\[
\Delta^{AC-pair,standalone}_i
=
AP(AC\_ALL\_NATIVE_i\_ONLY)
-
AP(AC\_ALL\_DONOR_i\_ONLY)
\]

---

## A4-C — Conditional AC pairedness

target scale 用 AC。

其他两个尺度保持 native full residual。

### Native

```text
AC_ALL_NATIVE_i_COND
```

### Donor

```text
AC_ALL_DONOR_i_COND
```

正式 paired effect：

\[
\Delta^{AC-pair,conditional}_i
\]

---

## A4-R — Centering rescue

### Standalone rescue

\[
\Delta^{center,standalone}_i
=
AP(AC\_ALL\_NATIVE_i\_ONLY)
-
AP(FULL\_NATIVE_i\_ONLY)
\]

其中 full native standalone baseline 直接复用 A2：

```text
P3 -> M100
P4 -> M010
P5 -> M001
```

### Conditional rescue

\[
\Delta^{center,conditional}_i
=
AP(AC\_ALL\_NATIVE_i\_COND)
-
AP(M111)
\]

---

## A4-F — 2^3 DC-removal factorial

每个 bit 表示该 scale：

```text
0 = FULL native residual
1 = AC_ALL native residual
```

固定 8 条：

```text
C000  FULL/FULL/FULL
C100  AC/FULL/FULL
C010  FULL/AC/FULL
C001  FULL/FULL/AC
C110  AC/AC/FULL
C101  AC/FULL/AC
C011  FULL/AC/AC
C111  AC/AC/AC
```

这里：

```text
C000 == A2 M111 == A3 native
```

必须逐样本 raw detection tensor bitwise identity。

定义 centering main effects：

\[
R_3 = AP(C100)-AP(C000)
\]

\[
R_4 = AP(C010)-AP(C000)
\]

\[
R_5 = AP(C001)-AP(C000)
\]

同时报告：

```text
pair interactions
three-way interaction
```

但不得用 factorial 结果替换 A4-S/A4-C 的 paired-causality endpoint。

---

## A4-P — Padding/content DC diagnostic

每尺度评估：

### Standalone

```text
AC_CONTENT_NATIVE_i_ONLY
AC_CONTENT_DONOR_i_ONLY
```

### Conditional

```text
AC_CONTENT_NATIVE_i_COND
AC_CONTENT_DONOR_i_COND
```

同时比较：

```text
AC_ALL_NATIVE
vs
AC_CONTENT_NATIVE
```

以及：

```text
AC_ALL native-donor
vs
AC_CONTENT native-donor
```

A4-P 是 diagnostic-only。

它不得独立产生 training GO。

---

# 10. Primary / secondary / control

## Primary hypothesis

```text
P5
```

理由：

```text
A2:
  paired standalone  STRONG_NEGATIVE
  paired conditional STRONG_NEGATIVE

A3:
  spatial recipient specificity STRONG_RECIPIENT_SPECIFIC
  U_ac   positive across FIXED/SOFT
  U_dc   negative across FIXED/SOFT
  U_mean negative across FIXED/SOFT
```

---

## Secondary

```text
P3
```

问题：

> centering 是否能把已有 recipient-specific semantic information 转化成 paired AP value？

---

## Diagnostic control

```text
P4
```

P4 不允许因为单个 A4 正数字事后升级成主路线。

---

# 11. A4 paired-effect label

A4-S / A4-C 沿用 A2 sign-stability discipline。

### STRONG_POSITIVE

```text
FIXED full > 0
FIXED LOO median > 0
FIXED positive folds >= 4/6
SOFT full > 0
```

### STRONG_NEGATIVE

```text
FIXED full < 0
FIXED LOO median < 0
FIXED negative folds >= 4/6
SOFT full < 0
```

其余：

```text
INCONCLUSIVE
```

不加任意 AP margin。

---

# 12. Centering rescue label

对：

```text
AC native - FULL native
```

分别 standalone / conditional。

### STRONG_POSITIVE_RESCUE

```text
FIXED full > 0
FIXED LOO median > 0
FIXED positive folds >= 4/6
SOFT full > 0
```

### STRONG_NEGATIVE_RESCUE

同理反号。

否则：

```text
INCONCLUSIVE
```

---

# 13. A4 joint decision

P5 真正值得进入 centering training，最好同时满足：

```text
AC pairedness:
  STRONG_POSITIVE

AND

centering rescue:
  STRONG_POSITIVE_RESCUE
```

最强证据：

```text
AC recipient > AC donor
AND
AC recipient > FULL recipient
```

且 FIXED / SOFT / LOO 方向一致。

---

# 14. A4 分叉预注册

## Branch A — CENTERING TRAINING GO

若 P5：

```text
AC pairedness = STRONG_POSITIVE
AND
centering rescue = STRONG_POSITIVE_RESCUE
```

下一步：

```text
A4T — parameter-free residual centering training experiment
```

候选结构：

\[
F_i
=
R_i
+
q\left(
\delta_i
-
mean_{HW}(\delta_i)
\right)
\]

首次训练仍不引入：

```text
q3/q4/q5
new gate
static learned per-scale weights
```

---

## Branch B — paired specificity restored, AP not rescued

若：

```text
AC pairedness = STRONG_POSITIVE
but
centering rescue not positive
```

结论：

> centering improves recipient-specific causal use but may remove a generic component that still contributes to detector performance.

下一步：

```text
continue decomposition
NO training GO
```

---

## Branch C — AP rescued, pairedness not restored

若：

```text
centering rescue = STRONG_POSITIVE
but
AC pairedness != STRONG_POSITIVE
```

结论：

> centering acts as an architectural regularizer, but paired IR causal value remains unproven.

不得写：

```text
paired IR information restored
```

也不得把它作为 multimodal-fusion success 直接开训练。

---

## Branch D — AC remains paired-negative

若 P5：

```text
AC pairedness = STRONG_NEGATIVE
```

则：

```text
STOP CENTERING ROUTE
```

下一步进入：

```text
projection channel semantics
aux representation
object-conditioned residual
local spatial correspondence
```

继续 HOLD reliability gate。

---

## Branch E — full-map/content divergence

若：

```text
AC_all rescue positive
AC_content not positive
```

先进入：

```text
padding/global-statistics source audit
```

不得宣称真实内容 DC harmful。

若二者均稳定 positive：

> stronger evidence that the post-projection spatial DC itself is harmful.

---

# 15. DC harm ≠ projection bias parameter harm

当前：

\[
\delta = WA+b
\]

所以：

\[
mean(\delta)
=
W\,mean(A)+b
\]

A4 最多只能证明：

> **post-projection spatial DC component is causally harmful**

不能证明：

> Conv2d bias parameter is harmful

因为 DC 包含：

```text
projection bias b
+
content-induced W mean(A)
```

若 A4 成功，拆二者属于：

```text
A4b / A5
```

---

# 16. Runtime hard gates

## G1 — upstream freeze

必须验证：

```text
A2 result SHA
A3 summary SHA
A3 all_gates_passed
A2/A3 donor map SHA
F1-C summary SHA
```

A3 source-root summary raw SHA：

```text
121dacc0ed50f5d24a8108ea3710e981c3c0314210729c80ed339652ea579839
```

mirror canonical-LF SHA：

```text
3523cb526d7a0fde3b0f0f121f73f29326aa88167bf6ad60d0505d7fed50d9ed
```

raw 与 canonical 必须分字段比较，不得混用。

失败：

```text
A4_UPSTREAM_FREEZE_FAIL
```

---

## G2 — frozen dependency closure

继续验证：

- Step3 eval helper
- F1 model
- gate
- dataset
- F0
- AuxEncoder
- fusion
- trainability
- causality helpers
- raw sample index
- modality_preprocess
- contract
- torch / ultralytics
- manifests
- last.pt
- val6 identity/order

失败：

```text
A4_FROZEN_DEPENDENCY_CLOSURE_FAIL
```

---

## G3 — evaluation-only state

禁止：

```text
optimizer
backward
train()
parameter update
buffer mutation
```

要求：

```text
state_sha256_before == state_sha256_after
```

失败：

```text
A4_PARAMETER_MUTATION
```

---

## G4 — native equivalence

A4 native FULL forward 必须逐样本等于：

```text
checkpoint native
A2 M111
A3 native
```

失败：

```text
A4_NATIVE_EQUIVALENCE_FAIL
```

---

## G5 — q freeze

所有 FULL / AC_ALL / AC_CONTENT / donor condition：

```text
q == untouched recipient q_native
```

失败：

```text
A4_Q_FREEZE_FAIL
```

---

## G6 — donor freeze

donor map 必须逐项等于 A2/A3 map。

失败：

```text
A4_DONOR_MAP_DRIFT
```

---

## G7 — donor AC self-centering

对 donor AC：

```text
donor residual
-
donor own mean
```

必须动态 trace：

```text
residual_source_id == donor_id
mean_source_id == donor_id
```

AC-content 还必须：

```text
content_mask_source_id == donor_id
```

失败：

```text
A4_DONOR_AC_MEAN_SOURCE_FAIL
```

---

## G8 — full-map DC semantics

动态证明：

```text
DC_ALL == mean_HW(residual)
AC_ALL == residual - DC_ALL
```

失败：

```text
A4_DC_AC_DECOMPOSITION_FAIL
```

---

## G9 — content-mask provenance

content mask 只能来自：

```text
ori_shape + ratio_pad
```

不得来自：

```text
GT
prediction
AP
feature-correlation optimum
```

失败：

```text
A4_CONTENT_MASK_PROVENANCE_FAIL
```

---

## G10 — content DC coverage

每个 sample/scale：

```text
sum(coverage_i) > 0
```

若无 letterbox padding，允许 coverage 全 1，但必须显式记录：

```text
full_content_coverage = true
```

失败：

```text
A4_CONTENT_DC_COVERAGE_FAIL
```

---

## G11 — post-projection intervention

所有 DC/AC manipulation 必须只发生在：

```text
projection -> δ -> decomposition -> residual add
```

失败：

```text
A4_RESIDUAL_INTERVENTION_SEMANTICS_FAIL
```

---

## G12 — factorial completeness

必须恰好：

```text
C000 C100 C010 C001 C110 C101 C011 C111
```

并且：

```text
C000 == A2 M111 == A3 native
```

失败：

```text
A4_FACTORIAL_INCOMPLETE
```

---

## G13 — stock eval

所有 AP 继续使用 Step3 authoritative validator semantics。

失败：

```text
A4_STOCK_EVAL_SEMANTICS_FAIL
```

---

## G14 — provenance complete

必须记录：

- A4 DESIGN_FREEZE SHA
- evaluator SHA
- DC/AC helper SHA
- content-mask helper SHA
- tests SHA
- audit SHA
- A2 result SHA
- A3 summary raw SHA
- A3 summary canonical-LF SHA
- donor map SHA
- F1-C summary SHA
- checkpoint / manifest SHAs
- frozen dependency SHAs
- contract SHA
- modality_preprocess SHA/blob
- torch / ultralytics
- val6 ids/order
- all condition identities
- DC definition version
- content coverage definition

失败：

```text
A4_PROVENANCE_INCOMPLETE
```

---

# 17. Interpretation discipline

正式报告必须保留：

```text
AC utility != AC paired causality
centering rescue != paired restoration
DC harm != projection bias parameter harm
AC_content is diagnostic-only
FIXED primary / SOFT replication
P5 primary / P3 secondary / P4 control
```

任一 G1–G14 失败：

```text
A4_ABORT
```

不得输出正式机制标签。

---

# 18. 最低测试集合

至少覆盖：

1. A3 summary raw SHA mismatch fail
2. A3 all_gates_passed false fail
3. donor drift fail
4. val6 order drift fail
5. source/version drift fail
6. DC_ALL spatial constant per channel
7. AC_ALL full-map channel mean≈0
8. FULL == AC_ALL + DC_ALL
9. donor AC uses donor own mean
10. recipient mean contamination fail
11. weighted DC_content exact
12. AC_content == residual - DC_content
13. donor AC_content uses donor own content coverage
14. letterbox content rectangle reconstruction
15. feature coverage shape P3/P4/P5
16. content mask has no GT/pred dependency
17. zero coverage fail
18. full-content coverage explicit pass
19. 16:9 content coverage sanity
20. q computed before decomposition
21. q unchanged native vs donor AC
22. FIXED q==1
23. state before==after
24. only target scale replaced in S/C/R
25. donor cache never calls gate
26. AC native-vs-donor classification
27. centering rescue separately classified
28. paired positive + rescue negative cannot yield training GO
29. rescue positive + paired inconclusive cannot claim paired restoration
30. paired negative triggers STOP CENTERING ROUTE
31. exactly 8 factorial cells
32. C000 FULL/FULL/FULL
33. C111 AC/AC/AC
34. C000 native bitwise == A2/A3 native
35. factorial interactions use correct cells
36. DC-negative cannot emit “projection bias harmful”
37. U_ac>0 alone cannot emit “paired-positive”
38. AC_content-only rescue cannot trigger primary training GO
39. P4 isolated positive cannot promote P4
40. all G1–G14 dynamic, no unconditional True

---

# 19. 正式输出

```text
docs/step4_a4/DESIGN_FREEZE.md

reports/step4_a4/preexecution_audit.json
reports/step4_a4/ac_paired_standalone.json
reports/step4_a4/ac_paired_conditional.json
reports/step4_a4/centering_rescue.json
reports/step4_a4/dc_removal_factorial.json
reports/step4_a4/content_dc_diagnostic.json
reports/step4_a4/a4_summary.json
```

只有：

```text
a4_summary.json
```

拥有正式分叉权。

---

# 20. A4 summary 必须至少包含

```text
schema
all_gates_passed
G1...G14
interpretation_discipline

primary = P5
secondary = P3
control = P4

pairedness:
  standalone:
    P3/P4/P5
  conditional:
    P3/P4/P5

centering_rescue:
  standalone:
    P3/P4/P5
  conditional:
    P3/P4/P5

factorial:
  C000...C111
  main effects
  pair interactions
  three-way interaction

content_dc_diagnostic:
  AC_all vs AC_content
  native and donor
  per scale
  per system

joint_decision:
  P5 paired-restoration status
  P5 performance-rescue status
  next branch
```

---

# 21. 明确禁止

```text
NO training
NO optimizer
NO backward
NO new checkpoint
NO new seed
NO new val samples
NO donor-map change
NO q3/q4/q5
NO new gate
NO learned static weights
NO best.pt
NO Depth
NO pre-gate intervention
NO dataset-level IR shuffle
NO donor residual centered by recipient mean
NO donor content AC using recipient content mask
NO GT-derived content mask
NO AP-selected DC definition
NO replacing DC_all primary with DC_content
NO post-result threshold edits
NO calling DC harm "projection bias parameter harm"
```

---

# 22. 正式路线状态

```text
F1-C
  CLOSED / FAILED

A2
  CLOSED / DIAGNOSTIC COMPLETE

A3
  CLOSED / DIAGNOSTIC COMPLETE
  ACCEPTED @ 4e15c1ec

A4
  PRE-EXECUTION
  Residual DC/AC Paired Causality Audit
  EVALUATION-ONLY

PRIMARY:
  P5

SECONDARY:
  P3

CONTROL:
  P4

HOLD:
  all training
  static per-scale weights
  q3/q4/q5
  new reliability gate
```

A4 只有在 G1–G14 全部 PASS 后，才允许解释 AC pairedness、centering rescue 和后续 training branch。
