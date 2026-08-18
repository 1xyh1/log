# A5 DESIGN_FREEZE — Cross-scale AC Paired Interaction Audit

状态：**PRE-EXECUTION / EVALUATION-ONLY / DESIGN FROZEN**

## 0. 上游正式状态

```text
F1-C
  CLOSED / FAILED

A2
  CLOSED / DIAGNOSTIC COMPLETE

A3
  CLOSED / DIAGNOSTIC COMPLETE
  ACCEPTED @ 4e15c1ec2cd64af39031d3fcfde200f2d248b65a

A4 execution
  CLOSED / DIAGNOSTIC COMPLETE
  RESULT EVIDENCE ACCEPTED @ 36221d2f827c411bddd66350729dfd05a3b48f49
  G1–G14 PASS
  64 conditions frozen
  DO NOT RERUN / DO NOT REWRITE

A4 reviewer correction
  HEAD @ b7ee0d6803d949a8c512b11defcb2125a3f4c8a1
  corrected branch = MIXED_PAIRED_CONTEXT_NO_GO
  corrected training_go = false
  A4T = HOLD
```

A4 P5 primary evidence:

```text
P5 AC_ALL pairedness

standalone:
  STRONG_POSITIVE

conditional:
  STRONG_NEGATIVE
```

A4 P5 centering rescue:

```text
standalone:
  STRONG_POSITIVE_RESCUE

conditional:
  STRONG_POSITIVE_RESCUE
```

因此冻结科学问题：

> **P5 full-map centering 在 isolation 中恢复 paired causal value，但在完整多尺度 residual context 中 paired effect 反号。A5 只定位：P3、P4 或它们的 joint residual context，谁足以把 P5 AC paired-positive 翻成 paired-negative；以及对共尺度做 centering 是否能解除这种 antagonism。**

---

# 1. A5 唯一正式问题

对已训练冻结系统，定义 P5 target residual 为：

`AC_5 = δ_5 - mean_HW(δ_5)`

只操纵 **P5 residual identity**：

```text
recipient P5 AC
vs
donor P5 AC
```

同时把 recipient P3/P4 context 固定为三种状态之一：

```text
OFF  = residual coefficient 0
FULL = untouched recipient full projected residual
AC   = recipient AC_ALL residual
```

A5 要回答：

> **当 P5 AC 从 standalone paired-positive 变成 full-context paired-negative 时，P3 FULL、P4 FULL、P3×P4 joint context 中哪个是充分 antagonist；以及将 P3/P4 改为 AC 是否恢复 P5 pairedness？**

---

# 2. A5 明确不回答

A5 不回答：

- 是否直接训练 residual centering；
- 是否训练 selective centering；
- 是否训练 `q3/q4/q5`；
- 是否设计新 reliability gate；
- 是否修改 AuxEncoder；
- 是否修改 projection；
- 是否加入 Depth；
- 是否把 DC harm 归因为 Conv bias；
- padding/global statistics 的最终来源；
- object-conditioned fusion 的最终结构。

**A5 无论结果如何，都不直接发放 training GO。**

A5 只能决定下一步应该进入哪一类更窄的实验设计。

---

# 3. 冻结系统

## 3.1 Primary

`F1C-I-fixed / last.pt`

## 3.2 Replication

`F1C-I-soft / last.pt`

解释关系：

```text
FIXED = primary trained system
SOFT  = replication trained system
```

不是同权重 q ablation。

禁止：

```text
best.pt
new checkpoint
new seed
```

---

# 4. 冻结数据与 donor

继续使用 A2/A3/A4 完全相同的 val6：

```text
000003_013_00000085
000004_013_00000081
000004_014_00000001
000016
000016_001_00000001
000016_042_suppl_00000164
```

donor map：

```text
c5cd8e852663eae3243bc5e4c263c6f2c26c7b2faa977ae3b60cb5c1ea122af5
```

不得重新生成 donor。

---

# 5. q 冻结

所有 SOFT 条件必须：

```text
untouched recipient A3/A4/A5
        ↓
q_native
        ↓
freeze q_native
        ↓
construct P3/P4 context
        ↓
replace only P5 AC identity when native vs donor is compared
```

FIXED：

`q_native ≡ 1`

关键原则：

> **P3/P4 OFF/FULL/AC context 不允许回流改变 q。P5 donor identity 也不允许改变 q。**

---

# 6. Residual 定义

对 scale `i`：

```text
δ_i = P_i(A_i)
FULL_i = δ_i
AC_i   = δ_i - mean_HW(δ_i)
```

A5 primary 只使用：

`AC_ALL`

不使用 `AC_CONTENT` 作为 primary intervention。

A4 的 content diagnostic 仅作为 upstream caveat：

> full-map centering 可能包含 letterbox/global feature statistics effect。

A5 不再扩大 padding diagnostic；若 A5 找到明确 antagonist，再在后续窄实验里拆。

---

# 7. P5 donor hard rule

P5 donor AC 必须：

`AC_5^donor = δ_5^donor - mean_HW(δ_5^donor)`

必须用 donor 自己的 mean。

P3/P4 context 始终来自 **recipient**：

```text
recipient P3 state
recipient P4 state
recipient RGB
recipient GT
recipient q_native
```

因此 A5 只操纵：

`P5 paired identity`

而不是多尺度 donor shuffle。

---

# 8. 3×3 Cross-scale Context Matrix

定义：

```text
O = OFF
F = FULL recipient residual
A = AC_ALL recipient residual
```

context id 按 `(P3 state, P4 state)`。

| ID | P3 | P4 | 意义 |
|---|---|---|---|
| OO | OFF | OFF | P5 AC standalone；A4 anchor |
| FO | FULL | OFF | 只加入 P3 FULL |
| OF | OFF | FULL | 只加入 P4 FULL |
| FF | FULL | FULL | A4 conditional anchor |
| AO | AC | OFF | 只加入 P3 AC |
| OA | OFF | AC | 只加入 P4 AC |
| AF | AC | FULL | P3 centered，P4 保持 FULL |
| FA | FULL | AC | P3 FULL，P4 centered |
| AA | AC | AC | P3/P4 都 centered |

每个 context 必须有：

```text
P5_AC_NATIVE
P5_AC_DONOR
```

因此：

```text
9 contexts × 2 P5 identities
= 18 conditions / system

FIXED + SOFT
= 36 A5 condition instances
```

其中 `OO native/donor` 与 `FF native/donor` 必须对 A4 做 anchor closure。

---

# 9. Primary endpoint：P5 paired effect conditioned on context

对 context `c`：

`Δ5(c) = AP(P5AC_recipient | c) - AP(P5AC_donor | c)`

这是 A5 唯一 primary causal endpoint。

分别输出：

```text
full val6
LOO per recipient
LOO median
positive folds
negative folds
zero folds
```

FIXED/SOFT 分开。

---

# 10. Context pairedness label

每个 context 独立判：

## STRONG_POSITIVE

```text
FIXED full > 0
FIXED LOO median > 0
FIXED positive folds >= 4/6
SOFT full > 0
```

## STRONG_NEGATIVE

```text
FIXED full < 0
FIXED LOO median < 0
FIXED negative folds >= 4/6
SOFT full < 0
```

否则：

`INCONCLUSIVE`

禁止任意 AP margin。

---

# 11. Context shift：谁在改变 P5 pairedness

以 standalone `OO` 为冻结 baseline：

`Γ(c) = Δ5(c) - Δ5(OO)`

意义：

```text
Γ < 0:
  context makes P5 pairedness more antagonistic

Γ > 0:
  context makes P5 pairedness more favorable
```

对 Γ 也输出 full、LOO、LOO median、fold signs。

## STRONG_ANTAGONISTIC_SHIFT

```text
FIXED full < 0
FIXED LOO median < 0
FIXED negative folds >= 4/6
SOFT full < 0
```

## STRONG_RESCUING_SHIFT

反号同理。

否则：

`INCONCLUSIVE_SHIFT`

注意：

> negative shift 不等于 sign flip。正式“flip”必须结合 context pairedness label。

---

# 12. Full-context antagonism decomposition

A5 必须计算：

```text
D3F = Δ5(FO) - Δ5(OO)
D4F = Δ5(OF) - Δ5(OO)
IFF = Δ5(FF) - Δ5(FO) - Δ5(OF) + Δ5(OO)
```

回答：

```text
P3 FULL 是否单独压坏 P5 pairedness？
P4 FULL 是否单独压坏 P5 pairedness？
还是必须 P3+P4 joint context 才反号？
```

---

# 13. Centered-context decomposition

同时计算：

```text
D3A = Δ5(AO) - Δ5(OO)
D4A = Δ5(OA) - Δ5(OO)

IAA = Δ5(AA) - Δ5(AO) - Δ5(OA) + Δ5(OO)

IAF = Δ5(AF) - Δ5(AO) - Δ5(OF) + Δ5(OO)

IFA = Δ5(FA) - Δ5(FO) - Δ5(OA) + Δ5(OO)
```

---

# 14. “谁足以翻转 P5”正式判定

A5 不强迫单一 root cause；允许多个 flag 同时成立。

## P3_FULL_SUFFICIENT_FLIP

必须：

```text
OO = STRONG_POSITIVE
FO = STRONG_NEGATIVE
D3F = STRONG_ANTAGONISTIC_SHIFT
```

## P4_FULL_SUFFICIENT_FLIP

必须：

```text
OO = STRONG_POSITIVE
OF = STRONG_NEGATIVE
D4F = STRONG_ANTAGONISTIC_SHIFT
```

## BOTH_FULL_INDIVIDUALLY_SUFFICIENT

两者同时成立。

## JOINT_FULL_CONTEXT_REQUIRED

必须：

```text
OO = STRONG_POSITIVE
FF = STRONG_NEGATIVE

FO != STRONG_NEGATIVE
OF != STRONG_NEGATIVE
IFF = STRONG_ANTAGONISTIC_SHIFT
```

若 FF 反号但 IFF 不稳定：

`FULL_CONTEXT_FLIP_WITH_UNRESOLVED_INTERACTION`

不得强称 joint interaction causal root。

---

# 15. “center co-scale 后能不能救 P5 pairedness”判定

## P3_CENTERING_RESCUES_WITH_P4_FULL

比较 `FF -> AF`，要求：

```text
FF = STRONG_NEGATIVE
AF = STRONG_POSITIVE
Δ5(AF)-Δ5(FF) = STRONG_RESCUING_SHIFT
```

## P4_CENTERING_RESCUES_WITH_P3_FULL

比较 `FF -> FA`，同样要求负→正且 shift stable positive。

## BOTH_CENTERED_RESTORE

`AA = STRONG_POSITIVE`

只允许解释：

> P3/P4 context centering restores P5 paired causal value in the frozen checkpoints.

**仍然不能直接发 training GO。**

## CENTERING_FAILS_TO_RESTORE

若：

`AA = STRONG_NEGATIVE`

则：

> cross-scale antagonism persists even after full-map centering of P3/P4.

下一步进入更深 representation/channel semantics audit。

---

# 16. Native AP secondary diagnostic

A5 primary 是 paired effect，不是 native AP。

但所有 condition 都已有 native AP，因此必须报告：

```text
U_native(c) = AP(P5AC_recipient | c)

R_native(c) = U_native(c) - U_native(FF)
```

用途：

> 判断一个恢复 P5 pairedness 的 context 是否以 native AP 损失为代价。

secondary metric 不得单独触发 training GO，也不得覆盖 pairedness primary。

---

# 17. A4 anchor closure

## OO

A5：

```text
OO / P5 native AC
OO / P5 donor AC
```

必须逐样本 raw detection tensor 等于 A4：

```text
AC_ALL_NATIVE_P5_STANDALONE
AC_ALL_DONOR_P5_STANDALONE
```

并且 AP/LOO 完全等价。

## FF

A5：

```text
FF / P5 native AC
FF / P5 donor AC
```

必须逐样本 raw detection tensor 等于 A4：

```text
AC_ALL_NATIVE_P5_CONDITIONAL
AC_ALL_DONOR_P5_CONDITIONAL
```

并且 AP/LOO 完全等价。

任何 anchor 失败：

`A5_A4_ANCHOR_EQUIVALENCE_FAIL`

立即 ABORT。

---

# 18. A5 决策分叉

A5 **没有 training_go=true 分支**。

所有分支只决定下一设计阶段。

## Branch 1 — SINGLE_SCALE_FULL_ANTAGONIST

如果：

`P3_FULL_SUFFICIENT_FLIP xor P4_FULL_SUFFICIENT_FLIP`

下一步：

`A5b — implicated-scale centering/source audit`

优先拆：

```text
full-map vs content-aware mean
DC vs AC
projection/content statistics
```

## Branch 2 — BOTH_SCALES_INDIVIDUALLY_ANTAGONISTIC

若 P3/P4 FULL 都单独 sufficient：

下一步：

`A5b — dual cross-scale representation audit`

禁止直接全尺度 centering training，因为 A4 已有 P4 centering negative rescue evidence。

## Branch 3 — JOINT_ONLY_ANTAGONISM

若只有 FF 反号且 IFF 稳定 antagonistic：

下一步：

`A6 — joint cross-scale residual interaction mechanism`

优先研究：

```text
neck addition coupling
channel competition
scale redundancy
cross-scale object support
```

## Branch 4 — CENTERED_CONTEXT_RESTORES_P5

若 AF / FA / AA 中出现稳定恢复：

只允许：

`candidate selective-centering architecture identified`

下一步必须另写：

`selective-centering TRAINING DESIGN_FREEZE`

A5 本身：

`training_go = false`

## Branch 5 — CENTERING_DOES_NOT_RESTORE

若 AA 仍 STRONG_NEGATIVE：

`STOP DC-CENTERING AS ROOT-CAUSE EXPLANATION`

进入：

```text
projection channel semantics
AuxEncoder representation
object-conditioned residual
local spatial correspondence
```

## Branch 6 — INCONCLUSIVE

若 OO anchor 重现，但其余 context 没有跨系统稳定结构：

`A5_DIAGNOSIS_INCONCLUSIVE`

下一步不是训练，而是扩大诊断样本或检查 acquisition/data contract。

---

# 19. Runtime hard gates

## G1 — upstream reviewer adjudication

必须验证：

```text
A4 execution commit = 36221d2f827c411bddd66350729dfd05a3b48f49
A4 reviewer head    = b7ee0d6803d949a8c512b11defcb2125a3f4c8a1

reviewer_adjudication:
  experiment_result = ACCEPTED_DIAGNOSTIC_COMPLETE
  corrected_branch = MIXED_PAIRED_CONTEXT_NO_GO
  corrected_training_go = false
  a4t_status = HOLD
  rerun_required = false
```

失败：

`A5_UPSTREAM_ADJUDICATION_FAIL`

## G2 — frozen dependency closure

沿用 A4 dependency closure，并 pin：

```text
A4 frozen reports
A4 reviewer adjudication
A4 feedback
current corrected decision code
current test suite
model sources
dataset
modality_preprocess
Step3 authoritative evaluator
checkpoints
manifests
contract
torch / ultralytics
```

## G3 — evaluation-only state

禁止 optimizer/backward/train/parameter update/buffer mutation。

要求：

`state_sha_before == state_sha_after`

失败：

`A5_PARAMETER_MUTATION`

## G4 — q freeze

所有 9×2 context：

`q == untouched recipient q_native`

失败：

`A5_Q_FREEZE_FAIL`

## G5 — donor map freeze

P5 donor identity 必须逐样本等于 A2/A3/A4 donor map。

失败：

`A5_DONOR_MAP_DRIFT`

## G6 — P5-only identity intervention

native/donor pair 中：

```text
P3 context identical
P4 context identical
RGB identical
GT identical
q identical
```

唯一变化：

`P5 AC source identity`

失败：

`A5_P5_IDENTITY_ISOLATION_FAIL`

## G7 — P3/P4 context semantics

每个 context 动态证明：

```text
O -> alpha 0
F -> recipient FULL residual
A -> recipient AC_ALL residual
```

不得出现 donor P3/P4。

失败：

`A5_CONTEXT_SEMANTICS_FAIL`

## G8 — donor P5 self-centering

必须：

```text
P5 residual_source_id == donor_id
P5 mean_source_id == donor_id
```

失败：

`A5_DONOR_AC_MEAN_SOURCE_FAIL`

## G9 — OO A4 anchor closure

逐样本 raw tensor + AP/LOO identity。

失败：

`A5_OO_ANCHOR_FAIL`

## G10 — FF A4 anchor closure

逐样本 raw tensor + AP/LOO identity。

失败：

`A5_FF_ANCHOR_FAIL`

## G11 — exact context completeness

必须恰好：

`OO FO OF FF AO OA AF FA AA`

每个均 native+donor。

失败：

`A5_CONTEXT_MATRIX_INCOMPLETE`

## G12 — paired-effect computation

必须验证：

`Δ5(c) = native AP - donor AP`

LOO recipient set 完全一致。

失败：

`A5_PAIRED_EFFECT_SEMANTICS_FAIL`

## G13 — interaction contrast correctness

必须单元测试：

`D3F D4F IFF D3A D4A IAA IAF IFA`

失败：

`A5_INTERACTION_CONTRAST_FAIL`

## G14 — stock eval semantics

继续使用 Step3 authoritative validator semantics。

失败：

`A5_STOCK_EVAL_SEMANTICS_FAIL`

## G15 — provenance

必须记录：

- A5 DESIGN_FREEZE SHA
- evaluator/helper/tests/audit SHAs
- A4 execution commit
- A4 reviewer head
- A4 summary raw SHA
- reviewer_adjudication SHA
- A4 OO/FF anchor identities
- donor map SHA
- checkpoint/manifest SHAs
- dependency closure
- val6 ids/order
- q values
- context matrix identities
- versions
- interaction formula version

失败：

`A5_PROVENANCE_INCOMPLETE`

任一 G1–G15 fail：

`A5_ABORT`

---

# 20. Interpretation discipline

正式报告必须显式写：

```text
context shift != sign flip
native AP rescue != paired restoration
AC utility != paired causality
A5 identifies cross-scale context mechanism, not training efficacy
A5 never grants training GO
P3/P4 context is always recipient-native
only P5 identity is paired/donor manipulated
FIXED primary / SOFT replication
val6 limits population generalization
```

---

# 21. 最低测试集合

至少：

1. A4 adjudication corrected_training_go must be false
2. A4T HOLD required
3. A4 execution reports remain frozen
4. val6 order drift fail
5. donor drift fail
6. q computed before context intervention
7. q frozen across native/donor
8. FIXED q=1
9. P5 donor uses donor own mean
10. P3/P4 donor contamination fail
11. O state coefficient zero
12. F state exact recipient full residual
13. A state exact recipient AC_ALL
14. OO native anchor bitwise A4 standalone
15. OO donor anchor bitwise A4 standalone
16. FF native anchor bitwise A4 conditional
17. FF donor anchor bitwise A4 conditional
18. exact nine contexts
19. exactly native+donor per context
20. Δ5 native-minus-donor sign
21. context shift Γ vs OO
22. D3F coefficients
23. D4F coefficients
24. IFF coefficients
25. D3A coefficients
26. D4A coefficients
27. IAA coefficients
28. IAF coefficients
29. IFA coefficients
30. P3 sufficient flip classification
31. P4 sufficient flip classification
32. joint-only classification
33. shift negative but no sign flip must not call FLIP
34. AF rescue classification
35. FA rescue classification
36. AA restoration classification
37. AA positive cannot create training_go=true
38. native AP secondary cannot override paired primary
39. state SHA before==after
40. all G1–G15 dynamic
41. no optimizer/backward/train
42. no result-driven extra context
43. stock evaluator pinned
44. provenance includes executed-vs-corrected A4 dual track
45. regression: A4 mixed-context bug cannot reappear in A5 route logic

---

# 22. 正式输出

```text
docs/step4_a5/DESIGN_FREEZE.md

reports/step4_a5/preexecution_audit.json
reports/step4_a5/context_paired_effects.json
reports/step4_a5/context_shifts.json
reports/step4_a5/cross_scale_interactions.json
reports/step4_a5/native_ap_secondary.json
reports/step4_a5/a5_summary.json
```

唯一正式路线入口：

`a5_summary.json`

---

# 23. A5 summary 最低字段

```text
schema
all_gates_passed
G1...G15
interpretation_discipline

upstream:
  A4 execution commit
  A4 reviewer adjudication
  A4T HOLD

context_labels:
  OO FO OF FF AO OA AF FA AA

paired_effects:
  FIXED
  SOFT

context_shifts_from_OO

interactions:
  D3F
  D4F
  IFF
  D3A
  D4A
  IAA
  IAF
  IFA

mechanism_flags:
  P3_FULL_SUFFICIENT_FLIP
  P4_FULL_SUFFICIENT_FLIP
  BOTH_FULL_INDIVIDUALLY_SUFFICIENT
  JOINT_FULL_CONTEXT_REQUIRED
  P3_CENTERING_RESCUES_WITH_P4_FULL
  P4_CENTERING_RESCUES_WITH_P3_FULL
  BOTH_CENTERED_RESTORE
  CENTERING_FAILS_TO_RESTORE

native_ap_secondary

next_branch
training_go = false
```

---

# 24. 明确禁止

```text
NO training
NO optimizer
NO backward
NO new checkpoint
NO new seed
NO new val sample
NO donor-map change
NO q3/q4/q5
NO new gate
NO learned weights
NO Depth
NO P3/P4 donor intervention
NO dataset-level IR shuffle
NO recipient mean for donor P5 AC
NO AC_CONTENT as primary A5 intervention
NO result-driven context addition
NO arbitrary AP margin
NO changing A4 frozen reports
NO rewriting A4 machine GO as if it never happened
NO A5 training_go=true under any result
```

---

# 25. 当前正式路线

```text
A4
  CLOSED / DIAGNOSTIC COMPLETE
  EVIDENCE ACCEPTED @ 36221d2f

A4 reviewer adjudication
  CLOSED @ b7ee0d6
  corrected branch = MIXED_PAIRED_CONTEXT_NO_GO
  A4T HOLD

A5
  PRE-EXECUTION
  Cross-scale AC Paired Interaction Audit
  EVALUATION-ONLY
  P5-CENTERED

PRIMARY QUESTION:
  who flips P5 AC pairedness from positive to negative?

TRAINING:
  HOLD
```
