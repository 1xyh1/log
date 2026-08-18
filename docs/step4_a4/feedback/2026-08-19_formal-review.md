# A4 二轮审阅裁决（reviewer adjudication）：2026-08-19

状态：**A4 EXPERIMENT RESULT ACCEPTED / DIAGNOSTIC COMPLETE；A4 MACHINE ROUTE DECISION REJECTED；A4T HOLD**
冻结 artifact：commit `36221d2f827c411bddd66350729dfd05a3b48f49`（父提交 = A3 的 `4e15c1ec…`，无实验分叉）
结构化裁决：`reports/step4_a4/reviewer_adjudication.json`

## 1. A4 实验结果：ACCEPTED（不重跑）

- G1–G14 全部 PASS；64 conditions（FIXED 32 + SOFT 32）结果正式接受
- checkpoint state 不变、native equivalence、q freeze、donor self-centering、content coverage、C000 closure 均未因 decision bug 失效
- **不重跑 64 conditions；不修改已执行 result**；`reports/step4_a4/*` 保持冻结

## 2. A4 机器路线裁决：REJECTED（P0 decision precedence bug）

执行代码 `src/multimodal/step4_a4_decision.py::joint_p5_decision()` 先计算 `go_contexts`
（paired STRONG_POSITIVE + rescue STRONG_POSITIVE_RESCUE 的同 context 集合），命中即提前
`return CENTERING_TRAINING_GO`；`MIXED_PAIRED_CONTEXT_NO_GO`（positive_pair && negative_pair）
检查位于其后，conditional 的 STRONG_NEGATIVE 无机会否决。

**真实命中组合**（本次正式结果）：

| context | AC_ALL pairedness | AC_ALL centering rescue |
|---|---|---|
| standalone | STRONG_POSITIVE | STRONG_POSITIVE_RESCUE |
| conditional | STRONG_NEGATIVE | STRONG_POSITIVE_RESCUE |

机器输出 `CENTERING_TRAINING_GO / training_go: true` —— **无效**。

**正确优先级**：① positive_pair && negative_pair → `MIXED_PAIRED_CONTEXT_NO_GO`；② 否则
same-context paired-positive + rescue-positive → `CENTERING_TRAINING_GO`。

**修正后裁决**：`MIXED_PAIRED_CONTEXT_NO_GO / training_go: false`。

## 3. 60 tests 未捕获原因

`test_mixed_paired_context_never_goes()` 的 rescue 为双 INCONCLUSIVE，不触发 go_contexts
提前 return；缺失真实命中组合（paired standalone positive / conditional negative +
rescue 双 STRONG_POSITIVE_RESCUE）的用例。

## 4. P5 关键数值（原始，来自冻结 a4_summary.json，SHA `721198d0…`）

- **standalone**：ΔAC-pair FIXED **+0.00242020**（LOO median +0.00289061，4/6 正）、SOFT **+0.000989325**（LOO median +0.00135611，5/6 正）；Δcenter FIXED **+0.00302883**（6/6 正）、SOFT **+0.00176688**
- **conditional**：ΔAC-pair FIXED **−0.00946564**（LOO median −0.01020544，0/6 正）、SOFT **−0.000429751**（LOO median −0.000451638，1/5 正）；Δcenter FIXED **+0.00787241**（6/6 正）、SOFT **+0.000255627**（6/6 正）

**科学结论（审阅者冻结表述）**：P5 centering restores paired causal value in isolation,
but that restoration collapses and reverses sign in the full multiscale residual context.
问题进一步定位到 **cross-scale residual interaction / representation coupling**，
而非单纯 P5 DC。

## 5. Factorial 与 AC_CONTENT 附加证据

- Factorial main effects：**R3 STRONG_POSITIVE_RESCUE / R4 STRONG_NEGATIVE_RESCUE / R5 STRONG_POSITIVE_RESCUE** → 全尺度统一 `δ − mean_HW(δ)` 不是理所当然正确的架构（P4 为反方向 control）
- AC_CONTENT（diagnostic-only，冻结时即无权改 primary label）：P5 standalone paired INCONCLUSIVE（不复现）、rescue 双 STRONG_POSITIVE_RESCUE → full-map centering 的 paired restoration 可能不只是"真实内容 DC 被去掉"，letterbox/global feature statistics 参与其中

## 6. 修复记录（本机执行）

- `src/multimodal/step4_a4_decision.py`：mixed 冲突检查提前至 go_contexts 之前（评审裁决语义）
- `tests/test_step4_a4.py`：新增 `test_mixed_context_sign_conflict_beats_same_context_go`（真实命中组合）
- pytest 全量 **61 passed**
- **重要声明：修复后代码 ≠ 执行时代码**。执行时 decision.py SHA = `aea5d6c8a9fe5345e0b19ce816d3fe5f7bd0fec3edee4d2abd52e22cd6bbc196`（audit provenance 冻结记录），修复后 = `50650e2b1a3679325a2cd1b7d95eccebfeb9044d801ffb48dc08b83c89ad95a2`；tests 执行时 = `40f95f0c4fc269c41e6d642987b3fe430e165db6c1a104191461904b6e78f3b2`，修复后 = `ec9b294464237869b09c788dd1f23ebeeb2194cfb138514173934718e474cbc4`。未重跑 audit（`preexecution_audit.json` 保持执行时快照）。
- 未修改已冻结的 `reports/step4_a4/*` 正式报告。

## 7. 正式状态与下一步

- **A4 CLOSED / DIAGNOSTIC COMPLETE**（RESULT EVIDENCE ACCEPTED @ 36221d2f）
- **A4T: HOLD**（`training_go = false` 冻结）
- **下一步 = A5 — Cross-scale AC Paired Interaction Audit**（evaluation-only，P5 为中心：P5 AC alone / P3 FULL+P5 AC / P4 FULL+P5 AC / P3+P4 FULL+P5 AC / P3 AC+P5 AC / P4 AC+P5 AC / P3 AC+P4 FULL+P5 AC …，目标 = 找出谁把 P5 的 paired-positive 翻成 paired-negative；A2 的 FIXED P3×P5 antagonism 与 A4 的 P5 standalone→conditional sign inversion 可接续）；待审阅者 A5 DESIGN_FREEZE
