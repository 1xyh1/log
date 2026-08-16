# F1-C-A1 v2 执行反馈：2026-08-17

- 修复包：`f1_c_a1_v2_fix_de37323.zip`（SHA256 验证 MATCH）
- 应用方式：overlay 覆盖（本机源仓库无 de37323 git 基线；应用前全量备份至
  `_a1v2_backup/`）
- 推送：1xyh1/log @ 85ac8d8

## 本机执行发现并修复的包内缺陷（2 处，均已记录）

1. **q0−q1 浮点误差破坏 tie**：`0.7−0.3 = 0.39999999999999997` 使 utility 序列
   出现伪 tie 破坏，Spearman 错误降为 0.8。修复：`q0_minus_q1_map50_95` 统一
   `round(…, 6)`（与全链 delta 口径一致）。
2. **测试期望数学不可达**：`test_correlation_report_contains_continuous_and_
   family_holdout_axes` 期望 tie 场景下 Spearman == 1.0，但标准 tie-aware
   Spearman 在有 tie 时上限为 0.8944。修复：改测试扫描数据使 utility 严格
   递减（0.5/0.4/−0.4/−0.5），注释说明原因。

修复后：`pytest tests/test_step4_f1_closeout.py tests/test_step4_f1_c_
descriptor_audit.py` 13 项全过；py_compile 全过。

## 执行结果（按包内 EXECUTION_GUIDE）

- LOO 重建：`step4_f1_b_loo.json`（last 主协议，恢复 frozen 文件名）与
  `step4_f1_b_loo_best.json`（best 诊断）均重跑通过，新 provenance 含新
  closeout 源码 SHA。
- descriptor audit v2：`reports/step4_f1_c_agreement/descriptor_audit_v2_
  {last,best}.json` 生成（旧 v1 文件保留未覆盖）。
- B1 summarizer v2.2：`_summary_step4_f1_b.json` 重建，判级不变
  （B1_GATE_FAILED_CAUSAL_PROTOCOL，frozen）；fp16 checkpoint 等价与 fp32
  精确不变的区分已按包内口径写入。

## v2 门禁判读（EXECUTION_GUIDE 第 6 节）

**连续目标 AP(q0)−AP(q1) 的 tie-aware Spearman**：

| 描述子 | last | best | LOFO 反转 |
|---|---|---|---|
| log-RMS（三尺度一致） | **+0.928** | **+0.697** | 无（noise 最弱 +0.38 仍正） |
| RGB-relative energy | **+0.977 / +0.961** | **+0.754** | 无（noise 最弱 +0.50 仍正） |
| gate-LN slice RMS（当前输入摘要） | −0.548 | −0.649 | noise LOFO −0.04（不稳定） |
| spatial cos | P3 +0.66 / P5 −0.85 | 尺度间反号 | 不稳定 |

- 两 ckpt 连续目标方向稳定 ✓；LOFO 不被单个 family 反转 ✓（幅度类）；q scan
  可辨识 16-18/18 ✓。
- 与审阅者独立重算（last relative energy ≈ +0.96~+0.98）一致 ✓。
- 结论：**幅度描述子（log-RMS / relative energy）通过 audit**；当前 LN 标量
  摘要弱且 noise LOFO 近零；空间 cos 不稳定。

## 待审阅者

按裁决，A1-v2 修复 GO 已闭环；幅度 gate 实现（concat(LN(GAP 联合), logRMS(A3),
logRMS(A4), logRMS(A5))）、G1–G10 与三组 smoke 为条件 GO；80ep formal 仍 HOLD。
等审阅者放行幅度 gate 实现。
