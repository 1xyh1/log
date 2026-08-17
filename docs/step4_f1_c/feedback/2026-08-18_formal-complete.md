# F1-C 执行反馈：2026-08-18 / 四组 formal 完成 + closeout 判级冻结

审阅者 FORMAL GO（db15af5）后四组 80ep 全部完成；期间发现并修复一个
initial-state 门禁语义缺口（见下），按协议走完整重跑链（smoke r5 +
readiness v2 重生成 + 三组 formal 重跑），最终 closeout 判级
**F1C_GATE_FAILED_CAUSAL_PROTOCOL**（frozen）。

## 门禁与执行

- 四组 formal（C0/fixed/magsoft/soft）80ep 全过：G5/G6/G8/G9/G10.7、
  fp32 RGB SHA、manifest 身份全部 PASS；g8 trace 四组字节一致 `bece47d5…`
- 四组 manifest 与 r5 readiness（`d0196134…`）全部闭合；
  base_checkpoint_sha256 均 == 646f8bc3…
- integrity / eval / loo / quality / posthoc provenance 全部 verified；
  summarize_source_sha256 = `782d32b9…`

## 语义缺口修复（2026-08-18，formal 启动时发现）

**问题**：F1C-I-soft（original-gate matched control）按 DESIGN_FREEZE 无
smoke、无冻结基准，但 initial-state equality 门禁对四组无条件比对 →
soft 构模后被 `ABORT_INITIAL_STATE_MISMATCH` 拦截（original gate 初始
state 与 magnitude 三组必然不同）。

**修复**（最小改动）：readiness 冻结块显式声明 `frozen_groups`（magnitude
三组）；runner 仅对 `frozen_groups` 内组做逐位比对，soft 记录 SHA 供审计
并继续。回归测试 `test_frozen_state_declares_coverage_groups`。
按协议完整重跑链：pytest 全绿 → audit v3 → smoke r5 三组 → readiness v2
重生成（r5）→ 自证 True [] → 三组 formal 重跑（旧目录
`F1C-C0-preclosure` 等留档；中断目录 `F1C-C0-interrupted` 留档）。

## 判级（closeout，frozen）

`decision = F1C_GATE_FAILED_CAUSAL_PROTOCOL`；`next_step: stop; inspect
intervention signs`

**primary（last.pt val6）**：
- C0 0.2917 / FIXED 0.2549 / **MAGSOFT 0.2157** / ORIGSOFT 0.3040
- MAGSOFT−C0 **−0.0760**（LOO 1/6 正）；MAGSOFT−FIXED −0.0392（1/6 正）；
  MAGSOFT−ORIGSOFT −0.0883（0/6 正）；vs 历史 B1-soft 0.304028 未超
- ORIGSOFT 0.3040282…与历史 B1-soft 0.304028 逐位一致（original-gate
  链路跨代可复现的交叉验证）

**晋级证据**：
- own QCLEAN 0.4998（≈0.5，仍是近似常数衰减——显式幅度输入未转化为
  输入条件化）
- macro 0.2265 vs own QCLEAN 0.4998：**macro_pass false**；
  worst4 0.2120 vs 0.2120：worst4_pass true（平）
- learned−QCLEAN 1/17（<9）；adaptive_pass false；
  diagnostic vs fixed/origsoft 全 false；beats_historical false

**科学解读（供审阅者确认）**：幅度 gate 不仅未证明输入条件化增益，
MAGSOFT 0.2157 反而显著劣于 C0/fixed/origsoft——显式 log-RMS 输入的
加入在该训练动力学下是**净负贡献**；q≈0.5 常数衰减再次出现，
learned 分支与 FORCE-QCLEAN 逐位打平。paired-IR 干预符号（N vs Z/S）在
formal 中未复现 F1-B 时期的稳定正值（待 LOO 明细复核）。

## 产物路径（绝对路径）

- summary：`C:\Users\xyh23\Documents\ChatGPT\多模态模型\multimodal_yolo26_qaf_v0_3\runs\step4_f1_c\_summary_step4_f1_c.json`
- 四组 formal：`...\runs\step4_f1_c\{F1C-C0,F1C-I-fixed,F1C-I-magsoft,F1C-I-soft}\`
- 留档：`...\runs\step4_f1_c\F1C-C0-preclosure`（修复前）/ `F1C-C0-interrupted`（23/80 中断）等
- readiness：`...\reports\step4_f1_c\smoke_readiness.json`（v2，r5）
- audit：`...\reports\step4_f1_c\pretrain_audit.json`（v3）
- 评估链日志：`D:\pycharm\Python Develop\YOLO_1\f1c_eval_chain2.log`

## 待审阅者

1. F1-C 判级确认（GATE_FAILED 是否冻结；下一步路线：no more F1-C variants？
   回到 RGB–IR agreement 诊断？其他候选？）
2. P2 待办（接口一致性）：runner 两处 `build_reference_3ch()` 改
   `build_reference_3ch(weights=str(a.base_checkpoint))` + integration
   test。**注意**：此改动会令 runner SHA 变化 → readiness 按设计失效 →
   需 smoke r6 + readiness 重生成 + 四组 formal 重跑（~40 分钟），
   建议在下一轮实验链起点执行，不破坏已冻结的 F1-C 判级。
