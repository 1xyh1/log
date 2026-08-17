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
- ORIGSOFT preregistered last-val mAP50-95 = 0.30402820783485807，与历史
  B1-soft 0.30402820783485807 数值精确复现到当前记录精度——强力支持评估
  与训练主协议跨代稳定（注意：两代 last.pt SHA 不同——F1-C `b3a944…`
  vs B1 `e9108c…`——不构成 checkpoint/全训练链逐位一致的证据）

**晋级证据**：
- own QCLEAN 的 q = 0.499762（≈0.5，仍是近似常数衰减——显式幅度输入
  未转化为输入条件化）
- **macro（量纲修正）**：macro_learned = 0.2265046 vs
  macro_FORCE-QCLEAN = 0.2266646，差 **−0.0001600** → **macro_pass false**
  （注意：own QCLEAN 的 0.499762 是 q 值不是 AP，不与 macro 直接比较）
- **worst4（margin 修正）**：0.2119577 vs 0.2119561，margin 仅
  **+1.66e-6**——冻结的严格 `>` 判据下形式上 PASS，但工程上应视为
  **近似持平，不构成有效正证据**
- learned−QCLEAN 1/17（<9）；adaptive_pass false；
  diagnostic vs fixed/origsoft 全 false；beats_historical false

**科学结论（审阅者冻结版，2026-08-18）**：在当前 RGB-anchor + IR residual、
detached gate input、R3 corruption schedule、80ep/seed20260812 的训练
协议下，给 global scalar reliability gate 增加显式三尺度 log-RMS
magnitude side-channel，**没有学出有效的输入条件化可靠性，并导致显著的
matched-control 性能退化**。

证据强度：magsoft clean val q ∈ [0.4977, 0.5020]（mean 0.499762，仍几乎
常数 0.5）；learned clean AP 0.21569992997 == FORCE-QCLEAN
0.21569992997；degraded 17 条件仅 1/17 learned 优于 FORCE-QCLEAN。
**magnitude 信息存在 ≠ optimizer 会自动把它学成可靠性控制**——A1 的
相关性没有错；失败的是"把这个统计量塞进当前 global gate，就能把相关性
转化为 causal control"这个假设。posthoc 已证明非 wiring bug（gate
params gradient active、residual→aux 梯度正常、gate→aux detach 语义
正确）。

**paired-IR 表述收紧**：B1 preregistered last.pt 本身
NORMAL−ZERO=+0.001376 但 NORMAL−SHUFFLE=−0.000315——paired-IR 因果价值
**原本就未在预注册主 checkpoint 上建立**；F1-C magsoft 进一步恶化：
NORMAL−ZERO=−0.05104、NORMAL−SHUFFLE=−0.04853，**明显负因果效应**。

## 产物路径（绝对路径）

- summary：`C:\Users\xyh23\Documents\ChatGPT\多模态模型\multimodal_yolo26_qaf_v0_3\runs\step4_f1_c\_summary_step4_f1_c.json`
- 四组 formal：`...\runs\step4_f1_c\{F1C-C0,F1C-I-fixed,F1C-I-magsoft,F1C-I-soft}\`
- 留档：`...\runs\step4_f1_c\F1C-C0-preclosure`（修复前）/ `F1C-C0-interrupted`（23/80 中断）等
- readiness：`...\reports\step4_f1_c\smoke_readiness.json`（v2，r5）
- audit：`...\reports\step4_f1_c\pretrain_audit.json`（v3）
- 评估链日志：`D:\pycharm\Python Develop\YOLO_1\f1c_eval_chain2.log`

## 审阅者最终裁决（2026-08-18，冻结）

**F1-C CLOSED / FAILED**：`F1C_GATE_FAILED_CAUSAL_PROTOCOL` frozen at
`bf983c4`。Magnitude-aware global scalar gate rejected under the frozen
protocol。**不追加 F1-C variants、不补确认 seed**。F1-C-soft initial-state
coverage 修复不构成结果污染（DESIGN_FREEZE 本就规定 original-soft 无
magnitude smoke；frozen_groups 限定是门禁语义修正，不是看结果后改晋级
条件；修复后完整重跑 r5 → readiness → formal）。

## 下一步：A2 — Scale-wise IR Causality / RGB–IR Agreement Audit（evaluation-only）

不训练新模型，用已冻结的 `F1C-I-soft` 与 `F1C-I-fixed` checkpoint 做
诊断（不再叫 F1-C）。四类 intervention：
1. **DROP-one-scale**：P3/P4/P5 residual 各自=0；
2. **KEEP-only-one-scale**：只保留 P3/P4/P5 之一；
3. **per-scale SHUFFLE**：只把某个尺度的 aux feature 换 donor；
4. **per-scale q sweep**：q_i ∈ {0, 0.25, 0.5, 0.75, 1}，其余两尺度保持
   frozen normal。

指标：Δpair_i = AP(paired_i) − AP(shuffled_i)；Δdrop_i = AP(normal) −
AP(drop_i)；继续 val6 LOO，不只看单个 full AP。

**分叉规则**：
- 三尺度稳定正 → 尺度冲突被 causal intervention 确认 → 先做 **static
  per-scale coefficients / scale selection**（不直接上动态 q3/q4/q5），
  回答"允许各尺度不同权重本身能否救回来"，再问动态是否更好；
- 全无稳定正值 → **暂停 gate 研究**，回头查 IR/RGB registration、
  AuxEncoder representation、IR preprocessing、projection 语义、
  RGB–IR spatial/agreement；
- 单一尺度稳定正 → **single-scale IR residual**（更简单更稳）。

## P2 处理（审阅者更新裁决）

下一轮开始前修，但**不再动冻结的 F1-C runner**：保留 `bf983c4` 为
F1-C frozen evidence commit。新建 A2/F2 runner 时从一开始就正确写
`build_reference_3ch(weights=str(a.base_checkpoint))`，并补真正的
integration test（传入临时 checkpoint A、mock/default checkpoint B，
确认构模实际读取 A，而不是只验证 A 存在）。
