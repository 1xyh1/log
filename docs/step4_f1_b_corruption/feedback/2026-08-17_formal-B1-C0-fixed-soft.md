# F1-B 执行反馈：2026-08-17 / formal B1-C0 · B1-I-fixed · B1-I-soft

- commit SHA：本机源仓库（未跟踪提交）；远端镜像 watcher 自动批次
- machine / GPU：Windows 11 本机 / RTX 4060 Laptop 8GB
- PyTorch / Ultralytics：torch 2.5.1+cu121 / ultralytics 8.4.56
- command：`run_step4_f1_b.py --group <g> --run-kind formal`（三组串行，exit 0，
  每组约 6 分钟）
- physical run directory：
  - `runs/step4_f1_b_corruption/B1-C0/`（23:47–23:53）
  - `runs/step4_f1_b_corruption/B1-I-fixed/`（23:53–23:59）
  - `runs/step4_f1_b_corruption/B1-I-soft/`（23:59–00:05）

## 门禁

- audit 硬门禁：三组启动前全部通过（provenance 新鲜，含 runner SHA）
- G5/G6/G8：三组 PASS；G9：80 epoch × 11 样本 records 落盘，summarizer 逐行
  重判全部通过（schedule 逐样本比对、records SHA、ID 集合完整无重复、IR
  before/after 语义、跨组 schedule 一致）
- posthoc 梯度审计（B1 版）：passed（gate detach 语义、residual 路径、C0 proj
  中性全部保持）

## Formal 与因果结果（last.pt / val6 主口径）

| 组 | NORMAL | ZERO-AUX | SHUFFLE |
|---|---|---|---|
| B1-C0 | **0.2840** | 0.2840 | 0.2840 |
| B1-I-fixed | **0.2549** | 0.2680 | 0.2641 |
| B1-I-soft | **0.3040** | 0.3027 | 0.3043 |

- B1-C0 与 F1-C0 逐位一致（0.2840）——corruption schedule 对 C0 无模型影响 ✓
- B1-I-fixed = 0.2549，显著低于 F1-I-fixed 的 0.2992：**训练期 corruption 对
  q 恒 1 的 fixed 结构有害**（50% 退化样本的 IR residual 无法降权）。
- B1-I-soft = 0.3040：> C0（+0.0200）、> fixed（+0.0491）、> ZERO（+0.0013）；
  **< SHUFFLE（−0.0003）**——N>S 微负，在 6-val 噪声内但严格判据不成立。
- SOFT−C0 LOO：6/6 正，median **+0.019017** ✓；SOFT−FIXED LOO：6/6 正，
  median **+0.048066** ✓。
  （更正 2026-08-17：上一版把两个 median 写反——SOFT−C0 是 +0.019017、
  SOFT−FIXED 是 +0.048066。）

## B1 晋级证据（_summary_step4_f1_b.json，frozen）

- own FORCE-QCLEAN = 0.5006（B1-soft 自己的 clean q，仍近似常数 0.5）
- macro AP（17 退化条件）：soft 0.2992 vs fixed 0.2589 vs qclean 0.2991
  → macro_pass true（soft 超 fixed 大幅、超 qclean 仅 **+3.49e-5**）
  （更正 2026-08-17：上一版写 +6e-5，实际 macro_soft − macro_qclean = +3.49e-5）
- identity learned−QCLEAN = **0.0**（更正：B1 的 identity 差为 0.0，不是 F1 的
  −1.6e-5——上一版沿用 F1 数值属笔误）
- worst-4 AP（noise:0.50/0.75、shift:0.50/0.25）：soft 0.2915 vs fixed 0.2493
  vs qclean 0.2915 → worst4_pass **false**（soft − qclean ≈ −3e-6）
- learned−QCLEAN 正数：**4/17**（要求 ≥9）→ adaptive_pass false
- q–severity 单调下降 family：**0**（仅诊断）
- decision：**B1_GATE_FAILED_CAUSAL_PROTOCOL**
- next_step：stop before spatial gate/QAF and inspect intervention signs

## 事实层面的解读（供审阅者判级）

1. B1 的训练期 corruption 让"q≈0.5 的恒定降权"在退化集上显著优于
   separately-trained fixed（macro +0.0403、worst4 +0.0422）——**但这不是
   自适应可靠性**：learned 与 FORCE-QCLEAN 几乎逐位打平（macro **+3.48977e-5**、
   worst4 −3e-6、identity 差 **0.0**），gate 仍未学到输入条件化。
   （更正 2026-08-17：identity 差为 0.0，不是 F1 的 −1.6e-5。）
2. 严格因果协议 N>S = −0.0003 微负；ZERO 与 SHUFFLE 都接近 N（差 ±0.0013/0.0003），
   与 F1 的 paired 信号（N−Z +0.0415）相比，B1 的配对优势几乎消失——
   corruption 训练模糊了"正确配对"与"缺失/错配"的边界。
3. B1-C0 精确复现 F1-C0 说明 corruption schedule 的 matched control 语义干净。

## 待审阅者

按 DESIGN_FREEZE 第 7 节，B1 失败且 q 仍近常数 → 下一候选是 **RGB–IR agreement
描述输入**（先做诊断，不直接上 Transformer/spatial gate）。等审阅者判级与下一轮
设计指示。
