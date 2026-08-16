# F1 执行反馈：2026-08-16 / formal F1-C0 · F1-I-fixed · F1-I-soft

- commit SHA：本机源仓库（未跟踪提交）；远端镜像 watcher 自动批次
- machine / GPU：Windows 11 本机 / RTX 4060 Laptop 8GB
- PyTorch / Ultralytics：torch 2.5.1+cu121 / ultralytics 8.4.56
- command：`run_step4_f1_ir_gate.py --group <g> --run-kind formal`（三组串行，exit 0）
- physical run directory：
  - `runs/step4_f1_ir_gate/F1-C0/`（08:23–08:29）
  - `runs/step4_f1_ir_gate/F1-I-fixed/`（08:29–08:35）
  - `runs/step4_f1_ir_gate/F1-I-soft/`（08:35–08:40）

## 门禁

- G0～G5 audit：PASS（detach 修复后重跑，all_passed=true）
- smoke：三组 PASS（detach 修复 + G6 阈值 epoch 线性缩放后）
- G6 formal：三组 PASS（80ep 用原始 1e-3 门槛，缩放因子=1）
- G8：三组 80 行 actual-yield，expected/actual order、flip 逐项一致
- posthoc 梯度审计（勘误清单）：`reports/step4_f1_ir_gate/posthoc_gradient_audit.json` 全 PASS
  - grad(aux_from_gate)=0 ✓；gate 参数 grad active ✓；residual 路径 aux grad 非零 ✓；
  - F1-C0 proj weight 精确零 ✓、bias 1.6e-6/1.0e-6/1.5e-6（衰减级中性，<1e-4 阈值）✓

## Formal 与因果结果（last.pt / val6 主口径）

| 组 | NORMAL | ZERO-AUX | SHUFFLE | FORCE-Q0 | FORCE-Q1 |
|---|---|---|---|---|---|
| F1-C0 | **0.2840** | 0.2840 | 0.2840 | 0.2840 | 0.2840 |
| F1-I-fixed | **0.2992** | 0.2577 | 0.2937 | 0.2618 | 0.2992 |
| F1-I-soft | **0.2977** | 0.2828 | 0.2873 | 0.2829 | 0.3154 |

best.pt（辅助）：C0 0.3195 / fixed 0.3176 / soft 0.2998。

- SOFT vs C0：**+0.0137**（>0）；SOFT vs ZERO：+0.0149；SOFT vs SHUFFLE：+0.0104
  → IR 互补候选的因果证据成立（前四项晋级条件全过）。
- SOFT vs FIXED：**−0.0015**（<0）→ 第 5 项晋级条件不成立，gate 未证明优于 q=1。
- val6 per-image q：mean 0.504、std 0.002、range 0.006（p10 0.5017 / p90 0.5051）
  → **gate 近似常数**（略高于 0.5）。
- 固定结构交叉验证：F1-I-fixed 的 NORMAL/ZERO/SHUFFLE 与 F0-I 逐位一致
  （0.2992 / 0.2577 / 0.2937），固定 q=1 的 F1 实现与 F0 等价。

## 质量退化诊断（eval_step4_f1_quality_last.json）

- identity q mean = 0.504；17 个退化条件中 8 个（noise/shift 全部 8 个）mean q
  低于 identity 超 1e-4（要求 ≥9 → 不达标）；blur/contrast 的 q 未系统性下降。
- learned gate 在 noise 四档以及 blur:0.50、contrast:0.50/0.75/1.00 上 AP 高于
  FORCE-Q1；**shift 四档均低于 FORCE-Q1**（learned 约 0.281–0.295 vs
  FORCE-Q1 约 0.312–0.323）。
- reliability_supported = false（8/17 < 9）→ 第 6 项晋级条件不成立。
- 更正（2026-08-16，审阅者指正）：上一版此处写"q≈0.5 在噪声/移位下保留了更多
  性能"为事实错误——shift 四档全部是 FORCE-Q1 更好。JSON 判级不受影响。

## LOO（val6 leave-one-out，last.pt）

- SOFT−C0：median > 0 且 ≥4/6 正（通过第 4 项）。
- SOFT−FIXED：full −0.0015，2/6 正，median −0.0028，
  被 000016_042_suppl_00000164 单图主导（−0.0304）→ 第 5 项 LOO 不成立。

## Summarizer decision

`_summary_step4_f1.json`（schema v2，verdict_frozen=true）：

- **IR_COMPLEMENTARY_BUT_GATE_NOT_PROVEN_BETTER_THAN_Q1**
- next_step：keep the simpler fixed residual unless quality probe shows robustness gain
- 对应 DESIGN_FREEZE 晋级规则："前四项成立但第 5 项不成立 → 只能说 IR feature
  fusion 成为互补候选，不能说 gate 优于 q=1"。

## 原始失败证据与处理

- smoke-F1-C0-e1 首跑 G6 失败（proj W 1.04e-4）：根因 gate 环路 + Muon 零梯度放大，
  详见 `feedback/2026-08-16_smoke-F1-C0-e1.md`；目录保留。
- 修复 1：gate 输入 detach（优化图变更，已获审阅者认可并写入 EXECUTION_GUIDE 勘误）；
  修复 2：G6 阈值按 epochs 线性缩放（仅 smoke 活性检查，formal 用原始 1e-3）。
- posthoc 首跑失败（ckpt 参数 requires_grad=False，Ultralytics 保存行为）：
  加载后显式 requires_grad_(True) 修复，产物通过。
