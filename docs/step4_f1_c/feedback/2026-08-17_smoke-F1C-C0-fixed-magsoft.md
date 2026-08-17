# F1-C 执行反馈：2026-08-17 / smoke F1C-C0 · F1C-I-fixed · F1C-I-magsoft

- command：`run_step4_f1_c.py --group <g> --run-kind smoke --epochs 1`（三组串行，exit 0）
- physical run directory：`runs/step4_f1_c/smoke-<g>-e1/`

## 门禁

- F1C audit（G1–G9 沿用 + G10 七项）：`reports/step4_f1_c/pretrain_audit.json`
  all_passed=true（G10.1/2 log-RMS 一致性、G10.3 初始逐位等价、G10.4 梯度、
  G10.5 detach 语义、G10.6 q 有限 B×1 输入不变、G10.7 runner fp32 SHA 记录）
- 单测：`tests/test_step4_f1_c_gate.py` 11 项全过；
  `test_step4_f1_b_corruption.py` + `test_step4_f1_b_closeout.py` 24 项全过
- smoke：三组 PASS（G5/G6/G8/G9）
- **G10.7**：三组 `step4_fp32_rgb_sha.json` 落盘，fp32 RGB SHA 与 manifest
  initial_rgb_backbone_sha256 逐字节一致（C0/fixed/magsoft 3/3）

## smoke G6 实测

- F1C-C0：proj [0.0, 0.0, 0.0]（精确零，C0 不变量保持）
- F1C-I-fixed：proj [0.00453, 0.00429, 0.00839]（1ep 学习量正常）
- F1C-I-magsoft：proj [0.0031, 0.00319, 0.00548]（1ep 学习量正常）

## 待审阅者

按裁决，formal 仍 HOLD。G1–G10 证据、三组 smoke、初始化等价与梯度证据已备齐，
等 FORMAL GO（四组：C0 / fixed / magsoft / 同链 original-gate soft matched
control）。

## P0 整改与 -r2 重跑（审阅者 HOLD 四项，2026-08-17 晚）

P0-1：runner 现在只接受 `step4-f1-c-audit-v2`，重哈希 9 项 pin
（corruption/runner/audit/gate/model/F1C DESIGN_FREEZE/A1-v2 last+best/
B1-v2.2）；audit v2 pin 当前 run_step4_f1_c.py、G9 静态审计 F1C runner。
P0-2：语义清理——C0: learned+magnitude、fixed: fixed_one+magnitude、
magsoft: learned+magnitude、soft: learned+original；删除 gate_mode="magnitude"
混合语义（模型 gate_module 唯一决定模块）；manifest 的 gate_module 从模型
实际 `gate_module_kind` 取值；design_freeze pin 改为 docs/step4_f1_c/。
P0-3：watcher 白名单加 runs/step4_f1_c/，54 个 smoke 原始产物已推送；
summarizer（summarize_step4_f1_c.py）逐行重判 G6/G8/G9/G10.7。
P0-4：F1C 四组评估链齐备——eval_step4_f1_c_causality.py（含 gate_module
身份检查）、eval_step4_f1_c_quality.py（own QCLEAN + original-soft 对照）、
step4_f1_c_loo.py（四组，MAGSOFT−C0/FIXED/ORIGSOFT 冻结条件 full>0、median>0、
≥4/6）、audit_step4_f1_c_posthoc.py、summarize_step4_f1_c.py（含历史 B1-soft
0.304028 外部基线比较）。

测试加强：detach 测试输入 requires_grad=True（强测试）；residual 测试走实际
`_gated_residual`；G10.6 新增 6ch 输入 RGB/Depth 通道逐字节不变检查。
G10.7 JSON 升级为 schema v1（expected/actual/match），runner 训练结束即时
断言 match。

-r2 重跑：三组 smoke 全 PASS；旧 e1 目录保留（陈旧证据）；G6/G8/G9 全过、
fp32 RGB SHA match 3/3、G9 records 11 条/epoch、跨组 schedule 一致。
