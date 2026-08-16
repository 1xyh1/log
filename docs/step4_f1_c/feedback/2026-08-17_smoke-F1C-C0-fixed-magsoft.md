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
