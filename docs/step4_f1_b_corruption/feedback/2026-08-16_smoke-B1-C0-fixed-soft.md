# F1-B 执行反馈：2026-08-16 / smoke B1-C0 · B1-I-fixed · B1-I-soft

- commit SHA：本机源仓库（未跟踪提交）；远端镜像 watcher 自动批次
- machine / GPU：Windows 11 本机 / RTX 4060 Laptop 8GB
- PyTorch / Ultralytics：torch 2.5.1+cu121 / ultralytics 8.4.56
- command：`run_step4_f1_b.py --group <g> --run-kind smoke --epochs 1`（三组串行，exit 0）
- physical run directory：
  - `runs/step4_f1_b_corruption/smoke-B1-C0-e1/`
  - `runs/step4_f1_b_corruption/smoke-B1-I-fixed-e1/`
  - `runs/step4_f1_b_corruption/smoke-B1-I-soft-e1/`

## 门禁

- B1 训练前审计（schedule 冻结 / SHA256 驱动 / G9 逻辑 / F1 v4 引用 / 文档）：
  `reports/step4_f1_b_corruption/pretrain_audit.json` all_passed=true
- 对抗测试：`tests/test_step4_f1_b_corruption.py` 18 项全过
- smoke：三组 PASS（G5/G6/G8/G9 全过）

## G9 首轮实战结果

- 三组 expected/actual schedule SHA 逐 epoch 一致（epoch 0）
- 三组 expected schedule 字节一致（同 seed 同 schedule）
- IR before/after 语义正确：B1-C0 全不变（corruption 未施加、证据照记）；
  I 组 clean 不变、非 clean 必变
- RGB/Depth/label/bbox 未变断言全过
- epoch 0 kind 分布（11 样本）：clean 6 / noise 2 / blur 2 / contrast 1 / zero 0
  （单 epoch 采样波动正常，80 epoch 分布由 formal 的 G9 汇总）

## 首跑失败与修复（fail-fast 保留证据）

- 首跑 B1-C0 G9 失败：expected 与 actual schedule SHA 不一致——规范化差异：
  `schedule_sha256` 使用 `json.dumps(sort_keys=True, separators=(",", ":"))`，
  而 runner 的 `_sha_json` 未开 sort_keys（行 dict 键序不同导致字节不同）。
  修复：runner `_sha_json` 加 `sort_keys=True`，与 canonical schedule 序列化
  完全一致；本地独立重算 MATCH=True 后重跑三组 smoke 全过。
- B1 audit 首跑失败：`no_builtin_hash` 检查把 docstring 里的说明文字误判为调用；
  改为 AST 级检查（ast.Call/Name('hash')），并把诊断列表移出 checks。

## 待审阅者放行

按审阅者指示，**formal 三组 × 80 epoch 暂不启动**，等放行。
