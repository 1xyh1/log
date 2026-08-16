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

## HOLD 整改（审阅者 2026-08-16 复审，formal 前必修）

审阅者判定 HOLD，三项必修已全部落地并重跑 smoke（新目录 -r2 修订号）：

1. **B1 pretrain audit 成为硬门禁**：`run_step4_f1_b.py` 增加 `--audit-report`
   参数，核验 schema/all_passed/全部 provenance（corruption/runner/audit/F1
   summary 源 SHA），陈旧即 `B1_PRETRAIN_AUDIT_STALE` 拒绝训练；manifest 新增
   `pretrain_audit_sha256`、`f1_v4_summary_sha256`、`design_freeze_sha256`。
2. **G9 逐样本证据落盘**：新增 `step4_b1_g9_records.jsonl`（每 epoch 每样本的
   sample_id/kind/severity/ir_sha_before/ir_sha_after/三通道与标注不变断言），
   G9 trace 行新增 `records_sha256`（canonical records SHA）；B1 summarizer
   将逐行重判，不信任 trace 布尔。
3. **smoke 目录唯一化**：所有已存在目录一律拒绝（formal/smoke 都不覆盖）；
   smoke 自动带 `-rN` 修订号（旧 smoke 证据保留）。

建议项同步完成：B1 专用评估链（`eval_step4_f1_b_causality.py` /
`step4_f1_b_loo.py` / `eval_step4_f1_b_quality.py` / `summarize_step4_f1_b.py`，
含 G9 逐行重判、own-QCLEAN、macro/worst-4、9/17 判据）；quality 文本改为
"ADAPTIVITY IS NOT PROVEN"。

重跑结果：audit 全过（新 provenance 含新 runner SHA）→ 三组 smoke
（`smoke-*-e1-r2`）全 PASS，G9 records 落盘 11 条/epoch、records_sha256 入 trace。
全部产物编译通过、18 项对抗测试全过。**继续等 FORMAL GO**。
