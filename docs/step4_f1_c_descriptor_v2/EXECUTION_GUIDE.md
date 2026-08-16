# F1-C-A1 v2 修复执行指导

本目录只放修复后的执行契约；实际运行反馈写入 `feedback/`。本补丁不重训，
但 LOO 与 descriptor audit 需要本机 `.pt`、原始样例和既有运行环境，GitHub
镜像不能独立完成。

## 1. 修复内容

- Spearman 改为并列值平均秩，避免 q 扫描大量 ties 时的顺序偏差。
- 实际 gate 输入按
  `LayerNorm(concat(GAP(A3), GAP(A4), GAP(A5)))` 计算，不再把三个尺度分别
  LayerNorm；报告只把 joint-LN 的分尺度 slice RMS 当作标量摘要，不声称其
  能代表完整向量方向的信息量。
- 空间 cosine 改为每个空间位置沿通道维计算（`dim=1`）。
- hard q* 降为探索轴；新增 scan range、best-second margin、可辨识阈值、
  连续目标 `AP(q=0)-AP(q=1)`、Pearson/Spearman 和 leave-one-family-out。
- quality 证据会核 schema、checkpoint、checkpoint/contract SHA、18 条件和
  完整五点 q grid；报告补 torch/Ultralytics 与依赖源码 provenance。
- LOO 恢复 frozen last 文件名 `step4_f1_b_loo.json`，last/best 共用完整 payload
  validator，不再使用 best 的同源恒真复算。
- B1 summary v2.2 明确区分历史 fp16 checkpoint 等价与 fp32 精确不变。

旧 `descriptor_audit_{last,best}.json` 与旧 agreement 报告保留为历史 v1 证据；
修复脚本写入带 v2/v1_1 后缀的新文件，不静默覆盖。

## 2. 先跑无权重测试

```powershell
python -m pytest tests/test_step4_f1_closeout.py tests/test_step4_f1_c_descriptor_audit.py -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m py_compile scripts/diagnose_step4_f1_c_descriptors.py scripts/step4_f1_b_loo.py scripts/summarize_step4_f1_b.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

## 3. 重建 LOO provenance（不重训）

LOO producer 与共享 validator 的源码 SHA 已变化，因此必须先重跑；否则
v2.2 summarizer 应按设计拒绝陈旧 LOO。

```powershell
python scripts/step4_f1_b_loo.py --checkpoint last.pt --overwrite
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/step4_f1_b_loo.py --checkpoint best.pt --overwrite
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

期望文件：

- `runs/step4_f1_b_corruption/step4_f1_b_loo.json`（last 主协议）
- `runs/step4_f1_b_corruption/step4_f1_b_loo_best.json`（best 诊断）

## 4. 重跑 descriptor audit v2（不重训）

```powershell
python scripts/diagnose_step4_f1_c_descriptors.py --checkpoint last.pt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/diagnose_step4_f1_c_descriptors.py --checkpoint best.pt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

期望文件为 `reports/step4_f1_c_agreement/descriptor_audit_v2_{last,best}.json`。
先看连续目标和 family holdout；不得只拿 hard q* 的单个 Spearman 判定放行。

可选重建带明文 shuffle map 的 A0 报告：

```powershell
python scripts/diagnose_step4_f1_c_agreement.py
python scripts/diagnose_step4_f1_c_agreement_all17.py
```

## 5. 重建 closeout v2.2

```powershell
python scripts/summarize_step4_f1_b.py --overwrite
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

历史 B1 formal 的 `step4_update_gate.json` 没有训练结束时的 fp32 RGB SHA；
v2.2 会保留 half-checkpoint 等价证据，并明确写出它无法排除 sub-fp16-ULP
变化。不得把这一 fallback 描述为“从 last.pt 严格证明 fp32 完全未变”。下一次
新训练 runner 必须在 checkpoint 半精度序列化前记录 final fp32 RGB SHA。

## 6. 判读门禁

只有当两个 checkpoint 上的连续目标相关方向稳定、关键 descriptor 的
leave-one-family-out 不被单个 corruption family 反转，并且 q scan 有足够动态
范围时，才进入单变量 gate-input 训练。A1 v2 仍是诊断，不改 B1 frozen verdict。
