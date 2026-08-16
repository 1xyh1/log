# 执行指导

以下命令在包含 `.pt` 权重、原始样例数据和现有 Python 环境的项目根目录执行。GitHub
镜像没有 checkpoint，不能在镜像独立完成动态门禁或训练。

## 0. 先重新闭合 F0 summary self-pin

远端最新 `step4_loo.json` 已是 v2，内部 dependency/evaluator/model/shuffle provenance、
G8 和 G6 均通过；但 `_summary_step4.json` 记录的 `loo_file_sha256` 仍是 LOO 最后一次
变更前的值。数值结论不受影响，且**不需要重跑 LOO**。在含权重的本机只重跑 summary，
让它重新验证全部门禁并钉住当前 LOO 文件字节：

先把旧 summary 复制到带时间戳的历史目录，再执行：

```powershell
python scripts/summarize_step4.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

命令必须完整 PASS；禁止手工修改 SHA 或 summary 绕过门禁。F1 audit 的 `G0_f0_closeout`
会独立复核 summary schema、verdict、LOO/self-source pin、G8、G6 与全部 provenance；任一
不一致都会阻止 smoke/formal。

## 1. F1 训练前审计

```powershell
python scripts/audit_step4_f1_modality_quality.py --split val
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/audit_step4_f1_ir_gate.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

先读 `reports/step4_f1_ir_gate/`。质量统计只是描述性证据；不能因为某图动态范围低就
直接删除、改标签或把手工质量分数接到 gate。

## 2. 三组 smoke

```powershell
python scripts/run_step4_f1_ir_gate.py --group F1-C0 --run-kind smoke --epochs 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/run_step4_f1_ir_gate.py --group F1-I-fixed --run-kind smoke --epochs 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/run_step4_f1_ir_gate.py --group F1-I-soft --run-kind smoke --epochs 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

检查三组 G5/G6/G8、无 NaN/OOM、RGB SHA 不变、P3/P4/P5 更新符合组别预期。
smoke 目录不得使用 formal 名称。

## 3. Formal 训练

只在 smoke 全 PASS 后串行执行：

```powershell
python scripts/run_step4_f1_ir_gate.py --group F1-C0 --run-kind formal
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/run_step4_f1_ir_gate.py --group F1-I-fixed --run-kind formal
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/run_step4_f1_ir_gate.py --group F1-I-soft --run-kind formal
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

正式目录已存在时 runner 会拒绝覆盖。不要用 `exist_ok`、重命名 smoke 或手工拼接结果
绕过保护。

## 4. 主因果评估

```powershell
python scripts/eval_step4_f1_causality.py --group F1-C0 --run-name F1-C0
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/eval_step4_f1_causality.py --group F1-I-fixed --run-name F1-I-fixed
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/eval_step4_f1_causality.py --group F1-I-soft --run-name F1-I-soft
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

先看 last.pt 的 NORMAL/ZERO/SHUFFLE；FORCE-Q0/Q1 仅解释机制。FORCE-Q0 使用的是
已经与 IR 共同训练过的 neck/head，不能代替独立 `F1-C0`。

## 5. 质量退化诊断、LOO 与汇总

```powershell
python scripts/eval_step4_f1_quality.py --run-name F1-I-soft
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/step4_f1_loo.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/summarize_step4_f1.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

质量诊断成功必须同时满足 q 对退化有合理响应、learned gate 相比 FORCE-Q1 保留更多
检测性能；只看到 q 变化不能宣称可靠性成功。最终判定只由汇总脚本在 provenance、
G6、G8、LOO 与 last.pt 质量报告全部通过后写出。三个 eval/LOO/summary 脚本默认拒绝
覆盖；确需重算时先归档旧派生产物，再显式传 `--overwrite`，不得把不同源码版本混在
同一个 formal 证据目录。

## 6. 当前验证边界

本补丁提交环境没有 PyTorch、Ultralytics、本机 YOLO26 checkpoint 和原始数据，因此只能
完成源码静态编译、JSON/AST 级检查，不能代替上述动态 audit/smoke。执行反馈必须写入
`feedback/`，不要反向修改本文件中的预注册阈值。

源码镜像 watcher 已允许同步 `runs/step4_f1_ir_gate/` 下的 JSON、JSONL、CSV、YAML、
TXT 和 `results.png`；`.pt`、数据集和其他图像仍被白名单规则排除。把本分支合并到受
watcher 管理的本机源码前，先确认本机源码也已包含同一批文件，避免双向目录状态不一致。

## 7. 勘误：gate 输入的 detach 语义（2026-08-16 本机 smoke 后追加）

本机 smoke 发现 F1-C0 的 projection weight 不再保持精确零，根因是 gate 环路的
"数值尘埃放大"：gate 输入 `LayerNorm(GAP(A))` 在零输入处梯度为 1/√eps≈1000；proj
bias 先动 → `dL/dq ≠ 0` → A 经 gate 环路获得非零梯度 → 零初始化 projection 得到
~1e-11 级梯度 → MuSGD 的 Muon 分支把任何非零动量按 `X /= X.norm()+eps` 归一化放大为
O(1) 更新。修复为：**gate 每次仍读取当前 A（无信息变化），但 gate 路径不向 aux
encoder 反传梯度；gate 参数仍然更新，aux encoder 仅由 residual 路径训练**。这是优化
图变更，不是数值修复，正式判级前需按下列清单分别验证：

- grad(aux_from_gate) == 0（受控 backward：冻结其余、只留 gate 回传路径，aux 参数梯度为零）；
- soft gate 参数确实更新（G6 `gate_max_abs_change > 0`）；
- active aux 仍通过 residual 路径更新（G6 `aux_encoder_global_rel_l2` 超过衰减阈值）；
- F1-C0 的 projection weight 精确为零、bias 保持衰减级中性（G6 记录 `proj_bias_norms`）。

G6 的 epoch 线性缩放阈值（`1e-3 × epochs/80`）**只用于 smoke 的训练链活性检查**，
不用于判断模型质量；80-epoch formal 保持原始 1e-3 门槛。汇总时必须同时保存缩放公式、
epoch 数、实测变化量与 control noise floor，供审阅者核对。
