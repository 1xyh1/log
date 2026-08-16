# F1-C 执行指导

## 0. 前置

- B1 summary 必须为 v2.2 冻结状态；F1-C audit 全过才能启动训练。

## 1. 训练前审计与测试

```powershell
python -m pytest tests/test_step4_f1_c_gate.py tests/test_step4_f1_b_corruption.py tests/test_step4_f1_b_closeout.py -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/audit_step4_f1_c.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

## 2. 三组 smoke（等审阅者放行后执行）

```powershell
python scripts/run_step4_f1_c.py --group F1C-C0 --run-kind smoke --epochs 1
python scripts/run_step4_f1_c.py --group F1C-I-fixed --run-kind smoke --epochs 1
python scripts/run_step4_f1_c.py --group F1C-I-magsoft --run-kind smoke --epochs 1
```

检查 G5/G6/G8/G9 + `step4_fp32_rgb_sha.json`（G10.7）落盘。

## 3. Formal（四组，审阅者 GO 后）

```powershell
python scripts/run_step4_f1_c.py --group F1C-C0 --run-kind formal
python scripts/run_step4_f1_c.py --group F1C-I-fixed --run-kind formal
python scripts/run_step4_f1_c.py --group F1C-I-magsoft --run-kind formal
python scripts/run_step4_f1_c.py --group F1C-I-soft --run-kind formal
```

## 4. 反馈纪律

执行反馈写入 `feedback/`；不回写 DESIGN_FREEZE 预注册阈值；失败目录保留。
