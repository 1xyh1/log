# F1-B 执行指导

以下命令在包含 `.pt` 权重、原始样例数据和现有 Python 环境的项目根目录执行。
GitHub 镜像没有 checkpoint，不能在镜像独立完成动态门禁或训练。

## 0. 前置

- F1 的 `_summary_step4_f1.json` 必须处于 v4 冻结状态（CONSTANT ATTENUATION
  DOMINATES; ADAPTIVITY IS NOT PROVEN）。
- 训练前审计与对抗测试必须先通过。

## 1. B1 训练前审计与测试

```powershell
python -m pytest tests/test_step4_f1_b_corruption.py -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/audit_step4_f1_b.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

audit 检查：corruption schedule 冻结值、SHA256 随机性驱动（非 hash()）、
noise 场含 epoch、G9 断言逻辑、F1 v4 冻结状态引用、文档存在性。

## 2. 三组 smoke

```powershell
python scripts/run_step4_f1_b.py --group B1-C0 --run-kind smoke --epochs 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/run_step4_f1_b.py --group B1-I-fixed --run-kind smoke --epochs 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/run_step4_f1_b.py --group B1-I-soft --run-kind smoke --epochs 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

检查三组 G5/G6/G8/**G9**、无 NaN/OOM、RGB SHA 不变、G9 的 expected/actual
schedule SHA 一致、IR SHA before/after 语义正确（clean 不变、非 clean 变；
B1-C0 全不变）。smoke 目录不得使用 formal 名称。

## 3. Formal 训练（等审阅者放行后执行）

```powershell
python scripts/run_step4_f1_b.py --group B1-C0 --run-kind formal
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/run_step4_f1_b.py --group B1-I-fixed --run-kind formal
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/run_step4_f1_b.py --group B1-I-soft --run-kind formal
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

正式目录已存在时 runner 会拒绝覆盖；禁止 exist_ok 绕过、重命名 smoke、手工拼接。

## 4. 评估链（结构不变，复用 F1 脚本）

因果评估、质量诊断（FORCE-QCLEAN 必须从 B1-soft 的 clean identity 重取）、
posthoc 梯度审计、LOO、B1 summarizer（晋级条件见 DESIGN_FREEZE 第 6 节）。

## 5. 反馈纪律

执行反馈写入 `feedback/`（带日期 + physical run 名），不回写本文件与
DESIGN_FREEZE 中的预注册阈值；失败目录保留、原始错误保留。
