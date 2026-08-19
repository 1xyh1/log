# T1-S implementation bundle

T1-S 是 **evaluation-only** 的 P5 FULL residual source-specificity audit。

它不训练模型，只读取已经接受的：

```text
T1-F_P5_FULL_seed20260812/weights/last.pt
```

并回答：

> T1 的 architecture-level performance gain 是否真正依赖正确 recipient ↔ IR residual source identity？

## 文件

```text
docs/step4_t1s/DESIGN_FREEZE.md
src/multimodal/t1s_source_specificity.py
scripts/audit_t1s.py
scripts/eval_t1s_source_specificity.py
tests/test_t1s.py
T1S_README.md
T1S_IMPLEMENTATION_VALIDATION.json
```

## 核心计算

```text
T1 residual cache: 6 sources
6×6 recipient/source matrix: 36 cells
ZERO residual: 6 recipients
native direct forward anchor: 6 recipients
all derangements: !6 = 265, assembled offline from cached matrix stats
```

265 个 derangement **不会重新 forward 265 次**。

## 两个硬 anchor

### Native diagonal

对每个 recipient：

```text
matrix(r <- r) detection SHA
== normal T1 checkpoint detection SHA
== accepted posttrain_paired T1 native detection SHA
```

### Frozen A2 donor mapping

```text
matrix(r <- frozen_donor[r]) detection SHA
== accepted posttrain_paired T1 donor detection SHA
```

所以 T1-S 不是另起 evaluator 语义，而是扩展已经接受的 T-series paired audit。

## 执行顺序

从项目根目录运行。

### 1. package tests

```powershell
.venv\Scripts\python.exe -m pytest tests/test_t1s.py -q

.venv\Scripts\python.exe -m py_compile `
  src/multimodal/t1s_source_specificity.py `
  scripts/audit_t1s.py `
  scripts/eval_t1s_source_specificity.py `
  tests/test_t1s.py
```

### 2. static audit

```powershell
.venv\Scripts\python.exe scripts/audit_t1s.py --phase static
```

从 static audit PASS 后，不修改 T1-S bundle 文件。

### 3. formal preexecution audit

```powershell
.venv\Scripts\python.exe scripts/audit_t1s.py --phase formal `
  --contract "D:/pycharm/Python Develop/YOLO_1/step3_data_contract.json"
```

必须：

```text
G1–G15 all true
all_passed = true
exit = 0
```

这个 audit **故意不再使用 git ancestry 作为证据**。T-series 上游通过结果文件、checkpoint、manifest、donor map、源文件 raw SHA 闭合，避免重复 R11 历史分叉问题。

### 4. evaluator

```powershell
.venv\Scripts\python.exe scripts/eval_t1s_source_specificity.py `
  --device 0 `
  --contract "D:/pycharm/Python Develop/YOLO_1/step3_data_contract.json"
```

默认拒绝 overwrite。

## 输出

```text
reports/step4_t1s/preexecution_static_audit.json
reports/step4_t1s/preexecution_audit.json
reports/step4_t1s/source_matrix.json
reports/step4_t1s/derangements.json
reports/step4_t1s/t1s_summary.json
```

## summary 主要看什么

```text
identity/native AP = I
ZERO AP            = Z
frozen donor AP    = F
265 derangement distribution = D
```

以及：

```text
I-Z
I-median(D)
median(D)-Z
identity rank/percentile
frozen donor rank/percentile
exact one-sided randomization p
```

预注册：

```text
alpha = 0.05
```

## 分支

```text
WRONG_SOURCE_TYPICALLY_OUTPERFORMS_NATIVE
INFERENCE_RESIDUAL_NOT_SUPPORTED_TRAINING_DYNAMICS_CANDIDATE
PAIRED_SOURCE_SPECIFICITY_SUPPORTED_SINGLE_SEED
GENERIC_RESIDUAL_BENEFIT_SOURCE_IDENTITY_UNPROVEN
SOURCE_SPECIFICITY_INCONCLUSIVE
```

只有：

```text
PAIRED_SOURCE_SPECIFICITY_SUPPORTED_SINGLE_SEED
```

可以给：

```text
replication_seed_go = true
```

无论结果如何，本轮始终：

```text
training_go   = false
depth_go      = false
production_go = false
```
