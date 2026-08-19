# T-series implementation bundle

This bundle implements the frozen **P5-only Direct IR Injection** training experiment.

## What it contains

```text
docs/step4_tseries/TRAINING_DESIGN_FREEZE.md
docs/step4_tseries/IMPLEMENTATION_ADJUDICATION.md
docs/step4_a4/feedback/2026-08-19_erratum.md

src/multimodal/tseries_core.py
src/multimodal/tseries_p5_model.py
src/multimodal/tseries_runtime.py

scripts/audit_tseries.py
scripts/run_tseries.py
scripts/smoke_tseries_suite.py
scripts/run_tseries_formal_suite.py
scripts/eval_tseries_posttrain.py
scripts/eval_tseries_paired.py
scripts/summarize_tseries.py

tests/test_tseries.py
```

## Scientific arms

```text
T0-N: architecture-matched NULL
      P3/P4 no direct IR
      P5 prediction path = R5

T1-F: P5 FULL
      F5 = R5 + delta5

T2-A: P5 AC_ALL
      F5 = R5 + delta5 - mean_HW(delta5)
```

All arms use the same model class, parameter names, initialization path, optimizer
parameter set, dataset, seed, and 80-epoch recipe. There is no reliability gate and
Depth is held out.

## Important P0 hardening

T0 does **not** use `R5 + 0*delta5` in the loss graph. Its aux/projection path is
evaluated under `no_grad` for matched BN exposure/logging, while detection prediction
is exactly `R5`.

T2 keeps the exact frozen post-projection AC_ALL forward. FP32 reduction can leave
small raw projection-bias gradient dust even though the exact mathematical derivative
is zero. The trainer therefore zeroes only T2 `proj.bias.grad` immediately before the
real optimizer step, records the pre-zero dust, requires zero bias weight decay, and
requires the one-epoch MuSGD smoke to leave the bias bitwise unchanged.

## Required execution order

Run from the project root after copying the bundle files into the repository.

### 1. Package/source regression

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tseries.py -q
.venv\Scripts\python.exe -m py_compile `
  src/multimodal/tseries_core.py `
  src/multimodal/tseries_p5_model.py `
  src/multimodal/tseries_runtime.py `
  scripts/audit_tseries.py `
  scripts/run_tseries.py `
  scripts/smoke_tseries_suite.py `
  scripts/run_tseries_formal_suite.py `
  scripts/eval_tseries_posttrain.py `
  scripts/eval_tseries_paired.py `
  scripts/summarize_tseries.py
```

### 2. Static audit

```powershell
.venv\Scripts\python.exe scripts/audit_tseries.py --phase static
```

Do not edit any T-series design/source/test file after this point.

### 3. Real one-epoch matched smokes

```powershell
.venv\Scripts\python.exe scripts/smoke_tseries_suite.py `
  --device 0 `
  --contract "D:/pycharm/Python Develop/YOLO_1/step3_data_contract.json" `
  --data "D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml" `
  --base-checkpoint "E:/odin/yolo26s.pt"
```

This runs T0/T1/T2 through the real trainer/MuSGD/data path for one epoch.

### 4. Formal G1-G18 pretraining audit

```powershell
.venv\Scripts\python.exe scripts/audit_tseries.py --phase formal `
  --contract "D:/pycharm/Python Develop/YOLO_1/step3_data_contract.json" `
  --data "D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml" `
  --base-checkpoint "E:/odin/yolo26s.pt"
```

Formal training is HOLD unless this exits 0.

### 5. Three formal 80-epoch runs

One command:

```powershell
.venv\Scripts\python.exe scripts/run_tseries_formal_suite.py `
  --device 0 `
  --contract "D:/pycharm/Python Develop/YOLO_1/step3_data_contract.json" `
  --data "D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml" `
  --base-checkpoint "E:/odin/yolo26s.pt"
```

The order is fixed:

```text
T0-N_P5_NULL_seed20260812
T1-F_P5_FULL_seed20260812
T2-A_P5_ACALL_seed20260812
```

Existing formal run directories are never overwritten.

### 6. Post-training performance

```powershell
.venv\Scripts\python.exe scripts/eval_tseries_posttrain.py --device 0
```

Re-evaluates `last.pt` for val6, train11, all17, val6 LOO and reads the final-10
training curve median. Best epoch is descriptive only.

### 7. Retrained paired causality

```powershell
.venv\Scripts\python.exe scripts/eval_tseries_paired.py --device 0
```

For T1 it compares recipient-vs-donor P5 FULL residual. For T2 it compares
recipient-vs-donor P5 AC_ALL, with donor AC computed from the donor's own projected
residual and own full-map mean. Native residual override must be bitwise equivalent to
normal checkpoint prediction per val sample before donor results are accepted.

### 8. Final single-seed summary

```powershell
.venv\Scripts\python.exe scripts/summarize_tseries.py
```

Primary performance contrasts use:

```text
last val6
late10 median val6
train11
```

A contrast is `STABLE_POSITIVE` only if all three are positive; no AP margin is
introduced. The first seed can at most authorize a replication seed. It cannot
authorize Depth or production.

## Formal outputs

```text
reports/step4_tseries/pretraining_static_audit.json
reports/step4_tseries/pretraining_smoke.json
reports/step4_tseries/pretraining_audit.json
reports/step4_tseries/posttrain_performance.json
reports/step4_tseries/posttrain_paired.json
reports/step4_tseries/tseries_summary.json
```

Per-run:

```text
runs/step4_tseries/<run>/
  manifest.json
  optimizer_manifest.json
  tseries_data_order.jsonl
  tseries_mechanism.jsonl
  results.csv
  weights/last.pt
```
