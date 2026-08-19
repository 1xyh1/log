# T1-TR implementation bundle

目标：

> 判断 T1 的训练收益是否依赖训练时正确 RGB↔IR source identity。

只新增一个训练臂：

```text
U2-S_P5_FULL_BALANCED_SHUFFLED_seed20260812
```

不重训 T0/T1。

## 核心设计

U2-S 每 epoch 使用 train11 的 deterministic cyclic derangement：

```text
shift = 1 + epoch % 10
donor[i] = train_ids[(i + shift) % 11]
```

80 epochs 中每个 recipient 对每个非自身 donor 恰好出现 8 次。

模型、初始化、optimizer、训练协议全部锚定旧 T1-F。唯一 treatment 是训练时 IR source identity。

最终三臂统一用 ZERO inference 比较：

```text
U0-N = old T0 -> ZERO/RGB
U1-P = old T1 -> ZERO
U2-S = new shuffled-training checkpoint -> ZERO
```

Primary endpoints：

```text
val6 ZERO mAP50-95
train11 ZERO mAP50-95
all17 ZERO mAP50-95
```

## 文件

```text
docs/step4_t1tr/DESIGN_FREEZE.md
src/multimodal/t1tr_training_source.py
scripts/run_t1tr.py
scripts/smoke_t1tr.py
scripts/audit_t1tr.py
scripts/verify_t1tr_run.py
scripts/eval_t1tr.py
scripts/summarize_t1tr.py
tests/test_t1tr.py
T1TR_README.md
T1TR_IMPLEMENTATION_VALIDATION.json
```

## 执行顺序

项目根目录：

```powershell
cd "C:\Users\xyh23\Documents\ChatGPT\多模态模型\multimodal_yolo26_qaf_v0_3"
```

### 1. Regression

```powershell
.venv\Scripts\python.exe -m pytest tests/test_t1tr.py -q

.venv\Scripts\python.exe -m py_compile `
  src/multimodal/t1tr_training_source.py `
  scripts/run_t1tr.py `
  scripts/smoke_t1tr.py `
  scripts/audit_t1tr.py `
  scripts/verify_t1tr_run.py `
  scripts/eval_t1tr.py `
  scripts/summarize_t1tr.py `
  tests/test_t1tr.py
```

### 2. Static audit

```powershell
.venv\Scripts\python.exe scripts/audit_t1tr.py --phase static
```

之后不要修改 T1-TR bundle 文件。

### 3. Real 1ep smoke

```powershell
.venv\Scripts\python.exe scripts/smoke_t1tr.py `
  --device 0 `
  --contract "D:/pycharm/Python Develop/YOLO_1/step3_data_contract.json" `
  --data "D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml" `
  --base-checkpoint "E:/odin/yolo26s.pt"
```

必须：

```text
all_dynamic_gates_passed = true
```

### 4. Formal G1-G18 audit

```powershell
.venv\Scripts\python.exe scripts/audit_t1tr.py --phase formal `
  --contract "D:/pycharm/Python Develop/YOLO_1/step3_data_contract.json" `
  --base-checkpoint "E:/odin/yolo26s.pt"
```

必须：

```text
all_passed = true
G1..G18 all true
```

### 5. 只训练 U2-S 80ep

```powershell
.venv\Scripts\python.exe scripts/run_t1tr.py `
  --run-kind formal `
  --device 0 `
  --contract "D:/pycharm/Python Develop/YOLO_1/step3_data_contract.json" `
  --data "D:/pycharm/Python Develop/YOLO_1/v031_step1_rgb_sample/dataset.yaml" `
  --base-checkpoint "E:/odin/yolo26s.pt"
```

### 6. Formal run integrity

```powershell
.venv\Scripts\python.exe scripts/verify_t1tr_run.py
```

必须：

```text
T1TR_U2_FORMAL_RUN_PASS
```

### 7. Common ZERO evaluator

```powershell
.venv\Scripts\python.exe scripts/eval_t1tr.py --device 0 `
  --contract "D:/pycharm/Python Develop/YOLO_1/step3_data_contract.json"
```

Hard anchors：

```text
T0 zero override == T0 native bitwise per val sample
T1 ZERO val6 == T1-S ZERO = 0.29596085371085373
```

### 8. Summary

```powershell
.venv\Scripts\python.exe scripts/summarize_t1tr.py
```

输出：

```text
reports/step4_t1tr/pretraining_static_audit.json
reports/step4_t1tr/pretraining_smoke.json
reports/step4_t1tr/pretraining_audit.json
reports/step4_t1tr/posttrain_zero_eval.json
reports/step4_t1tr/t1tr_summary.json
```

Formal run：

```text
runs/step4_t1tr/U2-S_P5_FULL_BALANCED_SHUFFLED_seed20260812/
```

## 结果判级

只有：

```text
PAIRED_TRAINING_SOURCE_SPECIFICITY_SUPPORTED_SINGLE_SEED
```

才允许：

```text
replication_seed_go = true
```

始终：

```text
depth_go = false
production_go = false
```

本 bundle 不修改 `trimodal_dataset.py`、T-series/T1-S frozen files 或既有 checkpoints。
