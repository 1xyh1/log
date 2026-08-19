# T1-GR E2–E5 Entry Freeze v2

状态：**IMPLEMENTATION READY FOR FORMAL DATA AUDIT / TRAINING NOT YET AUTHORIZED**
日期：2026-08-19

## 上游状态

```text
A2–A5 / T-series / T1-S / T1-TR
  CLOSED

T1-GR:
  E1 formal dataset available
  E2 contract       PENDING
  E3 fresh split    PENDING
  E4 split freeze   PENDING
  E5 Step1 baseline PENDING

Replication / Depth / Production
  HOLD
```

## 红线

```text
OLD val6:
  historical / diagnostic only
  NEVER treatment GO

FINAL HOLDOUT:
  sealed before any training
  NEVER hyperparameter tuning
  NEVER architecture selection
  NEVER seed selection
  NEVER early stopping
```

## v2 evidence model

v1 中以下字段属于“自我声明”，v2 禁止：

```text
final_holdout_used = False
final_holdout_evaluated = False
frozen_before_training = True
```

v2 必须由：

```text
private/public manifest
actual file ID sets
SHA256 commitments
runtime checkpoint hash
view dataset.yaml hash
freeze/train UTC timestamps
physical head inspection
actual Ultralytics effective args
```

推导合规事实。

## Private / public separation

repo 外 private：

```text
formal_data_contract_private.json
split_proposal_private.json
split_freeze_private.json
final_holdout_sealed.json
step1_rgb_view/
  dataset.yaml
  view_manifest.json
  images/train
  images/val
  labels/train
  labels/val
```

repo 内 public：

```text
reports/step4_t1gr/data_contract_public.json
reports/step4_t1gr/split_freeze_public.json
reports/step4_t1gr/step1_baseline_recipe.json
reports/step4_t1gr/step1_baseline_report.json
reports/step4_t1gr/e2_e5_entry_audit.json
```

public contract/freeze 不保存 paired/train/dev/holdout sample IDs。

注意：这叫 **repo nondisclosure + harness access seal**，不是声称人无法从原始数据目录推断样本集合。真正保证的是正式 Step1/T1-GR runner 不接受 raw dataset 或任意 dataset YAML，只接受冻结 view manifest。

## P0 hard fixes

### P0-1 Holdout seal

- private contract / proposal / split truth 强制 repo 外。
- public freeze 只保存 count + ID SHA commitment。
- `frozen_before_training=True` 删除。
- freeze 写 UTC timestamp；未来 runner 运行时计算 `freeze_timestamp < training_start_timestamp`。

### P0-2 Runner/evaluator access control

- Step1 runner **没有 `--data` 参数**。
- Step1 evaluator **没有 `--data` 参数**。
- 两者只接受 `--view-manifest`。
- view manifest pin：recipe/private contract/private split/dataset.yaml/train IDs/dev IDs/每个 RGB+label 文件 SHA。
- 实际 train/dev 目录 ID 集合重新扫描并与 commitment 比较。
- view 中增加任意额外样本（包括 holdout）均 fail。

### P0-3 Format/full-hash gate

Formal contract builder 无 `--full-hash` 可选开关：**full SHA256 是不可关闭行为**。

Hard gate：

```text
RGB/IR/Depth readable
expected dtype
expected ndim
expected channels
optional exact H/W
RGB/IR/Depth H/W match
exact sample count
class_names length == num_classes
strict YOLO detect labels
all files full SHA256
```

label parser：

```text
exactly 5 fields
class integer/in-range
finite numeric values
w/h > 0
normalized values in range
bbox x1/y1/x2/y2 inside image (tolerance frozen in layout spec)
```

### P0-4 Step1 recipe freeze

`step1_training_spec.json` 必须显式冻结：

```text
optimizer
lr0/lrf
momentum/weight_decay
nbs
warmup
batch/imgsz/epochs/patience
AMP/workers/cache
mosaic/close_mosaic
HSV / translate / scale / shear / perspective
flip / mixup / cutmix / copy_paste / erasing
end2end
seed
validation iou/conf/max_det/half
```

recipe 同时 pin：

```text
Ultralytics version
Torch version
base checkpoint SHA256
public/private contract commitment
public/private split commitment
```

runner：

```text
runtime checkpoint SHA recheck
runtime Ultralytics version recheck
pre-train effective args compare
post-run args.yaml effective args compare
physical head nc/mode pretrain check
```

## P1 hardening

- split = group-aware + sample/class-image/class-box joint balancing。
- 三个 split 必须非空。
- 每类同时报告 image count + box count。
- coverage minima 必须预注册；无法覆盖的类只能 explicit exemption + reason。
- duplicate leakage 检查 RGB / IR / Depth / multimodal triplet。
- group rule 在 contract 阶段实际执行并记录 validation，不再只判断 `type != null`。
- split freeze 强制 proposal schema + all gates PASS。

## Formal startup order

```text
1 package pytest / py_compile
2 static audit
3 synthetic integration gate
4 formal dataset probe (read-only)
5 fill layout spec from observed facts
6 E2 private/public contract
7 E3 private group-aware split proposal
8 review split support/leakage
9 E4 freeze private/public split + sealed holdout
10 freeze full Step1 training spec / recipe
11 build copy-only TRAIN+DEV RGB view
12 run Step1 baseline
13 DEV-only Step1 evaluation
14 E2–E5 formal audit
15 only if E2–E5 all PASS: build final T1-GR G0/G1/G2 multi-seed bundle
```

现在禁止执行第 12 步之前的任何训练。
