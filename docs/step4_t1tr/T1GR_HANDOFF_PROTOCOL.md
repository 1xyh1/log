# T1-GR Handoff Protocol
## Paired-Training Generalization Replication

状态：**WAITING FOR FORMAL DATA**
日期：2026-08-19

## 1. 唯一目标

用 fresh data evidence 验证：

> correctly paired IR training 是否稳定改善真正 held-out generalization，并优于 NULL 与 balanced fully-wrong IR training。

不再继续在旧 17 图 / val6 上做 treatment 选择。

## 2. 启动条件

```text
E1 正式约 2,000 图训练集可用
E2 新 data contract 完成
E3 fresh held-out split 从未参与 A2→T1-TR 决策
E4 split 在任何 T1-GR 训练前冻结
E5 真 Step 1 baseline 在正式数据上重新建立
```

任一失败：

```text
T1GR_HOLD
```

## 3. Split

优先：

```text
TRAIN
DEV
FINAL HOLDOUT
```

FINAL HOLDOUT：

```text
不得参与超参数选择
不得参与 treatment 选择
不得参与 early stopping
最终 adjudication 前保持封闭
```

必须检查 sequence / scene / source leakage。

## 4. 三个 matched arms

```text
G0-N
  NULL training

G1-P
  correctly paired P5 FULL IR training

G2-S
  balanced fully-wrong P5 FULL IR training
```

G2-S 继续满足：

```text
recipient RGB/label/geometry 不变
只改变 IR source identity
no self-match
balanced donor usage
deterministic and logged
```

## 5. Matching contract

三臂必须相同：

```text
base checkpoint
model class
state_dict keys
initialization
requires_grad map
optimizer groups
optimizer hyperparameters
batch / nbs
imgsz
augmentation
sampling
epoch count
checkpoint policy
validation cadence
```

唯一 treatment：

```text
training-time IR source condition
```

## 6. Seed policy

T1-GR 要做 cross-seed consistency。

建议最低：

```text
3 seeds
```

seed 值在 formal design freeze 时一次性冻结，禁止看完 seed1 再挑后续 seed。

## 7. Primary GO authority

只有 fresh held-out generalization 可进入 GO：

```text
held-out mAP50-95
cross-seed contrast consistency
held-out per-image/fold sensitivity
```

正式 contrasts：

```text
G1-G0
G1-G2
G2-G0
```

## 8. Diagnostics only

以下不得再作为 GO endpoint：

```text
training AP
train-set mAP
train+holdout all-set AP
old val6
best epoch alone
loss minimum alone
```

允许记录：

```text
train/holdout gap
late-k stability
optimization speed
residual RMS
native-vs-zero inference
```

## 9. Interpretation matrix

```text
G1 > G0
and
G1 > G2
across seeds
and held-out sensitivity stable
→ PAIRED_TRAINING_GENERALIZATION_SUPPORTED
```

```text
G1 ≈ G2 > G0
→ generic training treatment benefit;
  source identity not established
```

```text
G2 >= G1
→ paired source specificity fails replication
```

```text
G0 >= G1/G2
→ small-sample T1 signal did not transfer
```

## 10. Depth policy

T1-GR 完成前：

```text
depth_go = false
production_go = false
```

T1-GR 通过后再进入 D-series。

Depth topology 不预设为 P5-only。

## 11. Required formal outputs

```text
docs/step4_t1gr/DESIGN_FREEZE.md
reports/step4_t1gr/data_contract_audit.json
reports/step4_t1gr/split_freeze.json
reports/step4_t1gr/pretraining_audit.json
reports/step4_t1gr/per_seed_results.json
reports/step4_t1gr/cross_seed_summary.json
reports/step4_t1gr/t1gr_summary.json

runs/step4_t1gr/
  G0-N_seed...
  G1-P_seed...
  G2-S_seed...
```

## 12. Current status

```text
T1-GR
  WAITING FOR FORMAL DATA

Do not finalize runner until:
  dataset contract
  split IDs
  formal Step 1 baseline recipe
are frozen.
```

启动链：

```text
formal dataset
→ data contract audit
→ leakage-aware fresh split freeze
→ true Step 1 baseline
→ T1-GR design freeze
→ implementation/smoke
→ matched multi-seed training
→ fresh-held-out adjudication
```
