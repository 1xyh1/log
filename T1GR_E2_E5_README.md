# T1-GR E2–E5 Pre-training Tooling Bundle v2

状态：**PRE-TRAINING TOOLING / NO T1-GR TRAINING AUTHORIZATION**

v2 是对 v1 审阅中 4 个 P0 + P1 的修正版。

## 现在要不要训练？

**不要。**

当前允许做到：

```text
package validation
→ synthetic integration gate
→ formal dataset read-only probe
→ E2 contract
→ E3 split proposal
→ E4 split freeze
→ Step1 recipe/view freeze
```

只有 E2–E4 和 Step1 recipe/view 证据链闭合后，才能启动 Step1 RGB baseline。
真正 T1-GR G0/G1/G2 × multi-seed 仍必须等 E5 Step1 baseline 完成。

---

## v2 核心变化

### 1. private/public split

**不要把 full contract 或 split proposal 写进 repo。**

建议建立 repo 外 private root，例如：

```text
E:/t1gr_private/
```

里面放：

```text
formal_data_contract_private.json
split_proposal_private.json
split_freeze_private.json
final_holdout_sealed.json
step1_rgb_view/
```

repo 内只放 sanitised public reports。

### 2. runner/evaluator 不再接受任意 `--data`

formal Step1 只接受：

```text
--recipe
--view-manifest
```

`dataset.yaml` 必须来自 view manifest，且 SHA/实际 train/dev IDs/每个文件 SHA 全部闭合。

### 3. formal full hash 无关闭接口

`t1gr_build_contract.py` 每次 formal contract 都对全部：

```text
RGB
IR
Depth
label
```

做 SHA256；没有 `--full-hash` 可忘记。

### 4. Step1 effective recipe 全冻结

必须显式填写：

```text
config/step1_training_spec.json
```

没有 optimizer/LR/nbs/warmup/augmentation/max_det 等显式值，recipe builder 直接拒绝。

---

# A. 部署后先跑 package gates

项目根目录：

```powershell
cd "C:\Users\xyh23\Documents\ChatGPT\多模态模型\multimodal_yolo26_qaf_v0_3"
```

### A1. pytest

```powershell
.venv\Scripts\python.exe -m pytest tests/test_t1gr_e2e5.py -q
```

### A2. py_compile

```powershell
.venv\Scripts\python.exe -m py_compile `
  src/multimodal/t1gr_e2e5.py `
  scripts/t1gr_probe_dataset.py `
  scripts/t1gr_build_contract.py `
  scripts/t1gr_propose_split.py `
  scripts/t1gr_freeze_split.py `
  scripts/t1gr_build_step1_recipe.py `
  scripts/t1gr_build_step1_rgb_view.py `
  scripts/t1gr_run_step1_baseline.py `
  scripts/t1gr_eval_step1_baseline.py `
  scripts/t1gr_audit_e2_e5.py `
  scripts/t1gr_static_audit.py `
  scripts/t1gr_synthetic_integration_gate.py `
  tests/test_t1gr_e2e5.py
```

### A3. static audit

```powershell
.venv\Scripts\python.exe scripts/t1gr_static_audit.py
```

必须：

```text
37 / 37 PASS
```

### A4. synthetic integration gate

```powershell
.venv\Scripts\python.exe scripts/t1gr_synthetic_integration_gate.py
```

必须 6/6：

```text
bad_depth_contract_fails                         true
formal_full_hash_unconditional                   true
rare_class_coverage_blocks_split                 true
forged_holdout_in_view_fails                     true
arbitrary_data_cli_rejected                      true
checkpoint_same_path_content_change_fails        true
```

这一步 PASS 后才允许摸正式数据。

---

# B. 正式数据：现在第一步只跑 probe

```powershell
.venv\Scripts\python.exe scripts/t1gr_probe_dataset.py `
  --dataset-root "E:/BaiduNetdiskDownload/初赛数据集-面向城市场景的多模态目标检测/训练集"
```

输出：

```text
reports/step4_t1gr/dataset_probe.json
```

**probe 后先停。**

根据真实目录/格式/metadata 填：

```text
config/t1gr_layout_spec.json
```

禁止猜：

```text
IR dtype/channels
Depth dtype/channels
scene/sequence/source grouping rule
正式样本数
class names
```

---

# C. E2 Formal Data Contract

复制 template：

```powershell
Copy-Item config/t1gr_layout_spec.template.json config/t1gr_layout_spec.json
```

根据 probe/官方数据事实填写完整。

然后：

```powershell
.venv\Scripts\python.exe scripts/t1gr_build_contract.py `
  --layout-spec config/t1gr_layout_spec.json `
  --private-out "E:/t1gr_private/formal_data_contract_private.json"
```

默认 public output：

```text
reports/step4_t1gr/data_contract_public.json
```

E2 hard gate 包含：

```text
paired count == expected count
all required files paired
strict label format
class config
RGB/IR/Depth readable
frozen dtype/ndim/channels
cross-modal H/W
full SHA256
```

Group rule 如果已填写，会在 contract 阶段实际执行；未填写不会伪装成已验证，但 E3 必定 HOLD。

---

# D. E3 Fresh Split Proposal

layout spec 中必须先冻结：

```text
group_rule
split fractions
split_seed
objective weights
class coverage minima
explicit class exemptions (if any, with reason)
```

运行：

```powershell
.venv\Scripts\python.exe scripts/t1gr_propose_split.py `
  --private-contract "E:/t1gr_private/formal_data_contract_private.json" `
  --out-private "E:/t1gr_private/split_proposal_private.json"
```

**proposal 不会写进 repo。**

必须人工审：

```text
sample counts
group counts
per-class image counts
per-class box counts
rare-class feasibility
cross-split RGB/IR/Depth/triplet duplicates
```

若 coverage 受 group 约束无法满足：

```text
FAIL
```

除非 policy 里明确写：

```json
{"class_id": 10, "reason": "..."}
```

不得自动豁免。

---

# E. E4 Split Freeze / FINAL HOLDOUT seal

proposal 人工认可后：

```powershell
.venv\Scripts\python.exe scripts/t1gr_freeze_split.py `
  --proposal-private "E:/t1gr_private/split_proposal_private.json" `
  --split-manifest-private "E:/t1gr_private/split_freeze_private.json" `
  --sealed-holdout-out "E:/t1gr_private/final_holdout_sealed.json"
```

repo public：

```text
reports/step4_t1gr/split_freeze_public.json
```

public 只包含：

```text
count
IDs SHA commitments
group commitments
class support
freeze timestamp/commit
```

没有 sample IDs。

注意：public freeze **不会**再写 `frozen_before_training=true`。
未来 runner 根据 UTC timestamp 实际证明：

```text
freeze time < training start time
```

---

# F. Freeze Step1 recipe

复制：

```powershell
Copy-Item config/step1_training_spec.template.json config/step1_training_spec.json
```

把所有 null 替换为明确值。

然后：

```powershell
.venv\Scripts\python.exe scripts/t1gr_build_step1_recipe.py `
  --public-contract reports/step4_t1gr/data_contract_public.json `
  --split-freeze-public reports/step4_t1gr/split_freeze_public.json `
  --training-spec config/step1_training_spec.json `
  --base-checkpoint "E:/odin/yolo26s.pt"
```

输出：

```text
reports/step4_t1gr/step1_baseline_recipe.json
```

recipe 会 pin：

```text
base checkpoint SHA
Ultralytics version
Torch/Python version
optimizer/LR/nbs/warmup
augmentation
end2end
validation conf/iou/max_det
```

---

# G. Build TRAIN+DEV-only Step1 RGB view

**Formal 强制 copy，不允许 symlink。**

```powershell
.venv\Scripts\python.exe scripts/t1gr_build_step1_rgb_view.py `
  --private-contract "E:/t1gr_private/formal_data_contract_private.json" `
  --split-manifest-private "E:/t1gr_private/split_freeze_private.json" `
  --recipe reports/step4_t1gr/step1_baseline_recipe.json `
  --out-root "E:/t1gr_private/step1_rgb_view"
```

生成：

```text
E:/t1gr_private/step1_rgb_view/dataset.yaml
E:/t1gr_private/step1_rgb_view/view_manifest.json
```

view manifest 记录每个 train/dev RGB+label 文件映射与 SHA。
FINAL HOLDOUT 不复制到 view。

---

# H. 到这里才允许 Step1 baseline training

```powershell
.venv\Scripts\python.exe scripts/t1gr_run_step1_baseline.py `
  --recipe reports/step4_t1gr/step1_baseline_recipe.json `
  --view-manifest "E:/t1gr_private/step1_rgb_view/view_manifest.json" `
  --device 0
```

**没有 `--data` 参数。**

runner 会在 GPU training 前先检查：

```text
view recipe/contract/split pins
actual train/dev IDs
all copied file SHA
base checkpoint runtime SHA
freeze timestamp < training start
Ultralytics version
physical head nc/end2end
effective trainer args
```

训练后再检查 `args.yaml` 中 frozen effective args。

---

# I. DEV-only Step1 evaluation

```powershell
.venv\Scripts\python.exe scripts/t1gr_eval_step1_baseline.py `
  --recipe reports/step4_t1gr/step1_baseline_recipe.json `
  --view-manifest "E:/t1gr_private/step1_rgb_view/view_manifest.json" `
  --run-dir runs/step4_t1gr_step1/STEP1_RGB_BASELINE `
  --device 0
```

Evaluator 同样没有 `--data`。
实际 `images/val` IDs 必须等于 frozen DEV commitment。

---

# J. E2–E5 formal close

```powershell
.venv\Scripts\python.exe scripts/t1gr_audit_e2_e5.py `
  --public-contract reports/step4_t1gr/data_contract_public.json `
  --split-freeze-public reports/step4_t1gr/split_freeze_public.json `
  --recipe reports/step4_t1gr/step1_baseline_recipe.json `
  --view-manifest "E:/t1gr_private/step1_rgb_view/view_manifest.json" `
  --run-manifest runs/step4_t1gr_step1/STEP1_RGB_BASELINE/t1gr_step1_manifest.json `
  --baseline-report reports/step4_t1gr/step1_baseline_report.json
```

只有：

```text
E2 true
E3 true
E4 true
E5 true
all_passed true
```

才允许下一步：

```text
T1-GR DESIGN_FREEZE
→ G0-N / G1-P / G2-S × >=3 seeds implementation
```

Depth / Production 在这个阶段始终 HOLD。
