# Step 3-A 恢复执行手册（基于最新 `1xyh1/log` 问题定位）

## 结论先行

当前不应修改 R3 recipe，也不应先进入 Step4。最新保留的 C1-I / C2-D 80 epoch `results.csv` 已证明训练链本身能学习；真正需要先修的是：

1. post-hoc evaluator 的 GT box conversion；
2. C0 formal 与 smoke 产物混合；
3. formal 目录可覆盖；
4. G8 只记 planned schedule，没有记实际 DataLoader yield。

## 1. 应用补丁

```bat
python apply_patch.py ^
  --project-root "C:\Users\xyh23\Documents\ChatGPT\多模态模型\multimodal_yolo26_qaf_v0_3"
```

脚本会先备份被替换文件到：

```text
_hotfix_backup_reference_YYYYMMDD_HHMMSS/
```

不会删除权重、数据和已有 runs。

## 2. 先跑测试

```bat
cd /d "C:\Users\xyh23\Documents\ChatGPT\多模态模型\multimodal_yolo26_qaf_v0_3"

pytest -q tests/test_step3_recovery_contract.py
pytest -q tests/test_reference_fusion_blocks.py
pytest -q tests/test_step3_contract.py
```

`test_reference_fusion_blocks.py` 只是 Step4 候选模块单测，不会修改 Step3 模型。

## 3. 先审现有三个 run，不训练

```bat
python scripts/validate_step3_run.py runs/step3_earlyfusion/C0-N --expected-epochs 80
python scripts/validate_step3_run.py runs/step3_earlyfusion/C1-I --expected-epochs 80
python scripts/validate_step3_run.py runs/step3_earlyfusion/C2-D --expected-epochs 80
```

预期：

- **C0-N：FAIL**，因为当前目录只有 1 epoch 的 args/results/weights，而旧 eval JSON 含历史 formal 派生产物。
- **C1-I / C2-D：训练完整性应 PASS**；旧 eval JSON 没 provenance 时可能出现 warning，这正是下一步重评的原因。

## 4. 用现有完整 C1/C2 权重重新评估

先不依赖 C0 做 LOO：

```bat
python scripts/eval_step3_causality.py ^
  --group C1-I ^
  --run-name C1-I ^
  --project runs/step3_earlyfusion ^
  --contract reports/step3_data_contract.json ^
  --device 0

python scripts/eval_step3_causality.py ^
  --group C2-D ^
  --run-name C2-D ^
  --project runs/step3_earlyfusion ^
  --contract reports/step3_data_contract.json ^
  --device 0
```

重点看打印的：

```text
N / Z / S
preds
max_conf
mean_best_iou
```

如果修复后 NORMAL 不再是 0，则 P-C 已定位为 evaluator bug。

## 5. 恢复 C0，绝不写回旧目录

```bat
python scripts/run_step3_earlyfusion.py ^
  --group C0-N ^
  --run-kind recovery ^
  --name C0-N-recovery-20260814 ^
  --epochs 80 ^
  --batch 4 ^
  --snapshot step3_6ch_rgb_equiv_init.pt ^
  --contract reports/step3_data_contract.json ^
  --project runs/step3_earlyfusion ^
  --device 0
```

如果目标目录已有文件，runner 会直接拒绝。

## 6. 评估恢复后的 C0

```bat
python scripts/eval_step3_causality.py ^
  --group C0-N ^
  --run-name C0-N-recovery-20260814 ^
  --project runs/step3_earlyfusion ^
  --contract reports/step3_data_contract.json ^
  --device 0
```

然后给 C1/C2 补 LOO：

```bat
python scripts/eval_step3_causality.py ^
  --group C1-I --run-name C1-I ^
  --c0-run-name C0-N-recovery-20260814 ^
  --project runs/step3_earlyfusion ^
  --contract reports/step3_data_contract.json --device 0

python scripts/eval_step3_causality.py ^
  --group C2-D --run-name C2-D ^
  --c0-run-name C0-N-recovery-20260814 ^
  --project runs/step3_earlyfusion ^
  --contract reports/step3_data_contract.json --device 0
```

## 7. 最后总结

```bat
python scripts/summarize_step3.py ^
  --project runs/step3_earlyfusion ^
  --c0-run C0-N-recovery-20260814 ^
  --c1-run C1-I ^
  --c2-run C2-D
```

只有这一版 summary 允许进入正式结论。

## 8. G8 说明

新 runner 每 epoch 会：

1. `dataset.set_epoch(epoch)`；
2. `InfiniteDataLoader.reset()`；
3. 从真实 batch 的 `sample_id / flip_applied` 采集顺序；
4. epoch end 对比 expected vs actual；
5. mismatch 直接中止。

旧 C1/C2 的 G8 只有 planned hash，所以 summary 会把证据等级标成 `legacy_planned_or_mixed`。这不等于它们实际错序；在 `workers=0` 下旧实现大概率按计划消费，但严格证据等级必须如实标记。如果比赛前需要形式上完全闭环，可在后续决定是否重跑 C1/C2；不要把这件事和当前 evaluator 修复混在一起。
