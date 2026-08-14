# 当前问题定位（基于 `1xyh1/log@82655b7e...`）

## P0-1：post-hoc evaluator 的 GT 几何转换错误

当前旧版 `eval_step3_causality.py` 手工把 normalized xywh 转 xyxy。它先原地修改 x/y，再用修改后的值更新 w/h 列，导致右下角计算错误、GT 框缩小，IoU 被系统性压低。

症状非常典型：

- C1-I / C2-D 的训练时 stock validator 在后期分别约 0.25 / 0.21；
- post-hoc evaluator 却把 NORMAL 也评成 0 或接近 0。

这不是“模型没学到”的可信证据。

修复：新版 evaluator 不再复制 box/NMS/matching 逻辑，直接使用 Ultralytics `DetectionValidator` 的 `postprocess/_prepare_batch/_prepare_pred/_process_batch`。

## P0-2：C0-N formal 控制组被 1-epoch smoke 覆盖

当前：

```text
runs/step3_earlyfusion/C0-N/args.yaml -> epochs: 1
runs/step3_earlyfusion/C0-N/results.csv -> 1 row
```

但同目录旧 `eval_step3_causality.json` 的 `late10` 仍是：

```text
mean   0.2024
median 0.2035
min    0.1966
max    0.2049
```

这在同一个 1-epoch run 中不可能成立，说明目录混合了不同运行时期的 artifact。当前 C0 不能用于任何 candidate-vs-control 结论。

修复：

- formal/recovery run directory immutable；
- `validate_step3_run.py` 在评估前检查 epoch 数、results、G8、kernel growth、weights；
- 新 evaluator 写入所有关键文件 SHA256；文件一变，旧 eval 自动变 stale。

## P0-3：旧 G8 只证明 planned schedule

旧 trace 对 `sampler.perm` 做 hash，没有从真实 batch 记录 sample IDs。

它不是当前 mAP=0 的根因，但证据等级不足。新版 runner 从 `preprocess_batch()` 实际采集：

```text
sample_id
flip_applied
```

并在 epoch end 与预期顺序逐项比较。

## 不是当前主问题：R3 recipe 本身

C1-I preserved curve：

```text
epoch1  mAP50-95 ~0.0386
epoch80 mAP50-95 ~0.2539
train loss ~1.79 -> ~0.34
```

C2-D：

```text
epoch1  ~0.0386
epoch80 ~0.2106
train loss 持续下降
```

因此目前不能把问题归因成“关了 mosaic/几何增强导致三组都不收敛”。在 C0 恢复和 evaluator 修复之前，修改 R3 会把一个软件/实验管理问题变成新的实验变量。

## 当前允许的动作

```text
修 evaluator
→ 审 C1/C2
→ 新目录恢复 C0
→ 三组重新做 N/Z/S
→ Step3 总结
```

## 当前禁止的动作

```text
因为旧 post-hoc 全0就改 augmentation
因为 C0 目录坏了就推翻 6ch early fusion
直接把 RDTTrack prompt/orthogonal 加进 Step3
```
