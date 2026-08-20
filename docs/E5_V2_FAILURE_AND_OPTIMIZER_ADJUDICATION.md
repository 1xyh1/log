# E5 v2 失败证据与 AdamW/MuSGD 重新裁定

## 结论

- 保留原冻结的 MuSGD。
- 明确承认 Ultralytics 8.4.56 的 `optimizer=auto` 在当前 1504 个训练样本配置下会选 AdamW。
- v1 中“auto 等价展开为 MuSGD”和“estimated_training_iterations=30080”的表述撤销。
- v2 的科学含义是“显式冻结 MuSGD”，不是“跟随框架 auto”。
- v1 失败的直接异常仍无法从现有公开日志确定，因为 v1 把 traceback 统一压成了 `UNHANDLED_INTERNAL_ERROR`。
- 已确认的可复现性缺陷是：v1 在 `DetectionTrainer` 执行 seed 初始化之前就创建了新的 nc=12 head。这个缺陷能解释三次运行共享 epoch 指标全部不同，但不能在没有 traceback 的情况下被断言为中途失败的唯一原因。

## 输入证据复核

收到的 failure debug bundle 清单共 25 个成员，manifest SHA 校验 25/25 通过。三次运行的 `args.yaml` SHA256 一致：

```text
4376b38d...
```

但相同 seed/deterministic 配置下，共享 epoch 的指标并不一致：

| 对比 | 可比 epoch | 不一致 |
|---|---:|---:|
| run2 vs run3 | 4 | 4 |
| run3 vs interrupted run | 22 | 22 |

这不是“相同初始状态的确定性重复”。v1 的执行顺序是：

```text
build_model()
  -> 创建 nc=12 head（随机参数）
DetectionTrainer(...)
  -> init_seeds(...)
trainer.model = model
trainer.train()
```

因此 seed 来得太晚。v2 改为：

```text
init_seeds(seed + 1 + RANK)
  -> 创建 nc=12 head
  -> 迁移 shape-compatible checkpoint 参数
  -> 计算完整初始状态 SHA
DetectionTrainer(...)
  -> 再按框架流程初始化数据/训练随机状态
trainer.model = model
on_train_start
  -> 再次计算训练起点 SHA并与预检绑定
```

单进程 E5 强制 `RANK=-1`，所以有效初始化 seed 为：

```text
20260812 + 1 + (-1) = 20260812
```

## 三次失败能下什么结论

| 运行 | 已完成 CSV 行 | 日志最后阶段 | 可裁定 |
|---|---:|---|---|
| interrupted run | 25 | epoch 26 外部中断 | 用户/外部中断；未发 PASS |
| run2 | 4 | epoch 5 validation 之后 | trainer 内部未公开异常；直接原因未知 |
| run3 | 22 | 下一 epoch validation 之后 | trainer 内部未公开异常；直接原因未知 |

Ultralytics trainer 的 epoch 内顺序在 validation 后仍包含 NaN recovery、metrics CSV 写入和 checkpoint 保存。现有日志显示 validation 已结束但新 CSV 行未落盘，所以失败边界可能在这些步骤之一；没有 traceback 时不得进一步定死。

v2 在 `trainer.train()` 外围捕获 `BaseException`，把完整本地 traceback 写入：

```text
<private-run-dir>/E5_PRIVATE_FAILURE.json
```

该文件不会进入公开报告；公开 stderr 继续只输出脱敏错误码。

## optimizer 公式与裁定

Ultralytics v8.4.56 的 auto optimizer 使用有效 iteration 估算：

```text
iterations = ceil(len(dataset) / max(batch, nbs)) * epochs
```

代入本项目：

```text
ceil(1504 / max(4, 64)) * 80 = 1920
```

auto 分支阈值是 10000 iterations；当前 1920 会走 AdamW。相关上游版本源文件：

- https://github.com/ultralytics/ultralytics/blob/v8.4.56/ultralytics/engine/trainer.py

项目 v2 的最终冻结裁定：

| 项 | 裁定 |
|---|---|
| framework auto would select | AdamW |
| E5 v2 selected optimizer | MuSGD |
| 是否宣称 auto 等价 | 否 |
| 保留理由 | 保持原冻结母基线及后续 matched comparison 连续性 |
| 是否允许 CLI 改 optimizer | 否 |

## 赛制上限修正

v1 的 `eval_args.max_det=300` 与每图最多 100 个检测结果的赛制约束不一致。v2 冻结为：

```json
"max_det": 100
```

recipe validator 会拒绝任何其他值。

