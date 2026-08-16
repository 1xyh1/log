# F1 之后的逐级路线

本路线只定义触发条件，不预先承诺某个复杂模块。每级仍使用单模型、单 head、一次统一
推理，并保留 matched control、last.pt、NORMAL/ZERO/SHUFFLE、G5/G6/G8 与 LOO。

| 阶段 | 进入条件 | 唯一新增变量 | 对照与退出条件 | 代码状态 |
|---|---|---|---|---|
| F1-confirm | F1 六项晋级条件全过 | 仅更换一个预注册 seed | 重复方向失败即不晋级 | 等 F1 summary 后传 seed |
| F1-B | soft scalar 有益但可靠性退化门禁失败，或 IR 使用但仍不超 C0 | 训练期 deterministic IR corruption/dropout；结构不变 | 与未退化 F1-I-soft 同配方对照；clean 不退且 degraded 改善 | corruption primitive 已有，训练 schedule 暂不启用 |
| F1-S | clean/全局 gate 成功，但 shift/局部遮挡诊断明确失败 | 每尺度 `1xHxW` IR residual spatial gate | 同 seed scalar gate control；否则回退 scalar | 未实现，等待证据 |
| D1 | IR 路线经确认 seed 稳定 | Depth 单独套同一 scalar residual gate | C0 / D-fixed / D-soft；不得同时加入 IR | 未实现 |
| ID1 | IR 与 Depth 各自独立通过 | 单模型内联合 residual，先无 decorrelation | IR-only、Depth-only、joint matched ablation | 未实现 |
| BDDS/QAF | 小目标分层与固定 shift 证明形变/配准是主要误差，且取得完整公式或官方实现 | 一次只加双向融合或形变敏感 loss 之一 | 必须有不含新模块的 matched control | 仅研究候选 |

## F1-B 预留约束

如果 F1 summary 触发 F1-B，先冻结 corruption 采样契约：kind、severity、概率、按
`seed/epoch/sample_id` 生成的实际轨迹 SHA，以及 clean/corrupted 两条评估轴。必须复用
`TriModalDataset`，在其冻结输出后只改 IR channel；不能新写 bbox、validator 或主 loader。

训练 schedule 没有提前写进 formal runner，是为了避免在 F1 结果未知时把“结构门控”和
“退化增强”两个变量同时引入。当前 `step4_f1_interventions.py` 只允许 evaluation 使用。

## 论文方向的边界

- InfraNet/QualGate：当前 F1 已实现最小 task scalar；后续只在退化证据支持时扩展输入。
- EvaNet：始终保持独立诊断，不直接把 fused-image score 接入 detector。
- MoETrack：只保留“何时融合”的问题，不引入多 expert 选择。
- BDDS：等小目标/错位诊断与完整实现依据，不猜 loss 公式。
