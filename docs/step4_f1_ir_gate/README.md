# Step 4-F1：RGB 主导的 IR 可靠性门控

本目录只保存 F1 的设计、论文映射和执行指导；真实训练结果与复盘记录放在
`feedback/`，模型运行产物仍写入 `runs/step4_f1_ir_gate/`。这样不会把“预先冻结的
实验协议”和“训练之后才知道的反馈”混在一起。

## 当前状态

| 项目 | 状态 |
|---|---|
| F0 数值结论 | 保留：IR 被模型使用，但未证明超过 matched RGB control |
| F0 provenance | 新版 LOO 内部门禁已通过；当前 summary 的 LOO 文件 self-pin 不匹配，待本机仅重跑 summary |
| F1 结构代码 | 已实现，尚未在含 YOLO26 权重和样例数据的本机/GPU 环境运行 |
| F1 formal | 禁止启动，直到 F1 audit 的 G0～G5 与 smoke 全 PASS |
| Depth | 本阶段禁止加入 |

## 文件索引

- `DESIGN_FREEZE.md`：结构、对照组、门禁和晋级条件。
- `PAPER_MAPPING.md`：InfraNet、EvaNet、MoETrack、BDDS 分别借什么、不借什么。
- `EXECUTION_GUIDE.md`：从 F0 summary self-pin 重闭合到 F1 汇总的命令顺序。
- `ROADMAP_AFTER_F1.md`：按证据触发的 F1-B、spatial、Depth 与 joint 路线。
- `feedback/README.md`：只记录实际执行反馈，不回写修改预注册判据。

## 新增实现

- `src/multimodal/reliability_gate.py`
- `src/multimodal/step4_f1_ir_gate_model.py`
- `src/multimodal/modality_quality.py`
- `src/multimodal/step4_f1_interventions.py`
- `src/multimodal/step4_f1_closeout.py`
- `scripts/audit_step4_f1_ir_gate.py`
- `scripts/audit_step4_f1_modality_quality.py`
- `scripts/run_step4_f1_ir_gate.py`
- `scripts/eval_step4_f1_causality.py`
- `scripts/eval_step4_f1_quality.py`
- `scripts/step4_f1_loo.py`
- `scripts/summarize_step4_f1.py`
- `tests/test_step4_f1_ir_gate.py`

这些文件复用既有 `TriModalDataset`、预处理、stock validator 语义、shuffle、G8 和
RGB freeze 规则，没有新写 bbox 转换、主数据 loader 或检测指标。
