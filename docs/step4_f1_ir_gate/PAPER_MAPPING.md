# 论文与项目映射

## InfraNet / QualGate

论文：<https://arxiv.org/abs/2607.03795>

可借：两层 bottleneck MLP 从辅助特征 GAP 预测一个任务导向标量，并用受控退化检查
gate 是否随输入可靠性变化。

不能照抄：InfraNet 是 IR 主流、RGB 辅助，并同时执行 RGB suppression 与 IR
amplification，还有辅助检测损失。本项目恰好相反：YOLO26 是可见光预训练宿主，
所以只把 q 作用在 IR residual，绝不放大或门控 RGB，也暂不增加辅助检测头。

## EvaNet

论文：<https://arxiv.org/abs/2604.02896>
代码：<https://github.com/AWCXV/EvaNet>

可借：把模态质量/信息保留证据与融合网络的最终任务效果分开报告，并用 downstream
task 检验指标一致性。

不能照抄：EvaNet 评价的是生成后的红外-可见光融合图像，不是检测 feature gate。
F1 因此只实现独立输入统计和质量报告，不把 EvaNet 分数直接喂进模型。任何质量指标
必须同时证明能预测检测保留，才有资格在 F1-B 作为 prior 候选。

## MoETrack

论文：<https://arxiv.org/abs/2405.00168>
代码：<https://github.com/Zhangyong-Tang/MoETrack>

可借：把问题明确成“何时融合”，并专门测试模态失效场景；融合并非每张图都必然有益。

不能照抄：它由 RGB/TIR/RGBT 多个 expert 独立预测并按置信度选择结果，属于决策级专家
选择，接近比赛禁止的 ensemble/voting。F1 只在单一模型内部连续缩放 IR feature
residual，最终仍是一个 head、一个权重、一次统一推理。

## BDDS：双向融合与形变敏感损失

论文：<https://doi.org/10.1016/j.inffus.2025.103985>，Information Fusion 128，
文章号 103985。题名为 *Learning Bi-directional fusion and deformation-sensitive loss
for RGB-T tiny object detection*。

可借方向：小目标的多线索双向融合，以及对跨模态形变/错位敏感的训练目标。

当前不实现：公开摘要只能确认“bi-directional fusion strategy”和
“deformation-sensitive similarity metric loss”，没有足够公式与官方代码可安全复刻；
而当前 train11/val6 也不足以同时归因 gate、双向融合和新 loss。必须先完成 F1，随后用
固定 IR 小位移诊断证明错位确实是主要误差，拿到完整公式/代码后再预注册单独实验。

## 结论

F1 是 QualGate 的最小 RGB-primary 适配；EvaNet 进入独立诊断；MoETrack只提供问题定义；
BDDS 留在有证据触发的后续分支。没有从这些仓库复制实现代码。
