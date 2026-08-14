# RGB–Infrared–Depth YOLO26 QAF v0.3

面向“城市场景视觉多模态目标检测”赛题的运行门禁版工程。

v0.2 是用于验证数据接口、三模态张量连接和融合构想的 PoC；v0.3 修复了会影响复现、公平归因和推理一致性的关键问题，但仍不是已经完成正式训练或取得正式 mAP 的最终方案。

## 当前定位

- `yolo26n.pt`：本地连通性、官方运行时和 4–16 样本过拟合门禁。
- `yolo26s.pt`：同一训练循环下的 RGB-only 与 C 系列融合消融。
- `yolo26m.pt`：待 s 结构和训练配方有完整验证证据后再迁移。
- 正式模型保持完整 RGB/IR/Depth 输入、一个检测头和一次统一输出，不做简单多模型集成。

## v0.3 关键修复

- 依赖下限改为 `torch>=2.3`，与 `torch.amp.GradScaler` 接口一致；Ultralytics 固定为 `8.4.56`。
- Python、NumPy、PyTorch、CUDA 和 DataLoader 使用显式 seed；配置记录 `seed` 与 `deterministic`，checkpoint 和运行摘要保存实际值。
- 配置统一使用 `./splits`，消除包内目录与配置路径不一致。
- Concat 与 QAF 都采用零初始化残差增益，初始化输出严格等于 RGB 特征，不再受 BN/SiLU 或随机辅助特征影响。
- 官方运行时门禁先执行两次优化，使残差路径打开后再检查 RGB、IR、Depth、Fusion 梯度。
- 新增 `configs/b0_s_rgb_same_loop.yaml`：RGB-only 与 C 系列使用同一自定义训练循环和训练配方，作为融合收益的公平基线。
- 推理从 checkpoint 恢复 `imgsz`、IR 表示、深度范围和深度缩放方式；`--imgsz` 仅作为显式可选覆盖，关键预处理不允许静默漂移。
- RGB/IR/Depth 质量统计排除 letterbox padding，并在训练和推理时显式传给 QAF；兼容 Ultralytics 对单通道深度返回 `H×W×1` 的 OpenCV 包装。
- 新增 4–16 样本 train=val 过拟合门禁配置与划分脚本；它们已经可执行，但尚未实际训练通过。

## 模型与输入

六通道输入固定为：

```text
0:3  RGB，范围 0–1
3:4  红外灰度，范围 0–1
4:5  对数深度，范围 0–1
5:6  有效深度 Mask，0/1
```

```text
RGB(3) ── YOLO26预训练Backbone ── R-P3/R-P4/R-P5 ───────┐
IR(1)  ── 轻量独立Encoder ─────── I-P3/I-P4/I-P5 ───────┼─ 融合 ─ 原YOLO26 Neck ─ 12类统一Head
D+M(2) ─ 轻量独立Encoder ─────── D-P3/D-P4/D-P5 ───────┘
```

当前支持：

- `concat`：通道拼接残差基线；
- `qaf`：质量感知通道门控，Depth 无效区域由 Mask 硬屏蔽；
- `rgb_only`：不构造辅助编码器，保留相同训练循环用于公平比较。

## 已有证据与边界

本地自动测试当前为 `29 passed`（设置完整 18 组样例目录后）；未设置 `MMOD_SAMPLE_ROOT` 时为 `27 passed, 2 skipped`。已覆盖数据处理、深度门禁、padding-free 质量统计、增强、融合严格 RGB identity、两步后辅助分支梯度、checkpoint 格式校验与严格保存/恢复、推理预处理恢复、逆 letterbox、101 点 mAP 和提交 ZIP 格式等。

v0.2 的 n/s 真实权重前后向结果仍可在 `reports/local_smoke_*.json` 查看，但使用的是安全最小兼容执行器，只证明权重结构和张量连接可行，不等同于官方 Ultralytics 检测损失已经通过。

已在 Windows CPU、PyTorch `2.8.0+cpu`、Ultralytics `8.4.56`、真实 `yolo26n.pt`/`yolo26s.pt` 和 2 组赛事样例上通过六条官方运行时门禁：两种尺度各自覆盖 `rgb_only`、`concat`、`qaf`。全部完成了 12 类头重建、真实 loss、两次优化、第三次反向及 `[2,84,6]` 解码；四条多模态路径的 RGB/IR/Depth/Fusion 梯度均非零。结构化证据见 `reports/runtime_gate_{n,s}_*_cpu.json`。这仍是最小连通门禁，不是精度实验。

另用 n/QAF、2 组样例、64 像素、1 epoch 跑通了自定义训练循环、验证、`best.pt`/`last.pt` 保存、严格恢复、推理、反 letterbox、TXT 与提交 ZIP 校验。该临时 smoke run 只证明闭环能执行，mAP 为 0，不是过拟合或精度证据；大 checkpoint 未打包。

以下结果尚未获得，不应在此阶段宣称模型有效或方向已被比赛指标验证：

- CUDA 环境和正式分辨率下的对应运行时门禁；
- 4–16 个真实样本的 train=val 过拟合通过；
- 完整 2,000 组训练数据的质量审计和场景/序列分组划分；
- 同-loop B0 与 C 系列的正式 mAP@50:95；
- P3/P2、空间门控或更复杂跨模态模块的收益。

## 安装

推荐 Python 3.10–3.12 和 CUDA 版 PyTorch：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Windows PowerShell 激活命令为 `.venv\Scripts\Activate.ps1`。

## 建议执行顺序

### 1. 完整数据审计与分组划分

数据目录保持赛事原始命名：

```text
data/train/
├── visible/
├── infrared/
├── depth/
└── labels/
```

```bash
python scripts/audit_dataset.py \
  --data data/train \
  --json reports/full_audit.json \
  --md reports/full_audit.md

python scripts/make_group_split.py \
  --data data/train \
  --out splits \
  --val-ratio 0.2 \
  --group-parts 1
```

`make_group_split.py` 只是文件名前缀分组工具。若正式文件名含明确序列或场景编号，应按官方语义调整或人工核验，避免相邻帧泄漏到验证集。

### 2. 官方运行时门禁

```bash
python scripts/verify_cloud_runtime.py \
  --weights weights/yolo26n.pt \
  --data data/train \
  --imgsz 320 \
  --device cuda
```

门禁目标是在官方 Ultralytics 中完成 12 类检测头重建、六通道前向、真实检测 loss、两次优化、第三次反向的四组非零梯度检查，以及解码形状检查。不通过时不要启动长训练。可用 `--report reports/runtime_gate.json` 保存成功结果；长训前仍应在目标 CUDA 环境和目标权重上重跑。

可用 `--rgb-only` 对同-loop RGB-only 路径执行相同运行时检查。

### 3. 4–16 样本过拟合门禁

```bash
python scripts/make_overfit_split.py \
  --source splits/train.txt \
  --out splits/overfit.txt \
  --count 8 \
  --seed 0

python scripts/train.py --config configs/qaf_n_overfit.yaml
```

该配置令 train 与 val 指向同一小划分。当前只是门禁入口，尚未执行并通过；需检查 loss 是否持续下降、训练集预测是否接近记忆。checkpoint 格式已有严格 roundtrip 单测，且 1-epoch smoke 已实际完成保存、恢复和提交生成。

### 4. 同-loop 公平消融

先跑 RGB-only，再逐步增加融合：

```text
B0  RGB-only，同一训练循环       configs/b0_s_rgb_same_loop.yaml
C1  P5 Concat                   configs/c1_s_p5_concat.yaml
C2  P4+P5 Concat                configs/c2_s_p45_concat.yaml
C4  P4+P5 QAF                   configs/c4_s_p45_qaf.yaml
C7  QAF + 物理降质/缺失         configs/c7_s_qaf_robust.yaml
```

运行示例：

```bash
python scripts/train.py --config configs/b0_s_rgb_same_loop.yaml
python scripts/train.py --config configs/c1_s_p5_concat.yaml
```

只有数据划分、seed、训练循环、优化器、训练时长和增强策略对齐后，B0 与 C 系列的差异才可用于归因融合结构。

### 5. 官方单模态能力审计

`prepare_single_modality_baseline.py` 和 `train_official_baseline.py` 可生成 RGB/IR/Depth 的 B0/B1/B2 审计。它们使用官方 Ultralytics Trainer，只回答“各模态单独是否有检测信息”，训练循环与 C 系列不同，不能用来公平计算多模态融合增益。

## 推理与提交

```bash
python scripts/predict_submit.py \
  --checkpoint runs/c7_s_qaf_robust/best.pt \
  --base-weights weights/yolo26s.pt \
  --data data/test \
  --out predictions/round1 \
  --batch 8 --device cuda
```

默认从 checkpoint 恢复训练时预处理。只有确需改变推理尺寸时才显式增加 `--imgsz 960`；IR 模式、Depth 有效范围和缩放策略不提供静默覆盖。

脚本会生成同名 TXT、保留空检测文件、限制每图最多 100 框、检查坐标/类别/置信度/NaN/Inf，并生成根目录直接包含 TXT 的 ZIP。

## 后续按证据启用

- P3 融合；
- 空间质量门控；
- 受限特征对齐或 P5 轻量跨模态 Transformer；
- P2 小目标层或小目标定位损失；
- m 体量迁移。

这些功能应由完整数据审计和前序消融触发，不在门禁阶段同时堆叠。

## 许可提醒

本项目通过运行时依赖使用 Ultralytics。云端训练和最终提交前应确认其软件许可与比赛提交方式相容，并保留依赖版本及源码修改说明。
