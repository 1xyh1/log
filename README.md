# 1xyh1/log — Step 3-A 源码镜像与结果仓库

本仓库是 **YOLO26 多模态竞赛项目（Step 3-A 阶段）** 的审阅中转仓库，由本地 watcher 自动同步。**私有仓库，请勿外传**（含本地路径、实验日志与比赛实现细节）。

## 内容

| 路径 | 说明 |
|---|---|
| `src/multimodal/` | Step 3-A 核心源码镜像（数据契约 / 预处理 / 6ch 数据集 / 模型与门禁） |
| `scripts/` | 项目脚本镜像（Step 1/2 冻结脚本 + Step 3 audit/run/eval/summarize） |
| `tests/` | 测试镜像（含 Step 3 契约回归测试 test_step3_contract.py） |
| `docs/STEP3_IMPLEMENTATION_LOG.md` | 实现日志（每板块的完整改动代码 + 门禁运行结果） |
| `reports/step3_*` | 门禁报告与数据契约（G1-G7 gate JSON、data contract） |
| `runs/step3_earlyfusion/` | 三组训练结果（**仅文本与必要图**：eval/因果 JSON、G8 轨迹、kernel growth、results.csv、val 曲线 results.png；权重 .pt 不上传） |
| `SOURCE_MIRROR_MANIFEST.json` | 逐文件 SHA256 清单（自动生成） |
| `watch_step3_source_mirror.py` | 本地同步 watcher（2s 轮询 + 20s 静默 + 哈希比较 + 白名单 commit/push） |

## 不同步

权重（*.pt）、原始三模态数据、runs 下的二进制产物、venv、.env、密钥。

## 实验背景（一句话版）

赛题：面向城市场景的视觉多模态目标检测（RGB/IR/Depth 三模态，12 类，mAP@50-95 逐类平均）。
当前阶段：Step 3-A 输入级互补性与因果探针（6ch early fusion：C0-N `[RGB,0,0,0]` / C1-I `[RGB,I,0,0]` / C2-D `[RGB,0,D,M]`，同一 6ch O2M 初始化快照、R3 配方、配对训练调度；NORMAL/ZERO/SHUFFLE 三路因果评估，last.pt 为主口径）。
前置阶段：Step 1（RGB 母基线 B0-D = O2M + R2-ckpt-public，PASS）、Step 2（IR/Depth 单模态能力探针，PASS）。
