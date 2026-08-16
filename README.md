# 1xyh1/log — YOLO26 多模态实验源码与证据镜像

本仓库是 RGB/IR/Depth 城市场景目标检测项目的审阅中转镜像，由本地 watcher 同步。
GitHub 仓库当前为 public，因此禁止提交 checkpoint、原始数据、密钥或其他敏感材料。

## 内容

| 路径 | 说明 |
|---|---|
| `src/multimodal/` | Step 3/F0/F1 核心源码（冻结数据契约、模型与门禁） |
| `scripts/` | Step 1～F1 的 audit/run/eval/LOO/summarize 脚本 |
| `tests/` | 测试镜像（含 Step 3 契约回归测试 test_step3_contract.py） |
| `docs/STEP3_IMPLEMENTATION_LOG.md` | Step 3 实现日志与门禁结果 |
| `docs/STEP4_IMPLEMENTATION_PLAN.md` | 已完成 F0 的冻结实现协议 |
| `docs/step4_f1_ir_gate/` | F1 IR 可靠性门控的独立设计、论文映射、指导与反馈区 |
| `reports/step3_*` | 门禁报告与数据契约（G1-G7 gate JSON、data contract） |
| `runs/step3_earlyfusion/` | 三组训练结果（**仅文本与必要图**：eval/因果 JSON、G8 轨迹、kernel growth、results.csv、val 曲线 results.png；权重 .pt 不上传） |
| `SOURCE_MIRROR_MANIFEST.json` | 逐文件 SHA256 清单（自动生成） |
| `watch_step3_source_mirror.py` | 本地同步 watcher（2s 轮询 + 20s 静默 + 哈希比较 + 白名单 commit/push） |

## 不同步

权重（*.pt）、原始三模态数据、runs 下的二进制产物、venv、.env、密钥。

## 当前主线

赛题使用一个统一 YOLO26 模型处理 RGB/IR/Depth，12 类，以 mAP@50-95 为主指标，
不使用 ensemble/voting。Step 3 输入级 early fusion 已冻结；Step 4-F0 已证明模型真实使用
IR，但尚未证明超过 matched RGB baseline，Depth 暂无独立 val 泛化收益。下一阶段只做
RGB 主导的 IR residual 可靠性门控，Depth 暂不加入。正式结论始终以 last.pt、matched
control、NORMAL/ZERO/SHUFFLE、G5/G6/G8 和 LOO 为准。
