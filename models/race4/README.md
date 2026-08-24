# race4 比赛模型（4ch P5，2026-08-24 推送）

比赛跑分管线（协议外）训练的比赛提交候选模型。

## 文件

| 文件 | 说明 | DEV mAP50-95（ZERO-IR 口径） |
|---|---|---|
| `best.pt` | best checkpoint（ep33） | **0.2951** |
| `last.pt` | last checkpoint（133ep 早停） | 0.2692 |
| `race_summary.json` | 训练配置 + 双口径结果（含 native RGB+IR 0.2789） | — |

## 模型结构

- 4ch 输入：RGB 3ch + IR 灰度 1ch（正确配对训练，G1-P 定义）
- T1GRP5Model（P5 residual 拓扑，zero-init，end2end head，E2ELoss）
- base checkpoint：`E:/odin/yolo26s.pt`（官方预训练，backbone/neck 转移 + nc12 头随机初始化）
- 训练：MuSGD / lr0 0.01 / nbs 64 / batch 4 / 640 / 150ep（133ep 早停）/ seed 20260812

## 推理

```bash
# 纯 RGB 推理（IR 通道全 0，协议链主口径）
python scripts/race_predict_4ch.py --ckpt best.pt --test-root raw/test --ir-mode zero
# RGB+IR 推理
python scripts/race_predict_4ch.py --ckpt best.pt --test-root raw/test --ir-mode native
```

## 状态与问题

- 与协议链同结构同配方（T1GRP5Model + MuSGD）下，race 管线 mAP50-95 锁死 0.2951，
  低于协议链 G 实验（1504/198 split）的 0.3450~0.3643，差约 0.05——split 差异为主嫌疑，
  交叉验证待跑（详见 00_说明.md 第五节）。
- 无目标图须提交空 TXT；每图 ≤100 框；官方格式 `[class_id, cx, cy, w, h, conf]`。
