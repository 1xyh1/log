# F1-C 执行指导

## 0. 前置

- B1 summary 必须为 v2.2 冻结状态；F1-C audit 全过才能启动训练。
- **外部运行依赖闭包（reviewer 2026-08-17 P0，readiness v2）**：formal 构模
  的外部依赖全部进入 freshness 闭包——
  - base checkpoint：`E:/odin/yolo26s.pt` 的 SHA 必须等于
    `EXPECTED_BASE_CHECKPOINT_SHA256`（`646f8bc3…a1b`，与
    `reports/checkpoint_audit.md` 及审阅者上传文件一致）；跑 smoke 前先
    `sha256sum E:/odin/yolo26s.pt` 复核，不等即停；
  - builder：`src/multimodal/early_fusion_yolo26.py` 已进 audit/readiness/
    manifest 三处 pin 表；
  - 原始数据：17×4（RGB/IR/Depth/label）按 `contract["file_hashes"]` 重 hash，
    任一 mismatch 即 `ABORT_RAW_DATA_STALE`（smoke 与 formal 都检查）；
  - dataset.yaml：语义锁 `nc=12 + names == CLASS_NAMES` + 文件 SHA；
  - formal 构模后、Trainer 创建前，5 个 initial state SHA 与 readiness
    `initial_state_frozen` 逐位比对，不符 `ABORT_INITIAL_STATE_MISMATCH`。

## 1. 训练前审计与测试

```powershell
python -m pytest tests/ -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/audit_step4_f1_c.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

## 2. 三组 smoke（等审阅者放行后执行）

```powershell
python scripts/run_step4_f1_c.py --group F1C-C0 --run-kind smoke --epochs 1 --name smoke-F1C-C0-e1-r4
python scripts/run_step4_f1_c.py --group F1C-I-fixed --run-kind smoke --epochs 1 --name smoke-F1C-I-fixed-e1-r4
python scripts/run_step4_f1_c.py --group F1C-I-magsoft --run-kind smoke --epochs 1 --name smoke-F1C-I-magsoft-e1-r4
```

检查 G5/G6/G8/G9 + `step4_fp32_rgb_sha.json`（G10.7）落盘，并核每组
`manifest.json` 四项：schema v2、`base_checkpoint_sha256`==646f8bc3…、
`builder_source_sha256`、`data_yaml_sha256`。随后必须从
**原始 smoke 产物重新判定** readiness（不信已有 `passed=true`）：

```powershell
python scripts/audit_step4_f1_c_smoke_readiness.py `
  --c0-smoke smoke-F1C-C0-e1-r4 `
  --fixed-smoke smoke-F1C-I-fixed-e1-r4 `
  --magsoft-smoke smoke-F1C-I-magsoft-e1-r4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

生成 `reports/step4_f1_c/smoke_readiness.json`（schema v2，含
`base_checkpoint` / `initial_state_frozen` / `data_freshness` / `data_yaml`
四个新 evidence 块）。任一源码/audit/design/contract/权重/原始数据/smoke
原始证据随后变化都会使 formal runner 拒跑。按 DESIGN_FREEZE，
`F1C-I-soft` 是 formal matched control，**不额外要求 smoke**。

## 3. Formal（四组，审阅者 GO 后）

formal runner 硬锁 `epochs=80 / batch=4 / seed=20260812 / amp=False`，并在
构模前重新消费 `smoke_readiness.json`；readiness 缺失或 stale 时直接 ABORT；
随后重新核对 base checkpoint SHA（`ABORT_BASE_CHECKPOINT_STALE`）、
dataset.yaml SHA（`ABORT_DATA_YAML_STALE`）与构模后 initial-state 逐位比对
（`ABORT_INITIAL_STATE_MISMATCH`）。

```powershell
python scripts/run_step4_f1_c.py --group F1C-C0 --run-kind formal
python scripts/run_step4_f1_c.py --group F1C-I-fixed --run-kind formal
python scripts/run_step4_f1_c.py --group F1C-I-magsoft --run-kind formal
python scripts/run_step4_f1_c.py --group F1C-I-soft --run-kind formal
```

## 4. 反馈纪律

执行反馈写入 `feedback/`；不回写 DESIGN_FREEZE 预注册阈值；失败目录保留。
