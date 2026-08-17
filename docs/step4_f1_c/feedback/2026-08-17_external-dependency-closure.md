# F1-C 执行反馈：2026-08-17 / 外部运行依赖闭包（P0 HOLD 整改）

审阅者 HOLD 裁决（只补"运行依赖闭包"，不动模型/gate/判据）已按清单闭环：
readiness v2 / manifest v2 / audit v3 / smoke r4 三组 / 自证通过。

## 补丁内容

1. **锁 base checkpoint**：`EXPECTED_BASE_CHECKPOINT_SHA256` 常量
   （`646f8bc3…a1b`，与 reports/checkpoint_audit.md 及审阅者上传文件一致，
   本机实测 mtime 2026-08-06 未变）；smoke manifest 记录
   `base_checkpoint_sha256`；runner 在 `build_reference_3ch()` **之前**
   `verify_base_checkpoint`（smoke+formal），不符
   `ABORT_BASE_CHECKPOINT_MISSING/STALE`。
2. **锁 builder**：`src/multimodal/early_fusion_yolo26.py` 进三处 pin 表
   （audit provenance 10 项、readiness AUDIT_TARGETS / MANIFEST_PIN_TARGETS、
   runner audit_targets），manifest 记 `builder_source_sha256`。builder 文件
   本身零改动。
3. **formal 构模后 initial-state equality**：readiness 新增
   `evidence.initial_state_frozen`（5 个 initial SHA 冻结）；formal 构模后、
   Trainer 创建前重算 5 分量与冻结值逐位比对，不符
   `ABORT_INITIAL_STATE_MISMATCH`（至少 full model 逐位，实际全 5 个）。
4. **数据 freshness**：`verify_raw_data_freshness` 按 `contract["file_hashes"]`
   重 hash 磁盘 17×4（visible/infrared/depth/label）文件，任一 mismatch
   `ABORT_RAW_DATA_STALE`（smoke+formal）；readiness evidence 块
   `data_freshness`。dataset.yaml：语义锁（nc=12 + names==CLASS_NAMES）+
   SHA（`ABORT_DATA_YAML_MISSING/SEMANTICS/STALE`），manifest 记
   `data_yaml_sha256/names_sha256/n_classes`。
5. **对抗测试**：`tests/test_step4_f1_c_external_closure.py` 17 项
   （篡改权重→STALE、篡改原始文件→RAW_DATA_STALE、删 label→MISSING、
   yaml 语义漂移、pin 表与接线回归等），全走 tmp_path 不碰真实文件。

## 结果

- **pytest**：全量绿（2 skipped 为既有跳过项）
- **audit v3**：`reports/step4_f1_c/pretrain_audit.json` all_passed=true，
  8 sections（新增 external_dependency_closure，12 项 checks 全过），
  provenance 10 项含 builder_source_sha256
- **smoke r4 三组**（G6 全 PASS）：`runs/step4_f1_c/smoke-F1C-C0-e1-r4` /
  `smoke-F1C-I-fixed-e1-r4` / `smoke-F1C-I-magsoft-e1-r4`；
  manifest 均 schema v2，`base_checkpoint_sha256`==646f8bc3…、
  `data_yaml_n_classes`==12、三组 initial SHA 全等
  （与 r3 时代值逐位一致：rgb `aeeb732a…` / model `3ebea363…`，证明
  r3→r4 构造可复现）；g8 trace 三组字节一致 `b06cdb…`
- **readiness v2**：`reports/step4_f1_c/smoke_readiness.json` all_passed=true，
  4 个新 evidence 块齐全（base_checkpoint：smoke_recorded==current==常量；
  initial_state_frozen 5 hex；data_freshness 17×4=68 全过；data_yaml
  n_classes=12 names 匹配）
- **自证（formal 路径）**：以 F1C-I-magsoft 调 verify_readiness_report →
  `passed=True, errors=[]`

## 旧产物

r3 三组 smoke 目录与 v1 readiness 报告保留（未覆盖；readiness v2 由
--overwrite 重新生成，v1 内容见 r3 manifest 与旧报告历史）。

## 待审阅者复核

audit v2→v3 freshness、r4 三组 manifest identity、G6/G8/G9/G10.7、
readiness v2 四个新 evidence 块、producer/consumer key 对齐（data-yaml/
base-checkpoint 参数）、自证结果、对抗测试。复核后 FORMAL GO
（四组 80ep：C0/fixed/magsoft/soft matched control）。
