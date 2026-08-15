# Step 4-F0 实现计划（冻结版）

## 定位与继承

- Step 3-A 定稿：MODEL-USES-AUX-BUT-NO-BENEFIT（6ch shared-stem early fusion 未证明正确配对的互补增益；阴性 ≠ 模态无用）。不补 seed、不改 R3。
- Step 4-F0 回答：**"RGB pretrained anchor + 模态专属轻量编码 → P3/P4/P5 feature-level 融合"能否形成"正确配对 aux 的有益因果增益"**。

## 结构（F0 = RGB Anchor + Zero-init Residual Feature Injection）

```
RGB ─→ YOLO26 backbone(冻结) ─→ P3(256)/P4(256)/P5(512)
        ↑ +P3(A3)     ↑ +P4(A4)     ↑ +P5(A5)
Aux ─→ 共享 2ch 轻量 encoder ─→ A3/A4/A5
P_i = Conv1x1(bias=True)，weight=0 bias=0（zero-init）
F_i = R_i + P_i(A_i)   → 原 YOLO26 neck/head（可训练）
```

- **不用 IdentityConcat**：concat 改变 neck 输入维度与 BN 统计，破坏"严格 RGB anchor"。
- **zero-init 优于 α scalar**：W 自身的梯度 A·dL/dF > 0，一步后即解锁编码器（实测：step1 编码器梯度精确为 0 系 W=0 的数学必然——dL/dA=Wᵀ·dL/dF；step2 > 0。审计门禁按此事实定义）。
- aux 输入为统一 2ch：F0-C0=[0,0]、F0-I=[I,0]、F0-D=[D,M] —— 三组共享同一 encoder，参数完全一致（matched control）。
- 复用 Step-3 修复版 6ch 数据契约：模型内部 `_split_input` 按 aux_mode 拆分，训练器/评估器零改动复用（stock validator 语义、no-/255、G8、formal 目录门禁全部继承）。

## 冻结配置

- 训练：RGB backbone 冻结（BN eval 强制，`model.train()` 覆盖保持不变量）；可训练 = aux encoder + fusion projections + neck/head；R3 配方不变；seed 20260812；aux encoder 随机初始化 seed = MODEL_INIT_SEED(2026081200)。
- 矩阵：F0-C0 / F0-I / F0-D × seed12（对应 Step 3 的 C0-N/C1-I/C2-D 数据内容）。
- 因果（每个 checkpoint）：NORMAL / ZERO-AUX / SHUFFLE（输入级 bijective 无自配跨组 donor 置换）× last/best × train11/val6/all17。**正式 complementarity 判据：N > C0 且 N > ZERO-AUX 且 N > SHUFFLE**；"ZERO > SHUFFLE" 只说明错配比缺失更有害，**不是硬门槛**（如 N=0.30, S=0.24, Z=0.22 同样支持正确配对有价值）。
- 判级沿用四类协议；LOO 在 C0 与候选都完成后补。

## 门禁（audit_step4_f0.py，全 PASS 才允许训练）

| Gate | 内容 |
|---|---|
| G1 RGB 等价 | 同一 reference：3ch O2M vs F0(aux=0) 最终 detector 输出 max_abs_diff ≤ 1e-5 |
| G2 zero-init | 三个 P3/P4/P5 projection weight+bias 精确为 0 |
| G3 梯度流 | step1：proj 梯度 > 0、frozen backbone 梯度 None、F0-C0 权重梯度精确 0（bias 截距除外）；一步 SGD 后 step2：aux encoder 梯度 > 0 |
| G4 冻结锚点 | backbone 可训练参数 = 0 且 BN 恒 eval（含 model.train() 之后） |

## 文件

- `src/multimodal/{aux_encoder,feature_fusion,step4_f0_model}.py`（新）+ 复用 `trainability.py`
- `scripts/{audit_step4_f0,run_step4_f0,eval_step4_causality}.py`（新）
- `tests/test_step4_f0.py`（4 项：rgb 等价/zero-init/梯度流/shuffle 一致性）

## 执行顺序

1. audit 四门禁全 PASS（已完成：all_passed=true，reports/step4_f0_audit.json）
2. 三组各 1-epoch smoke（无 OOM/NaN；batch=4 预期可过，OOM 统一 batch=2/nbs=4 全组重来）
3. 三组 × 80 epochs（串行，formal 目录不可覆盖）
4. 三路因果评估（last 主口径）+ late10 + per-class
5. LOO + 四类判级 → **已完成（冻结，verdict_frozen=true，2026-08-16）**
   - 判级：F0-I / F0-D 均 MODEL-USES-AUX-BUT-NO-BENEFIT（`runs/step4_f0/_summary_step4.json` schema v2）
   - closeout 门禁：LOO payload 从 folds 重算精确比对、19 键 provenance（12 依赖 SHA + shuffle map SHA + 交叉）、G8 逐行 expected==actual、对抗测试 `tests/test_step4_closeout.py` 30 项
   - 下一步：F1 IR soft/reliability gate（暂不叠加 Depth）

## 后续（F1-F4，本阶段不实现）

F1 soft modality gate（权重和=1，quality prior 单调影响）→ F2a StrictOrthogonal / F2b RDTTrack-style（synthetic unit tests 先行）→ F3/F4 QAF 候选（需同时满足 clean 不差 + degraded 更稳 + causality 成立）。参考 `reference_implementation_notes.md`（RDTTrack 冻结纪律 / YOLOv5 multispectral P3P4P5 / CSSA soft gating / EvaNet 可靠性）。
