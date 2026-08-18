# A2 设计冻结：Scale-wise IR Residual Causality Audit

状态：**PRE-EXECUTION / EVALUATION-ONLY / dependency-closure-v2**  
上游冻结证据：`F1-C CLOSED / FAILED @ bf983c4`，`F1C_GATE_FAILED_CAUSAL_PROTOCOL`。

## 1. A2 只回答一个问题

A2 只回答：

> 已训练完成的 IR residual `δ3=P3(A3)`、`δ4=P4(A4)`、`δ5=P5(A5)`，在
> P3/P4/P5 哪些尺度上具有稳定、与正确 RGB–IR 配对相关的正因果价值？

A2 **不回答**：RGB–IR registration / spatial agreement 为什么好或坏；不训练新 gate；
不引入 Depth；不决定 `q3/q4/q5` 架构。若三个尺度均无稳定正 paired effect，后续才进入
A3（RGB–IR Spatial / Semantic Agreement Audit）。

## 2. 冻结 checkpoint 身份

Primary：`runs/step4_f1_c/F1C-I-fixed/weights/last.pt`

- `aux_mode=ir`
- `gate_mode=fixed_one`
- `gate_module=magnitude`
- `q≡1`
- 用于最干净地回答 IR encoder + projection residual 本身的尺度级因果价值。

Replication：`runs/step4_f1_c/F1C-I-soft/weights/last.pt`

- `aux_mode=ir`
- `gate_mode=learned`
- `gate_module=original`
- 作为独立训练系统上的复核，不得把 fixed/soft 当作“同一模型切 q”的 matched ablation。

Primary checkpoint 固定为 `last.pt`。`best.pt` 不参与 A2 主结论。

## 3. 因果识别边界：先算 q，再干预 residual

当前 F1 模型：

```text
A3,A4,A5 -> q(A3,A4,A5)
δi = Pi(Ai)
Fi = Ri + q * δi
```

A2 对 soft checkpoint 必须先使用**原始 recipient 的完整 paired A3/A4/A5**计算：

```text
qx = q(A3^x, A4^x, A5^x)
```

随后冻结 `qx`，全部 intervention 只能发生在 projection 之后的 residual `δi` 或其 gain 上。
不得因为 DROP/SHUFFLE/GAIN 重新计算 gate。

P3 conditional shuffle 的合法形式：

```text
F3^x = R3^x + qx * δ3^donor
F4^x = R4^x + qx * δ4^x
F5^x = R5^x + qx * δ5^x
```

禁止：先替换 `A3` 再把 mixed `(A3_donor,A4_x,A5_x)` 输入 gate。

fixed checkpoint 同一 evaluator 路径执行，`q_native` 必须恒等于 1。

## 4. Intervention 发生位置

允许：

```text
aux_encoder -> Ai -> projection Pi -> δi -> [A2 intervention] -> residual add
```

禁止：

- dataset 层对 IR 整图 ZERO/SHUFFLE；
- aux_encoder 输入层 shuffle；
- projection 之前 shuffle `Ai` 后又触发 gate；
- 修改 RGB；
- 修改 neck/head；
- 参数更新。

Donor cache 只允许计算 donor 的 `δ3/δ4/δ5`，不得调用 reliability gate。

## 5. 完整 2^3 residual-mask factorial

对每个 checkpoint 评估八个条件：

| condition | P3 | P4 | P5 |
|---|---:|---:|---:|
| M000 | 0 | 0 | 0 |
| M100 | 1 | 0 | 0 |
| M010 | 0 | 1 | 0 |
| M001 | 0 | 0 | 1 |
| M110 | 1 | 1 | 0 |
| M101 | 1 | 0 | 1 |
| M011 | 0 | 1 | 1 |
| M111 | 1 | 1 | 1 |

fixed：开启尺度 `αi=1`；关闭 `αi=0`。  
soft：开启尺度 `αi=q_native(x)`；关闭 `αi=0`。

`M111` 必须与冻结 checkpoint 的 native forward 在 detection tensor 上逐位一致；否则 A2 立即 ABORT。

报告至少计算：

```text
standalone_i = AP(keep-only-i) - AP(M000)
drop_i       = AP(M111) - AP(drop-i)
I34 = AP(M110)-AP(M100)-AP(M010)+AP(M000)
I35 = AP(M101)-AP(M100)-AP(M001)+AP(M000)
I45 = AP(M011)-AP(M010)-AP(M001)+AP(M000)
```

## 6. Scale-wise pairedness intervention

使用同一个冻结 val donor map，必须：bijection、no-self、deterministic；现有
`causality_interventions.bijective_derangement` / `assert_valid_shuffle_map` 为唯一 donor-map 实现。
fixed 与 soft 必须消费**同一 donor map 文件**。

每个尺度评估两类：

### 6.1 Conditional pairedness

其余尺度保持 native，仅 target scale residual 换 donor：

```text
Δ_i^conditional = AP(M111 paired) - AP(SHUFFLE_i_COND)
```

### 6.2 Standalone pairedness

只启用 target scale：

```text
Δ_i^standalone = AP(KEEP_i paired) - AP(SHUFFLE_i_ONLY)
```

这样 conditional effect 与 standalone effect 分开，不允许用其中一个替代另一个。

## 7. Scale-specific gain sweep

每个尺度分别做 conditional dose-response；其他尺度保持 native。

fixed：

```text
alpha_i in {0, 0.25, 0.5, 0.75, 1.0}
other alpha_j = 1
```

soft：

```text
alpha_i in {0, 0.25, 0.5, 0.75, 1.0, NATIVE}
other alpha_j = q_native(x)
NATIVE means alpha_i = q_native(x), not numerical 0.5
```

任何常数 gain 都只能替换 target scale 的 residual coefficient，不改变 gate。

## 8. LOO 与解释规则

val6 是主 probe。每个 condition 一次 forward 采集逐样本 validator stats，再从同一批 stats 重算：

- full val6；
- 6 个 leave-one-out folds。

不得为了 LOO 重新训练或改变 donor map。

A2 是诊断，不复用 F1-C promotion gate，不设置人为 `+0.01` 之类 margin。
对 `Δ_i^conditional` 与 `Δ_i^standalone` 分别输出：

```text
STRONG_POSITIVE:
  fixed full > 0
  fixed LOO median > 0
  fixed positive folds >= 4/6
  soft full > 0

STRONG_NEGATIVE:
  fixed full < 0
  fixed LOO median < 0
  fixed negative folds >= 4/6
  soft full < 0

INCONCLUSIVE:
  otherwise
```

不强制把 conditional 与 standalone 压成一个总标签；两者冲突本身就是 A2 结果。

## 9. 冻结依赖闭包（v2 新增，执行前必须满足）

A2 不仅锁两个 `last.pt` 文件，还必须锁**执行这些 checkpoint 的当前语义**。
`torch.load` 会使用当前工作树中的 Python 类定义，因此仅校验 checkpoint SHA 不足以证明
“冻结模型语义未漂移”。真实 eval 启动前必须：

1. `runs/step4_f1_c/_summary_step4_f1_c.json` 原始文件 SHA256 必须等于
   `bf983c4` 镜像清单记录值：
   `d4e64b86e221b102143bd98cc6056f8e84d7913680cad3c8c5826af4cf88942f`；
2. summary 必须仍为 `verdict_frozen=true` 且
   `decision=F1C_GATE_FAILED_CAUSAL_PROTOCOL`；
3. FIXED/SOFT 各自的冻结 `eval_step4_f1_c_causality.json` 必须存在，且 summary 中
   对应 `eval_provenance_verified` 为 PASS；
4. 当前 contract SHA、manifest SHA、val6 sample-id 顺序必须与冻结 F1-C causal eval 一致；
5. 当前 `step3_eval_utils.py`、F1 model、gate、dataset、F0 model、AuxEncoder、
   feature fusion、trainability、causality interventions、raw sample index 的 SHA 必须与
   冻结 F1-C causal eval provenance 一致；
6. 当前 torch / ultralytics 版本必须与冻结 F1-C causal eval provenance 一致；
7. A2 pre-execution audit 必须是 `step4-a2-preexecution-audit-v2`、`all_passed=true`，
   且 audit 记录的 design/engine/evaluator/tests/audit-source SHA 与真实 eval 启动瞬间的文件 SHA 一致。

任一项不满足：`A2_FROZEN_DEPENDENCY_CLOSURE_FAIL` 或
`A2_PREEXECUTION_AUDIT_STALE`，不得继续。

Donor map 还必须**精确等于**当前 val6 上
`bijective_derangement(val_ids)` 的 deterministic 结果；不能仅因为另一个已有 map 也满足
bijection/no-self 就接受。否则 `A2_DONOR_MAP_NOT_FROZEN_DETERMINISTIC`。

## 10. A2 硬门禁

- **A2-G1 CHECKPOINT**：只允许冻结 F1C-I-fixed / F1C-I-soft `last.pt`；checkpoint SHA 必须来自冻结 summary，manifest/contract/source/version/val6 identity 同时进入冻结依赖闭包。
- **A2-G2 EVAL-ONLY**：整个 A2 无 optimizer、无 backward、无参数更新；before/after model-state SHA 必须一致。
- **A2-G3 NATIVE-EQUIVALENCE**：M111 detection tensor 与原 checkpoint native forward 逐位一致。
- **A2-G4 Q-FREEZE**：soft 每个 sample 在所有 mask/shuffle/gain 条件中的 `q_native` 必须与 untouched recipient native q 逐位一致；fixed 必须恒为 1。
- **A2-G5 POST-GATE-ONLY**：donor residual cache 不得调用 gate；shuffle 只替换 target `δi`，source trace 证明其他尺度仍来自 recipient。
- **A2-G6 DONOR**：val donor map 必须精确等于 deterministic `bijective_derangement(val_ids)`，同时 bijective、no-self、fixed/soft 共用，map SHA 落盘。
- **A2-G7 FACTORIAL**：八个 mask 一个不少、一个不多；M000/M111/keep/drop 的身份固定。
- **A2-G8 STOCK-EVAL**：输入仍走 Step3 authoritative validator semantics；不调用 stock `/255` preprocess；Step3 eval helper 与 torch/ultralytics 版本必须与冻结 F1-C provenance 一致，禁止无条件 `true` 自证。
- **A2-G9 PROVENANCE**：记录 evaluator/intervention/design/Step3 eval utils/model/gate/dataset/F0/AuxEncoder/fusion/trainability/causality/raw-index/contract/audit、两个 manifest 和两个 last.pt 的 SHA；required provenance 字段动态完整性检查，禁止无条件 `true`。
- **A2-G10 INTERPRETATION**：主报告必须同时给 fixed primary 与 soft replication；不得把 fixed-vs-soft 差异解释成同模型 q ablation。

任一 G1–G9 失败：`A2_ABORT`，不得输出尺度因果标签。

## 11. 明确禁止

```text
NO training
NO optimizer
NO backward
NO parameter update
NO new gate
NO Depth
NO q3/q4/q5
NO dataset-level IR shuffle
NO intervention-before-gate for soft
NO best.pt as primary
NO donor-map change after seeing results
NO F1-C source/provenance rewrite
```

## 12. 后续分叉（不属于 A2 实现）

- 某些尺度 fixed + soft 都稳定 paired-positive：下一阶段先 static per-scale selection/weighting，再考虑动态 `q3/q4/q5`。
- 仅一个尺度稳定正：优先 single-scale IR fusion。
- 三尺度均无稳定正 paired effect：停止 gate 路线，进入 A3 agreement/registration/representation 诊断。
- fixed 正而 soft 负：优先怀疑 gate-conditioned training dynamics / representation coupling，而不是直接归因 registration。
