# T1-TR Formal Closeout

状态：**ACCEPTED / CLOSED**
远端基线：`9b14bdb028afcd6ce715c5ea68f8b2cf019e54a5`
Harness erratum closeout：`c87c455`
日期：2026-08-19

## Formal verdict

```text
T1-TR
  ACCEPTED / CLOSED

Formal route:
  TRAINING_TREATMENT_GAIN_NOT_STABLE

U2-S:
  VALID
  80 / 80 epochs
  balanced fully-wrong schedule VALID

Balanced wrong-source training:
  HARMFUL vs NULL
  SINGLE-SEED SUPPORTED

Paired-training source specificity:
  NOT FORMALLY ESTABLISHED

Replication:
  HOLD

Depth:
  HOLD

Production:
  HOLD
```

## Frozen primary result

统一 ZERO inference，mAP50-95：

```text
U0-N
  val6    = 0.26443877551020406
  train11 = 0.9725827205882351
  all17   = 0.7174711751029454

U1-P
  val6    = 0.29596085371085373
  train11 = 0.9577677986604124
  all17   = 0.7166615816435083

U2-S
  val6    = 0.2582846594333936
  train11 = 0.9704351190476188
  all17   = 0.7157771604281559
```

```text
U1-U0 = MIXED
  val6    +0.03152207820064967
  train11 -0.014814921927822744
  all17   -0.0008095934594370968

U2-U0 = STABLE_NEGATIVE
  val6    -0.006154116076810434
  train11 -0.0021476015406163285
  all17   -0.001694014674789579

U1-U2 = MIXED
  val6    +0.03767619427746011
  train11 -0.012667320387206416
  all17   +0.0008844212153524822
```

预注册正式结论保持：

```text
TRAINING_TREATMENT_GAIN_NOT_STABLE
```

## Strong secondary evidence

val6 LOO：

```text
U1-U0
  6/6 positive
  median +0.039270607349639636

U2-U0
  0/6 positive
  6/6 negative
  median -0.005053731356362931

U1-U2
  6/6 positive
  median +0.045811587625402536
```

因此保留：

```text
PAIRED-TRAINING GENERALIZATION SIGNAL
  STRONG SINGLE-SEED SECONDARY EVIDENCE
```

held-out 次序：

```text
paired training > NULL > balanced wrong-source training
```

但不得据此改写 formal causal GO。

## Scientific wording freeze

允许：

> Balanced fully-wrong IR source training 未复现 T1 effect，并在冻结的三个 primary endpoint 上稳定劣于 NULL。因此 T1 gain 不由该 source-agnostic shuffled-IR training mechanism 解释。

禁止：

> generic regularization 已全部排除。

其他 generic mechanism 仍未被排除。

## U2 secondary

```text
U2 ZERO   = 0.2582846594333936
U2 native = 0.29191154309825196
```

解释仅限：

> 错配训练破坏了训练状态；测试时恢复正确 native IR residual 能部分补偿。

## Harness erratum

原 test 在完整源码仓库错误检查 `trimodal_dataset.py` 不存在；执行版改为检查 bundle payload 不包含该冻结文件。

性质：

```text
TEST HARNESS ENVIRONMENT ASSUMPTION FIX
NON-SCIENTIFIC
```

不改变 design / runner / evaluator / summary / training / checkpoint / U2 result，无需重跑。

## Route close

```text
A2–A5
T-series
T1-S
T1-TR
  => CLOSED

SUPPORTED:
  P5-only topology/training treatment has single-seed benefit evidence.
  balanced fully-wrong IR training is harmful vs NULL.

SECONDARY:
  paired training has strong held-out val6 signal.

NOT FORMALLY ESTABLISHED:
  paired-training source specificity.

HOLD:
  replication
  Depth
  production

NEXT:
  T1-GR
  fresh-data generalization replication
```

现有 val6 从现在开始仅作 historical/diagnostic evidence，不再作为新 treatment GO 的 fresh holdout。
