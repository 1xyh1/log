# E4 Public Privacy Scanner Key Erratum（additive，harness-only）

日期：2026-08-20
状态：**HARNESS-ONLY / NON-SCIENTIFIC / AUTHORIZED BY REVIEWER**

## Observed failure

`assert_public_safe()` 拒绝了 E4 public 报告中的布尔字段：

```text
"contains_any_sample_ids": False
```

错误码：`PUBLIC_SENSITIVE_KEY:contains_any_sample_ids`（seal 与 verify 两个脚本均触发）。

## Root cause

`src/multimodal/t1gr_secure_io.py::assert_public_safe()` 对 dict key 名的规则：

```text
key.endswith("_ids") 且不含 "commitment"/"count" → FAIL
```

该规则**只看 key 名，不看值**。`contains_any_sample_ids` 以 `_ids` 结尾（`sample_ids` 的末 4 字符为 `_ids`），被误判为敏感 key。该字段语义是"本 public 报告不含任何 raw sample ID"的布尔自我声明（值为 False），并非泄漏内容。

## Change（reviewer authorized，2026-08-20）

两处最小字段 rename（值与语义不变）：

```text
seal  public（scripts/t1gr_e4_seal_split.py）:
  "contains_any_sample_ids": False
    → "any_raw_sample_id_present": False

verify public（scripts/t1gr_e4_verify_seal.py）:
  "contains_any_sample_ids": False
    → "any_raw_sample_id_present": False
```

## Unchanged

```text
t1gr_secure_io.py / assert_public_safe  未改（无 exemption、无放宽）
E3 split candidate                       未改
E4 commitments (TRAIN/DEV/FINAL)         未改
E4 private TRAIN/DEV access artifact     未改
E4 private FINAL HOLDOUT sealed artifact 未改
E4 receipt                               未改
sample counts 1504/198/298               未改
FINAL HOLDOUT access policy              未改
E4 seal 未重跑（已 PASS 的 seal 产物直接消费）
```

## Classification

```text
HARNESS-ONLY
NON-SCIENTIFIC
PRIVACY SEMANTICS UNCHANGED
NO RERUN OF SPLIT REQUIRED
E4 SCIENTIFIC / SPLIT VALIDITY NOT INVALIDATED
```

## Regression tests added（tests/test_t1gr_e4_seal.py，+3）

```text
test_public_boolean_no_raw_id_statement_is_safe   → assert_public_safe({"any_raw_sample_id_present": False}) 通过
test_raw_ids_still_rejected                       → assert_public_safe({"train_ids":["sample_001"]}) raises GateError
test_public_false_boolean_does_not_use_ids_suffix → not "any_raw_sample_id_present".endswith("_ids")
```

证明：修掉 false positive ≠ 放宽真正 raw-ID protection。

## SHA record

```text
verify script SHA before: 24df1f085349bb6917d8108e0ce7d2cf49534e51220f39743e166ad039b3924b
verify script SHA after:  84d6d5a3f50d695473c29feaa5fa46f53330b75b3caafda08a283f1d21a51a08
secure_io SHA:            18d88bad87dc51c54cee4881fabb31d03708c1b998e97dd1764317d8e1ddc691 (before == after)

e4_train_dev_access_private SHA:        770e52c63536f5f13f2958d28cc47bb7601afbe7fcbf59d2c281fa58a4ed3b28
e4_final_holdout_sealed_private SHA:    69155246a7457dc3a1638360f6d96695eec6ad2432a17540d02fd064e0f9020d
e4_seal_receipt_private SHA:            d7922b7f6cee48f43610424f408d1f043b827a771e6382e105d4853fbe76c77f

e4_split_freeze_public SHA:             f551403adaa11885563c388e1c51addf2fd918d8559433729d795752ed69a641
e4_seal_verification_public SHA:        725cd510df9ec18d91f4a610eff0320c4f3c03f0d5cfe7e5cc6d78050ea5c29e
```

`e4_split_freeze_public`（f551403a…）为本次 verify 所消费的 frozen seal evidence。
