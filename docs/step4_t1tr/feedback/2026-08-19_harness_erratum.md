# T1-TR Harness Erratum（additive，不修改任何冻结产物）

日期：2026-08-19
状态：**DOCUMENTATION ONLY / ADDITIVE / DOES NOT INVALIDATE U2 RUN**

## 背景

`tests/test_t1tr.py::test_no_dataset_modification_file_in_bundle` 在最终执行环境中做了 1 处断言修改。

**原 bundle 断言（Codex 沙箱环境可过）：**

```python
def test_no_dataset_modification_file_in_bundle():
    assert not (ROOT/"src/multimodal/trimodal_dataset.py").exists()
```

**执行环境修复后断言：**

```python
def test_no_dataset_modification_file_in_bundle():
    # Bundle must not modify the frozen dataset module. On a full checkout the
    # file legitimately exists, so verify against the bundle payload manifest
    # (test-harness environment fix, mirrors A3 FakeGate precedent).
    import json
    val = json.loads(
        (ROOT/"T1TR_IMPLEMENTATION_VALIDATION.json").read_text(encoding="utf-8")
    )
    payload = val["payload"]["files"]
    assert "src/multimodal/trimodal_dataset.py" not in payload
```

## 原因

原断言要求完整源码仓库中不存在 `src/multimodal/trimodal_dataset.py`。该文件是冻结的上游项目模块，在最终执行仓库中**本就存在且未被本 bundle 修改**（bundle payload 10 文件不含它）。Codex 构建沙箱无完整仓库，故原断言在沙箱通过、在完整仓库必败。

修复为验证 **bundle payload manifest 不含** 该文件——与测试意图（"bundle 不得包含数据集修改文件"）一致，且与执行仓库状态无关。

## 性质

- **TEST HARNESS ENVIRONMENT ASSUMPTION FIX**（同 A3 FakeGate 先例）
- 非 experiment treatment 修复
- 仅改 `tests/test_t1tr.py`；DESIGN / runner / evaluator / summary / training 代码 / checkpoint **均未改动**
- formal preexecution audit（73/73, G1–G18 ALL PASS）在修复后的 test/source 状态上生成并通过
- **不使 U2-S formal run 无效，不需要重跑 U2**

## SHA 记录

```text
bundle_original_test_sha256: c3da0c320215761bb337e990d359176239cd690ecec25cdfc79fd4e84f6afa91
executed_test_sha256:        c9d039a3d6a1b199e4d5b7f245257d03d3941a4f7017b0b80dbf698e13d5e3c8
```

两者差异完全由上述 harness 断言修复引起。
