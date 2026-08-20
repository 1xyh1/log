# E5 v2 实现日志

## 交付状态

`IMPLEMENTATION_PASS / GPU_INTEGRATION_NOT_RUN`

交付环境完成：

- Python compileall：PASS
- 冻结 training spec JSON 解析：PASS
- security policy JSON 解析：PASS
- 依赖轻量回归：62/62 PASS
- frozen config SHA 自校验：PASS
- failure bundle manifest：25/25 PASS
- v1/v2 输出命名隔离：PASS
- holdout operational input 静态门禁：PASS

交付环境没有项目冻结的 torch、Ultralytics 8.4.56 和 CUDA 栈，所以不声称完成 GPU preflight、1 epoch smoke 或 80 epoch formal。GPU 集成验证由 v2 preflight + mandatory smoke 在目标机完成。

## 板块 1：AdamW/MuSGD 裁定

修改文件：

- `config/t1gr_e5_training_spec.frozen.json`
- `config/t1gr_e5_training_spec.candidate.json`
- `config/t1gr_e5_training_spec.template.json`
- `src/multimodal/t1gr_e5_core.py`
- `scripts/t1gr_e5_freeze_recipe.py`

结果：

- 项目选择固定为 MuSGD。
- `optimizer=auto` 明确禁止。
- 裁定记录明确写出 auto 会选择 AdamW。
- 公式与输入逐项绑定，validator 重算必须等于 1920。
- 删除 v1 的 30080 iteration 和 auto=>MuSGD 等价表述。

## 板块 2：随机 head 初始化顺序

修改文件：

- `src/multimodal/t1gr_e5_core.py`
- `scripts/t1gr_e5_preflight.py`
- `scripts/t1gr_e5_run_step1.py`

新增公共构建函数 `build_seeded_model`：

1. 拒绝分布式 RANK。
2. 计算 `seed + 1 + RANK`。
3. 调用 Ultralytics `init_seeds`。
4. 创建 nc=12、end2end=true 的 YOLO26s。
5. 加载 checkpoint 中 shape-compatible 参数。
6. 计算完整初始状态和未迁移状态 SHA。

preflight 与 runner 使用同一函数，不再各自复制模型构建逻辑。

## 板块 3：初始状态与训练起点绑定

新增证据字段：

- `model_initialization_effective_seed`
- `model_initial_state_sha256`
- `untransferred_initial_state_sha256`
- `training_start_state_sha256`
- `untransferred_state_key_count`

门禁关系：

- runner 初始 SHA = preflight 初始 SHA
- smoke 初始 SHA = preflight 初始 SHA
- formal 初始 SHA = preflight 初始 SHA
- smoke/formal training-start SHA = 各自初始 SHA
- final audit 同时检查上述全部关系

## 板块 4：训练失败 traceback

修改文件：

- `config/t1gr_e5_security_policy.json`
- `config/t1gr_e5_training_spec.*.json`
- `src/multimodal/t1gr_e5_core.py`
- `scripts/t1gr_e5_run_step1.py`

私有失败证据：

```text
<run-dir>/E5_PRIVATE_FAILURE.json
```

包含：

- 当前 phase
- exception type/message
- 完整 traceback（受 1 MiB 上限保护）
- results.csv 是否存在及已完成行数
- args.yaml/last.pt/best.pt 是否存在
- `public_pass_issued=false`

公开 stderr 仍由 `safe_error_message` 输出脱敏错误。

## 板块 5：赛制 max_det 与 v2 输出隔离

修改文件：

- `config/t1gr_e5_training_spec.*.json`
- 全部 E5 operational scripts

结果：

- `eval_args.max_det=100`
- validator 强制必须为 100
- reports 使用 `e5_v2_*.json`
- run dirs 使用 `*_V2`
- v1 证据不覆盖、不复用

## 板块 6：验证

新增：

- `scripts/t1gr_e5_v2_regression_gate.py`
- 10 个 v2 专用回归测试，并加强原 frozen SHA 与 security policy 测试

最终零依赖门禁：

```text
PASS 62/62
```

## 完整改动代码

所有修改板块的完整 unified diff 位于：

- `docs/E5_V2_FULL_IMPLEMENTATION_DIFF.md`

该文件是从收到的 failure bundle 的 `config/`、`scripts/`、`src/`、`tests/` 与本 v2 对应目录直接生成，包含全部新增/删除/上下文代码块。

