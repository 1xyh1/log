# E3 Hardened Engineering Review v1.2

本文件只审工程安全性，不改变 E3 科学裁定。

## 科学规则

```text
18 historical contaminated IDs -> seed quarantine
808 strong edges               -> accepted provenance
180 review edges               -> MERGE_ALL_CONSERVATIVELY
final graph                    = strong ∪ review
final groups                   = connected components
historical seed ∩ component    -> whole component FORCE_TRAIN
```

## 风险检查矩阵

| 风险 | Fail-closed 控制 | 对抗/集成验证 |
|---|---|---|
| 空值/缺字段 | bounded JSON + schema + required non-null keys | null field / empty component / malformed JSON tests |
| 重复 ID | component 内、跨 component、force list 全查重 | duplicate component IDs tests |
| 重复 edge | strong/review 各自查重，跨列表禁止重复 | duplicate/self edge tests |
| 重复请求 | request fingerprint + payload integrity；同请求幂等复用 | atomic same-request + full-chain rerun |
| 并发 | public-output lock，O_EXCL，active lock 不抢占 | concurrent lock test |
| stale lock | 仅超过阈值且 PID 不存活时清理 | stale dead-lock recovery test |
| private 路径 | private input/output 必须 repo 外，symlink resolve 后再判 | repo-inside reject tests |
| public 路径 | 只能 `reports/step4_t1gr/`，禁 `..` escape | prefix/escape tests |
| private 权限 | temp/private POSIX 0600；Windows 不虚假宣称 ACL 已证明 | POSIX mode test + Windows caveat |
| 超时 | closure/split 全局 deadline，循环持续检查 | deadline expiry + synthetic timing |
| 输入并发修改 | policy/private/formal ZIP 前后 stat token 一致 | stat-token change test |
| ZIP 异常 | member 数/label 大小/总 label bytes/加密/path traversal 上限 | unsafe path test + static gate |
| 异常处理 | CLI 捕获所有异常，仅输出 safe error code，无 traceback | generic exception redaction test |
| public ID 泄露 | sensitive-key scan + string-list/path-string scan | IDs/path leak tests |
| public 文件完整性 | `payload_sha256` + request fingerprint | tamper detection test |
| 写入中断 | sibling temp + fsync + `os.replace()` | static + full-chain rerun |
| policy 漂移 | bundle 内 hard-coded frozen policy SHA | frozen SHA test/static audit |
| upstream private 漂移 | 重新构 strong DSU，与 strong-only partition 等价校验 | provenance mismatch test |
| upstream public/private 不一致 | 对 `b15f74a` public counts/commitments/provenance 交叉核验 | formal closure runtime gate |
| historical 隔离传播错误 | split 前重新验证 seed→whole-component force propagation | bad propagation test |
| split ID 集合错误 | nonempty + overlap empty + exact union==2000 + component not split | full 2000 synthetic gate |
| 任意 YAML 注入 | split proposal 无 `--data`/YAML 入口 | static audit |

## Windows 权限边界

Python 的 `chmod` 不能证明 Windows NTFS ACL 的完整保密性。因此工具只正式声明：

```text
private data is kept outside repo
private paths are never emitted to public reports
formal tooling does not copy private IDs into repo
```

不声明：

```text
Windows ACL confidentiality cryptographically proven
```

若 `E:\google\t1gr_private` 本身对其他 Windows 用户开放，需要在操作系统侧另行收紧 ACL；这不应由实验脚本偷偷修改。

## 并发/重复请求语义

```text
same request, no concurrent writer:
  existing payload integrity PASS -> idempotent reuse

same request while first writer still running:
  wait up to frozen lock_wait_seconds
  still locked -> CONCURRENT_RUN_LOCKED

same output path, different request fingerprint:
  OUTPUT_CONFLICT_DIFFERENT_REQUEST
```

不会自动覆盖正式 evidence。
