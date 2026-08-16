# F1 实际执行反馈区

这里只记录执行后事实，不保存预先计划。

每次执行新增一个带日期和物理 run 名的 Markdown 文件，至少包含：

可复制 `_TEMPLATE.md`，但不要直接在模板里累计多次运行。

- commit SHA、机器、GPU、PyTorch、Ultralytics；
- 命令与退出码；
- audit/smoke/formal 的物理目录；
- G5/G6/G8 结果；
- last/best/late10 与 NORMAL/ZERO/SHUFFLE；
- FORCE-Q0/Q1、val6 每图 q、质量退化结果；
- LOO 分布与 summarizer decision；
- 失败时保留目录和原始错误，不覆盖、不清理历史。

未实际执行前不要在这里预填 PASS。
