# 下一训练实验草案（尚未冻结）

本文件只定义 A1 v2 通过后可审阅的最小实验，不代表 FORMAL GO。

## 保持不变

- pretrained RGB backbone 继续冻结，RGB 为 anchor；P3/P4/P5 residual、
  zero-init projection、P5 路由、`amp=False`、train11/val6、G5/G6/G8/G9、
  B1 deterministic corruption 与 NORMAL/ZERO/SHUFFLE 全部不变。
- 仍是单模型推理，不引入 expert voting/ensemble。
- 不加 Depth、spatial gate、Transformer、QAF、BDDS。

## 唯一候选变量

gate 输入由原 joint-LN 向量扩展为：

`concat(joint_LN_gap_vector, log_rms_A3, log_rms_A4, log_rms_A5)`

三个 log-RMS 在 projection 前计算并 detach。优先不用 projected residual energy，
因为它同时受 projection 学习和 `q × P(A)` 尺度耦合影响；也暂不使用 spatial
cosine，因为 A1 已显示尺度间方向不稳。

为保持单变量，gate 仍输出一个 scalar q，MLP 宽度、初始化、residual 公式和
训练超参不变。matched 组至少包括 C0、fixed、原 gate、magnitude-input gate；
是否需要四组 formal 必须在新 DESIGN_FREEZE 中预注册后再执行。

## 必须先补的证据

1. A1 v2 的 continuous-target + leave-one-family-out 结果。
2. 新 gate 的 G1 RGB 等价、G2 zero-init、G3 梯度解锁与 gate-detach 审计。
3. 训练结束、checkpoint 半精度序列化之前的 final fp32 RGB backbone SHA。
4. FORMAL 晋级条件必须同时覆盖 clean causal、退化 macro/worst、
   learned-vs-own-QCLEAN，并保留 last.pt 主口径。
