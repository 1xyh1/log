# F1-C-A0 执行反馈：2026-08-17 / RGB–IR agreement diagnosis

- command：`diagnose_step4_f1_c_agreement.py`（val6 + 17 退化）+
  `diagnose_step4_f1_c_agreement_all17.py`（all17 + growth 漂移）
- 产物：
  - `reports/step4_f1_c_agreement/agreement_diagnosis.json`
  - `reports/step4_f1_c_agreement/agreement_all17_growth.json`
- 输入：B1-I-soft 的 last.pt / best.pt（不重训、不重评估 AP）

## 描述子（预注册）

- agreement_i = cos(GAP(R_i), GAP(P_i(A_i)))（1×C 余弦，P_i(A_i) 是 gate 缩放前
  的 projected aux residual）
- aux_rel_energy_i = ||P_i(A_i)||_2 / ||R_i||_2
- q = learned gate

## 结果：agreement 无法稳定区分正确配对与错配

**NORMAL − SHUFFLE agreement（per-image 差）**：

| 轴 | ckpt | P3 | P4 | P5 |
|---|---|---|---|---|
| val6 (n=6) | last | 3/6 正，median −0.0004 | 3/6，+0.0011 | 3/6，−0.0005 |
| val6 (n=6) | best | 3/6，+0.0029 | 3/6，−0.0045 | 2/6，−0.0071 |
| all17 (n=17) | last | 8/17，−0.0083 | 9/17，+0.0062 | 8/17，−0.0015 |
| all17 (n=17) | best | 9/17，+0.0016 | 9/17，+0.0015 | 11/17，+0.0015 |

两个轴、两个 checkpoint、三个尺度全部处于 ~50% 随机水平，median 幅度
≤0.01。**GAP-cosine agreement 描述子不满足"val6/all17 都有稳定方向"的
预注册条件**，不能直接作为 scalar agreement gate 的输入。

17 退化条件下 agreement 均值也无可解释方向（identity P3 +0.024/P4 +0.043；
noise:1.0 P4 +0.060；contrast:1.0 P3 −0.036），q 恒 0.50 附近。

## epoch 39 → 80 配对信号消失的定位证据

- val mAP：**e39 = 0.3241 峰值 → e80 = 0.3040**（掉 0.020）
- projection norm e39→e80 **继续增长**（P3 0.101→0.126，+25%；P5 0.187→0.227，
  +22%）——不是残差坍缩
- q：e39 0.5014 → e80 0.5003（恒常数）
- corruption 强度冻结（无变化）

候选归因（未定论）：后期 val 过拟合为主（e39 后 val 单调掉），residual
尺度漂移（proj 持续增长）为伴随现象；与 corruption 强度无关。

## 结论与建议

1. GAP-cosine agreement 方向失败 → 不进入 scalar agreement gate。
2. 下一个 agreement 描述子候选（如需继续）：空间对齐的局部 agreement
   （逐空间位置 cos）、或 RGB/IR 梯度方向一致性——需先做诊断再定。
3. 配对信号衰减的归因实验（如继续）：冻结 e39 snapshot 或 early-stop 对照
   需要重训，超出本阶段"不训练"指示，暂不启动。
