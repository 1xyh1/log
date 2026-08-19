# T-series implementation adjudication — FP32 bias-gradient exactness

Status: **PRE-EXECUTION P0 HARDENING / ACCEPTED INTO IMPLEMENTATION**

The frozen mathematical treatment remains unchanged:

```text
delta5 = Conv1x1(A5, bias=True)
used5  = delta5 - mean_HW(delta5)
fused5 = R5 + used5
```

Mathematically, a spatially constant projection bias cancels from `used5`.

## Numerical issue found before any formal T-series execution

In finite-precision FP32 autograd, the backward reductions for

```text
delta - mean_HW(delta)
```

can leave a small nonzero `proj.bias.grad` even though the exact derivative is zero.
This is especially unsafe with the existing MuSGD/Muon-style optimizer path because
small numerical gradients must not be allowed to become real treatment-side bias
updates.

Therefore the implementation adds a **T2-only optimizer-step numerical guard**:

1. forward remains the exact frozen post-projection `AC_ALL` treatment;
2. `proj.bias` remains physically present, `bias=True`, trainable flag unchanged,
   and remains in the same optimizer parameter set/group as T0/T1;
3. immediately before the optimizer step, T2 zeroes only `proj.bias.grad`;
4. smoke records the pre-zero numerical gradient magnitude;
5. formal pretraining requires the projection-bias optimizer group to have zero
   weight decay (or an equivalent proven no-update rule);
6. the one-epoch real MuSGD smoke requires projection bias to be bitwise unchanged.

This is a numerical-exactness guard for a derivative that is theoretically zero.
It is **not** a new learned gate, not a change to the forward treatment, and not a
post-result patch.

Formal reporting must distinguish:

```text
raw FP32 bias gradient dust
vs
optimizer-applied bias gradient (forced exact zero in T2)
vs
actual bias parameter delta (must remain zero)
```
