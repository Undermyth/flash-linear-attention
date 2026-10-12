# Simple GLA

Gating mechanism in [Gated RFA](https://arxiv.org/abs/2103.02143), [Mamba2](https://arxiv.org/abs/2405.21060) and [YOCO](https://arxiv.org/abs/2405.05254) (a.k.a., Gated RetNet).

Compared to GLA, the gating is head-wise instead of elementwise.
As a result, we can adapt the RetNet kernel for training using matmul w/o numerical instability.
It is faster than GLA but has less expressive power.
I will use it as a baseline for the GLA.

$S_{t+1} = g_{t+1} \odot S_{t} + K_{t+1} V_{t+1}^{\top}$ where $g$ is a scalar.

An optional head-wise write gate `beta` of shape `[B, T, H]` scales the rank-1 update:

$S_{t+1} = g_{t+1} \odot S_{t} + \beta_{t+1} K_{t+1} V_{t+1}^{\top}$

With `beta = 1` this reduces to the recurrence above, and because the gate multiplies the
write term only it is equivalent to scaling $V$ by `beta`.
