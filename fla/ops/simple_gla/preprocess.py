# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Shared input preprocessing for the Simple GLA operators.

`beta` is a head-wise write gate that scales the rank-1 update:

    S_t = a_t * S_{t-1} + beta_t * k_t v_t^T,    o_t = q_t S_t

Because `beta` multiplies the write term only, it is exactly equivalent to
scaling the values, so it is fused here as `v <- beta * v` instead of being
threaded through the shared chunk/recurrent kernels. `use_qk_l2norm_in_kernel`,
`use_gate_in_kernel` and `use_beta_sigmoid_in_kernel` are fused in the same pass,
mirroring how `fla/ops/gated_delta_rule/chunk.py` handles them.
"""

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.gate import fused_beta_sigmoid_bwd, fused_beta_sigmoid_fwd
from fla.ops.gated_delta_rule.gate import gdn_gate_bwd, gdn_gate_fwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


class SimpleGLAInputsFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor | None,
        g: torch.Tensor | None,
        A_log: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        g_input = g
        if use_gate_in_kernel:
            g = gdn_gate_fwd(g, A_log, dt_bias)

        v_input = v
        beta_eff = beta
        if beta is not None:
            if use_beta_sigmoid_in_kernel:
                beta_eff = fused_beta_sigmoid_fwd(beta_eff)
            # The product is formed in fp32 and rounded once, back to the value dtype.
            # Rounding `beta` to the value dtype before multiplying overshoots the
            # operator's tolerance in bf16, where the term would lose ~3 mantissa bits.
            v = (v_input.float() * beta_eff.float().unsqueeze(-1)).to(v_input.dtype)

        ctx.save_for_backward(q, q_rstd, k, k_rstd, v_input, beta_eff, beta, g_input, A_log, dt_bias)
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        return q, k, v, g

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor, dg: torch.Tensor | None):
        q, q_rstd, k, k_rstd, v_input, beta_eff, beta_raw, g_input, A_log, dt_bias = ctx.saved_tensors

        dbeta, dA_log, ddt_bias = None, None, None
        if beta_eff is not None:
            # `dv` is the gradient w.r.t. `beta * v`, so it feeds both the value
            # gradient (times `beta`) and the gate gradient (times `v`).
            dbeta = (dv.float() * v_input.float()).sum(-1)
            if ctx.use_beta_sigmoid_in_kernel:
                dbeta = fused_beta_sigmoid_bwd(beta_raw, dbeta)
            dbeta = dbeta.to(beta_raw.dtype)
            dv = (dv.float() * beta_eff.float().unsqueeze(-1)).to(v_input.dtype)
        if ctx.use_gate_in_kernel:
            dg, dA_log, ddt_bias = gdn_gate_bwd(g_input, A_log, dt_bias, dyg=dg)
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq, dk, dv, dbeta, dg, dA_log, ddt_bias, None, None, None


def validate_simple_gla_inputs(
    q: torch.Tensor,
    beta: torch.Tensor | None,
    g: torch.Tensor | None,
    g_gamma: torch.Tensor | None,
    A_log: torch.Tensor | None,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
) -> None:
    if beta is not None and beta.shape != q.shape[:3]:
        raise ValueError(
            f"`beta` is expected to have the leading `[B, T, H]` dims of `q`, i.e. {tuple(q.shape[:3])}, "
            f"but got {tuple(beta.shape)}.",
        )
    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("`A_log` must be provided when `use_gate_in_kernel=True`.")
        if g is None:
            raise ValueError("`g` (raw pre-activation) must be provided when `use_gate_in_kernel=True`.")
        if g_gamma is not None:
            raise ValueError("`g_gamma` cannot be combined with `use_gate_in_kernel=True`.")
    if use_beta_sigmoid_in_kernel and beta is None:
        raise ValueError("`beta` must be provided when `use_beta_sigmoid_in_kernel=True`.")


def prepare_simple_gla_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor | None,
    g: torch.Tensor | None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    *,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
):
    # Nothing to fuse: keep the caller on the exact code path it used before.
    if beta is None and not use_qk_l2norm_in_kernel and not use_gate_in_kernel:
        return q, k, v, g
    return SimpleGLAInputsFunction.apply(
        q,
        k,
        v,
        beta,
        g,
        A_log,
        dt_bias,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
    )
