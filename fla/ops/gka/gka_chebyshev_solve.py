"""
GKA Chebyshev Solver Implementation.

This module implements the core solving routines for the GKA layer, which uses
ridge regression over the entire sequence history to compute optimal states.

Core Algorithm:
    The GKA layer finds an optimal state matrix, S_t, at each timestep that minimizes
    a ridge regression objective over the entire history:

        S_t = argmin_S { λ·||S||²_F + Σ(i=1 to t) η_i·||S·k_i - v_i||² }

    where:
    - S ∈ ℝ^(D×D) is a linear map that should satisfy S·k_i ≈ v_i for all past tokens
    - λ is the ridge regularization strength (prevents overfitting, ensures stability)
    - η_i are exponentially decaying weights controlled by forgetting gates

    Unlike standard SSMs that update their state based only on the current token, this
    formulation considers the entire past, making it optimal in the Kalman Filter sense.

Mathematical Derivation:
    The optimal state has closed form:

        S_t = U_t · (H_t + λI)^(-1)

    where:
        U_t = Σ(i=1 to t) η_i v_i k_i^T    (cumulative value-key outer products)
        H_t = Σ(i=1 to t) η_i k_i k_i^T    (cumulative key-key covariance, Gram matrix)

    The output is computed as:

        y_t = S_t · q_t = U_t · (H_t + λI)^(-1) · q_t = U_t · x_t

    where x_t solves the ridge regression problem:

        (H_t + λI) · x = q_t

    This x_t represents q_t transformed by the inverse covariance of all past keys,
    accounting for correlations between keys rather than just individual similarities.

Ridge Regression Solver:
    The module uses Chebyshev Iteration as its solver, an iterative method using
    accelerated gradient descent with optimal convergence properties. Preferred
    for numerical stability in low-precision (bfloat16) arithmetic.

    Two modes are provided:
    - gka_chebyshev_gla: Parallel processing for entire sequences (training)
    - torch_decoding_one_step: Single-step processing with state management (inference)

Shape Notation:
    We use the following variables for shape hints:
        B: Batch size
        L: Sequence length
        N: Number of packed sequences for varlen inputs
        H: Number of attention heads
        D: Head dimension
"""


import torch
import triton
import triton.language as tl

from fla.ops.simple_gla import (
    chunk_simple_gla,
    fused_recurrent_simple_gla,
)
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp2

from .chebyshev_iteration import ChebyshevIteration

RCP_LN2_CONST = tl.constexpr(RCP_LN2)


@torch.compiler.disable
def latent_chebyshev(
    k, q, ridge_strength, num_iter, solver_type, bp_lambda, gk, chunk_size,
    cu_seqlens, cu_seqlens_cpu, solver_gain=None, solver_gain_mom=None,
    solver_gain_diag=None,
):
    """
    Dispatcher for ridge regression solvers: (H_t + λI)·x_t = q_t.

    Routes to different solver implementations based on solver_type. All solvers
    aim to solve the same problem but with different speed/accuracy tradeoffs.

    Args:
        k: Key tensor with shape [B, L, H, D]
        q: Query tensor with shape [B, L, H, D]
        ridge_strength: Regularization parameter λ
        num_iter: Number of iterations for iterative solvers
        solver_type: Solver method. Currently only supports 'chebyshev'.
        bp_lambda: Whether to backprop through λ (for Chebyshev solvers)
        gk: Forgetting gate for key-key covariance with shape [B, L, H]
        cu_seqlens: Cumulative sequence lengths with shape [N+1] for varlen inputs, or None
        cu_seqlens_cpu: Optional host copy of cu_seqlens

    Returns:
        x: Solution tensor with shape [B, L, H, D]
        kk_final: Final key-key covariance H_T with shape [N, H, D, D]
    """
    if solver_type == "chebyshev":
        x, h_kk, kk_final = ChebyshevIteration.apply(
            k, q, gk, num_iter, ridge_strength, bp_lambda, None, chunk_size,
            cu_seqlens, cu_seqlens_cpu, solver_gain, solver_gain_mom,
            solver_gain_diag,
        )
    else:
        raise ValueError(f"Unknown GKA solver type: {solver_type}")

    return x, h_kk, kk_final


def gka_chebyshev_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    gla_scale,
    ridge_strength: float = 0.02,
    solver_type: str = "chebyshev",
    num_iter: int = 30,
    bp_lambda: bool = True,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    solver_gain: torch.Tensor | None = None,
    solver_gain_mom: torch.Tensor | None = None,
    solver_gain_diag: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GKA Chebyshev solver for parallel training.

    ``solver_gain`` / ``solver_gain_mom`` are optional ``[H, num_iter]`` tables of
    learned multipliers on the Chebyshev step and on its momentum coefficient
    (see ``NeuralSolver`` in fla/layers/gka.py).  ``solver_gain_diag`` is an
    optional ``[H, num_iter, D]`` table of per-channel multipliers (exp2
    beta1L); it replaces the scalar step arm, so it requires
    ``solver_gain=None`` and a ``solver_gain_mom`` table.  All ``None`` = plain
    Chebyshev.

    Solves the ridge regression problem (H_t + λI)·x_t = q_t at each timestep,
    then computes output via GLA recurrence. Processes entire sequences in parallel.

    Args:
        q: Query tensor with shape [B, L, H, D]
        k: Key tensor with shape [B, L, H, D]
        v: Value tensor with shape [B, L, H, D]
        alpha: Mixing coefficient with shape  [B, L, H]. Controls interpolation: q_new = q + α·(x_t - q)
        g: Forgetting gate for key-value state with shape [B, L, H]
        gk: Forgetting gate for key-key covariance with shape [B, L, H]
        gla_scale: Scaling factor for GLA, typically head_dim^(-0.5)
        ridge_strength: Regularization parameter λ. Defaults to 0.02
        solver_type: Solver method. Currently only supports 'chebyshev' (Triton). Defaults to 'chebyshev'.
        num_iter: Iterations for iterative solvers. Set to 0 to skip ridge regression. Defaults to 30.
        bp_lambda: Whether to backprop through λ. Default True
        chunk_size: Chunk size for parallel processing. Defaults to 64
        cu_seqlens: Cumulative sequence lengths with shape [N+1] used for variable-length
            training, consistent with the FlashAttention API. `B` is expected to be 1 then.
        cu_seqlens_cpu: CPU copy of `cu_seqlens` to avoid unnecessary device synchronization.
            Defaults to None.

    Returns:
        o: Output tensor with shape [B, L, H, D]
        kk_final: Final key-key covariance (H_T in the regression problem) with shape
            [N, H, D, D], or None if num_iter=0.
        kv_final: Final key-value state from GLA with shape [N, H, D, D]

    Notes:
        For single-step autoregressive inference, use torch_decoding_one_step instead.
    """

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
            f"Please flatten variable-length inputs before processing.",
        )

    if num_iter > 0:
        # q_ solves (H_t + λI) · q_ = q_t
        q_, h_kk, kk_final = latent_chebyshev(
            k, q, ridge_strength, num_iter, solver_type, bp_lambda, gk, chunk_size,
            cu_seqlens, cu_seqlens_cpu, solver_gain, solver_gain_mom,
            solver_gain_diag,
        )

        q = q + alpha[..., None] * (q_ - q)  # Equivalent to ⍺q_ + (1-⍺)q
    else:
        kk_final = None

    o, kv_final = chunk_simple_gla(
        q=q, k=k, v=v, scale=gla_scale, g=g, output_final_state=True,
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
    )

    return o, h_kk, kk_final, kv_final


@triton.heuristics({
    "USE_GK": lambda args: args["gk_ptr"] is not None,
    "USE_GAIN": lambda args: args["gain_ptr"] is not None,
    "USE_GAIN_DIAG": lambda args: args["gain_diag_ptr"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.jit(do_not_specialize=["T"])
def chebyshev_iteration_fro_inference_forward_kernel(
    # inputs:
    prev_h_kk_ptr,
    k_ptr,
    q_ptr,
    gk_ptr,
    gain_ptr,
    gain_mom_ptr,
    gain_diag_ptr,
    # output:
    x_out_ptr,
    h_kk_ptr,
    cu_seqlens,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    # algorithm parameters:
    num_iter: tl.constexpr,
    ridge_ratio: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GAIN: tl.constexpr,
    USE_GAIN_DIAG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Triton kernel for single-step Chebyshev iteration during inference.

    Updates key-key covariance h_kk from previous state, then solves
    (h_kk + λI)·x = q using Chebyshev iteration. Optimized for autoregressive
    decoding where one token is processed at a time.

    ``gain_ptr`` / ``gain_mom_ptr`` point at ``[H, num_iter]`` fp32 tables of
    learned multipliers on the step and on the momentum coefficient; with
    ``USE_GAIN == 0`` the iteration is bit-identical to plain Chebyshev.  The
    tables are indexed by head only, so they are shared by all batch elements
    (and by all packed sequences under varlen).

    Grid: (B, H) - one thread block per batch element and head.
    For varlen inputs, dim 0 enumerates the `N` packed sequences, each holding a
    single token located through `cu_seqlens`, while the states stay indexed by
    the sequence id.
    Block size: Processes full D-dimensional vectors per block
    """
    i_n, h_id = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_t = tl.load(cu_seqlens + i_n).to(tl.int64)
    else:
        i_t = i_n.to(tl.int64) * T

    scalar_off = i_t * H + h_id

    range_D = tl.arange(0, D)

    vec_off = scalar_off * D
    q_ptr += vec_off + range_D
    k_ptr += vec_off + range_D
    x_out_ptr += vec_off + range_D

    q = tl.load(q_ptr)
    k = tl.load(k_ptr)

    prev_h_kk_ptr += (i_n * H + h_id).to(tl.int64) * D * D

    p_prev_h_kk = prev_h_kk_ptr + range_D[:, None] * D + range_D[None, :]
    h_kk = tl.load(p_prev_h_kk).to(tl.float32)

    if USE_GK:
        gk_ptr += scalar_off
        gk = tl.load(gk_ptr).to(tl.float32)
        gk_exp = exp2(gk * RCP_LN2_CONST)
    else:
        gk_exp = 1.0

    if USE_GAIN:
        # the tables are laid out as [H, num_iter]
        gain_ptr += h_id * num_iter
        gain_mom_ptr += h_id * num_iter
    if USE_GAIN_DIAG:
        # the diag table is laid out as [H, num_iter, D] (exp2 beta1L)
        gain_diag_ptr += h_id * num_iter * D
        gain_mom_ptr += h_id * num_iter

    h_kk_ = h_kk * gk_exp + k[:, None] * k[None, :]
    fro = tl.sqrt(tl.sum(h_kk_ * h_kk_))

    ridge_strength = ridge_ratio * fro

    stepsize = 2 / (2 * ridge_strength + fro)
    rho_sq_4 = fro / (2 * ridge_strength + fro) / 2.0
    rho_sq_4 = rho_sq_4 * rho_sq_4

    x_prev = tl.zeros_like(q).to(tl.float32)
    x = stepsize * q.to(tl.float32)
    omega = 2.0

    for i in range(num_iter):
        # `omega` is ALWAYS the pure Chebyshev recursion; the learned tables only
        # rescale the step and the momentum coefficient, exactly as in the chunked
        # training kernel (`chebyshev_iteration_fro_gating_forward_kernel`):
        #
        #     w = omega * gain[h, i] ;  mom = (w - 1) * gain_mom[h, i]
        #
        # and the coefficient of `x` is `1 + mom`, which is `w` only when the
        # momentum is tied (`gain_mom == 1`), so the untrained table and the whole
        # `USE_GAIN == 0` path stay bit-identical to the original kernel.
        omega = 1.0 / (1.0 - rho_sq_4 * omega)

        if USE_GAIN:
            w = omega * tl.load(gain_ptr + i)
            mom = (w - 1.0) * tl.load(gain_mom_ptr + i)
            wk = w + (mom - (w - 1.0))
        elif USE_GAIN_DIAG:
            # exp2 beta1L: per-channel gain replaces the scalar step arm; the
            # momentum follows the offline harness
            # (mom = (omega * gain_diag - 1) * gain_mom).  A table of exact
            # ones reproduces the scalar path bit-for-bit.
            d = tl.load(gain_diag_ptr + i * D + tl.arange(0, D))
            w = omega * d
            mom = (w - 1.0) * tl.load(gain_mom_ptr + i)
            wk = w + (mom - (w - 1.0))
        else:
            w = omega
            mom = omega - 1.0
            wk = w

        grad_and_x_prev = (
            stepsize
            * w
            * (
                tl.sum(h_kk * (x * gk_exp)[:, None], axis=0)
                + tl.sum(k[:, None] * k[None, :] * x[:, None], axis=0)
                + ridge_strength * x
                - q
            )
            + mom * x_prev
        )

        x_prev = x

        x = wk * x - grad_and_x_prev

    tl.store(x_out_ptr, x.to(x_out_ptr.dtype.element_ty))

    h_kk_ptr += (i_n * H + h_id).to(tl.int64) * D * D

    p_h_kk = h_kk_ptr + range_D[:, None] * D + range_D[None, :]
    tl.store(
        p_h_kk,
        h_kk_.to(p_h_kk.dtype.element_ty, fp_downcast_rounding="rtne"),
    )


def chebyshev_iteration_forward_triton_inference(
    prev_h_kk: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    gk: torch.Tensor,
    num_iter: int = 30,
    ridge_ratio: float = 0.1,
    cu_seqlens: torch.LongTensor | None = None,
    gain: torch.Tensor | None = None,
    gain_mom: torch.Tensor | None = None,
    gain_diag: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solves (h_kk + λI)·x = q for single-step inference using Triton-optimized Chebyshev iteration.

    Updates the key-key covariance state with the current key, then solves the ridge
    regression problem. Designed for autoregressive decoding (one token at a time).

    Args:
        prev_h_kk: Previous key-key covariance state with shape [B, H, D, D]
        k: Current key vector with shape [B, 1, H, D]
        q: Current query vector with shape [B, 1, H, D]
        gk: Forgetting gate for key-key covariance with shape [B, 1, H]
        num_iter: Number of Chebyshev iterations. Defaults to 30
        ridge_ratio: Base regularization parameter (scaled by ||h_kk||_F). Defaults to 0.1
        cu_seqlens: Cumulative sequence lengths with shape [N+1] for varlen inputs, or None.
            Each packed sequence holds exactly one token. Defaults to None
        gain: Optional ``[H, num_iter]`` fp32 multipliers on the Chebyshev step, one per
            (head, iteration); ``None`` = plain Chebyshev. Shared by all batch elements
            and by all packed sequences.
        gain_mom: Optional ``[H, num_iter]`` fp32 multipliers on the momentum coefficient
            ``(omega - 1)``, given together with ``gain``.
        gain_diag: Optional ``[H, num_iter, D]`` fp32 per-channel multipliers (exp2
            beta1L) replacing the scalar step arm; requires ``gain=None`` and a
            ``gain_mom`` table.

    Returns:
        x_out: Solution vector with shape [B, 1, H, D]
        h_kk: Updated key-key covariance state with shape [B, H, D, D]
    """
    B, L, H, D = q.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    assert (gain is None) == (gain_mom is None) or (gain is None and gain_diag is not None), \
        "gain and gain_mom must be given together; a diag table replaces the scalar step arm"
    if gain is not None:
        assert gain.shape == (H, num_iter) and gain_mom.shape == (H, num_iter), \
            f"gain tables must be [H, num_iter] = [{H}, {num_iter}], " \
            f"got {tuple(gain.shape)} and {tuple(gain_mom.shape)}"
    if gain_diag is not None:
        assert gain_mom is not None, "diag gain requires the scalar momentum table"
        assert gain_diag.shape == (H, num_iter, D), \
            f"gain_diag must be [H, num_iter, D] = [{H}, {num_iter}, {D}], got {tuple(gain_diag.shape)}"

    h_kk = torch.empty_like(prev_h_kk)

    grid = (N, H)

    x_out = torch.empty_like(q)

    chebyshev_iteration_fro_inference_forward_kernel[grid](
        # inputs:
        prev_h_kk,
        k,
        q,
        gk,
        gain,
        gain_mom,
        gain_diag,
        # output:
        x_out,
        h_kk,
        # varlen:
        cu_seqlens,
        # constants:
        L,
        H,
        D,
        # algorithm parameters:
        num_iter,
        ridge_ratio,
    )

    return x_out, h_kk


def torch_decoding_one_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    gla_scale: float,
    ridge_strength: float,
    solver_type: str,
    num_iter: int,
    prev_h_kk: torch.Tensor,
    prev_h_kv: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    solver_gain: torch.Tensor | None = None,
    solver_gain_mom: torch.Tensor | None = None,
    solver_gain_diag: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single-step GKA inference with state management for autoregressive decoding.

    Updates key-key covariance state, solves ridge regression, applies alpha mixing,
    and computes output via GLA. Processes one token at a time with explicit state passing.

    Args:
        q: Query tensor with shape [B, 1, H, D] or [1, H, D]
        k: Key tensor with shape [B, 1, H, D] or [1, H, D]
        v: Value tensor with shape [B, 1, H, D] or [1, H, D]
        alpha: Mixing coefficient with shape [B, 1, H] or [1, H]
        g: Forgetting gate for key-value state with shape [B, 1, H] or [1, H]
        gk: Forgetting gate for key-key covariance with shape [B, 1, H] or [1, H]
        gla_scale: Scaling factor for GLA, typically head_dim^(-0.5)
        ridge_strength: Regularization parameter λ
        solver_type: Solver method. Currently only supports 'chebyshev'.
        num_iter: Iterations for iterative solvers. Set to 0 to skip ridge regression
        prev_h_kk: Previous key-key covariance state with shape [B, H, D, D] or None
        prev_h_kv: Previous key-value state with shape [B, H, D, D] or None
        cu_seqlens: Cumulative sequence lengths with shape [N+1] used for variable-length
            decoding, consistent with the FlashAttention API. `B` is expected to be 1 then, and
            every packed sequence contributes exactly one token, so the packed length is `N`.
            Defaults to None.
        solver_gain: Optional ``[H, num_iter]`` fp32 table of learned multipliers on the
            Chebyshev step, matching the one used by ``gka_chebyshev_gla`` during training
            (see ``NeuralSolver`` in fla/layers/gka.py). ``None`` = plain Chebyshev.
        solver_gain_mom: Optional ``[H, num_iter]`` fp32 table of learned multipliers on the
            momentum coefficient ``(omega - 1)``, given together with ``solver_gain``.

    Returns:
        o: Output tensor with shape [B, 1, H, D]
        h_kk: Updated key-key covariance state with shape [N, H, D, D]
        h_kv: Updated key-value state with shape [N, H, D, D]

    Notes:
        Handles both batched [B, 1, H, D] and unbatched [1, H, D] inputs.
        For variable-length inputs, pass the packed [1, N, H, D] tensors together with
        `cu_seqlens`, one token per sequence.
        For parallel training, use gka_chebyshev_gla instead.
    """

    if len(q.shape) < 4:
        q = q[None, ...]
        k = k[None, ...]
        v = v[None, ...]

    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if q.shape[1] != N:
            raise ValueError(
                "A single decoding step consumes exactly one token per sequence, so the number of packed "
                f"tokens, i.e., {q.shape[1]}, is expected to be equal to the number of input sequences, "
                f"i.e., {N}.",
            )
        if prev_h_kk is not None and prev_h_kk.shape[0] != N:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {N} rather than {prev_h_kk.shape[0]}.",
            )

    q_ = q

    if num_iter > 0:
        if solver_type == "chebyshev":
            q_, h_kk = chebyshev_iteration_forward_triton_inference(
                prev_h_kk, k, q_, gk, num_iter=num_iter, ridge_ratio=ridge_strength,
                cu_seqlens=cu_seqlens, gain=solver_gain, gain_mom=solver_gain_mom,
                gain_diag=solver_gain_diag,
            )
        else:
            raise ValueError(f"Unknown GKA solver type: {solver_type}")

        q = q + alpha[..., None] * (q_ - q)
    else:
        h_kk = None

    o, h_kv = fused_recurrent_simple_gla(
        q=q,
        k=k,
        v=v,
        scale=gla_scale,
        g=g,
        initial_state=prev_h_kv,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    return o, h_kk, h_kv
