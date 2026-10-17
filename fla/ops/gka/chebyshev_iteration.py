
import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp2

from .chunk_dh_kk_to_dk_backward import chunk_dh_kk_to_dk
from .chunk_dh_kk_to_dk_gating_backward import chunk_dh_kk_to_dk_gating
from .chunk_dk_backward import chunk_bwd_dk
from .chunk_dk_backward_gating import chunk_bwd_dk_gating


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["D"],
)
@triton.jit(do_not_specialize=["T"])
def chebyshev_iteration_forward_kernel(
    # inputs:
    h_kk_ptr,
    k_ptr,
    q_ptr,
    # output:
    q_output_ptr,
    ridge_strength_ptr,
    rho_sq_4_ptr,
    stepsize_ptr,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    chunk_size: tl.constexpr,
    # algorithm parameters:
    num_iter: tl.constexpr,
):
    """
    Triton kernel for Chebyshev iteration solving (H_t + λI)·x = q in chunks.

    Grid: (num_chunks, B*H) - one thread block per chunk and batch-head pair
    Block size: Processes [chunk_size, D] vectors per block
    """

    chunk_id, bh_id = tl.program_id(0), tl.program_id(1)
    b_id, h_id = bh_id // H, bh_id % H

    num_chunks = tl.cdiv(T, chunk_size)

    # offsets
    scalar_off = b_id * T * H + h_id
    ridge_strength_ptr += scalar_off
    rho_sq_4_ptr += scalar_off
    stepsize_ptr += scalar_off

    vec_off = scalar_off * D
    q_ptr += vec_off
    q_output_ptr += vec_off
    k_ptr += vec_off

    h_kk_ptr += ((b_id * num_chunks + chunk_id) * H + h_id).to(tl.int64) * D * D

    o_t = chunk_id * chunk_size + tl.arange(0, chunk_size)
    m_t = o_t < T
    o_d = tl.arange(0, D)

    p_q = q_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    p_k = k_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    p_h_kk = h_kk_ptr + o_d[:, None] * D + o_d[None, :]

    q = tl.load(p_q, mask=m_t[:, None], other=0.0).to(tl.float32)
    k = tl.load(p_k, mask=m_t[:, None], other=0.0).to(q.dtype)
    h_kk = tl.load(p_h_kk).to(q.dtype)
    ridge_strength = tl.load(
        ridge_strength_ptr + o_t[:, None] * H, mask=m_t[:, None], other=0.0
    ).to(tl.float32)
    rho_sq_4 = tl.load(
        rho_sq_4_ptr + o_t[:, None] * H, mask=m_t[:, None], other=0.0
    ).to(tl.float32)
    stepsize = tl.load(
        stepsize_ptr + o_t[:, None] * H, mask=m_t[:, None], other=0.0
    ).to(tl.float32)

    normalizer = (
        1 / (tl.arange(1, chunk_size + 1) + chunk_id * chunk_size).to(q.dtype)
    )[
        :, None
    ]  # [chunk_size, 1]

    mask = tl.where(
        tl.arange(0, chunk_size)[:, None] >= tl.arange(0, chunk_size)[None, :],
        normalizer,
        0,
    ).to(q.dtype)

    w = tl.full((chunk_size, 1), 1, dtype=tl.float32)
    x_prev, x = tl.zeros_like(q), stepsize * q
    for i in range(num_iter):
        w = 1.0 / (1.0 - rho_sq_4 * w)

        grad_and_x_prev = (
            stepsize
            * w
            * (
                tl.dot(x * normalizer, h_kk)
                + tl.dot((tl.dot(x, tl.trans(k)) * mask), k)
                + ridge_strength * x
                - q
            )
            + (w - 1) * x_prev
        )  # [C, D]

        x_prev = x

        x = w * x - grad_and_x_prev

    p_q_output = q_output_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    tl.store(
        p_q_output,
        x.to(p_q_output.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=m_t[:, None],
    )


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["D", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def chebyshev_iteration_fro_forward_kernel(
    # inputs:
    h_kk_ptr,
    k_ptr,
    q_ptr,
    # output:
    x_out_ptr,
    fro_ptr,
    cu_seqlens,
    chunk_indices,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    chunk_size: tl.constexpr,
    # algorithm parameters:
    num_iter: tl.constexpr,
    ridge_ratio: tl.constexpr,
    load_fro: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Chebyshev iteration with adaptive ridge strength based on Frobenius norm.

    Computes or loads ||H_t||_F, sets λ = ridge_ratio * ||H_t||_F, then solves
    (H_t + λI)·x = q. Optionally saves the Frobenius norm for reuse.

    Grid: (num_chunks, B*H) - one thread block per chunk and batch-head pair.
    For varlen inputs, dim 0 enumerates the chunks of all sequences and
    `chunk_indices` maps each program to its sequence and in-sequence chunk.
    Block size: Processes [chunk_size, BD] vectors per block (D zero-padded to BD)
    """
    i_c, bh_id = tl.program_id(0), tl.program_id(1)
    b_id, h_id = bh_id // H, bh_id % H

    if IS_VARLEN:
        i_tg = i_c.to(tl.int64)
        i_n, i_c = (
            tl.load(chunk_indices + i_tg * 2).to(tl.int32),
            tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int64),
        )
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_tg = (b_id * tl.cdiv(T, chunk_size) + i_c).to(tl.int64)
        bos = (b_id * T).to(tl.int64)

    scalar_off = bos * H + h_id

    vec_off = scalar_off * D
    q_ptr += vec_off
    x_out_ptr += vec_off
    k_ptr += vec_off

    h_kk_ptr += (i_tg * H + h_id) * D * D

    o_t = i_c * chunk_size + tl.arange(0, chunk_size)
    m_t = o_t < T
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_tk = m_t[:, None] & m_d[None, :]
    m_kk = m_d[:, None] & m_d[None, :]

    p_q = q_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    p_k = k_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    p_h_kk = h_kk_ptr + o_d[:, None] * D + o_d[None, :]

    q = tl.load(p_q, mask=m_tk, other=0.0)
    k = tl.load(p_k, mask=m_tk, other=0.0)
    h_kk = tl.load(p_h_kk, mask=m_kk, other=0.0)

    mask = tl.where(
        tl.arange(0, chunk_size)[:, None] >= tl.arange(0, chunk_size)[None, :], 1, 0
    ).to(q.dtype)

    if load_fro == 1:
        fro_ptr += scalar_off
        fro = tl.load(
            fro_ptr + o_t[:, None] * H, mask=m_t[:, None], other=0.0
        )
    else:
        fro = tl.zeros((chunk_size, 1), dtype=tl.float32)
        h_kk_fro = tl.dot(k, tl.trans(k))  # CxC
        fro += tl.sum(
            tl.dot(mask, (h_kk_fro * h_kk_fro).to(q.dtype)) * mask,
            axis=1,
            keep_dims=True,
        )
        # h_kk grows with the accumulated history, so both the elementwise square and the
        # reduction have to leave fp16 to avoid inf in `fro`
        fro = tl.sqrt(
            tl.sum(h_kk.to(tl.float32) * h_kk.to(tl.float32))
            + 2 * tl.cumsum(tl.sum(k * tl.dot(k.to(h_kk.dtype), h_kk), axis=1))[:, None]
            + fro
        )

    ridge_strength = ridge_ratio * fro

    stepsize = 2 / (2 * ridge_strength + fro)
    rho_sq_4 = fro / (2 * ridge_strength + fro) / 2.0
    rho_sq_4 = rho_sq_4 * rho_sq_4

    x_prev = tl.zeros_like(q).to(tl.float32)
    x = stepsize * q.to(tl.float32)
    omega = tl.full((chunk_size, 1), 2.0, dtype=tl.float32)

    for _ in range(num_iter):
        omega = 1.0 / (1.0 - rho_sq_4 * omega)

        grad_and_x_prev = (
            stepsize
            * omega
            * (
                tl.dot(x.to(h_kk.dtype), h_kk)
                + tl.dot((tl.dot(x.to(k.dtype), tl.trans(k)) * mask).to(k.dtype), k)
                + ridge_strength * x
                - q
            )
            + (omega - 1) * x_prev
        )  # [C, D]
        x_prev = x

        x = omega * x - grad_and_x_prev

    # store results
    p_x_out = x_out_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    tl.store(
        p_x_out,
        x.to(p_x_out.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=m_tk,
    )

    if load_fro == 0:
        fro_ptr += scalar_off
        tl.store(
            fro_ptr + o_t[:, None] * H,
            fro.to(fro_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=m_t[:, None],
        )


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_GAIN': lambda args: args['gain_ptr'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["D", "IS_VARLEN", "USE_GAIN"],
)
@triton.jit(do_not_specialize=["T"])
def chebyshev_iteration_fro_gating_forward_kernel(
    # inputs:
    h_kk_ptr,
    k_ptr,
    q_ptr,
    gkk_ptr,
    gain_ptr,
    gain_mom_ptr,
    # output:
    x_out_ptr,
    fro_ptr,
    cu_seqlens,
    chunk_indices,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    chunk_size: tl.constexpr,
    # algorithm parameters:
    num_iter: tl.constexpr,
    ridge_ratio: tl.constexpr,
    load_fro: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GAIN: tl.constexpr,
):
    """
    Chebyshev iteration with gating and adaptive ridge strength.

    ``gain_ptr`` / ``gain_mom_ptr`` point at ``[H, num_iter]`` fp32 tables of
    learned multipliers on the step and on the momentum coefficient; with
    ``USE_GAIN == 0`` the iteration is bit-identical to plain Chebyshev.

    Like chebyshev_iteration_fro_forward_kernel but applies exponential gating
    (exp2) to key interactions for temporal decay, consuming `gkk` in log2 space.
    Solves (H_t + λI)·x = q
    with λ = ridge_ratio * ||H_t||_F.

    Grid: (num_chunks, B*H) - one thread block per chunk and batch-head pair.
    For varlen inputs, dim 0 enumerates the chunks of all sequences and
    `chunk_indices` maps each program to its sequence and in-sequence chunk.
    Block size: Processes [chunk_size, BD] vectors per block (D zero-padded to BD)
    """
    i_c, bh_id = tl.program_id(0), tl.program_id(1)
    b_id, h_id = bh_id // H, bh_id % H

    if IS_VARLEN:
        i_tg = i_c.to(tl.int64)
        i_n, i_c = (
            tl.load(chunk_indices + i_tg * 2).to(tl.int32),
            tl.load(chunk_indices + i_tg * 2 + 1).to(tl.int64),
        )
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_tg = (b_id * tl.cdiv(T, chunk_size) + i_c).to(tl.int64)
        bos = (b_id * T).to(tl.int64)

    scalar_off = bos * H + h_id
    gkk_ptr += scalar_off

    vec_off = scalar_off * D
    q_ptr += vec_off
    x_out_ptr += vec_off
    k_ptr += vec_off

    h_kk_ptr += (i_tg * H + h_id) * D * D

    o_t = i_c * chunk_size + tl.arange(0, chunk_size)
    m_t = o_t < T
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_tk = m_t[:, None] & m_d[None, :]
    m_kk = m_d[:, None] & m_d[None, :]

    if USE_GAIN:
        gain_ptr += h_id * num_iter
        gain_mom_ptr += h_id * num_iter

    gkk = tl.load(gkk_ptr + o_t * H, mask=m_t, other=0.0).to(tl.float32)

    q = tl.load(q_ptr + o_t[:, None] * (H * D) + o_d[None, :],
                mask=m_tk, other=0.0)
    k = tl.load(k_ptr + o_t[:, None] * (H * D) + o_d[None, :],
                mask=m_tk, other=0.0)
    h_kk = tl.load(h_kk_ptr + o_d[:, None] * D + o_d[None, :], mask=m_kk, other=0.0)

    mask = tl.where(
        (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :]),
        exp2(gkk[:, None] - gkk[None, :]),
        0,
    ).to(q.dtype)

    exp_gkk = exp2(gkk[:, None])

    if load_fro == 1:
        fro_ptr += scalar_off
        fro = tl.load(
            fro_ptr + o_t[:, None] * H, mask=m_t[:, None], other=0.0
        )
    else:
        fro = tl.zeros((chunk_size, 1), dtype=tl.float32)
        h_kk_fro = tl.dot(k, tl.trans(k))  # CxC
        fro += tl.sum(
            tl.dot(mask, (h_kk_fro * h_kk_fro).to(q.dtype)) * mask,
            axis=1,
            keep_dims=True,
        )

        # h_kk grows with the accumulated history, so both the elementwise square and the
        # reduction have to leave fp16 to avoid inf in `fro`
        fro += exp_gkk * (exp_gkk * tl.sum(h_kk.to(tl.float32) * h_kk.to(tl.float32)))
        fro = tl.sqrt(
            fro
            + 2
            * exp_gkk
            * tl.sum(
                mask * tl.sum(k * tl.dot(k.to(h_kk.dtype), h_kk), axis=1)[None, :],
                axis=1,
                keep_dims=True,
            )
        )

    ridge_strength = ridge_ratio * fro

    stepsize = 2 / (2 * ridge_strength + fro)
    rho_sq_4 = fro / (2 * ridge_strength + fro) / 2.0
    rho_sq_4 = rho_sq_4 * rho_sq_4

    x_prev = tl.zeros_like(q).to(tl.float32)
    x = stepsize * q.to(tl.float32)
    omega = tl.full((chunk_size, 1), 2.0, dtype=tl.float32)

    for i in range(num_iter):
        # `omega` is ALWAYS the pure Chebyshev recursion; the learned tables only
        # rescale the step and the momentum coefficient of the *same* three-term
        # expression below.
        #
        # The reference form (fla/layers/gka.py::NeuralSolver, and
        # exp/torch_solver.py::run_iteration) is
        #
        #     x <- x + w * s * (q - A x) + mom * (x - x_prev)
        #
        # i.e. the coefficient of `x` is ``1 + mom``, *not* ``w``.  Plain
        # Chebyshev happens to satisfy ``1 + mom == w`` because its momentum is
        # tied (``mom == w - 1``), but a learned `phi != 0` breaks that tie, so
        # the x-coefficient has to be ``wk`` below.  ``wk`` reduces to exactly
        # ``w`` when the tie holds (``mom - (w - 1) == 0.0`` *exactly* when
        # ``gain_mom == 1``, because ``mom`` is computed as ``(w-1)*1.0``), so
        # the untrained table and the whole ``USE_GAIN == 0`` path stay
        # bit-identical to the original kernel.
        omega = 1.0 / (1.0 - rho_sq_4 * omega)

        if USE_GAIN:
            #     omega_eff_k = omega_k * gain[h, k]
            #     mom_k       = (omega_eff_k - 1) * gain_mom[h, k]
            w = omega * tl.load(gain_ptr + i)
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
                tl.dot((x * exp_gkk).to(h_kk.dtype), h_kk)
                + tl.dot((tl.dot(x.to(k.dtype), tl.trans(k)) * mask).to(k.dtype), k)
                + ridge_strength * x
                - q
            )
            + mom * x_prev
        )
        x_prev = x
        x = wk * x - grad_and_x_prev

    # store results
    p_x_out = x_out_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    tl.store(
        p_x_out,
        x.to(p_x_out.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=m_tk,
    )

    if load_fro == 0:
        fro_ptr += scalar_off
        tl.store(
            fro_ptr + o_t[:, None] * H,
            fro.to(fro_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=m_t[:, None],
        )


def chebyshev_iteration_forward_triton(
    k: torch.Tensor,
    q: torch.Tensor,
    gkk: torch.Tensor,
    fro: torch.Tensor,
    num_iter: int = 30,
    ridge_ratio: float = 0.02,
    chunk_size: int = 64,
    load_fro: int = 0,
    output_final_state: bool = False,
    h0=None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    gain: torch.Tensor | None = None,
    gain_mom: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solves (H_t + λI)·x = q using Chebyshev iteration with chunked processing.

    Computes chunk-level key-key covariance h_kk, then dispatches to appropriate
    Triton kernel (with or without gating) to solve the ridge regression problem.

    Args:
        k: Key tensor with shape [B, L, H, D]
        q: Query tensor with shape [B, L, H, D]
        gkk: Cumulative gating for key-key covariance with shape [B, L, H], in log2 space, or None
        fro: Frobenius norm tensor with shape [B, L, H] (computed if load_fro=0)
        num_iter: Number of Chebyshev iterations. Defaults to 30
        ridge_ratio: Base regularization parameter. Defaults to 0.02
        chunk_size: Chunk size for parallel processing. Defaults to 64
        load_fro: Whether to load (1) or compute (0) Frobenius norm. Defaults to 0
        output_final_state: Whether to return final h_kk state. Defaults to False
        cu_seqlens: Cumulative sequence lengths with shape [N+1] for varlen inputs, or None
        chunk_indices: Chunk indices prepared from cu_seqlens, computed if None
        gain: optional ``[H, num_iter]`` fp32 multipliers on the Chebyshev step,
            one per (head, iteration); ``None`` = plain Chebyshev
        gain_mom: optional ``[H, num_iter]`` fp32 multipliers on the momentum
            coefficient ``(omega - 1)``

    Returns:
        x_out: Solution tensor with shape [B, L, H, D]
        fro: Frobenius norm tensor with shape [B, L, H]
        h_kk: Chunk-level key-key covariance with shape [B, num_chunks, H, D, D]
            (num_chunks counts the chunks of all sequences for varlen inputs)
        h_kk_final: Final state with shape [N, H, D, D], or None if output_final_state=False
    """
    h_kk, h_kk_final = chunk_fwd_h(
        k=k,
        v=k,
        g=gkk,
        gk=None,
        gv=None,
        h0=h0,  # initial_state
        output_final_state=output_final_state,
        states_in_fp32=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    B, T, H, D = q.shape

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    num_chunks = triton.cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)
    grid = (num_chunks, H * B)
    # the D x D state tiles are zero-padded to the next power of 2 so that
    # non-power-of-2 head dims can use tl.arange(0, BD)
    BD = triton.next_power_of_2(D)

    x_out = torch.empty_like(q)

    assert (gain is None) == (gain_mom is None), \
        "gain and gain_mom must be given together"
    if gain is not None:
        assert gkk is not None, "learned gains are only supported with gating"
        assert gain.shape == (H, num_iter) and gain_mom.shape == (H, num_iter), \
            f"gain tables must be [H, num_iter] = [{H}, {num_iter}], got {tuple(gain.shape)}"

    if gkk is None:
        chebyshev_iteration_fro_forward_kernel[grid](
            # inputs:
            h_kk_ptr=h_kk,
            k_ptr=k,
            q_ptr=q,
            # output:
            x_out_ptr=x_out,
            fro_ptr=fro,  # omega_last,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            # constants:
            T=T,
            H=H,
            D=D,
            BD=BD,
            chunk_size=chunk_size,
            # algorithm parameters:
            num_iter=num_iter,
            ridge_ratio=ridge_ratio,
            load_fro=load_fro,
        )
    else:
        chebyshev_iteration_fro_gating_forward_kernel[grid](
            # inputs:
            h_kk_ptr=h_kk,
            k_ptr=k,
            q_ptr=q,
            gkk_ptr=gkk,
            gain_ptr=gain,
            gain_mom_ptr=gain_mom,
            # output:
            x_out_ptr=x_out,
            fro_ptr=fro,  # omega_last,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            # constants:
            T=T,
            H=H,
            D=D,
            BD=BD,
            chunk_size=chunk_size,
            # algorithm parameters:
            num_iter=num_iter,
            ridge_ratio=ridge_ratio,
            load_fro=load_fro,
        )

    return x_out, fro, h_kk, h_kk_final


class ChebyshevIteration(torch.autograd.Function):
    """
    Autograd function for Chebyshev iteration solving (H_t + λI)·x = q.

    The forward pass solves the ridge regression problem and computes final key-key state.
    The backward pass reuses the forward solver on gradients for efficient backpropagation.
    Supports packed variable-length inputs via `cu_seqlens`.
    """

    @staticmethod
    def forward(
        ctx,
        k: torch.Tensor,
        q: torch.Tensor,
        gk: torch.Tensor | None = None,
        num_iter: int = 30,
        ridge_ratio: float = 0.02,
        bp_lambda: bool = True,
        h0=None,
        chunk_size: int = 64,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        gain: torch.Tensor | None = None,
        gain_mom: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, D = q.shape
        fro = torch.empty((B, T, H), dtype=torch.float32).to(q.device)
        chunk_indices = None
        if cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(
                cu_seqlens,
                chunk_size,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )
        gkk = None
        if gk is not None:
            gkk = chunk_local_cumsum(
                gk,
                chunk_size=chunk_size,
                scale=RCP_LN2,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )

        x_out, fro, h_kk, h_kk_final = chebyshev_iteration_forward_triton(
            k,
            q,
            gkk,
            fro,
            num_iter=num_iter,
            ridge_ratio=ridge_ratio,
            chunk_size=chunk_size,
            load_fro=0,
            output_final_state=True,
            h0=h0,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            gain=gain,
            gain_mom=gain_mom,
        )

        ctx.save_for_backward(k, x_out, fro, gkk, h0, chunk_indices)
        # The backward solves the transposed system with the *same* operator, so
        # it must use the same coefficients.  detach(): the tables are learned
        # by the auxiliary loss inside fla/layers/gka.py, not through the
        # kernel's (implicit-differentiation) backward.
        ctx.gain = None if gain is None else gain.detach()
        ctx.gain_mom = None if gain_mom is None else gain_mom.detach()
        ctx.num_iter = num_iter
        ctx.chunk_size = chunk_size
        ctx.ridge_ratio = ridge_ratio
        ctx.bp_lambda = bp_lambda
        ctx.cu_seqlens = cu_seqlens

        return x_out, h_kk, h_kk_final

    @staticmethod
    def backward(
        ctx, dx_out: torch.Tensor, d_h_kk: torch.Tensor | None = None, d_kk_final: torch.Tensor | None = None
    ) -> tuple[
        torch.Tensor | None,   # dk
        torch.Tensor | None,   # dq
        torch.Tensor | None,   # dgk
        None,                  # num_iter
        None,                  # ridge_ratio
        None,                  # bp_lambda
        torch.Tensor | None,   # dh0
        None,                  # chunk_size
        None,                  # cu_seqlens
        None,                  # cu_seqlens_cpu
        None,                  # gain      (trained by the aux loss, not here)
        None,                  # gain_mom
    ]:
        k, x_out, fro, gkk, h0, chunk_indices = ctx.saved_tensors
        num_iter = ctx.num_iter
        ridge_ratio = ctx.ridge_ratio
        chunk_size = ctx.chunk_size
        bp_lambda = ctx.bp_lambda
        cu_seqlens = ctx.cu_seqlens

        dq, fro, h_kk, _ = chebyshev_iteration_forward_triton(
            k,
            dx_out,
            gkk,
            fro,
            num_iter=num_iter,
            ridge_ratio=ridge_ratio,
            chunk_size=chunk_size,
            load_fro=1,
            output_final_state=False,
            h0=h0,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            gain=ctx.gain,
            gain_mom=ctx.gain_mom,
        )

        h_kk_final = None

        d_kk, dh0 = chunk_bwd_dh(
            q=dq,
            k=k,
            v=k,
            do=x_out,
            h0=h0,
            # The dk kernels below consume `d_kk[c]` as the negated transpose of
            # `dL/dh_kk[c+1]` (the solve path accumulates `dq⊗x`), so the incoming
            # final-state gradient must be injected with a flipped sign.
            dht=-d_kk_final if d_kk_final is not None else None,
            g=gkk,
            gk=None,
            gv=None,
            chunk_size=chunk_size,
            scale=1,
            cu_seqlens=cu_seqlens,
        )

        if gkk is None:
            if bp_lambda:
                dk = chunk_bwd_dk(
                    h_kk,
                    d_kk,
                    k,
                    x_out,
                    dq,
                    fro,
                    h_kk_final,
                    chunk_size=chunk_size,
                    ridge_ratio=ridge_ratio,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                )
            else:
                dk = chunk_dh_kk_to_dk(
                    k, x_out, d_kk, dq, ctx.chunk_size,
                    cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                )

            dgk = None
        else:  # use gating
            if bp_lambda:
                dk, dgk = chunk_bwd_dk_gating(
                    h_kk,
                    d_kk,
                    k,
                    x_out,
                    dq,
                    fro,
                    gkk,
                    chunk_size=chunk_size,
                    ridge_ratio=ridge_ratio,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                )
            else:
                dk, dgk = chunk_dh_kk_to_dk_gating(
                    k, x_out, h_kk, d_kk, dq, gkk, ctx.chunk_size,
                    cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                )

        return dk, dq, dgk, None, None, None, dh0, None, None, None, None, None
