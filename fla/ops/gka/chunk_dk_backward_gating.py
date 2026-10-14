
import torch
import triton
import triton.language as tl
from torch import Tensor

from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp2


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_dk_gating_kernel(
    # inputs:
    h_kk_ptr,
    d_kk_ptr,
    x_out_ptr,
    dq_ptr,
    k_ptr,
    fro_ptr,
    gkk_ptr,
    # outputs:
    h_kk_suffix_sum_partial_ptr,
    dk_ptr,
    dgkk_ptr,
    cu_seqlens,
    chunk_indices,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    chunk_size: tl.constexpr,
    # algorithm parameters:
    ridge_ratio: tl.constexpr,
    FLOOR_FRO: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Computes key and gating gradients with adaptive ridge strength (λ = ridge_ratio * ||H_t||_F).

    First pass of two-pass backward: computes per-chunk gradients and partial suffix sums
    for inter-chunk gradient propagation. Accounts for gating and adaptive regularization.

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
    fro_ptr += scalar_off
    gkk_ptr += scalar_off
    dgkk_ptr += scalar_off

    vec_off = scalar_off * D
    x_out_ptr += vec_off
    dq_ptr += vec_off
    k_ptr += vec_off
    dk_ptr += vec_off

    d_kk_ptr += (i_tg * H + h_id) * D * D
    h_kk_ptr += (i_tg * H + h_id) * D * D
    h_kk_suffix_sum_partial_ptr += (i_tg * H + h_id) * D * D

    chunk_range = tl.arange(0, chunk_size)
    o_t = i_c * chunk_size + chunk_range
    m_t = o_t < T
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_tk = m_t[:, None] & m_d[None, :]
    m_kk = m_d[:, None] & m_d[None, :]

    p_dq = dq_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    p_x_out = x_out_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    p_k = k_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    p_fro = fro_ptr + o_t * H
    p_gkk = gkk_ptr + o_t * H

    p_d_kk = d_kk_ptr + o_d[:, None] * D + o_d[None, :]
    p_h_kk = h_kk_ptr + o_d[:, None] * D + o_d[None, :]

    k = tl.load(p_k, mask=m_tk, other=0.0)
    dq = tl.load(p_dq, mask=m_tk, other=0.0)
    x_out = tl.load(p_x_out, mask=m_tk, other=0.0)
    gkk = tl.load(p_gkk, mask=m_t, other=0.0).to(tl.float32)

    bool_mask = (chunk_range[:, None] <= chunk_range[None, :]) & m_t[:, None] & m_t[None, :]
    mask = tl.where(bool_mask, exp2(gkk[None, :] - gkk[:, None]), 0)
    gkk_last = tl.sum(tl.where(chunk_range == tl.minimum(chunk_size - 1, T - i_c * chunk_size - 1), gkk, 0))

    s = tl.dot(k, tl.trans(x_out)) * mask
    ds = tl.dot(k, tl.trans(dq))

    dm = tl.where(bool_mask, s * ds, 0)
    dgkk = tl.zeros([chunk_size], dtype=tl.float32)
    dgkk += tl.sum(dm, axis=0)
    dgkk -= tl.sum(dm, axis=1)

    d_kk = tl.load(p_d_kk, mask=m_kk, other=0.0)
    dk = tl.zeros([chunk_size, BD], dtype=tl.float32)
    k2 = k * exp2(gkk_last - gkk)[:, None].to(k.dtype)
    dk += tl.dot(k2, (d_kk + tl.trans(d_kk)).to(k.dtype))

    dk += tl.dot(s.to(dq.dtype), dq)
    dk += tl.dot((ds * mask).to(x_out.dtype), x_out)

    h_kk = tl.load(p_h_kk, mask=m_kk, other=0.0)
    dgkk += tl.sum(tl.dot(dq, h_kk) * exp2(gkk)[:, None] * x_out, axis=1)

    dk2 = tl.dot(k2, d_kk.to(k.dtype))
    dgkk_last = tl.sum(dk2 * k, axis=1)
    dgkk -= dgkk_last
    dgkk_last = tl.sum(dgkk_last)
    dgkk_last += tl.sum(d_kk.to(tl.float32) * h_kk.to(tl.float32)) * exp2(gkk_last)

    dgkk = tl.where(chunk_range < tl.minimum(chunk_size - 1, T - i_c * chunk_size - 1), dgkk, dgkk + dgkk_last)

    fro = tl.load(p_fro, mask=m_t, other=0.0).to(tl.float32)
    # `dq`/`x_out` can be fp16, and a bare tl.sum on an fp16 tensor accumulates
    # (and therefore saturates at 65504) in fp16 -- reduce this ridge adjoint in fp32.
    tmp = ridge_ratio * tl.sum(dq.to(tl.float32) * x_out.to(tl.float32), axis=1)
    # `fro` is the *raw* norm; the forward floored it before forming lambda, and
    # max(fro, c) has zero derivative below c -- so the whole lambda adjoint must
    # vanish on the floored rows (previously the un-floored formula was applied
    # there, amplified by 1 / FLOOR_FRO).
    tmp = tl.where(fro >= FLOOR_FRO, tmp, 0.0)
    dgkk += tmp * fro

    w = 2 * tmp / fro
    w = tl.where(o_t < T, w, 0.0)

    weights1 = tl.sum(mask * (exp2(gkk) * w)[None, :], axis=1)
    weights2 = tl.dot(mask * w[None, :], tl.trans(mask))  # * tl.trans(mask)

    tmp = tl.dot((tl.dot(k, tl.trans(k)) * weights2).to(k.dtype), k).to(tl.float32)
    tmp += tl.dot((weights1[:, None] * k).to(k.dtype), h_kk.to(k.dtype))

    dk += tmp
    dgkk -= tl.sum(tmp * k, axis=1) * 0.5

    h_kk_suffix_sum_partial = tl.sum(exp2(2 * gkk) * w) * h_kk
    h_kk_suffix_sum_partial += tl.dot(tl.trans((weights1[:, None] * k).to(k.dtype)), k)

    p_dk = dk_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    p_h_kk_suffix_sum_partial = h_kk_suffix_sum_partial_ptr + o_d[:, None] * D + o_d[None, :]

    tl.store(
        p_h_kk_suffix_sum_partial,
        h_kk_suffix_sum_partial.to(p_h_kk_suffix_sum_partial.dtype.element_ty),
        mask=m_kk,
    )
    tl.store(p_dk, (-dk).to(p_dk.dtype.element_ty), mask=m_tk)

    p_dgkk = dgkk_ptr + o_t * H
    tl.store(p_dgkk, dgkk.to(p_dgkk.dtype.element_ty), mask=m_t)


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
def chunk_bwd_dk_gating_kernel_forloop(
    # inputs:
    h_kk_ptr,
    h_kk_suffix_sum_partial_ptr,
    k_ptr,
    gkk_ptr,
    # outputs:
    dk2_ptr,
    dgkk2_ptr,
    cu_seqlens,
    split_offsets,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    chunk_size: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Accumulates suffix sums and computes inter-chunk gradient contributions.

    Second pass of two-pass backward: processes chunks sequentially in reverse order,
    accumulating h_kk_suffix_sum to compute gradients from later timesteps.

    Grid: (N, H) - one thread block per sequence (batch element when not varlen) and
    head, looping over that sequence's chunks.
    Block size: Processes [chunk_size, BD] vectors per block (D zero-padded to BD)
    """
    b_id, h_id = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + b_id).to(tl.int64), tl.load(cu_seqlens + b_id + 1).to(tl.int64)
        T = eos - bos
        num_chunks = tl.cdiv(T, chunk_size)
        # per-sequence starting slot in the packed [1, total_chunks, H, D, D] state buffer
        chunk_start = tl.load(split_offsets + b_id).to(tl.int64)
    else:
        num_chunks = tl.cdiv(T, chunk_size)
        chunk_start = (b_id * num_chunks).to(tl.int64)
        bos = (b_id * T).to(tl.int64)

    scalar_off = bos * H + h_id
    gkk_ptr += scalar_off
    dgkk2_ptr += scalar_off

    vec_off = scalar_off * D
    k_ptr += vec_off
    dk2_ptr += vec_off

    chunk_range = tl.arange(0, chunk_size)
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_kk = m_d[:, None] & m_d[None, :]

    h_kk_suffix_sum = tl.zeros((BD, BD), dtype=tl.float32)
    for chunk_id in range(num_chunks - 1, -1, -1):
        # load dq and x_out
        o_t = chunk_id * chunk_size + chunk_range
        m_t = o_t < T
        m_tk = m_t[:, None] & m_d[None, :]
        p_k = k_ptr + o_t[:, None] * (H * D) + o_d[None, :]
        p_h_kk_suffix_sum_partial = (
            h_kk_suffix_sum_partial_ptr
            + ((chunk_start + chunk_id) * H + h_id) * D * D
            + o_d[:, None] * D
            + o_d[None, :]
        )
        h_kk_suffix_sum_partial = tl.load(p_h_kk_suffix_sum_partial, mask=m_kk, other=0.0)

        p_h_kk = (
            h_kk_ptr
            + ((chunk_start + chunk_id) * H + h_id) * D * D
            + o_d[:, None] * D
            + o_d[None, :]
        )
        h_kk = tl.load(p_h_kk, mask=m_kk, other=0.0)

        p_gkk = gkk_ptr + o_t * H
        gkk = tl.load(p_gkk, mask=m_t, other=0.0).to(tl.float32)
        k = tl.load(p_k, mask=m_tk, other=0.0)

        gkk_last = tl.sum(tl.where(chunk_range == tl.minimum(chunk_size - 1, T - chunk_id * chunk_size - 1), gkk, 0))
        gkk_last_exp = exp2(gkk_last)

        dk2 = -tl.dot(
            (k * exp2(gkk_last - gkk)[:, None]).to(k.dtype), h_kk_suffix_sum.to(k.dtype)
        )

        dgkk2 = tl.sum(dk2.to(tl.float32) * k, axis=1) * 0.5

        h_kk_suffix_sum *= gkk_last_exp
        dgkk2_last = tl.sum(h_kk_suffix_sum * h_kk) * 0.5
        dgkk2_last -= tl.sum(dgkk2)

        dgkk2 = tl.where(chunk_range < tl.minimum(chunk_size - 1, T - chunk_id * chunk_size - 1), dgkk2, dgkk2 + dgkk2_last)

        h_kk_suffix_sum = h_kk_suffix_sum + h_kk_suffix_sum_partial

        p_dk2 = dk2_ptr + o_t[:, None] * (H * D) + o_d[None, :]
        tl.store(p_dk2, dk2.to(p_dk2.dtype.element_ty), mask=m_tk)

        p_dgkk2 = dgkk2_ptr + o_t * H
        tl.store(p_dgkk2, dgkk2.to(p_dgkk2.dtype.element_ty), mask=m_t)


def chunk_bwd_dk_gating(
    h_kk: Tensor,
    d_kk: Tensor,
    k: Tensor,
    x_out: Tensor,
    dq: Tensor,
    fro: Tensor,
    gkk: Tensor,
    chunk_size: int = 64,
    ridge_ratio: float = 0.02,
    fro_floor: float = 1e-2,
    cu_seqlens: Tensor | None = None,
    chunk_indices: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Computes key and gating gradients with adaptive ridge strength.

    Two-pass algorithm: chunk_bwd_dk_gating_kernel computes per-chunk gradients
    in parallel, then chunk_bwd_dk_gating_kernel_forloop accumulates suffix sums
    sequentially for inter-chunk dependencies.

    Args:
        h_kk: Chunk-level key-key covariance with shape [B, num_chunks, H, D, D]
            (num_chunks counts the chunks of all sequences for varlen inputs)
        d_kk: Gradient of chunk-level covariance with shape [B, num_chunks, H, D, D]
        k: Key tensor with shape [B, L, H, D]
        x_out: Output from forward pass with shape [B, L, H, D]
        dq: Gradient of query with shape [B, L, H, D]
        fro: Frobenius norm tensor with shape [B, L, H]
        gkk: Cumulative gating values with shape [B, L, H], in log2 space
        chunk_size: Chunk size for parallel processing. Defaults to 64
        ridge_ratio: Base regularization parameter. Defaults to 0.02
        cu_seqlens: Cumulative sequence lengths with shape [N+1] for varlen inputs, or None
        chunk_indices: Chunk indices prepared from cu_seqlens, computed if None

    Returns:
        dk: Key gradient with shape [B, L, H, D]
        dgkk: Gating gradient with shape [B, L, H]
    """

    B, T, H, D = k.shape

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the number of sequences; num_chunks: total chunk slots of all sequences
    if cu_seqlens is None:
        N, num_chunks, chunk_offsets = B, triton.cdiv(T, chunk_size), None
    else:
        N, num_chunks = len(cu_seqlens) - 1, len(chunk_indices)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
    BD = triton.next_power_of_2(D)

    grid = (num_chunks, B * H)
    dk = torch.empty_like(k)
    h_kk_suffix_sum_partial = torch.empty_like(h_kk)

    dgkk = torch.empty([B, T, H]).to(k.device)

    chunk_bwd_dk_gating_kernel[grid](
        # inputs:
        h_kk_ptr=h_kk,
        d_kk_ptr=d_kk,
        x_out_ptr=x_out,
        dq_ptr=dq,
        k_ptr=k,
        fro_ptr=fro,
        gkk_ptr=gkk,
        # outputs:
        h_kk_suffix_sum_partial_ptr=h_kk_suffix_sum_partial,
        dk_ptr=dk,
        dgkk_ptr=dgkk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        # constants:
        T=T,
        H=H,
        D=D,
        BD=BD,
        chunk_size=chunk_size,
        ridge_ratio=ridge_ratio,
        FLOOR_FRO=fro_floor,
    )

    dk2 = torch.empty_like(k)
    dgkk2 = torch.empty_like(dgkk)
    chunk_bwd_dk_gating_kernel_forloop[(N, H)](
        # inputs:
        h_kk_ptr=h_kk,
        h_kk_suffix_sum_partial_ptr=h_kk_suffix_sum_partial,
        k_ptr=k,
        gkk_ptr=gkk,
        # inputs & outputs:
        dk2_ptr=dk2,
        dgkk2_ptr=dgkk2,
        cu_seqlens=cu_seqlens,
        split_offsets=chunk_offsets,
        # constants:
        T=T,
        H=H,
        D=D,
        BD=BD,
        chunk_size=chunk_size,
    )
    dk.add_(dk2)
    dgkk.add_(dgkk2)
    dgkk = chunk_local_cumsum(
        -dgkk,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    return dk, dgkk
