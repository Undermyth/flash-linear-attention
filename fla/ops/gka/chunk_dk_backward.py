# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl
from torch import Tensor

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets


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
def chunk_bwd_dk_kernel(
    # inputs:
    h_kk_ptr,
    d_kk_ptr,
    x_out_ptr,
    dq_ptr,
    k_ptr,
    fro_ptr,
    # outputs:
    h_kk_suffix_sum_partial_ptr,
    dk_ptr,
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
    Computes key gradients with adaptive ridge strength (no gating).

    First pass of two-pass backward: computes per-chunk gradients and partial suffix sums
    for inter-chunk gradient propagation. Non-gated version of chunk_bwd_dk_gating_kernel.

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

    vec_off = scalar_off * D
    x_out_ptr += vec_off
    dq_ptr += vec_off
    k_ptr += vec_off
    dk_ptr += vec_off

    d_kk_ptr += (i_tg * H + h_id) * D * D
    h_kk_ptr += (i_tg * H + h_id) * D * D
    h_kk_suffix_sum_partial_ptr += (i_tg * H + h_id) * D * D

    o_t = i_c * chunk_size + tl.arange(0, chunk_size)
    m_t = o_t < T
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_tk = m_t[:, None] & m_d[None, :]
    m_kk = m_d[:, None] & m_d[None, :]

    p_dq = dq_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    p_x_out = x_out_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    p_k = k_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    p_fro = fro_ptr + o_t * H
    p_d_kk = d_kk_ptr + o_d[:, None] * D + o_d[None, :]
    p_h_kk = h_kk_ptr + o_d[:, None] * D + o_d[None, :]

    d_kk = tl.load(p_d_kk, mask=m_kk, other=0.0)
    h_kk = tl.load(p_h_kk, mask=m_kk, other=0.0)

    k = tl.load(p_k, mask=m_tk, other=0.0)
    dq = tl.load(p_dq, mask=m_tk, other=0.0)
    x_out = tl.load(p_x_out, mask=m_tk, other=0.0)
    fro = tl.load(p_fro, mask=m_t, other=0.0).to(tl.float32)

    # masked rows would divide by a zero `fro`, and the NaNs would then leak
    # through the reverse cumsum into valid rows
    # fp32 reduction (fp16 tl.sum saturates) and zero the lambda adjoint on the
    # rows where the forward floored `fro` (max() has zero derivative there).
    tmp = tl.where(fro >= FLOOR_FRO, tl.sum(dq.to(tl.float32) * x_out.to(tl.float32), axis=1), 0.0)
    w = tl.where(m_t, ridge_ratio * tmp / fro, 0.0)
    weights = tl.cumsum(w, reverse=True)
    k_weight = (weights[:, None] * k).to(k.dtype)

    bool_mask = tl.arange(0, chunk_size)[:, None] <= tl.arange(0, chunk_size)[None, :]
    mask = tl.where(bool_mask, 1, 0)
    dk = tl.zeros([chunk_size, BD], dtype=tl.float32)
    dk += tl.dot(k, (d_kk + tl.trans(d_kk)).to(k.dtype)) + tl.dot(
        (tl.dot(k, tl.trans(x_out)) * mask).to(dq.dtype), dq
    )
    dk += tl.dot((tl.dot(k, tl.trans(dq)) * mask).to(x_out.dtype), x_out)

    dk += 2 * (
        tl.dot(
            (
                tl.dot(k, tl.trans(k))
                * tl.where(bool_mask, weights[None, :], weights[:, None])
            ).to(k.dtype),
            k,
        )
        + tl.dot(k_weight, h_kk.to(k.dtype))
    )

    h_kk_suffix_sum_partial = tl.sum(
        tl.where(tl.arange(0, chunk_size) == 0, 1, 0) * weights
    ) * h_kk + tl.dot(tl.trans(k_weight), k)

    p_dk = dk_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    p_h_kk_suffix_sum_partial = h_kk_suffix_sum_partial_ptr + o_d[:, None] * D + o_d[None, :]

    tl.store(
        p_h_kk_suffix_sum_partial,
        h_kk_suffix_sum_partial.to(p_h_kk_suffix_sum_partial.dtype.element_ty),
        mask=m_kk,
    )
    tl.store(p_dk, dk.to(p_dk.dtype.element_ty), mask=m_tk)


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
def chunk_bwd_dk_kernel_forloop(
    # inputs:
    h_kk_suffix_sum_partial_ptr,
    k_ptr,
    dk_ptr,
    # outputs:
    dk_final_ptr,
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
    Accumulates suffix sums and computes final key gradients.

    Second pass of two-pass backward: processes chunks sequentially in reverse order,
    accumulating h_kk_suffix_sum to compute final gradients from later timesteps.

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

    vec_off = scalar_off * D
    k_ptr += vec_off
    dk_ptr += vec_off
    dk_final_ptr += vec_off

    chunk_range = tl.arange(0, chunk_size)
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_tk_base = m_d[None, :]
    m_kk = m_d[:, None] & m_d[None, :]

    h_kk_suffix_sum = tl.zeros((BD, BD), dtype=tl.float32)
    for chunk_id in range(num_chunks - 1, -1, -1):
        # load dq and x_out
        o_t = chunk_id * chunk_size + chunk_range
        m_t = o_t < T
        m_tk = m_t[:, None] & m_tk_base
        p_k = k_ptr + o_t[:, None] * (H * D) + o_d[None, :]
        p_dk = dk_ptr + o_t[:, None] * (H * D) + o_d[None, :]
        p_h_kk_suffix_sum_partial = (
            h_kk_suffix_sum_partial_ptr
            + ((chunk_start + chunk_id) * H + h_id) * D * D
            + o_d[:, None] * D
            + o_d[None, :]
        )
        h_kk_suffix_sum_partial = tl.load(p_h_kk_suffix_sum_partial, mask=m_kk, other=0.0)

        k = tl.load(p_k, mask=m_tk, other=0.0)

        dk = tl.load(p_dk, mask=m_tk, other=0.0).to(tl.float32)
        dk += 2 * tl.dot(k, h_kk_suffix_sum.to(k.dtype))
        dk = -dk

        h_kk_suffix_sum += h_kk_suffix_sum_partial

        p_dk_final = dk_final_ptr + o_t[:, None] * (H * D) + o_d[None, :]

        tl.store(p_dk_final, dk.to(p_dk_final.dtype.element_ty), mask=m_tk)


def chunk_bwd_dk(
    h_kk: Tensor,
    d_kk: Tensor,
    k: Tensor,
    x_out: Tensor,
    dq: Tensor,
    fro: Tensor,
    h_kk_final=None,
    chunk_size: int = 64,
    ridge_ratio: float = 0.02,
    fro_floor: float = 1e-2,
    cu_seqlens: Tensor | None = None,
    chunk_indices: Tensor | None = None,
) -> Tensor:
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

    dk = torch.empty_like(k)
    if h_kk_final is None:
        grid = (num_chunks, B * H)

        h_kk_suffix_sum_partial = torch.empty_like(h_kk)
        chunk_bwd_dk_kernel[grid](
            # inputs:
            h_kk_ptr=h_kk,
            d_kk_ptr=d_kk,
            x_out_ptr=x_out,
            dq_ptr=dq,
            k_ptr=k,
            fro_ptr=fro,
            # outputs:
            h_kk_suffix_sum_partial_ptr=h_kk_suffix_sum_partial,
            dk_ptr=dk,
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

        dk_final = torch.empty_like(k)
        chunk_bwd_dk_kernel_forloop[(N, H)](
            # inputs:
            h_kk_suffix_sum_partial_ptr=h_kk_suffix_sum_partial,
            k_ptr=k,
            dk_ptr=dk,
            # output:
            dk_final_ptr=dk_final,
            cu_seqlens=cu_seqlens,
            split_offsets=chunk_offsets,
            # constants:
            T=T,
            H=H,
            D=D,
            BD=BD,
            chunk_size=chunk_size,
        )
        return dk_final
    else:
        if cu_seqlens is not None:
            raise NotImplementedError(
                "The single-pass `chunk_bwd_dk` path (h_kk_final given) does not support `cu_seqlens`."
            )
        grid = (B, H)
        chunk_bwd_dk_kernel_single_pass[grid](
            # inputs:
            h_kk_final,
            k,
            x_out,
            dq,
            fro,
            # outputs:
            dk,
            # constants:
            T,
            H,
            D,
            chunk_size,
            # algorithm parameters:
            ridge_ratio,
            FLOOR_FRO=fro_floor,
        )
        return dk


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["D"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_dk_kernel_single_pass(
    # inputs:
    h_kk_final_ptr,
    k_ptr,
    x_out_ptr,
    dq_ptr,
    fro_ptr,
    # outputs:
    dk_ptr,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    chunk_size: tl.constexpr,
    # algorithm parameters:
    ridge_ratio: tl.constexpr,
    FLOOR_FRO: tl.constexpr,
):
    """
    Single-pass backward for key gradients when final h_kk state is available.

    Processes chunks sequentially in reverse order, avoiding the need for a two-pass
    algorithm by using the final key-key covariance state.

    Grid: (B, H) - one thread block per batch-head pair (sequential over chunks)
    Block size: Processes [chunk_size, D] vectors per block
    """
    b_id, h_id = tl.program_id(0), tl.program_id(1)

    num_chunks = tl.cdiv(T, chunk_size)

    scalar_off = b_id * T * H + h_id
    fro_ptr += scalar_off

    vec_off = scalar_off * D
    k_ptr += vec_off
    x_out_ptr += vec_off
    dq_ptr += vec_off
    dk_ptr += vec_off

    h_kk_final_ptr += (b_id * H + h_id).to(tl.int64) * D * D
    o_d = tl.arange(0, D)
    p_h_kk_final = h_kk_final_ptr + o_d[:, None] * D + o_d[None, :]
    h_kk = tl.load(p_h_kk_final).to(tl.float32)

    d_kk = tl.zeros((D, D), dtype=tl.float32)

    bool_mask = tl.arange(0, chunk_size)[:, None] <= tl.arange(0, chunk_size)[None, :]
    mask = tl.where(bool_mask, 1, 0)

    h_kk_suffix_sum = tl.zeros_like(h_kk)

    for chunk_id in range(num_chunks - 1, -1, -1):
        # load dq and x_out
        o_t = chunk_id * chunk_size + tl.arange(0, chunk_size)
        m_t = o_t < T
        p_dq = dq_ptr + o_t[:, None] * (H * D) + o_d[None, :]
        p_x_out = x_out_ptr + o_t[:, None] * (H * D) + o_d[None, :]
        p_k = k_ptr + o_t[:, None] * (H * D) + o_d[None, :]
        p_fro = fro_ptr + o_t * H

        dq = tl.load(p_dq, mask=m_t[:, None], other=0.0)
        x_out = tl.load(p_x_out, mask=m_t[:, None], other=0.0)
        k = tl.load(p_k, mask=m_t[:, None], other=0.0)
        fro = tl.load(p_fro, mask=m_t, other=0.0).to(tl.float32)

        h_kk -= tl.dot(tl.trans(k), k)

        dk = tl.zeros([chunk_size, D], dtype=tl.float32)
        dk += tl.dot(k, (d_kk + tl.trans(d_kk)).to(k.dtype)) + tl.dot(
            (tl.dot(k, tl.trans(x_out)) * mask).to(dq.dtype), dq
        )
        dk += tl.dot((tl.dot(k, tl.trans(dq)) * mask).to(x_out.dtype), x_out)

        # fro norm
        dk += 2 * tl.dot(k, h_kk_suffix_sum.to(k.dtype))

        weights = ridge_ratio * tl.where(
            fro >= FLOOR_FRO, tl.sum(dq.to(tl.float32) * x_out.to(tl.float32), axis=1), 0.0
        ) / fro
        h_kk_suffix_sum += tl.sum(weights) * h_kk

        weights = tl.cumsum(weights, reverse=True)

        k_weight = (weights[:, None] * k).to(k.dtype)
        dk += 2 * (
            tl.dot(
                (
                    tl.dot(k, tl.trans(k))
                    * tl.where(bool_mask, weights[None, :], weights[:, None])
                ).to(k.dtype),
                k,
            )
            + tl.dot(k_weight, h_kk.to(k.dtype))
        )

        h_kk_suffix_sum += tl.dot(tl.trans(k_weight), k)

        d_kk += tl.dot(tl.trans(dq), x_out)

        p_dk = dk_ptr + o_t[:, None] * (H * D) + o_d[None, :]
        tl.store(p_dk, -dk.to(p_dk.dtype.element_ty), mask=m_t[:, None])
