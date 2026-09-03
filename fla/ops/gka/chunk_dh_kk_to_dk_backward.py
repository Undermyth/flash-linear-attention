# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/mesa_net/chunk_h_kk_intra_bwd.py

import torch
import triton
import triton.language as tl
from torch import Tensor

from fla.ops.utils import prepare_chunk_indices


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
def chunk_dh_kk_to_dk_kernel(
    # input
    k_ptr,
    x_out_ptr,
    dh_kk_ptr,
    dq_ptr,
    # output:
    dk_ptr,
    cu_seqlens,
    chunk_indices,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    chunk_size: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Computes key gradient dk from chunk-level covariance gradient dh_kk.

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

    # offset calculation
    vec_off = (bos * H + h_id) * D
    k_ptr += vec_off
    x_out_ptr += vec_off
    dk_ptr += vec_off
    dq_ptr += vec_off

    dh_kk_ptr += (i_tg * H + h_id) * D * D

    # load
    o_t = i_c * chunk_size + tl.arange(0, chunk_size)
    m_t = o_t < T
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_tk = m_t[:, None] & m_d[None, :]
    m_kk = m_d[:, None] & m_d[None, :]

    p_x_out = x_out_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    x_out = tl.load(p_x_out, mask=m_tk, other=0.0)
    p_dq = dq_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    dq = tl.load(p_dq, mask=m_tk, other=0.0)

    p_dh_kk = dh_kk_ptr + o_d[:, None] * D + o_d[None, :]
    dh_kk = tl.load(p_dh_kk, mask=m_kk, other=0.0)

    p_k = k_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    k = tl.load(p_k, mask=m_tk, other=0.0)

    mask = tl.trans(
        tl.where(
            tl.arange(0, chunk_size)[:, None] >= tl.arange(0, chunk_size)[None, :], 1, 0
        )
    )

    dk = tl.zeros([chunk_size, BD], dtype=tl.float32)

    dk += tl.dot(k, dh_kk + tl.trans(dh_kk)) + tl.dot(
        (tl.dot(k, tl.trans(x_out)) * mask).to(dq.dtype), dq
    )
    dk += tl.dot((tl.dot(k, tl.trans(dq)) * mask).to(x_out.dtype), x_out)
    dk = -dk

    p_dk = dk_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    tl.store(p_dk, dk.to(p_dk.dtype.element_ty), mask=m_tk)


def chunk_dh_kk_to_dk(
    k: Tensor,
    x_out: Tensor,
    dh_kk: Tensor,
    dq: Tensor,
    chunk_size: int = 64,
    cu_seqlens: Tensor | None = None,
    chunk_indices: Tensor | None = None,
) -> Tensor:
    """
    Computes key gradient dk from chunk-level covariance gradient dh_kk.

    Wrapper function that launches the Triton kernel for parallel gradient computation.

    Args:
        k: Key tensor with shape [B, L, H, D]
        x_out: Output from forward pass with shape [B, L, H, D]
        dh_kk: Gradient of chunk-level covariance with shape [B, num_chunks, H, D, D]
        dq: Gradient of query with shape [B, L, H, D]
        chunk_size: Chunk size for parallel processing. Defaults to 64
        cu_seqlens: Cumulative sequence lengths with shape [N+1] for varlen inputs, or None
        chunk_indices: Chunk indices prepared from cu_seqlens, computed if None

    Returns:
        dk: Key gradient with shape [B, L, H, D]
    """

    B, T, H, D = k.shape

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    num_chunks = triton.cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)
    BD = triton.next_power_of_2(D)

    dk = torch.empty_like(k)
    grid = (num_chunks, B * H)

    chunk_dh_kk_to_dk_kernel[grid](
        # inputs:
        k_ptr=k,
        x_out_ptr=x_out,
        dh_kk_ptr=dh_kk,
        dq_ptr=dq,
        # output:
        dk_ptr=dk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        # constants:
        T=T,
        H=H,
        D=D,
        BD=BD,
        chunk_size=chunk_size,
    )

    return dk
