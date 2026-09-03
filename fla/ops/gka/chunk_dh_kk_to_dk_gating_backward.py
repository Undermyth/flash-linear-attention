# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/mesa_net/chunk_h_kk_intra_bwd.py


import torch
import triton
import triton.language as tl
from torch import Tensor

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2


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
def chunk_dh_kk_to_dk_gating_kernel(
    # input
    k_ptr,
    x_out_ptr,
    kk_ptr,
    d_kk_ptr,
    dq_ptr,
    gkk_ptr,
    # output:
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
    IS_VARLEN: tl.constexpr,
):
    """
    Computes key and gating gradients (dk, dgkk) from covariance gradient d_kk.

    Gated version that backpropagates through exponential gating applied to
    key-key covariance. Computes gradients for both keys and gating values.

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
    scalar_off = bos * H + h_id
    gkk_ptr += scalar_off
    dgkk_ptr += scalar_off

    vec_off = scalar_off * D
    k_ptr += vec_off
    x_out_ptr += vec_off
    dk_ptr += vec_off
    dq_ptr += vec_off

    d_kk_ptr += (i_tg * H + h_id) * D * D
    kk_ptr += (i_tg * H + h_id) * D * D

    # load
    chunk_range = tl.arange(0, chunk_size)
    o_t = i_c * chunk_size + chunk_range
    m_t = o_t < T
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    m_tk = m_t[:, None] & m_d[None, :]
    m_kk = m_d[:, None] & m_d[None, :]

    p_x_out = x_out_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    x_out = tl.load(p_x_out, mask=m_tk, other=0.0)  # .to(tl.float32)
    p_dq = dq_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    dq = tl.load(p_dq, mask=m_tk, other=0.0)

    p_gkk = gkk_ptr + o_t * H
    gkk = tl.load(p_gkk, mask=m_t, other=0.0).to(tl.float32)

    p_d_kk = d_kk_ptr + o_d[:, None] * D + o_d[None, :]
    p_kk = kk_ptr + o_d[:, None] * D + o_d[None, :]

    p_k = k_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    k = tl.load(p_k, mask=m_tk, other=0.0)

    bool_mask = chunk_range[:, None] >= chunk_range[None, :]
    mask = tl.where(bool_mask, exp2(gkk[:, None] - gkk[None, :]), 0)

    dk = tl.zeros([chunk_size, BD], dtype=tl.float32)

    s = tl.dot(x_out, tl.trans(k)) * mask
    ds = tl.dot(dq, tl.trans(k))
    dk += tl.dot(tl.trans(s).to(dq.dtype), dq)

    dm = tl.where(bool_mask, s * ds, 0)

    dgkk = tl.zeros([chunk_size], dtype=tl.float32)
    dgkk += tl.sum(dm, axis=1)
    dgkk -= tl.sum(dm, axis=0)
    dk += tl.dot(tl.trans(ds * mask).to(x_out.dtype), x_out)

    gkk_last = tl.sum(tl.where(chunk_range == tl.minimum(chunk_size - 1, T - i_c * chunk_size - 1), gkk, 0))
    mask1 = exp2(gkk_last - gkk)[:, None]

    d_kk = tl.load(p_d_kk, mask=m_kk, other=0.0)

    dk += tl.dot((k * mask1).to(k.dtype), (d_kk + tl.trans(d_kk)).to(k.dtype))
    dk = -dk

    kk = tl.load(p_kk, mask=m_kk, other=0.0)
    dgkk += tl.sum(tl.dot(dq, kk) * exp2(gkk)[:, None] * x_out, axis=1)

    dk2 = tl.dot((k * mask1).to(k.dtype), d_kk.to(k.dtype))
    dgkk_last = tl.sum(dk2 * k, axis=1)
    dgkk -= dgkk_last
    dgkk_last = tl.sum(dgkk_last)
    dgkk_last += tl.sum(d_kk * kk) * exp2(gkk_last)

    dgkk = tl.where(chunk_range < tl.minimum(chunk_size - 1, T - i_c * chunk_size - 1), dgkk, dgkk + dgkk_last)

    dgkk = tl.cumsum(-dgkk, reverse=True)

    p_dgkk = dgkk_ptr + o_t * H
    tl.store(p_dgkk, dgkk.to(p_dgkk.dtype.element_ty), mask=m_t)

    p_dk = dk_ptr + o_t[:, None] * (H * D) + o_d[None, :]
    tl.store(p_dk, dk.to(p_dk.dtype.element_ty), mask=m_tk)


def chunk_dh_kk_to_dk_gating(
    k: Tensor,
    x_out: Tensor,
    h_kk: Tensor,
    d_kk: Tensor,
    dq: Tensor,
    gkk: Tensor,
    chunk_size: int = 64,
    cu_seqlens: Tensor | None = None,
    chunk_indices: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Computes key and gating gradients (dk, dgkk) from covariance gradient d_kk.

    Wrapper function that launches the Triton kernel for gated gradient computation.

    Args:
        k: Key tensor with shape [B, L, H, D]
        x_out: Output from forward pass with shape [B, L, H, D]
        h_kk: Chunk-level key-key covariance with shape [B, num_chunks, H, D, D]
        d_kk: Gradient of chunk-level covariance with shape [B, num_chunks, H, D, D]
        dq: Gradient of query with shape [B, L, H, D]
        gkk: Cumulative gating values with shape [B, L, H], in log2 space
        chunk_size: Chunk size for parallel processing. Defaults to 64
        cu_seqlens: Cumulative sequence lengths with shape [N+1] for varlen inputs, or None
        chunk_indices: Chunk indices prepared from cu_seqlens, computed if None

    Returns:
        dk: Key gradient with shape [B, L, H, D]
        dgkk: Gating gradient with shape [B, L, H]
    """

    B, T, H, D = k.shape

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    num_chunks = triton.cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)
    BD = triton.next_power_of_2(D)

    dk = torch.empty_like(k)
    grid = (num_chunks, B * H)

    dgkk = torch.empty([B, T, H]).to(k.device)
    chunk_dh_kk_to_dk_gating_kernel[grid](
        # inputs:
        k_ptr=k,
        x_out_ptr=x_out,
        kk_ptr=h_kk,
        d_kk_ptr=d_kk,
        dq_ptr=dq,
        gkk_ptr=gkk,
        # output:
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
    )

    return dk, dgkk
