# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gka.gka_chebyshev_solve import gka_chebyshev_gla, torch_decoding_one_step
from fla.ops.gka.naive import naive_gka_decoding_one_step, naive_gka_exact
from fla.utils import IS_INTEL_ALCHEMIST, assert_close, device, device_platform

# MesaNet relies on CG, which converges fast enough with max_CG_iteration = D;
# Chebyshev decays slower per iteration, so allow more rounds against the exact reference.
NUM_ITER = 30


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'gate_range', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-gate_range{}-{}".format(*test))
        for test in [
            (1, 63, 1, 64, [0.8, 0.99], torch.float16),
            (2, 500, 4, 60, [0.8, 0.99], torch.float16),
            (2, 1024, 8, 128, [0.8, 0.99], torch.float16),
            (2, 1024, 8, 128, [0.01, 0.1], torch.float16),
            (2, 1024, 8, 128, [1, 1], torch.float16),
            (4, 2048, 8, 64, [0.8, 0.99], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure',
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    gate_range: tuple[float, float],
    dtype: torch.dtype,
):
    if D & (D - 1):
        pytest.skip(reason='the Chebyshev kernels use tl.arange(0, D) and require a power-of-2 D')
    torch.manual_seed(42)
    q = torch.rand(B, T, H, D, dtype=dtype) / 10
    k = F.normalize(torch.rand(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.rand(B, T, H, D, dtype=dtype) / 10
    alpha = torch.rand(B, T, H, dtype=dtype).sigmoid()
    lower_gate, upper_gate = gate_range
    g = torch.rand(B, T, H, dtype=dtype).float().uniform_(lower_gate, upper_gate).log()
    gk = torch.rand(B, T, H, dtype=dtype).float().uniform_(lower_gate, upper_gate).log()
    q, k, v, alpha, g, gk = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, alpha, g, gk))
    do = torch.rand_like(v)

    scale = D ** -0.5
    d_h_kk_final = torch.rand(B, H, D, D, dtype=torch.float32, device=device)
    d_h_kv_final = torch.rand(B, H, D, D, dtype=torch.float32, device=device)

    tri, tri_kk_final, tri_kv_final = gka_chebyshev_gla(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        alpha=alpha.clone(),
        g=g.clone(),
        gk=gk.clone(),
        gla_scale=scale,
        num_iter=NUM_ITER,
    )

    ((tri * do).sum() + (tri_kk_final * d_h_kk_final).sum() + (tri_kv_final * d_h_kv_final).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dalpha, tri_dg, tri_dgk = q.grad, k.grad, v.grad, alpha.grad, g.grad, gk.grad
    q.grad = k.grad = v.grad = alpha.grad = g.grad = gk.grad = None

    ref, ref_hkk_final, ref_hkv_final = naive_gka_exact(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        alpha=alpha.clone(),
        g=g.clone(),
        gk=gk.clone(),
        gla_scale=scale,
    )

    ((ref * do).sum() +
     (ref_hkk_final * d_h_kk_final).sum() + (ref_hkv_final * d_h_kv_final).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dalpha, ref_dg, ref_dgk = q.grad, k.grad, v.grad, alpha.grad, g.grad, gk.grad
    q.grad = k.grad = v.grad = alpha.grad = g.grad = gk.grad = None

    assert_close('o', ref, tri, 0.007)
    assert_close('h_kk_final', ref_hkk_final, tri_kk_final, 0.008)
    assert_close('h_kv_final', ref_hkv_final, tri_kv_final, 0.008)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('dalpha', ref_dalpha, tri_dalpha, 0.008)
    assert_close('dg', ref_dg, tri_dg, 0.008)
    assert_close('dgk', ref_dgk, tri_dgk, 0.008)


@pytest.mark.parametrize(
    ('H', 'D', 'gate_range', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-gate_range{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (3, 50, [0.8, 0.99], [0, 15], torch.float16),
            (4, 64, [0.8, 0.99], [0, 14, 121, 421, 500], torch.float16),
            (4, 64, [0.01, 0.1], [0, 256, 500, 1000], torch.float16),
            (4, 100, [1, 1], [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_chunk_varlen(
    H: int,
    D: int,
    gate_range: tuple[float, float],
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='gka_chebyshev_gla is not supported on alchemist for D>128')
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    # randomly split the sequence into N segments
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.long, device=device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1
    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype) / 10
    k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn((1, T, H, D), dtype=dtype) / 10
    lower_gate, upper_gate = gate_range
    g = torch.rand(1, T, H, dtype=dtype).float().uniform_(lower_gate, upper_gate).log()
    gk = torch.rand(1, T, H, dtype=dtype).float().uniform_(lower_gate, upper_gate).log()
    alpha = torch.rand(1, T, H, dtype=dtype).sigmoid()

    q, k, v, alpha, g, gk = map(lambda x: x.to(device).requires_grad_(), (q, k, v, alpha, g, gk))
    do = torch.rand_like(v) / 10

    scale = D ** -0.5
    d_h_kk_final = torch.rand(N, H, D, D, dtype=torch.float32, device=device)
    d_h_kv_final = torch.rand(N, H, D, D, dtype=torch.float32, device=device)

    tri, tri_h_kk_final, tri_h_kv_final = gka_chebyshev_gla(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        alpha=alpha.clone(),
        g=g.clone(),
        gk=gk.clone(),
        gla_scale=scale,
        num_iter=NUM_ITER,
        cu_seqlens=cu_seqlens,
    )

    ((tri * do).sum() +
     (tri_h_kk_final * d_h_kk_final).sum() + (tri_h_kv_final * d_h_kv_final).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dalpha, tri_dg, tri_dgk = \
        q.grad, k.grad, v.grad, alpha.grad, g.grad, gk.grad
    q.grad = k.grad = v.grad = alpha.grad = g.grad = gk.grad = None

    ref = []
    ref_h_kk_t = []
    ref_h_kv_t = []
    for i in range(N):
        ref_i, ref_h_kk_i, ref_h_kv_i = naive_gka_exact(
            q=q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k=k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v=v[:, cu_seqlens[i]:cu_seqlens[i+1]],
            alpha=alpha[:, cu_seqlens[i]:cu_seqlens[i+1]],
            g=g[:, cu_seqlens[i]:cu_seqlens[i+1]],
            gk=gk[:, cu_seqlens[i]:cu_seqlens[i+1]],
            gla_scale=scale,
        )
        ref.append(ref_i)
        ref_h_kk_t.append(ref_h_kk_i)
        ref_h_kv_t.append(ref_h_kv_i)
    ref = torch.cat(ref, 1)
    ref_h_kk_t = torch.cat(ref_h_kk_t, 0)
    ref_h_kv_t = torch.cat(ref_h_kv_t, 0)

    ((ref * do).sum() + (ref_h_kk_t * d_h_kk_final).sum() + (ref_h_kv_t * d_h_kv_final).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dalpha, ref_dg, ref_dgk = \
        q.grad, k.grad, v.grad, alpha.grad, g.grad, gk.grad
    q.grad = k.grad = v.grad = alpha.grad = g.grad = gk.grad = None

    assert_close('o', ref, tri, 0.006)
    assert_close('h_kk_final', ref_h_kk_t, tri_h_kk_final, 0.008)
    assert_close('h_kv_final', ref_h_kv_t, tri_h_kv_final, 0.008)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('dalpha', ref_dalpha, tri_dalpha, 0.015)
    assert_close('dg', ref_dg, tri_dg, 0.015)
    assert_close('dgk', ref_dgk, tri_dgk, 0.015)


@pytest.mark.parametrize(
    ('B', 'H', 'D', 'gate_range', 'num_iter', 'dtype'),
    [
        pytest.param(*test, id="B{}-H{}-D{}-gate_range{}-num_iter{}-{}".format(*test))
        for test in [
            (1, 3, 50, [0.95, 0.99], 1, torch.float16),
            (2, 4, 60, [0.95, 0.99], 5, torch.float16),
            (2, 8, 128, [0.95, 0.99], 1, torch.float16),
            (2, 8, 128, [0.95, 0.99], 5, torch.float16),
            (2, 8, 128, [0.95, 0.99], 100, torch.float16),
        ]
    ],
)
def test_decoding_one_step(
    B: int,
    H: int,
    D: int,
    gate_range: tuple[float, float],
    num_iter: int,
    dtype: torch.dtype,
):
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='gka_chebyshev_gla is not supported on alchemist for D>128')
    if D & (D - 1):
        pytest.skip(reason='the single-step Chebyshev kernel uses tl.arange(0, D) and requires a power-of-2 D')
    torch.manual_seed(42)
    torch.set_default_device(device)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    # the single-step kernel reads RCP_LN2 as a python global inside @triton.jit
    os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'
    q = torch.rand((B, 1, H, D), dtype=dtype)
    k = F.normalize(torch.randn(B, 1, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.rand((B, 1, H, D), dtype=dtype)
    lower_gate, upper_gate = gate_range
    g = torch.rand(B, 1, H, dtype=dtype).float().uniform_(lower_gate, upper_gate).log()
    gk = torch.rand(B, 1, H, dtype=dtype).float().uniform_(lower_gate, upper_gate).log()
    alpha = torch.rand(B, 1, H, dtype=dtype).sigmoid()

    k_init_rand = F.normalize(torch.rand(B, H, D, device=device, dtype=dtype), dim=-1, p=2)
    prev_h_kk = (k_init_rand.unsqueeze(-1) * k_init_rand.unsqueeze(-2)).detach().clone().float()
    prev_h_kv = torch.rand(B, H, D, D, dtype=torch.float32, device=device)

    o, curr_h_kk, curr_h_kv = torch_decoding_one_step(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        alpha=alpha.clone(),
        g=g.clone(),
        gk=gk.clone(),
        gla_scale=D ** -0.5,
        ridge_strength=0.02,
        solver_type='chebyshev',
        num_iter=num_iter,
        prev_h_kk=prev_h_kk.clone(),
        prev_h_kv=prev_h_kv.clone(),
    )

    o_ref, curr_h_kk_ref, curr_h_kv_ref = naive_gka_decoding_one_step(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        alpha=alpha.clone(),
        g=g.clone(),
        gk=gk.clone(),
        gla_scale=D ** -0.5,
        ridge_strength=0.02,
        num_iter=num_iter,
        prev_h_kk=prev_h_kk.clone(),
        prev_h_kv=prev_h_kv.clone(),
    )

    assert_close('o', o_ref, o, 0.005)
    assert_close('curr_h_kk', curr_h_kk_ref, curr_h_kk, 0.005)
    assert_close('curr_h_kv', curr_h_kv_ref, curr_h_kv, 0.005)


@pytest.mark.parametrize(
    ('N', 'H', 'D', 'gate_range', 'num_iter', 'dtype'),
    [
        pytest.param(*test, id="N{}-H{}-D{}-gate_range{}-num_iter{}-{}".format(*test))
        for test in [
            (1, 3, 64, [0.95, 0.99], 5, torch.float16),
            (3, 4, 64, [0.95, 0.99], 1, torch.float16),
            (5, 2, 64, [0.8, 0.99], 5, torch.float16),
            (8, 8, 128, [0.95, 0.99], 100, torch.float16),
        ]
    ],
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_decoding_one_step_varlen because SKIP_TEST_CHUNK_VARLEN is set',
)
def test_decoding_one_step_varlen(
    N: int,
    H: int,
    D: int,
    gate_range: tuple[float, float],
    num_iter: int,
    dtype: torch.dtype,
):
    if IS_INTEL_ALCHEMIST and D > 128:
        pytest.skip(reason='gka_chebyshev_gla is not supported on alchemist for D>128')
    if D & (D - 1):
        pytest.skip(reason='the single-step Chebyshev kernel uses tl.arange(0, D) and requires a power-of-2 D')
    torch.manual_seed(42)
    torch.set_default_device(device)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    # the single-step kernel reads RCP_LN2 as a python global inside @triton.jit
    os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'
    # a decoding step consumes one token per sequence, so N packed tokens make N sequences
    q = torch.rand((1, N, H, D), dtype=dtype)
    k = F.normalize(torch.randn(1, N, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.rand((1, N, H, D), dtype=dtype)
    lower_gate, upper_gate = gate_range
    g = torch.rand(1, N, H, dtype=dtype).float().uniform_(lower_gate, upper_gate).log()
    gk = torch.rand(1, N, H, dtype=dtype).float().uniform_(lower_gate, upper_gate).log()
    alpha = torch.rand(1, N, H, dtype=dtype).sigmoid()
    cu_seqlens = torch.arange(N + 1, dtype=torch.long, device=device)

    k_init_rand = F.normalize(torch.rand(N, H, D, device=device, dtype=dtype), dim=-1, p=2)
    prev_h_kk = (k_init_rand.unsqueeze(-1) * k_init_rand.unsqueeze(-2)).detach().clone().float()
    prev_h_kv = torch.rand(N, H, D, D, dtype=torch.float32, device=device)

    o, curr_h_kk, curr_h_kv = torch_decoding_one_step(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        alpha=alpha.clone(),
        g=g.clone(),
        gk=gk.clone(),
        gla_scale=D ** -0.5,
        ridge_strength=0.02,
        solver_type='chebyshev',
        num_iter=num_iter,
        prev_h_kk=prev_h_kk.clone(),
        prev_h_kv=prev_h_kv.clone(),
        cu_seqlens=cu_seqlens,
    )

    # decode every sequence on its own as the reference
    ref, ref_h_kk_t, ref_h_kv_t = [], [], []
    for i in range(N):
        ref_i, ref_h_kk_i, ref_h_kv_i = naive_gka_decoding_one_step(
            q=q[:, i:i + 1],
            k=k[:, i:i + 1],
            v=v[:, i:i + 1],
            alpha=alpha[:, i:i + 1],
            g=g[:, i:i + 1],
            gk=gk[:, i:i + 1],
            gla_scale=D ** -0.5,
            ridge_strength=0.02,
            num_iter=num_iter,
            prev_h_kk=prev_h_kk[i:i + 1],
            prev_h_kv=prev_h_kv[i:i + 1],
        )
        ref.append(ref_i)
        ref_h_kk_t.append(ref_h_kk_i)
        ref_h_kv_t.append(ref_h_kv_i)
    ref = torch.cat(ref, 1)
    ref_h_kk_t = torch.cat(ref_h_kk_t, 0)
    ref_h_kv_t = torch.cat(ref_h_kv_t, 0)

    assert_close('o', ref, o, 0.005)
    assert_close('curr_h_kk', ref_h_kk_t, curr_h_kk, 0.005)
    assert_close('curr_h_kv', ref_h_kv_t, curr_h_kv, 0.005)
