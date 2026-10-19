# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import itertools

import pytest
import torch

from fla.layers.gka import GatedKalmaNet, NeuralSolver
from fla.utils import assert_close, device


def _chunk_padded(tensors: tuple[torch.Tensor, ...], lengths: list[int], chunk_size: int):
    """Pad every packed sequence up to a whole number of chunks.

    The result is the tensor the non-varlen path would see if the pack were laid
    out with the chunks of each sequence aligned to `chunk_size`; the padded rows
    are zero, so they are inert.
    """
    out = [[] for _ in tensors]
    src = 0
    for length in lengths:
        padded = -(-length // chunk_size) * chunk_size
        for dst, t in zip(out, tensors):
            tail = torch.zeros(1, padded - length, *t.shape[2:], dtype=t.dtype, device=t.device)
            dst.append(torch.cat([t[:, src:src + length], tail], dim=1))
        src += length
    return tuple(torch.cat(x, dim=1) for x in out)


@pytest.mark.parametrize(
    ('lengths', 'gate_range'),
    [
        pytest.param(*case, id="lengths{}-gate_range{}".format(*case))
        for case in [
            ([128, 128, 192], [0.8, 0.99]),
            ([100, 150, 37], [0.8, 0.99]),
            ([15, 63, 64, 65], [0.8, 0.99]),
            ([191, 200], [0.01, 0.1]),
        ]
    ],
)
def test_neural_solver_varlen_matches_chunk_padded(lengths: list[int], gate_range: tuple[float, float]):
    """The packed reference solve must tile each sequence on its own chunks.

    Padding every sequence to a whole number of chunks makes the plain path see
    exactly the same chunks (the extra rows carry `gk = 0`, so the local cumsum
    freezes), hence the loss and the theta/phi gradients have to agree.  The last
    case is the one where leaving `gt = 0` in the dead tail overflows fp32
    (`2 ** -gt_last`) and poisons the loss with nan.
    """
    torch.manual_seed(42)
    H, D, C, K = 3, 16, 64, 5
    T = sum(lengths)
    n_chunks = sum(-(-length // C) for length in lengths)
    k = torch.randn(1, T, H, D, device=device)
    q = torch.randn(1, T, H, D, device=device)
    gk = torch.rand(1, T, H, device=device).uniform_(*gate_range).log()
    alpha = torch.rand(1, T, H, device=device).sigmoid()
    cu_seqlens = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.long, device=device)
    h0 = torch.randn(1, n_chunks, H, D, D, device=device)

    solver = NeuralSolver(num_heads=H, num_steps=K, chunk_size=C, grad_checkpoint=False).to(device)
    with torch.no_grad():
        solver.theta.normal_(0.0, 0.2)
        solver.phi.normal_(0.0, 0.2)

    loss_varlen, logs = solver(h0, k, q, gk, alpha, cu_seqlens=cu_seqlens)
    grads_varlen = torch.autograd.grad(loss_varlen, [solver.theta, solver.phi])

    kp, qp, gkp, ap = _chunk_padded((k, q, gk, alpha), lengths, C)
    loss_flat, logs_flat = solver(h0, kp, qp, gkp, ap)
    grads_flat = torch.autograd.grad(loss_flat, [solver.theta, solver.phi])

    assert torch.isfinite(loss_varlen), 'the packed reference loss must stay finite'
    assert set(logs) == set(logs_flat)
    assert_close('loss', loss_flat, loss_varlen, 0.0)
    for name, ref, tri in zip(('dtheta', 'dphi'), grads_flat, grads_varlen):
        assert torch.isfinite(tri).all(), name
        assert_close(name, ref, tri, 0.0)


def test_gka_layer_trains_on_packed_inputs():
    """A training step on packed (cu_seqlens) inputs must drive the solver.

    Regression: the reference solve used to tile the flat token grid, so a pack
    whose total length is not a multiple of the chunk size crashed (and the padded
    rows of the last chunk of every sequence could produce nan).
    """
    torch.manual_seed(42)
    layer = GatedKalmaNet(
        hidden_size=128,
        head_dim=64,
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=4,
        num_iter=4,
        solver_steps=4,
        chunk_size=64,
        conv_size=4,
    ).to(device=device, dtype=torch.bfloat16)
    layer.train()

    x = torch.randn(1, 256, 128, device=device, dtype=torch.bfloat16)
    # aligned: two sequences of exactly two chunks; ragged: the second sequence
    # ends in a partial chunk.
    cu_aligned = torch.tensor([0, 128, 256], dtype=torch.long, device=device)
    cu_ragged = torch.tensor([0, 100, 256], dtype=torch.long, device=device)

    layer.zero_grad(set_to_none=True)
    o_packed, _, _, loss_packed = layer(x, cu_seqlens=cu_aligned)
    assert loss_packed is not None and torch.isfinite(loss_packed)
    assert layer.solver_residuals is not None and 'res_4' in layer.solver_residuals

    # A pack is the concatenation of its sequences: the conv, the GLA state and the
    # Chebyshev state are all reset at a sequence boundary, which running every
    # sequence on its own reproduces.
    o_a, _, _, loss_a = layer(x[:, :128])
    o_b, _, _, loss_b = layer(x[:, 128:])
    assert loss_a is not None and loss_b is not None
    assert_close('o', torch.cat([o_a, o_b], dim=1), o_packed, 0.02)
    assert_close('loss', (loss_a + loss_b) / 2, loss_packed, 0.02)

    layer.zero_grad(set_to_none=True)
    o_ragged, _, _, loss_ragged = layer(x, cu_seqlens=cu_ragged)
    assert torch.isfinite(loss_ragged)
    assert o_ragged.shape == x.shape and torch.isfinite(o_ragged).all()

    # the auxiliary solver loss must actually reach theta/phi
    loss_ragged.backward()
    for name, p in layer.solver.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f'no usable gradient for {name}'
