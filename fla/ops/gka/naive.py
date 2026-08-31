# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def _chebyshev_solve(
    h_kk: torch.Tensor,
    q: torch.Tensor,
    ridge_strength: float,
    num_iter: int,
) -> torch.Tensor:
    r"""
    Solve (H + λI)·x = q, with λ = ridge_strength·||H||_F, by Chebyshev iteration, following the
    same update order as the Triton solver.

    Args:
        h_kk (torch.Tensor):
            Key-key covariances of shape `[..., D, D]`.
        q (torch.Tensor):
            Right-hand sides of shape `[..., D]`.
        ridge_strength (float):
            Ridge strength as a fraction of the Frobenius norm of `h_kk`.
        num_iter (int):
            Number of Chebyshev iterations.

    Returns:
        x (torch.Tensor):
            Solutions of shape `[..., D]`.
    """
    fro = h_kk.norm(dim=(-2, -1))
    lamb = ridge_strength * fro
    # the spectral radius is bounded by ||H||_F, and the ridge shifts the lower bound to λ itself
    denom = 2 * lamb + fro
    stepsize = (2 / denom).unsqueeze(-1)
    rho_sq_4 = (fro / denom / 2).unsqueeze(-1) ** 2
    omega = torch.full_like(stepsize, 2.0)

    x = stepsize * q
    x_prev = torch.zeros_like(x)
    for _ in range(num_iter):
        omega = 1 / (1 - rho_sq_4 * omega)
        grad = (x.unsqueeze(-1) * h_kk).sum(-2) + lamb.unsqueeze(-1) * x - q
        grad_and_x_prev = stepsize * omega * grad + (omega - 1) * x_prev
        x_prev = x
        x = omega * x - grad_and_x_prev
    return x


def naive_gka_decoding_one_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    gla_scale: float,
    ridge_strength: float = 0.02,
    num_iter: int = 30,
    prev_h_kk: torch.Tensor | None = None,
    prev_h_kv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    r"""
    Naive reference for `torch_decoding_one_step`: one GKA decoding step, where the ridge problem
    is solved for the key-key covariance that already contains the current token.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, H, D]` or `[B, 1, H, D]`.
        k (torch.Tensor):
            Keys of the same shape as `q`.
        v (torch.Tensor):
            Values of the same shape as `q`.
        alpha (torch.Tensor):
            Mixing coefficients of shape `[B, H]` or `[B, 1, H]`.
        g (torch.Tensor, Optional):
            Per-token log decay of the key-value state with shape `[B, H]` or `[B, 1, H]`. Default: `None`.
        gk (torch.Tensor, Optional):
            Per-token log decay of the key-key covariance with shape `[B, H]` or `[B, 1, H]`. Default: `None`.
        gla_scale (float):
            Scale applied to the query that reads out the key-value state, typically `D ** -0.5`.
        ridge_strength (float):
            Ridge strength as a fraction of ||H_t||_F. Default: `0.02`.
        num_iter (int):
            Number of Chebyshev iterations; `0` skips both the solve and the alpha mixing, and
            leaves the key-key state behind the solver unreported, as the reference does. Default: `30`.
        prev_h_kk (torch.Tensor, Optional):
            Key-key covariance carried over from previous steps, of shape `[B, H, D, D]`. Default: `None`.
        prev_h_kv (torch.Tensor, Optional):
            Key-value state carried over from previous steps, of shape `[B, H, D, DV]`. Default: `None`.

    Returns:
        o (torch.Tensor):
            Output with the rank of the input queries: `[B, H, DV]` or `[B, 1, H, DV]`.
        h_kk (torch.Tensor):
            Updated key-key covariance of shape `[B, H, D, D]`, or `None` if `num_iter` is `0`.
        h_kv (torch.Tensor):
            Updated key-value state of shape `[B, H, D, DV]`.
    """
    # NOTE: `torch_decoding_one_step` promotes a 3D input by prepending a batch dimension, which
    # decodes correctly only for B = 1; here a 3D input is read as [B, H, D] instead.
    rank_3 = q.dim() == 3
    if rank_3:
        q, k, v = (x.unsqueeze(1) for x in (q, k, v))
        alpha, g, gk = (x.unsqueeze(1) if torch.is_tensor(x) else None for x in (alpha, g, gk))
    q, k, v, alpha = (x.float() for x in (q, k, v, alpha))
    B, _, H, D = q.shape
    DV = v.shape[-1]

    decay_kk = gk.float().exp()[:, 0, :, None, None] if gk is not None else 1.0
    decay_kv = g.float().exp()[:, 0, :, None, None] if g is not None else 1.0
    k_i, v_i, q_i = k[:, 0], v[:, 0], q[:, 0]
    h_kk = (k.new_zeros(B, H, D, D) if prev_h_kk is None else prev_h_kk.float()) * decay_kk
    h_kv = (k.new_zeros(B, H, D, DV) if prev_h_kv is None else prev_h_kv.float()) * decay_kv
    h_kk = h_kk + k_i.unsqueeze(-1) * k_i.unsqueeze(-2)
    h_kv = h_kv + k_i.unsqueeze(-1) * v_i.unsqueeze(-2)

    if num_iter > 0:
        x = _chebyshev_solve(h_kk, q_i, ridge_strength, num_iter)
        q_star = q_i + alpha[:, 0].unsqueeze(-1) * (x - q_i)
    else:
        h_kk = None
        q_star = q_i

    o = gla_scale * (q_star.unsqueeze(-1) * h_kv).sum(-2)
    if not rank_3:
        o = o.unsqueeze(1)
    return o, h_kk, h_kv


def naive_gka_exact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    gla_scale: float,
    ridge_strength: float = 0.02,
    h_kk_init: torch.Tensor | None = None,
    h_kv_init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Naive reference for `gka_chebyshev_gla` that materializes every per-timestep covariance and
    solves (H_t + λ_t I)·x_t = q_t by direct factorization, so comparing a solver against it
    isolates the iteration error.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, L, H, D]`.
        k (torch.Tensor):
            Keys of shape `[B, L, H, D]`.
        v (torch.Tensor):
            Values of shape `[B, L, H, DV]`.
        alpha (torch.Tensor):
            Mixing coefficients of shape `[B, L, H]`.
        g (torch.Tensor, Optional):
            Per-token log decay of the key-value state with shape `[B, L, H]`. Default: `None`.
        gk (torch.Tensor, Optional):
            Per-token log decay of the key-key covariance with shape `[B, L, H]`. Default: `None`.
        gla_scale (float):
            Scale applied to the query that reads out the key-value state, typically `D ** -0.5`.
        ridge_strength (float):
            Ridge strength as a fraction of ||H_t||_F. Default: `0.02`.
        h_kk_init (torch.Tensor, Optional):
            Key-key covariance before the first token, of shape `[B, H, D, D]`. Default: `None`.
        h_kv_init (torch.Tensor, Optional):
            Key-value state before the first token, of shape `[B, H, D, DV]`. Default: `None`.

    Returns:
        o (torch.Tensor):
            Output of shape `[B, L, H, DV]`.
        h_kk (torch.Tensor):
            Final key-key covariance of shape `[B, H, D, D]`.
        h_kv (torch.Tensor):
            Final key-value state of shape `[B, H, D, DV]`.
    """
    B, L, H, D = q.shape
    DV = v.shape[-1]
    q, k, v, alpha = (x.float() for x in (q, k, v, alpha))
    decay_kk = gk.float().exp()[..., None, None] if gk is not None else k.new_ones(B, L, H, 1, 1)
    decay_kv = g.float().exp()[..., None, None] if g is not None else k.new_ones(B, L, H, 1, 1)
    h_kk = k.new_zeros(B, H, D, D) if h_kk_init is None else h_kk_init.float()
    h_kv = k.new_zeros(B, H, D, DV) if h_kv_init is None else h_kv_init.float()

    h_kk_all, h_kv_all = [], []
    for i in range(L):
        k_i, v_i = k[:, i], v[:, i]
        h_kk = h_kk * decay_kk[:, i] + k_i.unsqueeze(-1) * k_i.unsqueeze(-2)
        h_kv = h_kv * decay_kv[:, i] + k_i.unsqueeze(-1) * v_i.unsqueeze(-2)
        h_kk_all.append(h_kk)
        h_kv_all.append(h_kv)
    h_kk_all = torch.stack(h_kk_all, 1)
    h_kv_all = torch.stack(h_kv_all, 1)

    lamb = ridge_strength * h_kk_all.norm(dim=(-2, -1))
    eye = torch.eye(D, device=q.device, dtype=q.dtype)
    x = torch.linalg.solve(h_kk_all + lamb[..., None, None] * eye, q)
    q_star = q + alpha.unsqueeze(-1) * (x - q)
    o = gla_scale * (q_star.unsqueeze(-1) * h_kv_all).sum(-2)
    return o, h_kk_all[:, -1], h_kv_all[:, -1]


def naive_gka_chebyshev(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    gla_scale: float,
    ridge_strength: float = 0.02,
    num_iter: int = 30,
    h_kk_init: torch.Tensor | None = None,
    h_kv_init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    r"""
    Naive reference for `gka_chebyshev_gla`: solve (H_t + λ_t I)·x_t = q_t per timestep with the
    Chebyshev iteration, mix the solution into the query with `alpha`, then read out the
    key-value state.

    Args:
        q (torch.Tensor):
            Queries of shape `[B, L, H, D]`.
        k (torch.Tensor):
            Keys of shape `[B, L, H, D]`.
        v (torch.Tensor):
            Values of shape `[B, L, H, DV]`.
        alpha (torch.Tensor):
            Mixing coefficients of shape `[B, L, H]`.
        g (torch.Tensor, Optional):
            Per-token log decay of the key-value state with shape `[B, L, H]`. Default: `None`.
        gk (torch.Tensor, Optional):
            Per-token log decay of the key-key covariance with shape `[B, L, H]`. Default: `None`.
        gla_scale (float):
            Scale applied to the query that reads out the key-value state, typically `D ** -0.5`.
        ridge_strength (float):
            Ridge strength as a fraction of ||H_t||_F. Default: `0.02`.
        num_iter (int):
            Number of Chebyshev iterations; `0` skips both the solve and the alpha mixing, and
            reports no final key-key covariance, as `gka_chebyshev_gla` does. Default: `30`.
        h_kk_init (torch.Tensor, Optional):
            Key-key covariance before the first token, of shape `[B, H, D, D]`. Default: `None`.
        h_kv_init (torch.Tensor, Optional):
            Key-value state before the first token, of shape `[B, H, D, DV]`. Default: `None`.

    Returns:
        o (torch.Tensor):
            Output of shape `[B, L, H, DV]`.
        h_kk (torch.Tensor):
            Final key-key covariance of shape `[B, H, D, D]`, or `None` if `num_iter` is `0`.
        h_kv (torch.Tensor):
            Final key-value state of shape `[B, H, D, DV]`.
    """
    B, L, H, D = q.shape
    DV = v.shape[-1]
    q, k, v, alpha = (x.float() for x in (q, k, v, alpha))
    decay_kk = gk.float().exp()[..., None, None] if gk is not None else k.new_ones(B, L, H, 1, 1)
    decay_kv = g.float().exp()[..., None, None] if g is not None else k.new_ones(B, L, H, 1, 1)
    h_kk = k.new_zeros(B, H, D, D) if h_kk_init is None else h_kk_init.float()
    h_kv = k.new_zeros(B, H, D, DV) if h_kv_init is None else h_kv_init.float()
    o = torch.empty(B, L, H, DV, device=q.device, dtype=q.dtype)

    for i in range(L):
        k_i, v_i = k[:, i], v[:, i]
        h_kk = h_kk * decay_kk[:, i] + k_i.unsqueeze(-1) * k_i.unsqueeze(-2)
        h_kv = h_kv * decay_kv[:, i] + k_i.unsqueeze(-1) * v_i.unsqueeze(-2)
        q_i = q[:, i]
        if num_iter > 0:
            x = _chebyshev_solve(h_kk, q_i, ridge_strength, num_iter)
            q_star = q_i + alpha[:, i].unsqueeze(-1) * (x - q_i)
        else:
            q_star = q_i
        o[:, i] = gla_scale * (q_star.unsqueeze(-1) * h_kv).sum(-2)

    return o, h_kk if num_iter > 0 else None, h_kv
