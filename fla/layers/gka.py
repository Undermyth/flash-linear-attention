"""
Gated KalmaNet (GKA) layer implementation with Grouped Query Attention (GQA) support.

This module implements the Gated KalmaNet layer from the paper:
"Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression" (https://arxiv.org/abs/2511.21016).

Overview:
    GKA is a state space model (SSM) that updates its state based on the entire history
    of previously seen tokens, rather than just the current token. At each timestep t,
    the optimal state S_t minimizes a ridge regression objective over all past tokens:

        S_t = argmin_S { λ·||S||²_F + Σ(i=1 to t) η_i·||S·k_i - v_i||² }

    where λ is the ridge regularization strength and η_i are exponentially decaying weights
    controlled by forgetting gates. This has a closed-form solution:

        S_t = U_t · (H_t + λI)^(-1)

    where:
        U_t = Σ(i=1 to t) η_i v_i k_i^T    (cumulative value-key outer products)
        H_t = Σ(i=1 to t) η_i k_i k_i^T    (cumulative key-key covariance)

    The output is computed as y_t = S_t · q_t, which reduces to solving the ridge
    regression problem (H_t + λI)·x_t = q_t for x_t, then computing y_t = U_t · x_t.
    This is solved numerically via the Chebyshev iteration algorithm.

    The layer supports GQA-style head grouping where the number of key/value heads
    can be different from the number of query heads. The hyperparameters in this module
    are set so that we can initialize the module's weights from existing Attention layers.

Key Variables:
    - head_dim: Head dimension (same for q/k/v)
    - num_q_heads: Number of query heads
    - num_k_heads: Number of key heads
    - num_v_heads: Number of value heads
    - expanded_num_heads: Maximum of num_q_heads, num_k_heads, num_v_heads
    - num_q_groups: Repetition factor for Q heads, computed as expanded_num_heads // num_q_heads
    - num_k_groups: Repetition factor for K heads, computed as expanded_num_heads // num_k_heads
    - num_v_groups: Repetition factor for V heads, computed as expanded_num_heads // num_v_heads
    - query_dim: Total query dimension, num_q_heads * head_dim
    - key_dim: Total key dimension, num_k_heads * head_dim
    - value_dim: Total value dimension, num_v_heads * head_dim
    - expanded_dim: Expanded dimension, expanded_num_heads * head_dim
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution

if TYPE_CHECKING:
    from fla.models.utils import Cache

# from ..sp_p2p_utils import reorder_for_ssm_p2p
# from ..low_rank_linear import LinearLowRank
# from .ops.chebyshev.chunk_dk_backward_gating_sp import compute_kk_final_with_gating
# from .sequence_parallel.gka_sp_utils import (
#     state_passing_gka_p2p,
#     recompute_with_initial_states,
# )

from fla.ops.gka.gka_chebyshev_solve import gka_chebyshev_gla, torch_decoding_one_step

# if TYPE_CHECKING:
#     from hmf.model.hybrid_zoo.models.cache import GatedKalmaNetCache


class LinearLowRank(nn.Module):
    """
    Low-rank linear layer that factorizes a linear projection into two smaller ones:
        y = linearB(act_fn(linearA(x)))

    where linearA projects from in_features to rank, and linearB projects from rank to out_features.

    linearB is zero-initialized so that the module outputs zeros at init time. This makes it safe
    to use as a residual branch (e.g. added on top of a repeat/expand) without perturbing the
    pretrained signal at the start of training.

    Note: linearB is marked with ``_is_hf_initialized = True`` to prevent transformers'
    ``post_init`` from overwriting the zero initialization with a random normal.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: Bottleneck dimension between the two linear layers.
        act_fn: Optional activation applied between linearA and linearB. Defaults to Identity.
        bias: Whether to include bias terms in both linear layers. Defaults to False.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        act_fn: Any | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.linearA = nn.Linear(in_features, rank, bias=bias)
        self.linearB = nn.Linear(rank, out_features, bias=bias)

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.bias = bias
        self.act_fn = act_fn if act_fn is not None else nn.Identity()

        # Initialize B weights and bias to zero
        nn.init.zeros_(self.linearB.weight)
        if self.linearB.bias is not None:
            nn.init.zeros_(self.linearB.bias)
        self.linearB._is_hf_initialized = (
            True
        )  # prevent transformers' post_init from re-initializing

    def forward(self, x: torch.Tensor):
        return self.linearB(self.act_fn(self.linearA(x)))


class GatedKalmaNet(nn.Module):
    """
    GKA layer with Chebyshev solver and GQA support.

    Args:
        hidden_size: The hidden size of the input. Defaults to 2048.
        head_dim: The dimension of each head. Assumes q/k/v have the same head dimension.
            Defaults to 128.
        num_q_heads: The number of query heads. Defaults to 16.
        num_k_heads: The number of key heads for GQA. If None, defaults to num_q_heads.
            Defaults to None.
        num_v_heads: The number of value heads for GQA. If None, defaults to num_q_heads.
            Defaults to None.
        use_beta_gate: Whether to use beta gating mechanism. Defaults to True.
        use_alpha_connection: Whether to use alpha connection for residual paths. Defaults to True.
        use_v_conv: Whether to apply convolution to value vectors. Defaults to True.
        use_forgetting_gate: Whether to use forgetting gate mechanism. Defaults to True.
        use_forgetting_gate_kk: Whether to use forgetting gate for key-key interactions. Defaults to True.
        gla_rescale: Whether to apply GLA-style rescaling. Defaults to True.
        solver_type: Solver method. Currently, we only support "chebyshev". Defaults to "chebyshev".
        bp_lambda: Whether to backpropagate through lambda parameters. Defaults to True.
        num_iter: Number of iterations for iterative solvers. Defaults to 30.
        ridge_strength: Regularization strength for ridge regression. Defaults to 0.02.
        use_gate: Whether to use output gating mechanism. Defaults to True.
        conv_size: Size of the convolution kernel. Defaults to 4.
        layer_idx: The index of the layer. Used for caching during inference. Defaults to None.
        norm_eps: The epsilon value for the normalization layer. Defaults to 1e-6.
        chunk_size: Size of chunks for parallel processing in Triton kernels. Defaults to 64.
        kv_proj_rank: Rank for low-rank KV projection to expand K/V heads. If None, uses
            standard GQA repeat. Defaults to None.
        kv_learnable_residual: Whether to use a learnable residual for the KV head expansion
            instead of a fixed repeat. Only used when kv_proj_rank is set. Defaults to False.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 128,
        num_q_heads: int = 16,
        num_k_heads: int | None = None,
        num_v_heads: int | None = None,
        use_beta_gate: bool = True,
        use_alpha_connection: bool = True,
        use_v_conv: bool = True,
        use_forgetting_gate: bool = True,
        use_forgetting_gate_kk: bool = True,
        gla_rescale: bool = True,
        solver_type: str = "chebyshev",
        bp_lambda: bool = True,
        num_iter: int = 30,
        ridge_strength: float = 0.02,
        use_gate: bool = True,
        conv_size: int = 4,
        layer_idx: int | None = None,
        norm_eps: float = 1e-6,
        chunk_size: int = 64,
        kv_proj_rank: int | None = None,
        kv_learnable_residual: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        assert solver_type == "chebyshev", f"Received {solver_type=}. Currently, GKA only supports 'chebyshev' solver."

        self.hidden_size = hidden_size

        self.use_gate = use_gate
        self.conv_size = conv_size
        self.sequence_parallel_group = None
        self.chunk_size = chunk_size

        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_k_heads = num_k_heads if num_k_heads is not None else num_q_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_q_heads
        self.expanded_num_heads = max(
            self.num_q_heads, self.num_k_heads, self.num_v_heads
        )
        self.num_q_groups = self.expanded_num_heads // self.num_q_heads
        self.num_k_groups = self.expanded_num_heads // self.num_k_heads
        self.num_v_groups = self.expanded_num_heads // self.num_v_heads

        self.query_dim = self.num_q_heads * self.head_dim
        self.key_dim = self.num_k_heads * self.head_dim
        self.value_dim = self.num_v_heads * self.head_dim
        self.expanded_dim = self.expanded_num_heads * self.head_dim

        self.layer_idx = layer_idx

        self.ridge_strength = ridge_strength
        self.num_iter = num_iter
        self.use_beta_gate = use_beta_gate
        self.use_alpha_connection = use_alpha_connection
        self.solver_type = solver_type
        self.bp_lambda = bp_lambda
        self.use_v_conv = use_v_conv
        self.use_forgetting_gate = use_forgetting_gate
        self.use_forgetting_gate_kk = use_forgetting_gate_kk

        if gla_rescale:
            self.gla_scale = self.head_dim ** -0.5
        else:
            self.gla_scale = 1.0

        # Projection layers
        self.q_proj = nn.Linear(hidden_size, self.query_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.a_proj = nn.Linear(hidden_size, self.expanded_num_heads, bias=False)

        self.b_proj = (
            nn.Linear(hidden_size, self.expanded_num_heads, bias=True)
            if self.use_beta_gate
            else None
        )

        if self.use_alpha_connection:
            self.alpha_proj = nn.Linear(hidden_size, self.expanded_num_heads, bias=True)

        # Initialize A parameter (log scale)
        A = torch.empty(self.expanded_num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # Initialize dt bias parameter
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.expanded_num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Short convolution layers
        self.conv_size = conv_size
        self.q_conv1d = ShortConvolution(
            hidden_size=self.query_dim, kernel_size=conv_size, activation="silu"
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=self.key_dim, kernel_size=conv_size, activation="silu"
        )
        if self.use_v_conv:
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim, kernel_size=conv_size, activation="silu"
            )

        # Output gating and normalization
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.expanded_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.expanded_dim, hidden_size, bias=False)

        # Optional KV projection for GQA
        self.kv_proj_rank = kv_proj_rank
        self.kv_learnable_residual = kv_learnable_residual
        if self.kv_proj_rank is not None:
            act_fn = nn.SiLU()
            self.proj_k = LinearLowRank(
                in_features=self.key_dim,
                out_features=self.expanded_num_heads * self.head_dim,
                rank=self.kv_proj_rank,
                act_fn=act_fn,
            )
            self.proj_v = LinearLowRank(
                in_features=self.value_dim,
                out_features=self.expanded_num_heads * self.head_dim,
                rank=self.kv_proj_rank,
                act_fn=act_fn,
            )

            if self.kv_learnable_residual:
                # Learnable mixing matrices initialized to mimic repeat_kv
                eye_k = torch.eye(self.num_k_heads)
                ones_k = torch.ones((self.num_k_groups, 1))
                self.k_rep = nn.Parameter(
                    torch.kron(eye_k, ones_k)
                )  # [expanded_num_heads, num_k_heads]

                eye_v = torch.eye(self.num_v_heads)
                ones_v = torch.ones((self.num_v_groups, 1))
                self.v_rep = nn.Parameter(
                    torch.kron(eye_v, ones_v)
                )  # [expanded_num_heads, num_v_heads]

    def _expand_kv(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Expand k and v from their native head counts to expanded_num_heads.

        Three modes:
        1. kv_proj_rank + kv_learnable_residual: learnable mixing residual + low-rank projection
        2. kv_proj_rank without learnable residual: plain repeat residual + low-rank projection
        3. No projection: simple repeat (or rearrange if groups == 1). This is the standard GQA repeat.

        Args:
            k: [B, L, Dk] raw key tensor (flat, not yet reshaped to heads)
            v: [B, L, Dv] raw value tensor (flat, not yet reshaped to heads)

        Returns:
            k: [B, L, H, D] expanded keys
            v: [B, L, H, D] expanded values
        """
        if self.kv_proj_rank is not None and (
            self.num_k_groups > 1 or self.num_v_groups > 1
        ):
            # Build residual
            if self.kv_learnable_residual:
                k_heads = rearrange(k, "... (h d) -> ... h d", h=self.num_k_heads)
                k_res = torch.einsum("eh,blhd->bled", self.k_rep, k_heads)
                v_heads = rearrange(v, "... (h d) -> ... h d", h=self.num_v_heads)
                v_res = torch.einsum("eh,blhd->bled", self.v_rep, v_heads)
            else:
                k_res = repeat(
                    rearrange(k, "... (h d) -> ... h d", h=self.num_k_heads),
                    "... h d -> ... (h g) d",
                    g=self.num_k_groups,
                )
                v_res = repeat(
                    rearrange(v, "... (h d) -> ... h d", h=self.num_v_heads),
                    "... h d -> ... (h g) d",
                    g=self.num_v_groups,
                )

            # Low-rank projection to expanded dim + add residual
            k = (
                rearrange(
                    self.proj_k(k), "... (h d) -> ... h d", h=self.expanded_num_heads
                )
                + k_res
            )
            v = (
                rearrange(
                    self.proj_v(v), "... (h d) -> ... h d", h=self.expanded_num_heads
                )
                + v_res
            )
        else:
            # Plain rearrange + repeat
            k = rearrange(k, "... (h d) -> ... h d", h=self.num_k_heads)
            v = rearrange(v, "... (h d) -> ... h d", h=self.num_v_heads)
            if self.num_k_groups > 1:
                k = repeat(k, "... h d -> ... (h g) d", g=self.num_k_groups)
            if self.num_v_groups > 1:
                v = repeat(v, "... h d -> ... (h g) d", g=self.num_v_groups)

        return k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        """
        GKA forward pass with optional caching.

        NOTE: We use the following variables for tensor shape hints:
            - B: Input batch size
            - L: Sequence length (full sequence length before SP distribution)
            - Dh: Hidden dimension (self.hidden_size)
            - H: Expanded number of heads (self.expanded_num_heads)
            - Hq: Number of query heads (self.num_q_heads)
            - Hk: Number of key heads (self.num_k_heads)
            - Hv: Number of value heads (self.num_v_heads)
            - D: Head dimension (self.head_dim)
            - Dq: Query dimension (self.query_dim)
            - Dk: Key dimension (self.key_dim)
            - Dv: Value dimension (self.value_dim)
            - Dexp: Expanded dimension (self.expanded_dim)
            - C: Convolution size (self.conv_size)
            - SP: Sequence parallel size. 1 if not using sequence parallel

            - B_sp: Batch size after SP reordering. B_sp = B*2 if self.sequence_parallel_group else B_sp = B
            - L_sp: Sequence length after SP reordering. L_sp = L/(SP*2) + C-1 if self.sequence_parallel_group else L_sp = L/SP
            - Lp: Sequence length after SP reordering and conv padding removal. Lp = L/(SP*2) if self.sequence_parallel_group else Lp = L/SP

        Args:
            hidden_states: Input tensor with shape [B, L/SP, Dh].
            attention_mask: Optional 0-1 mask of shape [B, L] used to unpad padded
                inputs into a packed sequence (see `unpad_hidden_states`).
            past_key_values: Optional cache containing conv_state and recurrent_state
                for inference. When provided, performs incremental decoding.
            use_cache: Whether to return updated cache states. Defaults to False.
            output_attentions: Whether to return attention weights (not used). Defaults to False.
            **kwargs: Additional keyword arguments, may include cu_seqlens for packed sequences.

        Returns:
            Tuple of (output, attention_weights, past_key_values) where:
                - output: Output tensor with same shape as hidden_states [B, L, Dh]
                - attention_weights: None (not computed for GKA)
                - past_key_values: Updated cache (same as input if provided, else None)
        """
        # During inference with padding, the mask must be a 0-1 matrix; it is
        # converted to `cu_seqlens` so the varlen kernels can consume it packed.
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape

        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get("cu_seqlens")
        hidden_states, indices, cu_seqlens = unpad_hidden_states(
            hidden_states, cu_seqlens, attention_mask, q_len
        )

        # Apply projections and convolutions
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if last_state is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        # q: [B_sp, L_sp, Dq]
        # conv_state_q: [B, Dq, C] or None

        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        # k: [B_sp, L_sp, Dk]
        # conv_state_k: [B, Dk, C] or None

        if self.use_v_conv:
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            # v: [B_sp, L_sp, Dv]
            # conv_state_v: [B, Dv, C] or None
        else:
            v = self.v_proj(hidden_states)  # [B_sp, L_sp, Dv]
            conv_state_v = None

        # Reshape q to heads and expand k/v (handles projection + residual if configured)
        q = rearrange(
            q, "... (h d) -> ... h d", h=self.num_q_heads
        )  # [B_sp, Lp, Hq, D]
        k, v = self._expand_kv(k, v)  # [B_sp, Lp, H, D] each

        # Apply GQA/head repeats for q to match expanded_num_heads
        if self.num_q_groups > 1:
            q = repeat(q, "... h d -> ... (h g) d", g=self.num_q_groups)
        # q: [B_sp, Lp, H, D]

        # Get recurrent state from cache
        last_h_kk, last_h_kv = (
            last_state["recurrent_state"] if last_state is not None else (None, None)
        )

        # Normalize queries and keys
        qk_norm_eps = 1e-6
        q = q / (
            torch.linalg.vector_norm(q, dim=-1, keepdim=True).to(q) + qk_norm_eps
        )  # [B_sp, Lp, H, D]
        k = k / (
            torch.linalg.vector_norm(k, dim=-1, keepdim=True).to(k) + qk_norm_eps
        )  # [B_sp, Lp, H, D]

        # Apply beta gating if enabled
        if self.use_beta_gate:
            # NOTE: not sure if should be float32 here. only MesaNet use that
            beta = self.b_proj(hidden_states).float().sigmoid()  # [B_sp, Lp, H]
            eps = 1e-6
            k = ((beta[..., None] + eps) * k).to(k.dtype)  # [B_sp, Lp, H, D]
            v = ((beta[..., None] + eps) * v).to(v.dtype)  # [B_sp, Lp, H, D]

        # Compute alpha connection parameter
        if self.use_alpha_connection:
            alpha = self.alpha_proj(hidden_states).sigmoid()  # [B_sp, Lp, H]
        else:
            alpha = torch.ones_like(q[..., 0])  # [B_sp, Lp, H]

        # Compute forgetting gate
        if self.use_forgetting_gate:
            g = -self.A_log.float().exp() * F.softplus(
                self.a_proj(hidden_states).float() + self.dt_bias
            )  # [B_sp, Lp, H]
        else:
            g = None

        gk = g if self.use_forgetting_gate_kk else None

        # Backward kernels require sequence length to be divisible by chunk size;
        # the varlen kernels instead mask the ragged chunk tails.
        if gk is None and self.training and cu_seqlens is None:
            seq_len = hidden_states.shape[1]
            assert (
                seq_len % self.chunk_size == 0
            ), f"Triton chunk size must divide the sequence length. Got seq_len={seq_len}, chunk_size={self.chunk_size}"

        # Apply GKA Chebyshev solver
        if last_state is None:
            with torch.cuda.device(q.device):
                o, h_kk, h_kv = gka_chebyshev_gla(
                    q=q,
                    k=k,
                    v=v,
                    alpha=alpha,
                    g=g,
                    gk=gk,
                    gla_scale=self.gla_scale,
                    ridge_strength=self.ridge_strength,
                    solver_type=self.solver_type,
                    num_iter=self.num_iter,
                    bp_lambda=self.bp_lambda,
                    chunk_size=self.chunk_size,
                    cu_seqlens=cu_seqlens,
                )
                # o: [B_sp, Lp, H, D]
                # h_kk: [N, H, D, D] (N = B, or the number of packed sequences)
                # h_kv: [N, H, D, D]
        else:
            if cu_seqlens is not None:
                raise NotImplementedError(
                    "GKA single-step decoding does not support `cu_seqlens` yet."
                )
            with torch.cuda.device(q.device):
                o, h_kk, h_kv = torch_decoding_one_step(
                    q=q.squeeze(0),
                    k=k.squeeze(0),
                    v=v.squeeze(0),
                    alpha=alpha.squeeze(0),
                    g=g,
                    gk=gk,
                    gla_scale=self.gla_scale,
                    ridge_strength=self.ridge_strength,
                    solver_type=self.solver_type,
                    num_iter=self.num_iter,
                    prev_h_kk=last_h_kk,
                    prev_h_kv=last_h_kv,
                )
                # o: [Lp, H, D] (batch squeezed)
                # h_kk: [H, D, D]
                # h_kv: [H, D, D]
            if len(o.shape) == 3:
                o = o.unsqueeze(0)  # [1, Lp, H, D]

            o = o.to(q)

        # Update cache
        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=(h_kk, h_kv),
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            offset=q_len
        )

        # Apply output gating and normalization
        if self.use_gate:
            g = rearrange(
                self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim
            )  # [B_sp, Lp, H, D]
            o = self.o_norm(o, g)  # [B_sp, Lp, H, D]
        else:
            o = self.o_norm(o)  # [B_sp, Lp, H, D]

        # Reshape and project output
        o = rearrange(o, "b t h d -> b t (h d)")  # [B_sp, Lp, Dexp]
        o = self.o_proj(o)  # [B_sp, Lp, Dh]
        o = repad_hidden_states(o, indices, batch_size, q_len)  # [B, L, Dh]

        return o, None, past_key_values
