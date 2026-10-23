# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""
Optional capture channel for the internals of the GKA Chebyshev solver.

The solver already computes everything an offline analysis needs:

- ``h_kk_chunks``: the chunk-level key-key covariances returned by ``chunk_fwd_h``
  inside :func:`fla.ops.gka.chebyshev_iteration.chebyshev_iteration_forward_triton`,
  i.e. the state ``H`` at every chunk boundary, shape ``[B, NT, H, D, D]``;
- ``x_out``: the Chebyshev iterate after ``num_iter`` steps, shape ``[B, L, H, D]``;
- ``fro``: the per-position Frobenius norm ``||H_t||_F`` computed by the kernel,
  shape ``[B, L, H]``; the ridge strength is ``lambda_t = ridge_strength * fro``;
- ``gkk``: the chunk-local cumulative decay (log2 units) used by the kernel.

This module makes those tensors observable from the outside **without** touching
the Triton kernels and **without** changing the public signature of any solver
entry point.  The channel is disabled by default: when it is off, the solver
keeps no reference to its intermediates and the training path is unchanged.

Consumers (e.g. data-extraction scripts) enable the channel right before the
forward pass, run the model, and read the published tensors right after it.
"""

from __future__ import annotations

import torch

__all__ = [
    "clear",
    "is_enabled",
    "publish",
    "set_enabled",
    "take_last",
]

_ENABLED: bool = False
_LAST: dict[str, torch.Tensor] = {}


def set_enabled(flag: bool = True) -> None:
    """Enable or disable the capture channel.

    Args:
        flag: Whether the channel should collect the solver intermediates.
    """
    global _ENABLED
    _ENABLED = bool(flag)
    if not _ENABLED:
        clear()


def is_enabled() -> bool:
    """Whether the capture channel is currently enabled."""
    return _ENABLED


def publish(**tensors: torch.Tensor) -> None:
    """Publish the intermediates of the most recent solver call.

    The call is a no-op while the channel is disabled, so the solver may invoke
    it unconditionally without paying for it during training.

    Args:
        **tensors: Named tensors; entries whose value is ``None`` are skipped.
    """
    if not _ENABLED:
        return
    _LAST.clear()
    for name, value in tensors.items():
        if value is not None:
            _LAST[name] = value


def take_last() -> dict[str, torch.Tensor]:
    """Return a copy of the tensors published by the most recent solver call."""
    return dict(_LAST)


def clear() -> None:
    """Drop the published intermediates."""
    _LAST.clear()
