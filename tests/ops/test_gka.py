# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gka.gka_chebyshev_solve import gka_chebyshev_gla
from fla.ops.gka.naive import naive_gka_chebyshev
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'num_iter', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-iter{}-{}".format(*test))
        for test in [
            (1, 15, 1, 32, 1, torch.float32),
            (1, 63, 2, 32, 5, torch.float32),
            (2, 64, 2, 64, 10, torch.float32),
            (2, 256, 4, 64, 1, torch.float32),
            (2, 256, 4, 64, 5, torch.float32),
            (2, 256, 4, 64, 30, torch.float32),
            (2, 1024, 4, 128, 30, torch.float32),
            (2, 1024, 4, 128, 30, torch.bfloat16),
        ]
    ],
)
@pytest.mark.parametrize(
    ('use_g', 'use_gk'),
    [
        pytest.param(False, False, id="no-g-no-gk"),
        pytest.param(True, False, id="g-no-gk"),
        pytest.param(False, True, id="no-g-gk"),
        pytest.param(True, True, id="g-gk"),
    ],
)
def test_gka_chebyshev_gla(
    B: int,
    T: int,
    H: int,
    D: int,
    num_iter: int,
    dtype: torch.dtype,
    use_g: bool,
    use_gk: bool,
):
    torch.manual_seed(42)

    q = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    v = torch.randn(B, T, H, D, dtype=torch.float32, device=device)
    alpha = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device)) if use_g else None
    gk = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device)) if use_gk else None
    scale = D ** -0.5

    # the naive reference always computes in fp32; the kernels run in `dtype`
    ref_o, ref_hkk, ref_hkv = naive_gka_chebyshev(
        q=q,
        k=k,
        v=v,
        alpha=alpha,
        g=g,
        gk=gk,
        gla_scale=scale,
        num_iter=num_iter,
    )
    tri_o, tri_hkk, tri_hkv = gka_chebyshev_gla(
        q=q.to(dtype),
        k=k.to(dtype),
        v=v.to(dtype),
        alpha=alpha.to(dtype),
        g=g.to(dtype) if g is not None else None,
        gk=gk.to(dtype) if gk is not None else None,
        gla_scale=scale,
        num_iter=num_iter,
    )

    ratio = 1e-2 if dtype == torch.float32 else 5e-2
    assert_close('gla.o', ref_o, tri_o, ratio)
    if num_iter > 0:
        assert ref_hkk is not None and tri_hkk is not None
        assert_close('gla.hkk', ref_hkk, tri_hkk, ratio)
    else:
        assert tri_hkk is None
    assert_close('gla.hkv', ref_hkv, tri_hkv, ratio)
