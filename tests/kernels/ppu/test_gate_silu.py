# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
import vllm._C  # noqa: F401

import vllm._custom_ops  # noqa: F401
from vllm.platforms import current_platform

K = 2048
D = 6144

pytestmark = pytest.mark.skipif(not current_platform.is_ppu(), reason="PPU-only kernel")


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    gate_up = F.linear(x, weight)
    output = torch.empty((1, D), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(output, gate_up)
    return output


@pytest.mark.parametrize("seed", [20260625, 20260626, 20260627])
def test_ppu_gate_silu_matches_vllm_fp16_path(seed: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((1, K), generator=generator, device="cuda", dtype=torch.float16)
    weight = (
        torch.randn(
            (2 * D, K),
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        )
        / K**0.5
    ).contiguous()

    reference = _reference(x, weight)
    actual = torch.ops._C.ppu_gate_silu(x, weight)
    torch.cuda.synchronize()

    difference = (reference.float() - actual.float()).abs()
    cosine = F.cosine_similarity(
        reference.float().flatten(), actual.float().flatten(), dim=0
    )
    assert torch.isfinite(actual.float()).all()
    assert difference.max().item() <= 0.02
    assert cosine.item() >= 0.9999


def test_ppu_gate_silu_zero_input_is_exact_zero() -> None:
    x = torch.zeros((1, K), device="cuda", dtype=torch.float16)
    weight = torch.empty((2 * D, K), device="cuda", dtype=torch.float16)
    actual = torch.ops._C.ppu_gate_silu(x, weight)
    torch.cuda.synchronize()
    assert torch.count_nonzero(actual).item() == 0


def test_ppu_gate_silu_rejects_prefill_shape() -> None:
    x = torch.empty((2, K), device="cuda", dtype=torch.float16)
    weight = torch.empty((2 * D, K), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match=r"x must have shape \[1, 2048\]"):
        torch.ops._C.ppu_gate_silu(x, weight)
