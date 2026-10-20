# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Marlin MoE block_size_m must respect the device's shared-memory budget.

Regression test for the "Invalid thread config ... max_shared_mem=101376"
startup crash of 8-bit Marlin MoE on 99 KB shared-memory GPUs (SM86/SM89/
SM12x) when the expert intermediate size is not a multiple of 128.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    _marlin_moe_tile_smem_bytes,
    marlin_moe_fit_block_size_m,
)

SMEM_99KB = 101376  # RTX PRO 6000 / RTX 5090 / RTX 4090 opt-in limit
SMEM_228KB = 232448  # H100 / B200

# Nemotron 3.5 Lightning 30B-A3B: hidden 2688, intermediate 1856, non-gated.
LIGHTNING = dict(hidden_size=2688, intermediate_size=1856, w13_num_shards=1)


def test_tile_smem_matches_kernel_error_message():
    # The kernel reported exactly 101,376 B for MKN=[8192, 2688, 1856] at
    # block 64 on SM120 (thread_k=128, thread_n=64, MXFP8: 8-bit, group 32).
    assert (
        _marlin_moe_tile_smem_bytes(
            thread_k=128,
            thread_n=64,
            tb_m=64,
            num_bits=8,
            group_size=32,
            has_zp=False,
            is_a_8bit=False,
        )
        == SMEM_99KB
    )


@pytest.mark.parametrize(
    "smem,expected",
    [(SMEM_99KB, 48), (SMEM_228KB, 64)],
)
def test_8bit_narrow_intermediate(smem, expected):
    bm = marlin_moe_fit_block_size_m(
        64,
        **LIGHTNING,
        num_bits=8,
        group_size1=32,
        group_size2=32,
        has_zp=False,
        is_a_8bit=False,
        device=torch.device("cpu"),
        max_shared_mem=smem,
    )
    assert bm == expected


def test_4bit_unchanged_on_99kb():
    bm = marlin_moe_fit_block_size_m(
        64,
        **LIGHTNING,
        num_bits=4,
        group_size1=16,
        group_size2=16,
        has_zp=False,
        is_a_8bit=False,
        device=torch.device("cpu"),
        max_shared_mem=SMEM_99KB,
    )
    assert bm == 64


def test_8bit_wide_intermediate_unchanged_on_99kb():
    # intermediate % 128 == 0 unlocks the (64, 128) tile, which fits at 64.
    bm = marlin_moe_fit_block_size_m(
        64,
        hidden_size=2688,
        intermediate_size=1920,
        w13_num_shards=1,
        num_bits=8,
        group_size1=32,
        group_size2=32,
        has_zp=False,
        is_a_8bit=False,
        device=torch.device("cpu"),
        max_shared_mem=SMEM_99KB,
    )
    assert bm == 64


def test_never_grows_and_unknown_limit_is_noop():
    kwargs = dict(
        **LIGHTNING,
        num_bits=8,
        group_size1=32,
        group_size2=32,
        has_zp=False,
        is_a_8bit=False,
    )
    assert (
        marlin_moe_fit_block_size_m(
            16, device=torch.device("cpu"), max_shared_mem=SMEM_99KB, **kwargs
        )
        == 16
    )
    # Non-CUDA device and no explicit limit: leave the caller's choice alone.
    assert marlin_moe_fit_block_size_m(64, device=torch.device("cpu"), **kwargs) == 64


def test_8bit_activation_keeps_block_floor():
    # With 1-byte activations the kernel needs block_size_m >= 16, so a
    # budget that only fits at 8 must leave the caller's 16 alone.
    bm = marlin_moe_fit_block_size_m(
        16,
        **LIGHTNING,
        num_bits=8,
        group_size1=32,
        group_size2=32,
        has_zp=False,
        is_a_8bit=True,
        device=torch.device("cpu"),
        max_shared_mem=40_000,
    )
    assert bm == 16
