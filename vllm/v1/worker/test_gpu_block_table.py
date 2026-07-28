# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.gpu.block_table import BlockTables

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="requires CUDA",
)


def test_block_tables_apply_staged_writes_fuses_kv_groups(monkeypatch):
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16, 32, 8],
        max_num_reqs=4,
        max_num_batched_tokens=64,
        max_num_blocks_per_group=[8, 8, 8],
        device=device,
        kernel_block_sizes=[16, 16, 8],
    )

    def fail_if_apply_write_called():
        pytest.fail("multi-group writes should use the fused apply kernel")

    for block_table in block_tables.block_tables:
        monkeypatch.setattr(block_table, "apply_write", fail_if_apply_write_called)

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([1, 2], [10, 11], []),
        overwrite=True,
    )
    block_tables.append_block_ids(
        req_index=1,
        new_block_ids=([3], [12], [5, 6]),
        overwrite=True,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :2],
        torch.tensor([1, 2], dtype=torch.int32, device=device),
    )
    # Group 1 has blocks_per_kv_block == 2, so each KV block expands to two
    # kernel block IDs.
    assert torch.equal(
        block_tables.block_tables[1].gpu[0, :4],
        torch.tensor([20, 21, 22, 23], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[0].gpu[1, :1],
        torch.tensor([3], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[1].gpu[1, :2],
        torch.tensor([24, 25], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[2].gpu[1, :2],
        torch.tensor([5, 6], dtype=torch.int32, device=device),
    )
    assert block_tables.num_blocks.np[0, 0] == 2
    assert block_tables.num_blocks.np[1, 0] == 4
    assert block_tables.num_blocks.np[2, 0] == 0
    assert block_tables.num_blocks.np[0, 1] == 1
    assert block_tables.num_blocks.np[1, 1] == 2
    assert block_tables.num_blocks.np[2, 1] == 2
    assert torch.equal(
        block_tables.num_blocks.gpu[:, :2],
        torch.tensor([[2, 1], [4, 2], [0, 2]], dtype=torch.int32, device=device),
    )

    for block_table in block_tables.block_tables:
        assert not block_table._staged_write_indices
        assert not block_table._staged_write_starts
        assert not block_table._staged_write_contents
        assert not block_table._staged_write_cu_lens

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([7], [13], [8]),
        overwrite=False,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :3],
        torch.tensor([1, 2, 7], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[1].gpu[0, :6],
        torch.tensor([20, 21, 22, 23, 26, 27], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[2].gpu[0, :1],
        torch.tensor([8], dtype=torch.int32, device=device),
    )
    assert block_tables.num_blocks.np[0, 0] == 3
    assert block_tables.num_blocks.np[1, 0] == 6
    assert block_tables.num_blocks.np[2, 0] == 1


def test_block_tables_apply_staged_writes_single_group():
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16],
        max_num_reqs=2,
        max_num_batched_tokens=16,
        max_num_blocks_per_group=[4],
        device=device,
        kernel_block_sizes=[16],
    )

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([1, 2],),
        overwrite=True,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :2],
        torch.tensor([1, 2], dtype=torch.int32, device=device),
    )


def test_v1_multi_group_block_table_uses_one_packed_commit(monkeypatch):
    device = torch.device("cuda")
    block_tables = MultiGroupBlockTable(
        max_num_reqs=3,
        max_model_len=128,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=device,
        block_sizes=[16, 32],
        kernel_block_sizes=[16, 16],
        max_num_blocks=[4, 4],
    )

    assert block_tables._packed_block_table is not None
    assert block_tables.block_tables[0].block_table.gpu.stride(0) == (
        block_tables._packed_block_table.gpu.stride(0)
    )
    assert block_tables.block_tables[1].block_table.gpu.stride(0) == (
        block_tables._packed_block_table.gpu.stride(0)
    )

    def fail_individual_commit(*args, **kwargs):
        pytest.fail("multi-group commit must use the packed buffer")

    for table in block_tables.block_tables:
        monkeypatch.setattr(table, "commit_block_table", fail_individual_commit)

    block_tables.add_row(([2, 3], [5]), 0)
    block_tables.add_row(([7, 8], [9, 10]), 1)
    block_tables.commit_block_table(num_reqs=2)
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables[0].block_table.gpu[0, :2],
        torch.tensor([2, 3], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables[0].block_table.gpu[1, :2],
        torch.tensor([7, 8], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables[1].block_table.gpu[0, :2],
        torch.tensor([10, 11], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables[1].block_table.gpu[1, :4],
        torch.tensor([18, 19, 20, 21], dtype=torch.int32, device=device),
    )

    def fail_redundant_copy(*args, **kwargs):
        pytest.fail("unchanged block tables must not be copied again")

    monkeypatch.setattr(
        block_tables._packed_block_table, "copy_to_gpu", fail_redundant_copy
    )
    block_tables.commit_block_table(num_reqs=2)


def test_v1_multi_group_slot_mapping_is_fused(monkeypatch):
    device = torch.device("cuda")
    block_tables = MultiGroupBlockTable(
        max_num_reqs=2,
        max_model_len=128,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=device,
        block_sizes=[16, 32],
        kernel_block_sizes=[16, 16],
        max_num_blocks=[4, 4],
    )
    block_tables.add_row(([2, 3], [5]), 0)
    block_tables.add_row(([7, 8], [9, 10]), 1)
    block_tables.commit_block_table(num_reqs=2)

    def fail_individual_mapping(*args, **kwargs):
        pytest.fail("multi-group slot mapping must use one fused kernel")

    for table in block_tables.block_tables:
        monkeypatch.setattr(table, "compute_slot_mapping", fail_individual_mapping)

    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    positions = torch.tensor([0, 15, 16, 0, 17], dtype=torch.int64, device=device)
    block_tables.compute_slot_mapping(2, query_start_loc, positions)
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables[0].slot_mapping.gpu,
        torch.tensor(
            [32, 47, 48, 112, 129, -1, -1, -1],
            dtype=torch.int64,
            device=device,
        ),
    )
    assert torch.equal(
        block_tables[1].slot_mapping.gpu,
        torch.tensor(
            [160, 175, 176, 288, 305, -1, -1, -1],
            dtype=torch.int64,
            device=device,
        ),
    )

    block_tables.compute_slot_mapping(
        1,
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([0], dtype=torch.int64, device=device),
    )
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables[0].slot_mapping.gpu,
        torch.tensor([32, -1, -1, -1, -1, -1, -1, -1], device=device),
    )
    assert torch.equal(
        block_tables[1].slot_mapping.gpu,
        torch.tensor([160, -1, -1, -1, -1, -1, -1, -1], device=device),
    )


def test_v1_single_request_uses_fused_two_group_slot_mapping(monkeypatch):
    device = torch.device("cuda")
    block_tables = MultiGroupBlockTable(
        max_num_reqs=2,
        max_model_len=128,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=device,
        block_sizes=[16, 32],
        kernel_block_sizes=[16, 16],
        max_num_blocks=[4, 4],
    )
    block_tables.add_row(([2, 3], [5]), 0)
    block_tables.commit_block_table(num_reqs=1)

    def fail_individual_mapping(*args, **kwargs):
        pytest.fail("two-group slot mapping must use one fused kernel")

    for table in block_tables.block_tables:
        monkeypatch.setattr(table, "compute_slot_mapping", fail_individual_mapping)

    block_tables.compute_slot_mapping(
        1,
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([17], dtype=torch.int64, device=device),
    )
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables[0].slot_mapping.gpu,
        torch.tensor([49, -1, -1, -1, -1, -1, -1, -1], device=device),
    )
    assert torch.equal(
        block_tables[1].slot_mapping.gpu,
        torch.tensor([177, -1, -1, -1, -1, -1, -1, -1], device=device),
    )
