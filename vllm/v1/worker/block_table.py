# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.cp_utils import get_total_cp_world_size
from vllm.v1.worker.gpu.buffer_utils import _load_ptr

logger = init_logger(__name__)


class _CpuGpuBufferView(CpuGpuBuffer):
    """A group-local view into a packed ``CpuGpuBuffer``."""

    def __init__(
        self,
        cpu: torch.Tensor,
        gpu: torch.Tensor,
        np_array: np.ndarray,
    ) -> None:
        self.cpu = cpu
        self.gpu = gpu
        self.np = np_array

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)


class BlockTable:
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,
        block_table_buffer: _CpuGpuBufferView | None = None,
        slot_mapping_buffer: _CpuGpuBufferView | None = None,
    ):
        """
        Args:
            block_size: Block size used for KV cache memory allocation
            max_num_reqs: Maximum number of concurrent requests supported.
            max_num_blocks_per_req: Maximum number of blocks per request.
            max_num_batched_tokens: Maximum number of tokens in a batch.
            pin_memory: Whether to pin memory for faster GPU transfers.
            device: Target device for the block table.
            kernel_block_size: The block_size of underlying attention kernel.
                Will be the same as `block_size` if `block_size` is supported
                by the attention kernel.
        """
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device

        if kernel_block_size == block_size:
            # Standard case: allocation and computation use same block size
            # No block splitting needed, direct mapping
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            # Hybrid case: allocation block size differs from kernel block size
            # Memory blocks are subdivided to match kernel requirements
            # Example: 32-token memory blocks with 16-token kernel blocks
            # → Each memory block corresponds to 2 kernel blocks
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )

            self.block_size = kernel_block_size
            self.blocks_per_kv_block = block_size // kernel_block_size
            self.use_hybrid_blocks = True

        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        self.block_table = block_table_buffer or self._make_buffer(
            self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
        )
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        self.slot_mapping = slot_mapping_buffer or self._make_buffer(
            self.max_num_batched_tokens, dtype=torch.int64
        )

        if self.use_hybrid_blocks:
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            self._kernel_block_arange = None

        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            # PCP might not be initialized in testing
            self.pcp_world_size = 1
            self.pcp_rank = 0
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        if not block_ids:
            return

        if self.use_hybrid_blocks:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )

        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def clear_row(self, row_idx: int) -> None:
        num_blocks = self.num_blocks_per_row[row_idx]
        if num_blocks > 0:
            self.block_table.np[row_idx, :num_blocks] = 0
        self.num_blocks_per_row[row_idx] = 0

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        block_table_np = self.block_table.np
        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        num_tokens = positions.shape[0]
        total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        _compute_slot_mapping_kernel[(num_reqs + 1,)](
            num_tokens,
            self.max_num_batched_tokens,
            query_start_loc,
            positions,
            self.block_table.gpu,
            self.block_table.gpu.stride(0),
            self.block_size,
            self.slot_mapping.gpu,
            TOTAL_CP_WORLD_SIZE=total_cp_world_size,
            TOTAL_CP_RANK=total_cp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=self.cp_kv_cache_interleave_size,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )

    def commit_block_table(self, num_reqs: int) -> None:
        self.block_table.copy_to_gpu(num_reqs)

    def clear(self) -> None:
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)

    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        """Convert kv_manager_block_id IDs to kernel block IDs.

        Example:
            # kv_manager_block_ids: 32 tokens,
            # Kernel block size: 16 tokens
            # blocks_per_kv_block = 2
            >>> kv_manager_block_ids = np.array([0, 1, 2])
            >>> Result: [0, 1, 2, 3, 4, 5]

            # Each kv_manager_block_id maps to 2 kernel block id:
            # kv_manager_block_id 0 → kernel block id [0, 1]
            # kv_manager_block_id 1 → kernel block id [2, 3]
            # kv_manager_block_id 2 → kernel block id [4, 5]
        """
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids

        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )

        return kernel_block_ids.reshape(-1)

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        """Returns the device tensor of the block table."""
        return self.block_table.gpu[:num_reqs]

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table.np

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )


class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks: list[int] | None = None,
        cp_kv_cache_interleave_size: int = 1,
    ) -> None:
        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )
        if max_num_blocks is None:
            # Note(hc): each dcp rank only store
            # (max_model_len//dcp_world_size) tokens in kvcache,
            # so the block_size which used for calc max_num_blocks_per_req
            # must be multiplied by dcp_world_size.
            total_cp_world_size = get_total_cp_world_size()
            max_num_blocks = [
                cdiv(max_model_len, block_size * total_cp_world_size)
                for block_size in block_sizes
            ]

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        # Align to a multiple of (128 / block_size) as required
        # by some attention backends such as TRTLLM (#39324)
        max_num_blocks = [
            cdiv(n, 128 // bs) * (128 // bs) if bs <= 128 else n
            for n, bs in zip(max_num_blocks, block_sizes)
        ]

        self._packed_block_table: CpuGpuBuffer | None = None
        self._packed_slot_mapping: CpuGpuBuffer | None = None
        self._block_table_ptrs: torch.Tensor | None = None
        self._block_table_strides: torch.Tensor | None = None
        self._kernel_block_sizes: torch.Tensor | None = None
        self._block_table_dirty = True
        self._slot_mapping_dirty_tokens = 0

        if not block_sizes:
            self.block_tables = []
            return

        if len(block_sizes) == 1:
            self.block_tables = [
                BlockTable(
                    block_sizes[0],
                    max_num_reqs,
                    max_num_blocks[0],
                    max_num_batched_tokens,
                    pin_memory,
                    device,
                    kernel_block_sizes[0],
                    cp_kv_cache_interleave_size,
                )
            ]
            return

        group_widths = []
        for block_size, kernel_block_size, num_blocks in zip(
            block_sizes, kernel_block_sizes, max_num_blocks
        ):
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )
            group_widths.append(num_blocks * (block_size // kernel_block_size))

        # Keep every group base and row 16-byte aligned for pointer-based Triton
        # loads. The padding is metadata-only and is typically a few integers.
        group_offsets = []
        packed_width = 0
        for group_width in group_widths:
            packed_width = (packed_width + 3) // 4 * 4
            group_offsets.append(packed_width)
            packed_width += group_width
        packed_width = (packed_width + 3) // 4 * 4

        # Pack groups by columns so the active request rows are contiguous and
        # can be copied to the device with one H2D operation per step.
        self._packed_block_table = CpuGpuBuffer(
            max_num_reqs,
            packed_width,
            dtype=torch.int32,
            device=device,
            pin_memory=pin_memory,
        )
        self._packed_slot_mapping = CpuGpuBuffer(
            len(block_sizes),
            max_num_batched_tokens,
            dtype=torch.int64,
            device=device,
            pin_memory=pin_memory,
        )
        self._packed_slot_mapping.gpu.fill_(PAD_SLOT_ID)

        self.block_tables = []
        for group_id, (
            block_size,
            kernel_block_size,
            max_num_blocks_per_req,
            group_width,
            column_offset,
        ) in enumerate(
            zip(
                block_sizes,
                kernel_block_sizes,
                max_num_blocks,
                group_widths,
                group_offsets,
            )
        ):
            column_end = column_offset + group_width
            block_table_buffer = _CpuGpuBufferView(
                self._packed_block_table.cpu[:, column_offset:column_end],
                self._packed_block_table.gpu[:, column_offset:column_end],
                self._packed_block_table.np[:, column_offset:column_end],
            )
            slot_mapping_buffer = _CpuGpuBufferView(
                self._packed_slot_mapping.cpu[group_id],
                self._packed_slot_mapping.gpu[group_id],
                self._packed_slot_mapping.np[group_id],
            )
            self.block_tables.append(
                BlockTable(
                    block_size,
                    max_num_reqs,
                    max_num_blocks_per_req,
                    max_num_batched_tokens,
                    pin_memory,
                    device,
                    kernel_block_size,
                    cp_kv_cache_interleave_size,
                    block_table_buffer=block_table_buffer,
                    slot_mapping_buffer=slot_mapping_buffer,
                )
            )

        self._block_table_ptrs = torch.tensor(
            [table.block_table.gpu.data_ptr() for table in self.block_tables],
            dtype=torch.uint64,
            device=device,
        )
        self._block_table_strides = torch.tensor(
            [table.block_table.gpu.stride(0) for table in self.block_tables],
            dtype=torch.int64,
            device=device,
        )
        self._kernel_block_sizes = torch.tensor(
            [table.block_size for table in self.block_tables],
            dtype=torch.int32,
            device=device,
        )

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        if any(block_ids):
            self._block_table_dirty = True
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        self._block_table_dirty = True
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def clear_row(self, row_idx: int) -> None:
        self._block_table_dirty = True
        for block_table in self.block_tables:
            block_table.clear_row(row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        self._block_table_dirty = True
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        self._block_table_dirty = True
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if not self.block_tables:
            return
        if len(self.block_tables) == 1:
            self.block_tables[0].compute_slot_mapping(
                num_reqs, query_start_loc, positions
            )
            return
        if len(self.block_tables) == 2:
            first_table, second_table = self.block_tables
            num_tokens = positions.shape[0]
            pad_end = max(num_tokens, self._slot_mapping_dirty_tokens)
            self._slot_mapping_dirty_tokens = num_tokens
            total_cp_world_size = (
                first_table.pcp_world_size * first_table.dcp_world_size
            )
            total_cp_rank = (
                first_table.pcp_rank * first_table.dcp_world_size
                + first_table.dcp_rank
            )
            _compute_slot_mappings_two_groups_kernel[(num_reqs + 1,)](
                num_tokens,
                pad_end,
                query_start_loc,
                positions,
                first_table.block_table.gpu,
                second_table.block_table.gpu,
                first_table.slot_mapping.gpu,
                second_table.slot_mapping.gpu,
                BLOCK_TABLE_STRIDE_0=first_table.block_table.gpu.stride(0),
                BLOCK_TABLE_STRIDE_1=second_table.block_table.gpu.stride(0),
                KV_BLOCK_SIZE_0=first_table.block_size,
                KV_BLOCK_SIZE_1=second_table.block_size,
                TOTAL_CP_WORLD_SIZE=total_cp_world_size,
                TOTAL_CP_RANK=total_cp_rank,
                CP_KV_CACHE_INTERLEAVE_SIZE=first_table.cp_kv_cache_interleave_size,
                PAD_ID=PAD_SLOT_ID,
                TILE_SIZE=1024,
            )
            return
        if num_reqs == 1:
            # A fused launch does not amortize its dynamic group metadata for
            # three or more groups on the latency-sensitive serving path.
            for block_table in self.block_tables:
                block_table.compute_slot_mapping(
                    num_reqs, query_start_loc, positions
                )
            return

        assert self._block_table_ptrs is not None
        assert self._block_table_strides is not None
        assert self._kernel_block_sizes is not None
        assert self._packed_slot_mapping is not None

        first_table = self.block_tables[0]
        total_cp_world_size = (
            first_table.pcp_world_size * first_table.dcp_world_size
        )
        total_cp_rank = (
            first_table.pcp_rank * first_table.dcp_world_size
            + first_table.dcp_rank
        )
        _compute_slot_mappings_kernel[(len(self.block_tables), num_reqs + 1)](
            positions.shape[0],
            first_table.max_num_batched_tokens,
            query_start_loc,
            positions,
            self._block_table_ptrs,
            self._block_table_strides,
            self._kernel_block_sizes,
            self._packed_slot_mapping.gpu,
            self._packed_slot_mapping.gpu.stride(0),
            TOTAL_CP_WORLD_SIZE=total_cp_world_size,
            TOTAL_CP_RANK=total_cp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=first_table.cp_kv_cache_interleave_size,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )

    def commit_block_table(self, num_reqs: int) -> None:
        if not self.block_tables or not self._block_table_dirty:
            return
        if self._packed_block_table is not None:
            self._packed_block_table.copy_to_gpu(num_reqs)
            self._block_table_dirty = False
            return
        self.block_tables[0].commit_block_table(num_reqs)
        self._block_table_dirty = False

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()
        if self._packed_slot_mapping is not None:
            self._packed_slot_mapping.gpu.fill_(PAD_SLOT_ID)
        self._slot_mapping_dirty_tokens = 0
        self._block_table_dirty = False

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]


@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mapping_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,  # [num_reqs + 1], int32
    positions_ptr,  # [num_tokens], int64
    block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32 (flat)
    block_table_stride,  # max_num_blocks_per_req
    block_size,
    slot_mapping_ptr,  # [max_num_tokens], int64
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    if req_idx == tl.num_programs(0) - 1:
        # Pad remaining slots for CUDA graph compatibility.
        for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
    row_offset = req_idx * block_table_stride
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        block_indices = pos // virtual_block_size
        block_numbers = tl.load(block_table_ptr + row_offset + block_indices).to(
            tl.int64
        )

        virtual_block_offsets = pos - block_indices * virtual_block_size
        is_local = (
            virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        local_block_offsets = (
            virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
        )

        slot_ids = block_numbers * block_size + local_block_offsets
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mappings_two_groups_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,
    positions_ptr,
    block_table_0_ptr,
    block_table_1_ptr,
    slot_mapping_0_ptr,
    slot_mapping_1_ptr,
    BLOCK_TABLE_STRIDE_0: tl.constexpr,
    BLOCK_TABLE_STRIDE_1: tl.constexpr,
    KV_BLOCK_SIZE_0: tl.constexpr,
    KV_BLOCK_SIZE_1: tl.constexpr,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    if req_idx == tl.num_programs(0) - 1:
        for i in range(num_tokens, max_num_tokens, TILE_SIZE):
            offsets = i + tl.arange(0, TILE_SIZE)
            mask = offsets < max_num_tokens
            tl.store(slot_mapping_0_ptr + offsets, PAD_ID, mask=mask)
            tl.store(slot_mapping_1_ptr + offsets, PAD_ID, mask=mask)
        return

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)
    row_offset_0 = req_idx * BLOCK_TABLE_STRIDE_0
    row_offset_1 = req_idx * BLOCK_TABLE_STRIDE_1
    virtual_block_size_0 = KV_BLOCK_SIZE_0 * TOTAL_CP_WORLD_SIZE
    virtual_block_size_1 = KV_BLOCK_SIZE_1 * TOTAL_CP_WORLD_SIZE

    for i in range(start_idx, end_idx, TILE_SIZE):
        offsets = i + tl.arange(0, TILE_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)

        block_indices_0 = pos // virtual_block_size_0
        block_numbers_0 = tl.load(
            block_table_0_ptr + row_offset_0 + block_indices_0
        ).to(tl.int64)
        virtual_offsets_0 = pos - block_indices_0 * virtual_block_size_0
        is_local_0 = (
            virtual_offsets_0 // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        local_offsets_0 = (
            virtual_offsets_0
            // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_offsets_0 % CP_KV_CACHE_INTERLEAVE_SIZE
        )
        slot_ids_0 = block_numbers_0 * KV_BLOCK_SIZE_0 + local_offsets_0
        tl.store(
            slot_mapping_0_ptr + offsets,
            tl.where(is_local_0, slot_ids_0, PAD_ID),
            mask=mask,
        )

        block_indices_1 = pos // virtual_block_size_1
        block_numbers_1 = tl.load(
            block_table_1_ptr + row_offset_1 + block_indices_1
        ).to(tl.int64)
        virtual_offsets_1 = pos - block_indices_1 * virtual_block_size_1
        is_local_1 = (
            virtual_offsets_1 // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        local_offsets_1 = (
            virtual_offsets_1
            // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_offsets_1 % CP_KV_CACHE_INTERLEAVE_SIZE
        )
        slot_ids_1 = block_numbers_1 * KV_BLOCK_SIZE_1 + local_offsets_1
        tl.store(
            slot_mapping_1_ptr + offsets,
            tl.where(is_local_1, slot_ids_1, PAD_ID),
            mask=mask,
        )


@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mappings_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,
    positions_ptr,
    block_table_ptrs,
    block_table_strides,
    block_sizes,
    slot_mappings_ptr,
    slot_mappings_stride,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    req_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if req_idx == tl.num_programs(1) - 1:
        for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    block_size = tl.load(block_sizes + group_id)

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)
    virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
    row_offset = req_idx * block_table_stride

    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        block_indices = pos // virtual_block_size
        block_numbers = tl.load(block_table_ptr + row_offset + block_indices).to(
            tl.int64
        )

        virtual_block_offsets = pos - block_indices * virtual_block_size
        is_local = (
            virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        local_block_offsets = (
            virtual_block_offsets
            // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
        )

        slot_ids = block_numbers * block_size + local_block_offsets
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)
