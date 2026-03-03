import math
from collections import deque

import numpy as np
import xxhash

from engine.sequence import Sequence


class Block:
    # Index in BlockManager.blocks for this physical KV-cache slot.
    block_id: int
    # Number of live sequences currently referencing this block.
    ref_count: int
    # Prefix-cache hash key for a full block; -1 means not cacheable/stale.
    hash: int
    # Exact token ids stored in this block, used to verify cache hits.
    token_ids: list[int]

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]) -> None:
        self.hash = hash
        self.token_ids = token_ids

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1

        # Note: Do not use clear(): token_ids may alias external lists from update().
        self.token_ids = []


class BlockManager:
    """
    Blocks (or tokens) layout:

    ----------------------------------------------------------------------
    | < computed > | < new_computed > |       < new >       |
    ----------------------------------------------------------------------
    |     < Prefix-cached tokens >    |  < to be computed > |
    ----------------------------------------------------------------------
                                      | < to be allocated > |
    ----------------------------------------------------------------------
                                      |   < to be cached >  |
    ----------------------------------------------------------------------

    """

    # Number of token positions per KV-cache block.
    block_size: int
    # Fixed pool of all blocks; each item represents one cache slot.
    blocks: list[Block]
    # Prefix-cache index from block hash to block_id.
    hash_to_block_id: dict[int, int]
    # Queue of block ids that are free and can be allocated.
    free_block_ids: deque[int]
    # Set of block ids currently allocated/in use.
    used_block_ids: set[int]

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(block_id) for block_id in range(num_blocks)]
        self.hash_to_block_id = dict()
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        """Computes a 64-bit hash for a block of token IDs.

        The hash can be chained with a prefix hash so each full block hash
        depends on all previous full blocks in the sequence prefix.

        Args:
            token_ids: Token IDs in the current block.
            prefix: Hash of the previous full block, or -1 if there is no prefix.

        Returns:
            A 64-bit integer digest for the current block content and prefix.
        """
        h = xxhash.xxh64()

        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))

        h.update(np.array(token_ids).tobytes())

        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """Marks a free block as in-use and clears stale cache indexing.

        A block can still be present in `hash_to_block_id` while free
        (`ref_count == 0`) so it can be reused by prefix caching. When we
        repurpose that block for a new allocation, we must remove its previous
        hash entry to avoid stale hash->block mappings.

        Args:
            block_id: ID of the free block to allocate.

        Returns:
            The allocated block object.
        """
        block = self.blocks[block_id]

        if block.ref_count != 0:
            raise RuntimeError(
                f"Cannot allocate block {block_id}: expected ref_count == 0, "
                f"got {block.ref_count}."
            )

        # Re-purpose block for a new allocation.
        if self.hash_to_block_id.get(block.hash) == block_id:
            self.hash_to_block_id.pop(block.hash, None)

        block.reset()

        if self.free_block_ids and self.free_block_ids[0] == block_id:
            self.free_block_ids.popleft()
        else:
            self.free_block_ids.remove(block_id)

        self.used_block_ids.add(block_id)

        return block

    def _deallocate_block(self, block_id: int) -> None:
        """Returns an unreferenced block to the free-block pool.

        This helper only updates allocator bookkeeping. The caller must ensure
        all sequence references were released before deallocation.

        Args:
            block_id: ID of the block to return to the free pool.

        Raises:
            RuntimeError: If the block is still referenced by one or more
                sequences.
        """
        block = self.blocks[block_id]
        if block.ref_count != 0:
            raise RuntimeError(
                f"Cannot deallocate block {block_id}: expected ref_count == 0, "
                f"got {block.ref_count}."
            )
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate_sequence(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def can_allocate(self, num_tokens: int) -> bool:
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return len(self.free_block_ids) >= blocks_needed

    def _allocate_cached_prefix(self, seq: Sequence) -> int:
        """Allocates reusable prefix-cached blocks for a waiting sequence.

        This method walks sequence-local blocks from the beginning and reuses
        matching full blocks from prefix cache until the first cache miss
        boundary. It updates `seq.num_cached_tokens` and `seq.block_table`.

        Args:
            seq: Waiting sequence being allocated.

        Returns:
            Rolling hash value at the cache-miss boundary, or -1 when boundary
            is a non-cacheable partial block.
        """
        h = -1
        for logical_block_index in range(seq.num_blocks):
            token_ids = seq.block(logical_block_index)

            # Don't cache partial blocks.
            if len(token_ids) != self.block_size:
                h = -1
                break

            h = self.compute_hash(token_ids, prefix=h)

            # Cache miss boundary: keep the last block in new_tokens.
            if logical_block_index == seq.num_blocks - 1:
                break

            block_id = self.hash_to_block_id.get(h, -1)

            # Cache miss.
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break

            # Cache hit.
            seq.num_cached_tokens += self.block_size

            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                block = self._allocate_block(block_id)

            block.update(h, token_ids)

            self.hash_to_block_id[h] = block_id

            seq.block_table.append(block_id)

        return h

    def allocate(self, seq: Sequence) -> None:
        """Allocates/attaches KV-cache blocks for a waiting sequence.

        This method has two phases:
        1. Reuse a cached prefix from `hash_to_block_id` when full-block hashes
           and token contents match. These are tracked as cached tokens.
        2. Allocate fresh physical blocks for remaining scheduled tokens
           (`seq.num_new_tokens`), optionally indexing full blocks into prefix
           cache.

        `logical_block_index` is the sequence-local block position, while
        `block_id` is a physical cache slot ID in `self.blocks`.

        Args:
            seq: Waiting sequence to allocate blocks for.

        Raises:
            RuntimeError: If `seq.block_table` is not empty.
        """
        if seq.block_table:
            raise RuntimeError(
                "BlockManager.allocate expects seq.block_table to be empty for waiting "
                f"sequences, but got {len(seq.block_table)} existing blocks."
            )

        h = self._allocate_cached_prefix(seq)

        # Allocate new_blocks
        for token_start_index in range(
            seq.num_cached_tokens,
            seq.num_cached_tokens + seq.num_new_tokens,
            self.block_size,
        ):
            token_end_index = min(
                token_start_index + self.block_size,
                seq.num_cached_tokens + seq.num_new_tokens,
            )

            token_ids = seq[token_start_index:token_end_index]

            # Avoid hashing the first new block twice by reusing 'h' from prefix cache.
            if token_start_index != seq.num_cached_tokens:
                h = (
                    self.compute_hash(token_ids, h)
                    if len(token_ids) == self.block_size
                    else -1
                )

            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)

            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence) -> None:
        """Releases a sequence's block references and resets its block state.

        This decrements `ref_count` for each block in `seq.block_table` and
        returns blocks to the free pool when their reference count reaches zero.
        After release, sequence-side cache bookkeeping is reset so the sequence
        can be treated as unallocated.

        Args:
            seq: Sequence being finished or preempted.
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.num_new_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence, num_new_tokens: int) -> bool:
        """Checks whether enough free blocks exist to append new tokens.

        For running sequences, some appended tokens may fit in remaining space
        of the current tail block. This method subtracts that tail capacity and
        then checks whether the remaining tokens require more blocks than are
        currently free.

        Args:
            seq: Running sequence to append into.
            num_new_tokens: Number of tokens planned for append in this step.

        Returns:
            True if append can proceed with current free blocks, otherwise False.
        """
        tail_block_capacity = self.block_size - (
            seq.num_cached_tokens % self.block_size
        )

        if tail_block_capacity == self.block_size:
            tail_block_capacity = 0

        if tail_block_capacity >= num_new_tokens:
            return True

        blocks_needed = math.ceil(
            (num_new_tokens - tail_block_capacity) / self.block_size
        )

        if blocks_needed <= len(self.free_block_ids):
            return True

        return False

    def allocate_for_append(self, seq: Sequence) -> None:
        """Prepares/allocates blocks for appending tokens to a running sequence.

        This method updates `seq.block_table` for the range
        `[seq.num_cached_tokens, seq.num_cached_tokens + seq.num_new_tokens)`.
        It reuses already-present blocks when available, allocates missing
        blocks from the free pool, and updates prefix-cache metadata for full
        blocks. Partial tail blocks are allocated but not inserted into
        `hash_to_block_id`.

        Expected call pattern is:
        1. `can_append(seq, num_new_tokens)` to verify capacity.
        2. Set `seq.num_new_tokens`.
        3. `allocate_for_append(seq)` to realize block allocations/metadata.

        Args:
            seq: Running sequence being prepared for append.
        """
        for token_start_index in range(
            seq.num_cached_blocks * self.block_size,
            seq.num_cached_tokens + seq.num_new_tokens,
            self.block_size,
        ):
            token_end_index = min(
                token_start_index + self.block_size,
                seq.num_cached_tokens + seq.num_new_tokens,
            )

            token_ids = seq[token_start_index:token_end_index]

            logical_block_index = token_start_index // self.block_size

            current_block_id = (
                seq.block_table[logical_block_index]
                if logical_block_index < len(seq.block_table)
                else -1
            )

            # Don't cache partial blocks.
            if len(token_ids) != self.block_size:
                if current_block_id == -1:
                    new_block_id = self.free_block_ids[0]
                    self._allocate_block(new_block_id)
                    seq.block_table.append(new_block_id)
                continue

            previous_block_id = (
                seq.block_table[logical_block_index - 1]
                if token_start_index >= self.block_size
                else -1
            )

            prefix_hash = (
                self.blocks[previous_block_id].hash if previous_block_id != -1 else -1
            )

            h = self.compute_hash(token_ids, prefix=prefix_hash)

            if current_block_id == -1:
                new_block_id = self.free_block_ids[0]
                current_block = self._allocate_block(new_block_id)
                seq.block_table.append(new_block_id)
            else:
                current_block = self.blocks[current_block_id]
                if current_block.hash != -1:
                    # Idempotent re-visit can occur with chunked scheduling.
                    # If the hash already matches, keep the block and refresh
                    # stored tokens for consistency.
                    if current_block.hash == h:
                        if current_block.token_ids != token_ids:
                            current_block.token_ids = token_ids
                        self.hash_to_block_id[h] = current_block.block_id
                        continue
                    raise RuntimeError(
                        "Expected append target block to be non-hashed (hash == -1), "
                        f"but block {current_block_id} has hash {current_block.hash}. "
                        f"Expected hash {h} for token_ids length {len(token_ids)}."
                    )

            current_block.update(h, token_ids)
            self.hash_to_block_id[h] = current_block.block_id

    def get_token_layout(self, seq: Sequence) -> tuple[int, int, int]:
        """Computes token layout for a waiting sequence against prefix cache.

        The sequence is scanned block-by-block to classify tokens into:
        1. Full cached-prefix tokens whose block is currently in use.
        2. Full cached-prefix tokens whose block is currently free.
        3. New tokens to be computed from the first cache-miss boundary onward.

        A cache miss boundary is triggered by hash miss, token mismatch, or the
        last block (kept in `new_tokens` in this implementation).

        Args:
            seq: Waiting sequence to inspect.

        Returns:
            A tuple:
                - num_new_computed_tokens_in_used
                - num_new_computed_tokens_in_free
                - num_new_tokens

        Raises:
            RuntimeError: If `seq.block_table` is not empty.
        """
        if seq.block_table:
            raise RuntimeError(
                "BlockManager.get_token_layout expects seq.block_table to be empty "
                f"for waiting sequences, but got {len(seq.block_table)} existing "
                "blocks."
            )

        num_new_tokens = 0
        num_new_computed_tokens_in_used = 0
        num_new_computed_tokens_in_free = 0

        rolling_hash = -1
        seen_cache_miss_boundary = False

        for logical_block_index in range(seq.num_blocks):
            token_ids = seq.block(logical_block_index)

            if not seen_cache_miss_boundary:
                # Don't cache partial blocks.
                if len(token_ids) != self.block_size:
                    rolling_hash = -1
                else:
                    rolling_hash = self.compute_hash(token_ids, rolling_hash)

                cached_block_id = self.hash_to_block_id.get(rolling_hash, -1)

                if (
                    cached_block_id == -1
                    or self.blocks[cached_block_id].token_ids != token_ids
                    or logical_block_index == seq.num_blocks - 1
                ):
                    seen_cache_miss_boundary = True

            if seen_cache_miss_boundary:
                num_new_tokens += len(token_ids)
                continue

            if cached_block_id in self.used_block_ids:
                num_new_computed_tokens_in_used += len(token_ids)
            else:
                num_new_computed_tokens_in_free += len(token_ids)

        return (
            num_new_computed_tokens_in_used,
            num_new_computed_tokens_in_free,
            num_new_tokens,
        )
