from engine.generation.block_manager import BlockManager
from engine.sequence import Sequence


def test_get_token_layout_counts_all_following_blocks_as_new_after_cache_miss_boundary() -> None:
    block_size = Sequence.block_size
    manager = BlockManager(num_blocks=4, block_size=block_size)

    block0 = [10] * block_size
    block1 = [20] * block_size
    block2 = [30] * block_size
    block3 = [40] * block_size

    h0 = manager.compute_hash(block0)
    h1 = manager.compute_hash(block1, h0)
    h2 = manager.compute_hash(block2, h1)
    h3 = manager.compute_hash(block3, h2)

    # Block 0 is a valid cached hit and currently in use.
    manager.blocks[0].ref_count = 1
    manager.blocks[0].update(h0, block0)
    manager.used_block_ids.add(0)
    manager.hash_to_block_id[h0] = 0

    # Block 1 has a hash entry but token content mismatch: this is the miss boundary.
    manager.blocks[1].update(h1, [999] * block_size)
    manager.hash_to_block_id[h1] = 1

    # Later blocks would otherwise be cache hits, but must still be counted as new.
    manager.blocks[2].ref_count = 1
    manager.blocks[2].update(h2, block2)
    manager.used_block_ids.add(2)
    manager.hash_to_block_id[h2] = 2

    manager.blocks[3].update(h3, block3)
    manager.hash_to_block_id[h3] = 3

    seq = Sequence(block0 + block1 + block2 + block3)

    used, free, new = manager.get_token_layout(seq)

    assert used == block_size
    assert free == 0
    assert new == 3 * block_size

