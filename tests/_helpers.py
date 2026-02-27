from pathlib import Path

from engine.generation.config import Config
from engine.generation.scheduler import Scheduler


def make_scheduler(
    model_dir: Path,
    *,
    num_kvcache_blocks: int = 8,
    kvcache_block_size: int = 256,
    max_num_batched_tokens: int = 512,
    max_num_seqs: int = 8,
    max_model_len: int = 4096,
    chunked_prefill: bool = False,
) -> Scheduler:
    return Scheduler(
        Config(
            model=str(model_dir),
            num_kvcache_blocks=num_kvcache_blocks,
            kvcache_block_size=kvcache_block_size,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            chunked_prefill=chunked_prefill,
        )
    )
