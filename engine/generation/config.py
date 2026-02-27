import os
from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature <= 1e-10:
            raise ValueError("greedy sampling is not permitted")


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    # hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    chunked_prefill: bool = False
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if not os.path.isdir(self.model):
            raise FileNotFoundError(f"Model directory does not exist: {self.model}")

        if self.kvcache_block_size % 256 != 0:
            raise ValueError(
                "kvcache_block_size must be a multiple of 256, got "
                f"{self.kvcache_block_size}."
            )

        if not (1 <= self.tensor_parallel_size <= 8):
            raise ValueError(
                "tensor_parallel_size must be in [1, 8], got "
                f"{self.tensor_parallel_size}."
            )
        # self.hf_config = AutoConfig.from_pretrained(self.model)
        # self.max_model_len = min(
        #     self.max_model_len, self.hf_config.max_position_embeddings
        # )

        # TODO: Revisit this.
        # assert self.max_num_batched_tokens >= self.max_model_len
