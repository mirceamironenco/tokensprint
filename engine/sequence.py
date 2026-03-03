from collections.abc import Iterator
from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count
from typing import ClassVar, overload

import torch

from engine.generation.config import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class SequenceInfo:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    seq_need_compute_logits: torch.Tensor
    block_tables: torch.Tensor | None = None


class Sequence:
    # Number of token positions in one KV-cache block.
    block_size: ClassVar[int] = 256
    # Monotonic counter used to assign unique sequence IDs.
    counter: ClassVar[Iterator[int]] = count()

    # Unique identifier for this sequence request.
    seq_id: int
    # Current scheduler lifecycle state.
    status: SequenceStatus
    # Full token history: prompt plus generated completion.
    token_ids: list[int]
    # Most recently appended token ID.
    last_token: int
    # Total number of tokens currently in token_ids.
    num_tokens: int
    # Number of prompt tokens in token_ids.
    num_prompt_tokens: int
    # Number of prefix tokens already backed by cache blocks.
    num_cached_tokens: int
    # Number of tokens to process in the current scheduling step.
    num_new_tokens: int
    # Assigned KV-cache block IDs for this sequence.
    block_table: list[int]
    # Sampling temperature used when selecting next tokens.
    temperature: float
    # Maximum number of completion tokens allowed.
    max_tokens: int
    # Whether EOS should be ignored as a stopping condition.
    ignore_eos: bool

    def __init__(
        self, token_ids: list[int], sampling_params: SamplingParams = SamplingParams()
    ) -> None:
        if not token_ids:
            raise ValueError("token_ids must be non-empty.")

        self.token_ids = copy(token_ids)
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_new_tokens = 0
        self.block_table = []

    def __len__(self) -> int:
        return self.num_tokens

    @overload
    def __getitem__(self, key: int) -> int: ...

    @overload
    def __getitem__(self, key: slice) -> list[int]: ...

    def __getitem__(self, key: int | slice) -> int | list[int]:
        return self.token_ids[key]

    def append_token(self, token_id: int) -> None:
        """Appends one token and updates sequence bookkeeping.

        Args:
            token_id: Token ID to append.

        Raises:
            RuntimeError: If num_tokens diverges from len(token_ids).
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

        if self.num_tokens != len(self.token_ids):
            raise RuntimeError(
                "Inconsistent token count: expected num_tokens to equal "
                f"len(token_ids), got {self.num_tokens} and {len(self.token_ids)}."
            )

    @property
    def is_finished(self) -> bool:
        """Checks whether this sequence is finished.

        Returns:
            True if the sequence status is FINISHED, otherwise False.
        """
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        """Gets the number of generated completion tokens.

        Returns:
            Number of tokens generated beyond the prompt length.
        """
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_context_tokens(self) -> int:
        """Gets the number of tokens currently in scope for the next forward pass.

        Returns:
            Sum of cached tokens and newly scheduled tokens.
        """
        return self.num_cached_tokens + self.num_new_tokens

    @property
    def prompt_token_ids(self) -> list[int]:
        """Gets prompt token IDs for this sequence.

        Returns:
            Token IDs corresponding to the original prompt.
        """
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> list[int]:
        """Gets generated completion token IDs.

        Returns:
            Token IDs generated after the prompt region.
        """
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self) -> int:
        """Gets the number of fully cached blocks.

        Returns:
            Number of blocks covered by num_cached_tokens.
        """
        return self.num_cached_tokens // self.block_size

    @property
    def num_current_blocks(self) -> int:
        """Gets the number of blocks currently tracked in block_table.

        Returns:
            Length of block_table.

        Raises:
            RuntimeError: If computed expected block count does not match
                block_table length.
        """
        expected_num_current_blocks = (
            self.num_cached_tokens + self.num_new_tokens + self.block_size - 1
        ) // self.block_size

        if expected_num_current_blocks != len(self.block_table):
            raise RuntimeError(
                "Inconsistent block_table length: expected "
                f"{expected_num_current_blocks}, got {len(self.block_table)}."
            )

        return len(self.block_table)

    @property
    def num_blocks(self) -> int:
        """Gets the number of blocks needed for all tokens in the sequence.

        Returns:
            Total block count for token_ids using ceiling division.
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    # @property
    # def last_block_num_tokens(self):
    #     return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, block_index: int) -> list[int]:
        """Gets token IDs in the specified block.

        Args:
            block_index: Zero-based block index.

        Returns:
            Token IDs for the selected block slice.

        Raises:
            IndexError: If block_index is outside [0, num_blocks).
        """
        if block_index < 0 or block_index >= self.num_blocks:
            raise IndexError(
                "Block index "
                f"{block_index} is out of range for {self.num_blocks} blocks."
            )

        return self.token_ids[
            block_index * self.block_size : (block_index + 1) * self.block_size
        ]

    def __getstate__(
        self,
    ) -> tuple[list[int], int, int, int, int, int, list[int], float]:
        return (
            self.token_ids,
            self.last_token,
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_new_tokens,
            self.block_table,
            self.temperature,
        )

    def __setstate__(
        self, state: tuple[list[int], int, int, int, int, int, list[int], float]
    ) -> None:
        (
            self.token_ids,
            self.last_token,
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_new_tokens,
            self.block_table,
            self.temperature,
        ) = state
