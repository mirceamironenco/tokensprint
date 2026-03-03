from __future__ import annotations

import logging
from collections import deque

from engine.generation.block_manager import BlockManager
from engine.generation.config import Config
from engine.sequence import Sequence, SequenceStatus

log = logging.getLogger(__name__)


class Scheduler:
    # Whether prefill can be chunked to fit current token budget.
    enable_chunked: bool
    # Maximum total sequence length allowed by the model.
    max_model_len: int
    # Maximum number of concurrently running sequences.
    max_num_seqs: int
    # Maximum number of tokens processed in one scheduler step.
    max_num_batched_tokens: int
    # End-of-sequence token ID.
    eos: int
    # KV-cache block allocator/manager shared by scheduler decisions.
    block_manager: BlockManager
    # Queue of waiting sequences that have not started running yet.
    waiting: deque[Sequence]
    # Ordered list of currently running sequences.
    running: list[Sequence]

    def __init__(self, config: Config) -> None:
        self.enable_chunked = config.chunked_prefill
        self.max_model_len = config.max_model_len
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos

        self.block_manager = BlockManager(
            num_blocks=config.num_kvcache_blocks, block_size=config.kvcache_block_size
        )
        self.waiting = deque()
        self.running = []

    def is_finished(self) -> bool:
        return len(self.waiting) == 0 and len(self.running) == 0

    def add_sequence(self, seq: Sequence) -> None:
        if len(seq) > self.max_model_len - 1:
            raise ValueError(
                "Sequence length exceeds max_model_len: "
                f"got {len(seq)}, max allowed is {self.max_model_len - 1}."
            )
        self.waiting.append(seq)

    def schedule(self) -> list[Sequence]:
        """Schedule one execution step under the current token/block budget.

        The algorithm runs in two phases:
        1. Try to schedule currently running sequences first. If a running
           sequence cannot append due to block pressure, preempt from the tail
           of the running queue until append is feasible or the current index is
           reached.
        2. If no preemption happened in phase 1, admit waiting sequences in FIFO
           order while both token budget and sequence slots allow it.

        Returns all sequences selected for this step, and raises if no sequence
        could be scheduled.
        """
        scheduled_seqs: list[Sequence] = []
        scheduled_running_seqs: list[Sequence] = []
        scheduled_new_reqs: list[Sequence] = []
        preempted_seqs: list[Sequence] = []
        token_budget = self.max_num_batched_tokens

        # Schedule from the running queue
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            seq = self.running[req_index]

            num_new_tokens = len(seq) - seq.num_cached_tokens

            if self.enable_chunked:
                num_new_tokens = min(num_new_tokens, token_budget)

            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - seq.num_cached_tokens
            )

            if num_new_tokens <= 0:
                raise RuntimeError(
                    "Expected a positive num_new_tokens for running sequence "
                    f"{seq.seq_id}, but got {num_new_tokens}."
                )

            while True:
                if self.block_manager.can_append(seq, num_new_tokens):
                    seq.num_new_tokens = num_new_tokens
                    self.block_manager.allocate_for_append(seq)
                    break
                else:
                    preempted_seq = self.running.pop()
                    self.preempt(preempted_seq)
                    preempted_seqs.append(preempted_seq)
                    if len(self.running) == req_index:
                        break

            if len(self.running) == req_index:
                break

            scheduled_running_seqs.append(seq)
            token_budget -= seq.num_new_tokens
            req_index += 1

        # Schedule from the waiting queue
        if not preempted_seqs:
            while (
                self.waiting
                and token_budget > 0
                and len(self.running) < self.max_num_seqs
            ):
                seq = self.waiting[0]

                if seq.block_table:
                    raise RuntimeError(
                        "Expected waiting sequence to have empty block_table, "
                        f"but sequence {seq.seq_id} has {len(seq.block_table)} blocks."
                    )

                (
                    num_new_computed_tokens_in_used,
                    num_new_computed_tokens_in_free,
                    num_new_tokens,
                ) = self.block_manager.get_token_layout(seq)

                if self.enable_chunked:
                    num_new_tokens = min(num_new_tokens, token_budget)

                if num_new_tokens <= 0:
                    raise RuntimeError(
                        "Expected a positive num_new_tokens for waiting sequence "
                        f"{seq.seq_id}, but got {num_new_tokens}."
                    )

                if num_new_tokens > token_budget or not self.block_manager.can_allocate(
                    num_new_computed_tokens_in_free + num_new_tokens
                ):
                    break

                seq.num_new_tokens = num_new_tokens

                self.block_manager.allocate(seq)

                expected_cached_tokens = (
                    num_new_computed_tokens_in_free + num_new_computed_tokens_in_used
                )

                if seq.num_cached_tokens != expected_cached_tokens:
                    raise RuntimeError(
                        "Cached token count mismatch after allocate for sequence "
                        f"{seq.seq_id}: expected {expected_cached_tokens}, "
                        f"got {seq.num_cached_tokens}."
                    )

                token_budget -= num_new_tokens
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_new_reqs.append(seq)

        scheduled_seqs = scheduled_running_seqs + scheduled_new_reqs

        if not scheduled_seqs:
            raise RuntimeError("Scheduler.schedule produced an empty schedule.")

        return scheduled_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self, seqs: list[Sequence], token_ids: list[int], seq_need_compute_logits
    ) -> None:
        if len(token_ids) != len(seq_need_compute_logits):
            raise ValueError(
                "token_ids and seq_need_compute_logits must have equal length, "
                f"got {len(token_ids)} and {len(seq_need_compute_logits)}."
            )

        for seq_index, token_id in zip(seq_need_compute_logits, token_ids):
            seq = seqs[seq_index]
            seq.append_token(token_id)

            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
                or len(seq) >= self.max_model_len
            ):
                if len(seq) >= self.max_model_len:
                    log.info(
                        "Sequence %s reached max_model_len %s.",
                        seq.seq_id,
                        self.max_model_len,
                    )

                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

        for seq in seqs:
            if seq.status != SequenceStatus.FINISHED:
                seq.num_cached_tokens = seq.num_cached_tokens + seq.num_new_tokens
                seq.num_new_tokens = 0
