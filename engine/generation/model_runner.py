from bisect import bisect_left

import torch

from engine.generation.config import Config
from engine.generation.sampler import Sampler
from engine.model import Transformer
from engine.sequence import Sequence, SequenceInfo
from engine.utils import to_tensor


class ModelRunner:
    kv_caches: list[torch.Tensor]
    graph_bs: list[int]
    graph_pool: object | None
    graphs: dict[int, torch.cuda.CUDAGraph]
    graph_vars: dict[int, dict[str, torch.Tensor]]

    def __init__(self, config: Config, rank: int, model: Transformer) -> None:
        self.config = config
        self.block_size = config.kvcache_block_size
        self.rank = rank
        self.model = model
        self.sampler = Sampler()
        self.kv_caches = []

        self.graph_bs = []
        self.graph_pool = None
        self.graphs = {}
        self.graph_vars = {}

        if torch.cuda.is_available():
            self.warmup_model()
            self.allocate_kv_cache()
            self.capture_cudagraph()

    def reset_perf_stats(self) -> None:
        return None

    def _add_perf(self, key: str, dt: float) -> None:
        return None

    def get_perf_stats(self) -> dict[str, float]:
        return {}

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        """Builds a padded block-table tensor for a batch of sequences.

        Each sequence block table is right-padded with `-1` to the batch max
        table length and moved to CUDA.

        Args:
            seqs: Batch sequences with per-sequence `block_table` values.

        Returns:
            CUDA tensor of shape `[batch, max_blocks]` with dtype `int32`.
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        return to_tensor(block_tables, dtype=torch.int32, device=torch.device("cuda"))

    def _slot_mapping_for_sequence(self, seq: Sequence) -> list[int]:
        """Computes flattened KV-cache slot indices for one sequence.

        The mapping covers blocks touched by the current model step, beginning at
        `num_cached_blocks` and ending at `num_context_tokens`.

        Args:
            seq: Sequence whose block layout is being materialized.

        Returns:
            Flat list of slot indices for gather/scatter into KV cache.
        """
        if not seq.block_table:  # warmup
            return []

        slot_mapping: list[int] = []
        for logical_block_index in range(seq.num_cached_blocks, len(seq.block_table)):
            block_id = seq.block_table[logical_block_index]
            block_start = block_id * self.block_size

            if logical_block_index == seq.num_cached_blocks:
                slot_start = block_start + seq.num_cached_tokens % self.block_size
            else:
                slot_start = block_start

            if logical_block_index == len(seq.block_table) - 1:
                context_tail = seq.num_context_tokens % self.block_size
                slot_end = block_start + (
                    context_tail if context_tail != 0 else self.block_size
                )
            else:
                slot_end = block_start + self.block_size

            slot_mapping.extend(range(slot_start, slot_end))

        return slot_mapping

    def prepare_model_input(
        self, seqs: list[Sequence]
    ) -> tuple[torch.Tensor, torch.Tensor, SequenceInfo]:
        """Builds model input tensors and runtime sequence info for a batch.

        Args:
            seqs: Scheduled sequences for the current forward step.

        Returns:
            Tuple of:
                - input_ids tensor on CUDA.
                - positions tensor on CUDA.
                - `SequenceInfo` with cu-seqlens, slot mapping, block tables, and
                  sequence indices needing logits.
        """
        input_ids: list[int] = []
        positions: list[int] = []
        cu_seqlens_q: list[int] = [0]
        cu_seqlens_k: list[int] = [0]
        slot_mapping: list[int] = []
        context_lens: list[int] = []
        seq_need_compute_logits: list[int] = []
        max_seqlen_q = 0
        max_seqlen_k = 0

        for seq_index, seq in enumerate(seqs):
            context_start = seq.num_cached_tokens
            context_end = seq.num_context_tokens

            if len(seq) == context_start + seq.num_new_tokens and seq.block_table:
                seq_need_compute_logits.append(seq_index)

            context_lens.append(context_end)
            input_ids.extend(seq[context_start:context_end])
            positions.extend(range(context_start, context_end))

            seqlen_q = seq.num_new_tokens
            seqlen_k = context_end
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)

            slot_mapping.extend(self._slot_mapping_for_sequence(seq))

        block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache or decoding
            block_tables = self.prepare_block_tables(seqs)

        cuda_device = torch.device("cuda")

        seqinfo = SequenceInfo(
            cu_seqlens_q=to_tensor(cu_seqlens_q, dtype=torch.int32, device=cuda_device),
            cu_seqlens_k=to_tensor(cu_seqlens_k, dtype=torch.int32, device=cuda_device),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=to_tensor(slot_mapping, dtype=torch.int32, device=cuda_device),
            context_lens=to_tensor(context_lens, dtype=torch.int32, device=cuda_device),
            block_tables=block_tables,
            seq_need_compute_logits=to_tensor(
                seq_need_compute_logits, dtype=torch.long, device=cuda_device
            ),
        )
        return (
            to_tensor(input_ids, dtype=torch.int64, device=cuda_device),
            to_tensor(positions, dtype=torch.int64, device=cuda_device),
            seqinfo,
        )

    def prepare_sample(
        self, seqs: list[Sequence], seqinfo: SequenceInfo
    ) -> torch.Tensor:
        temperatures = [seq.temperature for seq in seqs]

        temperatures = to_tensor(
            temperatures,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )

        return temperatures.index_select(0, seqinfo.seq_need_compute_logits)

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, seqinfo: SequenceInfo
    ) -> torch.Tensor:
        if self._can_use_cudagraph(input_ids, seqinfo):
            return self._run_model_cudagraph(input_ids, positions, seqinfo)

        hidden = self.model.decode(input_ids, input_pos=positions, seqinfo=seqinfo)

        return self.model.project_inference(hidden, seqinfo=seqinfo)

    def run(self, seqs: list[Sequence]) -> tuple[list[int], list[int]]:
        input_ids, positions, seqinfo = self.prepare_model_input(seqs)
        temperatures = self.prepare_sample(seqs, seqinfo)
        logits = self.run_model(input_ids, positions, seqinfo)
        sampled = self.sampler(logits, temperatures)
        token_ids = sampled.tolist()
        seq_need_compute_logits = seqinfo.seq_need_compute_logits.tolist()
        return token_ids, seq_need_compute_logits

    def _can_use_cudagraph(
        self, input_ids: torch.Tensor, seqinfo: SequenceInfo
    ) -> bool:
        if self.config.enforce_eager:
            return False

        if not self.graphs:
            return False

        bs = input_ids.size(0)
        graph_bs = self._select_graph_bucket(bs)
        if graph_bs is None:
            return False

        # Decode-only graph path.
        if seqinfo.max_seqlen_q != 1:
            return False

        if seqinfo.block_tables is None:
            return False

        if seqinfo.context_lens.numel() != bs:
            return False

        if seqinfo.seq_need_compute_logits.numel() != bs:
            return False

        if seqinfo.block_tables.size(1) > self.graph_vars[graph_bs][
            "block_tables"
        ].size(1):
            return False

        return True

    def _select_graph_bucket(self, bs: int) -> int | None:
        index = bisect_left(self.graph_bs, bs)
        if index >= len(self.graph_bs):
            return None
        return self.graph_bs[index]

    def _run_model_cudagraph(
        self, input_ids: torch.Tensor, positions: torch.Tensor, seqinfo: SequenceInfo
    ) -> torch.Tensor:
        bs = input_ids.size(0)
        graph_bs = self._select_graph_bucket(bs)
        if graph_bs is None:
            raise RuntimeError(f"No CUDA graph bucket available for batch size {bs}.")
        graph_vars = self.graph_vars[graph_bs]
        graph = self.graphs[graph_bs]

        graph_vars["input_ids"].zero_()
        graph_vars["input_ids"][:bs].copy_(input_ids)
        graph_vars["positions"].zero_()
        graph_vars["positions"][:bs].copy_(positions)

        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][: seqinfo.slot_mapping.numel()].copy_(
            seqinfo.slot_mapping
        )

        # Keep inactive lanes valid when replaying a larger bucket.
        graph_vars["context_lens"].fill_(1)
        graph_vars["context_lens"][:bs].copy_(seqinfo.context_lens)

        graph_vars["cu_seqlens_k"][0] = 0
        torch.cumsum(
            graph_vars["context_lens"],
            dim=0,
            out=graph_vars["cu_seqlens_k"][1 : graph_bs + 1],
        )

        graph_vars["block_tables"][:bs].fill_(-1)
        if bs < graph_bs:
            graph_vars["block_tables"][bs:].zero_()
        if seqinfo.block_tables is not None:
            block_table_width = seqinfo.block_tables.size(1)
            graph_vars["block_tables"][:bs, :block_table_width].copy_(
                seqinfo.block_tables
            )

        graph.replay()

        hidden = graph_vars["hidden"][:bs]
        return self.model.project_inference(hidden, seqinfo=seqinfo)

    def warmup_model(self) -> None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        num_tokens = self.config.max_model_len
        num_seqs = max(
            min(
                self.config.max_num_batched_tokens // self.config.max_model_len,
                self.config.max_num_seqs,
            ),
            1,
        )

        seqs = [Sequence([0] * num_tokens) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_new_tokens = num_tokens

        # Warmup should only exercise model path; sampled outputs are ignored.
        self.run(seqs)

        torch.cuda.empty_cache()

    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        self.graph_bs = []
        self.graph_pool = None
        self.graphs = {}
        self.graph_vars = {}

        if self.config.enforce_eager:
            return

        max_bs = min(self.config.max_num_seqs, 512)
        if max_bs < 1:
            return

        max_num_blocks = (
            self.config.max_model_len + self.block_size - 1
        ) // self.block_size

        graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        graph_bs = [bs for bs in graph_bs if bs <= max_bs]
        self.graph_bs = graph_bs

        cuda_device = torch.device(f"cuda:{self.rank}")

        for bs in reversed(graph_bs):
            graph = torch.cuda.CUDAGraph()

            input_ids = torch.zeros(bs, dtype=torch.int64, device=cuda_device)
            positions = torch.zeros(bs, dtype=torch.int64, device=cuda_device)
            slot_mapping = torch.full((bs,), -1, dtype=torch.int32, device=cuda_device)
            context_lens = torch.ones(bs, dtype=torch.int32, device=cuda_device)
            block_tables = torch.zeros(
                (bs, max_num_blocks), dtype=torch.int32, device=cuda_device
            )
            cu_seqlens_q = torch.arange(bs + 1, dtype=torch.int32, device=cuda_device)
            cu_seqlens_k = torch.arange(bs + 1, dtype=torch.int32, device=cuda_device)
            seq_need_compute_logits = torch.arange(
                bs, dtype=torch.long, device=cuda_device
            )
            hidden = torch.zeros(
                bs,
                self.model.model_dim,
                dtype=self.config.dtype,
                device=cuda_device,
            )

            seqinfo = SequenceInfo(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=1,
                max_seqlen_k=self.config.max_model_len,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                seq_need_compute_logits=seq_need_compute_logits,
                block_tables=block_tables,
            )

            # Warmup current graph shape.
            hidden.copy_(
                self.model.decode(input_ids, input_pos=positions, seqinfo=seqinfo)
            )
            torch.cuda.synchronize(cuda_device)

            with torch.cuda.graph(graph, self.graph_pool):
                hidden.copy_(
                    self.model.decode(input_ids, input_pos=positions, seqinfo=seqinfo)
                )

            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[bs] = graph
            self.graph_vars[bs] = {
                "input_ids": input_ids,
                "positions": positions,
                "slot_mapping": slot_mapping,
                "context_lens": context_lens,
                "block_tables": block_tables,
                "cu_seqlens_k": cu_seqlens_k,
                "hidden": hidden,
            }

    def allocate_kv_cache(self) -> None:
        free_mem, _ = torch.cuda.mem_get_info()
        peak_mem_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current_mem_usage = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        total_free_mem = free_mem * self.config.gpu_memory_utilization

        # Reserve some room for peak memory usage during model execution
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)

        bytes_per_block = (
            self.block_size
            * 2
            * len(self.model.layers)
            * self.model.num_kv_heads
            * self.model.head_dim
            * self.config.dtype.itemsize
        )

        self.config.num_kvcache_blocks = int(available_mem // bytes_per_block)

        if self.config.num_kvcache_blocks < 1:
            raise RuntimeError(
                "Insufficient GPU memory for KV cache allocation: computed "
                f"num_kvcache_blocks={self.config.num_kvcache_blocks}, "
                f"bytes_per_block={bytes_per_block}, available_mem={int(available_mem)}."
            )

        self.kv_caches = [
            torch.zeros(
                2,
                self.config.num_kvcache_blocks,
                self.block_size,
                self.model.num_kv_heads,
                self.model.head_dim,
                dtype=self.config.dtype,
                device=f"cuda:{self.rank}",
            )
            for _ in self.model.layers
        ]

        for layer, kv_cache in zip(self.model.layers, self.kv_caches, strict=True):
            layer.attn_layer.bind_kv_cache(
                kv_cache[0],
                kv_cache[1],
            )
