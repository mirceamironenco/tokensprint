from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from engine.nn.layers import (
    NormOrder,
    RMSNorm,
    TransformerBlock,
)
from engine.sequence import SequenceInfo


class Transformer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        embedding: nn.Embedding,
        layers: list[TransformerBlock],
        output_proj: nn.Module,
        norm_order: NormOrder = NormOrder.PRE,
        norm_eps: float = 1e-6,
        norm_impl: Literal["torch", "triton", "helion"] = "torch",
        norm_op: RMSNorm.Op | None = None,
    ) -> None:
        super().__init__()

        if model_dim != embedding.weight.size(1):
            raise ValueError(
                "model_dim must match embedding dimension, "
                f"got model_dim={model_dim} and embedding_dim={embedding.weight.size(1)}."
            )

        self.model_dim = model_dim
        self.embedding = embedding
        self.layers = nn.ModuleList(layers)
        self.output_proj = output_proj
        self.norm_order = norm_order
        self.norm_eps = norm_eps

        self.vocab_dim = embedding.weight.size(0)
        self.num_heads = layers[0].attn_layer.num_heads
        self.head_dim = layers[0].attn_layer.head_dim
        self.num_kv_heads = layers[0].attn_layer.num_kv_heads

        if norm_order == NormOrder.PRE:
            self.norm = RMSNorm(
                model_dim,
                eps=norm_eps,
                impl=norm_impl,
                norm_op=norm_op,
            )
        else:
            self.register_module("norm", None)

    def decode(
        self,
        seqs: torch.Tensor,
        *,
        input_pos: torch.Tensor | None = None,
        seqinfo: SequenceInfo | None = None,
    ) -> torch.Tensor:
        seqs = self.embedding(seqs)

        for layer in self.layers:
            seqs = layer(seqs, input_pos=input_pos, seqinfo=seqinfo)

        if self.norm is not None:
            seqs = self.norm(seqs)

        return seqs

    def project(self, seqs: torch.Tensor) -> torch.Tensor:
        return self.output_proj(seqs)

    def project_inference(
        self, seqs: torch.Tensor, seqinfo: SequenceInfo
    ) -> torch.Tensor:
        seq_indices = seqinfo.seq_need_compute_logits
        last_indices = (seqinfo.cu_seqlens_q[1:] - 1).to(torch.long)
        last_indices = last_indices.index_select(0, seq_indices)

        seqs = seqs.index_select(0, last_indices).contiguous()
        return self.project(seqs)

    def forward(
        self,
        seqs: torch.Tensor,
        *,
        input_pos: torch.Tensor | None,
        seqinfo: SequenceInfo | None = None,
    ) -> torch.Tensor:
        return self.project(self.decode(seqs, input_pos=input_pos, seqinfo=seqinfo))
