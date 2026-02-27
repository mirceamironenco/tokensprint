from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from engine.nn.layers import (
    Attention,
    NormOrder,
    RMSNorm,
    SwiGLUFFN,
    TiedProjectionLayer,
    TransformerBlock,
)
from engine.nn.rope import RopeEncoding
from engine.nn.sdpa import FlashSDPA
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

    @classmethod
    def build(
        cls,
        *,
        model_dim: int,
        vocab_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        attn_window_len: int | None = None,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        rope_impl: Literal["table", "triton", "helion"] = "triton",
        rope_approx_trigo: bool = False,
        norm_order: NormOrder = NormOrder.PRE,
        norm_eps: float = 1e-6,
        rmsnorm_impl: Literal["torch", "triton", "helion"] = "torch",
        tie_weights: bool = False,
        head_dim: int | None = None,
        attn_bias: bool = False,
        attn_output_bias: bool = False,
        attn_qk_norm: bool = False,
        attn_qk_norm_eps: float = 1e-6,
        ffn_inner_dim: int | None = None,
        ffn_multiple_of: int = 4096,
        ffn_multiplier: float = 2 / 3,
        ffn_bias: bool = False,
    ) -> Transformer:
        frontend = nn.Embedding(vocab_dim, model_dim)
        norm_op = RMSNorm.resolve_norm_op(rmsnorm_impl)

        layers = []

        if model_dim % num_heads != 0:
            raise ValueError(
                "model_dim must be divisible by num_heads, "
                f"got model_dim={model_dim} and num_heads={num_heads}."
            )

        head_dim = head_dim or (model_dim // num_heads)

        if rope_impl == "table":
            rope_encoder = RopeEncoding(head_dim, max_seq_len, rope_theta)
        elif rope_impl == "triton":
            from engine.kernels.triton.rope import TritonRopeEncoding

            rope_encoder = TritonRopeEncoding(
                head_dim,
                max_seq_len,
                rope_theta,
                approx_trigo=rope_approx_trigo,
            )
        elif rope_impl == "helion":
            from engine.kernels.helion.rope import HelionRopeEncoding

            rope_encoder = HelionRopeEncoding(
                head_dim,
                max_seq_len,
                rope_theta,
                approx_trigo=rope_approx_trigo,
            )
        else:
            raise ValueError(f"Unsupported rope_impl: {rope_impl}.")

        for _ in range(num_layers):
            attn = Attention(
                model_dim,
                num_heads,
                pos_encoder=rope_encoder,
                sdpa=FlashSDPA(),
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                bias=attn_bias,
                output_bias=attn_output_bias,
                qk_norm=attn_qk_norm,
                qk_norm_eps=attn_qk_norm_eps,
                norm_impl=rmsnorm_impl,
                norm_op=norm_op,
            )

            ffn = SwiGLUFFN(
                model_dim,
                inner_dim=ffn_inner_dim,
                dim_multiplier=ffn_multiplier,
                multiple_of=ffn_multiple_of,
                bias=ffn_bias,
            )

            layers.append(
                TransformerBlock(
                    attn,
                    ffn,
                    model_dim,
                    norm_order,
                    norm_eps,
                    norm_impl=rmsnorm_impl,
                    norm_op=norm_op,
                )
            )

        if not tie_weights:
            output_proj = nn.Linear(model_dim, vocab_dim)
        else:
            output_proj = TiedProjectionLayer(frontend)

        return Transformer(
            model_dim,
            frontend,
            layers,
            output_proj,
            norm_order,
            norm_eps,
            norm_impl=rmsnorm_impl,
            norm_op=norm_op,
        )

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
