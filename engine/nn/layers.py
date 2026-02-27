from enum import Enum
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.nn.attention import Attention
from engine.nn.base_layers import Linear, RMSNorm
from engine.sequence import SequenceInfo


class TiedProjectionLayer(nn.Module):
    """Needed for FSDP2 compatibility."""

    def __init__(self, embed: nn.Embedding) -> None:
        super().__init__()
        self.embed = embed
        self._shape = self.embed.weight.size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embed.weight)

    def extra_repr(self) -> str:
        return f"in_dim={self._shape[1]}, out_dim={self._shape[0]}"

    @property
    def weight(self) -> torch.Tensor:
        return self.embed.weight


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        model_dim: int,
        inner_dim: int | None = None,
        dim_multiplier: float = 1.0,
        multiple_of: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if inner_dim is None:
            inner_dim = 4 * model_dim

        if dim_multiplier != 1.0:
            inner_dim = int(inner_dim * dim_multiplier)

        if multiple_of != 1:
            inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)

        self.gate_proj = Linear(model_dim, inner_dim, bias=bias)
        self.up_proj = Linear(model_dim, inner_dim, bias=bias)
        self.down_proj = Linear(inner_dim, model_dim, bias=bias)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(seqs)) * self.up_proj(seqs))


class NormOrder(Enum):
    PRE = 0
    POST = 1


class TransformerBlock(nn.Module):
    def __init__(
        self,
        attn_layer: Attention,
        ffn_layer: SwiGLUFFN,
        model_dim: int,
        norm_order: NormOrder = NormOrder.PRE,
        norm_eps: float = 1e-6,
        norm_impl: Literal["torch", "triton", "helion"] = "torch",
        norm_op: RMSNorm.Op | None = None,
    ) -> None:
        super().__init__()

        self.attn_layer = attn_layer
        self.ffn_layer = ffn_layer

        self.attn_norm = RMSNorm(
            model_dim,
            eps=norm_eps,
            impl=norm_impl,
            norm_op=norm_op,
        )
        self.ffn_norm = RMSNorm(
            model_dim,
            eps=norm_eps,
            impl=norm_impl,
            norm_op=norm_op,
        )
        self.norm_order = norm_order

    def forward(
        self,
        seqs: torch.Tensor,
        *,
        input_pos: torch.Tensor | None = None,
        seqinfo: SequenceInfo | None = None,
    ) -> torch.Tensor:
        if self.norm_order == NormOrder.PRE:
            seqs = seqs + self.attn_layer(
                self.attn_norm(seqs), input_pos=input_pos, seqinfo=seqinfo
            )
        else:
            seqs = self.attn_norm(
                seqs + self.attn_layer(seqs, input_pos=input_pos, seqinfo=seqinfo)
            )

        if self.norm_order == NormOrder.PRE:
            seqs = seqs + self.ffn_layer(self.ffn_norm(seqs))
        else:
            seqs = self.ffn_norm(seqs + self.ffn_layer(seqs))

        return seqs
