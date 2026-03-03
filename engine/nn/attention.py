from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import triton
import triton.language as tl

from engine.nn.base_layers import Linear, RMSNorm
from engine.nn.rope import RopeEncoding
from engine.nn.sdpa import SDPA
from engine.sequence import SequenceInfo


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    if key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError(
            "key/value must be contiguous in head_dim (stride(-1) == 1), "
            f"got key.stride(-1)={key.stride(-1)} and value.stride(-1)={value.stride(-1)}."
        )
    if key.stride(1) != head_dim or value.stride(1) != head_dim:
        raise ValueError(
            "key/value tensors must have packed head layout with stride(1) == head_dim, "
            f"got key.stride(1)={key.stride(1)}, value.stride(1)={value.stride(1)}, "
            f"head_dim={head_dim}."
        )
    if k_cache.stride(1) != D or v_cache.stride(1) != D:
        raise ValueError(
            "k_cache/v_cache tensors must have stride(1) == num_heads * head_dim, "
            f"got k_cache.stride(1)={k_cache.stride(1)}, v_cache.stride(1)={v_cache.stride(1)}, "
            f"expected={D}."
        )
    if slot_mapping.numel() != N:
        raise ValueError(
            "slot_mapping size must match number of tokens in key/value, "
            f"got slot_mapping.numel()={slot_mapping.numel()} and N={N}."
        )
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


@dataclass(kw_only=True)
class AttentionConfig:
    model_dim: int
    num_heads: int
    num_kv_heads: int | None = None
    head_dim: int | None = None
    bias: bool = False
    output_bias: bool = False
    qk_norm: bool = False
    qk_norm_eps: float = 1e-6


class Attention(nn.Module):
    pos_encoder: RopeEncoding
    sdpa: SDPA
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    q_norm: RMSNorm | None
    k_norm: RMSNorm | None

    def __init__(
        self,
        config: AttentionConfig,
        *,
        pos_encoder: RopeEncoding,
        sdpa: SDPA,
        norm_impl: Literal["torch", "triton", "helion"] = "torch",
        norm_op: RMSNorm.Op | None = None,
    ) -> None:
        super().__init__()
        if config.model_dim % config.num_heads != 0:
            raise ValueError(
                "model_dim must be divisible by num_heads, "
                f"got model_dim={config.model_dim} and num_heads={config.num_heads}."
            )
        if config.head_dim is not None and config.model_dim % config.head_dim != 0:
            raise ValueError(
                "model_dim must be divisible by head_dim when head_dim is provided, "
                f"got model_dim={config.model_dim} and head_dim={config.head_dim}."
            )

        self.config = config
        self.model_dim = config.model_dim
        self.head_dim = config.head_dim or (config.model_dim // config.num_heads)
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads or config.num_heads

        self.q_proj = Linear(
            self.model_dim, self.num_heads * self.head_dim, bias=config.bias
        )
        self.k_proj = Linear(
            self.model_dim, self.num_kv_heads * self.head_dim, bias=config.bias
        )
        self.v_proj = Linear(
            self.model_dim, self.num_kv_heads * self.head_dim, bias=config.bias
        )
        self.o_proj = Linear(
            self.num_heads * self.head_dim,
            self.model_dim,
            bias=config.output_bias,
        )

        self.sdpa = sdpa
        self.pos_encoder = pos_encoder

        if config.qk_norm:
            self.q_norm = RMSNorm(
                self.head_dim,
                eps=config.qk_norm_eps,
                impl=norm_impl,
                norm_op=norm_op,
            )
            self.k_norm = RMSNorm(
                self.head_dim,
                eps=config.qk_norm_eps,
                impl=norm_impl,
                norm_op=norm_op,
            )
        else:
            self.register_module("q_norm", None)
            self.register_module("k_norm", None)

        self.register_buffer("k_cache", torch.empty(0), persistent=False)
        self.register_buffer("v_cache", torch.empty(0), persistent=False)

    def bind_kv_cache(self, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
        self.k_cache = k_cache
        self.v_cache = v_cache

    def forward(
        self,
        seqs: torch.Tensor,
        *,
        input_pos: torch.Tensor | None = None,
        seqinfo: SequenceInfo | None = None,
    ) -> torch.Tensor:
        q, k, v = self.q_proj(seqs), self.k_proj(seqs), self.v_proj(seqs)

        q = q.unflatten(-1, (self.num_heads, -1))

        k = k.unflatten(-1, (self.num_kv_heads, -1))
        v = v.unflatten(-1, (self.num_kv_heads, -1))

        if self.q_norm is not None:
            q = self.q_norm(q)

        if self.k_norm is not None:
            k = self.k_norm(k)

        q, k = self.pos_encoder(q, k, input_pos)

        if seqinfo is not None:
            if self.k_cache.numel() and self.v_cache.numel():
                store_kvcache(k, v, self.k_cache, self.v_cache, seqinfo.slot_mapping)

            if seqinfo.block_tables is not None:
                k, v = self.k_cache, self.v_cache

        # Attention
        output = self.sdpa(q, k, v, seqinfo=seqinfo)

        output = output.flatten(-2)

        output = self.o_proj(output)

        return output
