from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, override

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.sequence import SequenceInfo
from engine.utils import replace_method_signature_with

try:
    from flash_attn import (  # type: ignore
        flash_attn_func,
        flash_attn_varlen_func,
    )
except ImportError:
    _has_flash_attn_2 = False
    _has_flash_attn_kvcache = False
else:
    _has_flash_attn_2 = True
    try:
        from flash_attn import flash_attn_with_kvcache  # type: ignore
    except ImportError:
        _has_flash_attn_kvcache = False
    else:
        _has_flash_attn_kvcache = True


class SDPA(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        seqinfo: SequenceInfo | None = None,
    ) -> torch.Tensor: ...

    if TYPE_CHECKING:

        @replace_method_signature_with(forward)
        def __call__(self, *args: Any, **kwds: Any) -> Any:
            return super().__call__(*args, **kwds)


class TorchFlashSDPA(SDPA):
    @override
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        seqinfo: SequenceInfo | None = None,
    ) -> torch.Tensor:

        cu_seqlen_q, cu_seqlen_k, q_max_seqlen, k_max_seqlen = (
            None,
            None,
            query.size(1),
            key.size(1),
        )

        if seqinfo is not None:
            cu_seqlen_q, cu_seqlen_k, q_max_seqlen, k_max_seqlen = (
                seqinfo.cu_seqlens_q,
                seqinfo.cu_seqlens_k,
                seqinfo.max_seqlen_q,
                seqinfo.max_seqlen_k,
            )

        attn_output, *_ = torch.ops.aten._flash_attention_forward(
            query,
            key,
            value,
            cu_seqlen_q,
            cu_seqlen_k,
            q_max_seqlen,
            k_max_seqlen,
            is_causal=True,
            return_debug_mask=False,
        )

        return attn_output


class FlashSDPA(SDPA):
    @override
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        seqinfo: SequenceInfo | None = None,
    ) -> torch.Tensor:
        if not _has_flash_attn_2:
            raise RuntimeError(
                "FlashSDPA requires flash-attn, but it is not available in this environment."
            )

        softmax_scale = query.size(-1) ** -0.5

        # Dense path (e.g. training / simple forward): B x S x H x D.
        if seqinfo is None:
            if query.ndim == 3:
                query, key, value = (
                    query.unsqueeze(0),
                    key.unsqueeze(0),
                    value.unsqueeze(0),
                )
                return flash_attn_func(
                    query, key, value, softmax_scale=softmax_scale, causal=True
                ).squeeze(0)

            return flash_attn_func(
                query, key, value, softmax_scale=softmax_scale, causal=True
            )

        # Decode fast path: one query token per sequence with paged KV cache.
        if (
            _has_flash_attn_kvcache
            and seqinfo.max_seqlen_q == 1
            and seqinfo.block_tables is not None
        ):
            if query.ndim == 3:
                query = query.unsqueeze(1)
            elif query.ndim != 4:
                raise ValueError(
                    "Expected decode query rank to be 3 or 4 for KV-cache path, "
                    f"but got shape {tuple(query.shape)}."
                )

            output = flash_attn_with_kvcache(
                query,
                key,
                value,
                cache_seqlens=seqinfo.context_lens,
                block_table=seqinfo.block_tables,
                softmax_scale=softmax_scale,
                causal=True,
            )
            return output.squeeze(1)

        # Varlen/paged path (inference engine): total_tokens x H x D + cu_seqlens.
        return flash_attn_varlen_func(
            query,
            key,
            value,
            max_seqlen_q=seqinfo.max_seqlen_q,
            cu_seqlens_q=seqinfo.cu_seqlens_q,
            max_seqlen_k=seqinfo.max_seqlen_k,
            cu_seqlens_k=seqinfo.cu_seqlens_k,
            block_table=seqinfo.block_tables,
            softmax_scale=softmax_scale,
            causal=True,
        )


class NaiveSDPA(SDPA):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        seqinfo: SequenceInfo | None = None,
    ) -> torch.Tensor:
        nh, nkv = query.size(-2), key.size(-2)

        q, k, v = (
            query.transpose(-3, -2),
            key.transpose(-3, -2),
            value.transpose(-3, -2),
        )

        if (query_per_kv := nh // nkv) > 1:
            k = torch.repeat_interleave(k, query_per_kv, -3)
            v = torch.repeat_interleave(v, query_per_kv, -3)

        attn_scale = q.size(-1) ** -0.5

        attn_score = torch.matmul(q, k.transpose(-1, -2)) * attn_scale

        if attn_mask is not None:
            attn_score = attn_score + attn_mask

        attn_score = F.softmax(attn_score, dim=-1, dtype=torch.float32).type_as(q)

        output = torch.matmul(attn_score, v)

        output = output.transpose(1, 2)

        return output


class TorchSDPA(SDPA):
    @override
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        seqinfo: SequenceInfo | None = None,
    ) -> torch.Tensor:
        # q,k,v: (*, seqlen, nh/nkv, head_dim)
        nh, nkv = query.size(-2), key.size(-2)

        q, k, v = (
            query.transpose(-3, -2),
            key.transpose(-3, -2),
            value.transpose(-3, -2),
        )

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask, is_causal=attn_mask is None, enable_gqa=nh != nkv
        )

        # (bsz, num_heads, seqlen, head_dim) -> (bsz, seqlen, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2)

        return attn_output
