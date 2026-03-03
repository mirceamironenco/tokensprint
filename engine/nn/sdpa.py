from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, override

import torch
import torch.nn as nn

from engine.sequence import SequenceInfo
from engine.utils import replace_method_signature_with

try:
    from flash_attn import (  # type: ignore
        flash_attn_func,
        flash_attn_varlen_func,
    )
except ImportError:
    _has_flash_attn_2 = False
else:
    _has_flash_attn_2 = True


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
            had_batch_dim = query.ndim == 4

            if query.ndim == 3:
                query, key, value = query[None, ...], key[None, ...], value[None, ...]

            output = flash_attn_func(
                query,
                key,
                value,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=False,
            )

            if not had_batch_dim:
                output.squeeze_(0)

            return output

        # Varlen/paged path (inference engine): total_tokens x H x D + cu_seqlens.
        return flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=seqinfo.cu_seqlens_q,
            cu_seqlens_k=seqinfo.cu_seqlens_k,
            max_seqlen_q=seqinfo.max_seqlen_q,
            max_seqlen_k=seqinfo.max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
            block_table=seqinfo.block_tables,
        )
