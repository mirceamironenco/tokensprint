from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(kw_only=True)
class LLaMARoPEScaleConfig:
    """
    Frequency scaling config for LLaMA-style RoPE extension.
    """

    factor: float = 8.0
    frequency_factors: tuple[float, float] = (1.0, 4.0)
    original_context_length: int = 8192


def build_rope_inv_freq(
    *,
    dim: int,
    rope_theta: float,
    device: torch.device,
    rope_scale: LLaMARoPEScaleConfig | None = None,
) -> torch.Tensor:
    idx = torch.arange(0, dim, step=2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (rope_theta ** (idx / dim))

    if rope_scale is None or device.type == "meta":
        return inv_freq

    if rope_scale.factor <= 0.0:
        raise ValueError(f"rope_scale.factor must be > 0, got {rope_scale.factor}.")

    if rope_scale.original_context_length <= 0:
        raise ValueError(
            "rope_scale.original_context_length must be > 0, "
            f"got {rope_scale.original_context_length}."
        )

    low_freq_factor, high_freq_factor = rope_scale.frequency_factors
    if high_freq_factor <= low_freq_factor:
        raise ValueError(
            "rope_scale.frequency_factors must satisfy high > low, "
            f"got {rope_scale.frequency_factors}."
        )

    old_context = float(rope_scale.original_context_length)
    low_freq_wavelen = old_context / low_freq_factor
    high_freq_wavelen = old_context / high_freq_factor

    wavelen = (2.0 * math.pi) / inv_freq

    smooth = (old_context / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )

    scaled = inv_freq / rope_scale.factor
    smooth_scaled = (1.0 - smooth) * scaled + smooth * inv_freq

    inv_freq = torch.where(
        wavelen < high_freq_wavelen,
        inv_freq,
        torch.where(wavelen > low_freq_wavelen, scaled, smooth_scaled),
    )

    return inv_freq


class RopeEncoding(nn.Module):
    freqs: torch.Tensor

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        rope_scale: LLaMARoPEScaleConfig | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rope_scale = rope_scale

        self.register_buffer(
            "freqs",
            torch.empty(max_seq_len, dim * 2, dtype=torch.float32),
            persistent=False,
        )

    def reset_parameters(self) -> None:
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        device = self.freqs.device
        inv_freq = build_rope_inv_freq(
            dim=self.dim,
            rope_theta=self.rope_theta,
            device=device,
            rope_scale=self.rope_scale,
        )
        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(positions, inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)
        out_freqs = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

        self.freqs.copy_(out_freqs)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.size(-1) // 2]
        x2 = x[..., x.size(-1) // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x_ = x.float()
        x_ = (x_ * cos) + (self._rotate_half(x_) * sin)
        return x_.type_as(x)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        input_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # query/key: (bsz, seqlen, nh/nkv, head_dim) or (total, nh/nkv, head_dim)
        # input_pos: (*, seqlen) or (total,)
        if query.ndim not in (3, 4):
            raise ValueError(
                "Expected query to have rank 3 or 4, "
                f"but got shape {tuple(query.shape)}."
            )

        if key.ndim != query.ndim:
            raise ValueError(
                "Expected query and key to have the same rank, "
                f"but got {query.ndim} and {key.ndim}."
            )

        if query.shape[:-2] != key.shape[:-2]:
            raise ValueError(
                "Expected query and key to share prefix shape up to heads, "
                f"but got {tuple(query.shape[:-2])} and {tuple(key.shape[:-2])}."
            )

        seqlen, head_dim = query.size(-3), query.size(-1)

        if head_dim != self.dim or key.size(-1) != self.dim:
            raise ValueError(
                "Expected query/key head_dim to match rope dim "
                f"{self.dim}, but got {head_dim} and {key.size(-1)}."
            )

        if input_pos is None:
            freqs_cis = self.freqs[0:seqlen]
        else:
            freqs_cis = self.freqs[input_pos]

        target_prefix = query.shape[:-2]
        if freqs_cis.ndim == 2:
            prefix = (1,) * (len(target_prefix) - 1)
            freqs_cis = freqs_cis.view(*prefix, seqlen, head_dim * 2)
        elif freqs_cis.ndim != len(target_prefix) + 1:
            raise ValueError(
                "input_pos shape is incompatible with query/key shape: "
                "input_pos produced "
                f"{tuple(freqs_cis.shape)}, query is {tuple(query.shape)}."
            )

        freqs_cis = torch.broadcast_to(freqs_cis, target_prefix + (head_dim * 2,))
        freqs_cis = freqs_cis.unsqueeze(-2)

        cos = freqs_cis[..., :head_dim]
        sin = freqs_cis[..., head_dim:]

        return self._apply_rope(query, cos, sin), self._apply_rope(key, cos, sin)
