import torch
import torch.nn as nn


class RopeEncoding(nn.Module):
    freqs: torch.Tensor

    def __init__(self, dim: int, max_seq_len: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta

        self.register_buffer(
            "freqs",
            torch.empty(max_seq_len, dim * 2, dtype=torch.float32),
            persistent=False,
        )

    def reset_parameters(self) -> None:
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        device = self.freqs.device
        idx = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
        freqs = 1.0 / (self.rope_theta ** (idx / self.dim))
        positions = torch.arange(self.max_seq_len, device=device)

        freqs = torch.outer(positions, freqs)
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
