import torch
import torch.nn as nn
import triton
import triton.language as tl

from engine.nn.rope import LLaMARoPEScaleConfig, build_rope_inv_freq


@triton.jit
def rope_kernel(
    x_ptr,
    pos_ptr,
    freqs_ptr,
    n_heads,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM2: tl.constexpr,
    HEADS_PER_ELEM: tl.constexpr,
    HEAD_GROUP_SIZE: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
    APPROX_TRIGO: tl.constexpr,
):
    block_idx = tl.program_id(0).to(tl.int64)
    freqs_ofs = tl.arange(0, HEAD_DIM2)
    head_mask = freqs_ofs < (HEAD_DIM // 2)
    freqs = tl.load(freqs_ptr + freqs_ofs, mask=head_mask)

    for head_idx in range(
        block_idx * HEADS_PER_BLOCK,
        (block_idx + 1) * HEADS_PER_BLOCK,
        HEAD_GROUP_SIZE,
    ):
        grp_ofs = head_idx + tl.arange(0, HEAD_GROUP_SIZE)
        head_pos = tl.load(pos_ptr + grp_ofs // HEADS_PER_ELEM)
        angles = head_pos[:, None] * freqs[None, :]
        tl.static_assert(angles.dtype == tl.float32)

        if APPROX_TRIGO:
            sines, cosines = tl.inline_asm_elementwise(
                asm="""
                sin.approx.f32 $0, $2;
                cos.approx.f32 $1, $2;
                """,
                constraints="=r,=r,r",
                args=[angles],
                dtype=(tl.float32, tl.float32),
                is_pure=True,
                pack=1,
            )
        else:
            sines = tl.sin(angles)
            cosines = tl.cos(angles)

        re_ofs = grp_ofs[:, None] * HEAD_DIM + freqs_ofs[None, :]
        im_ofs = re_ofs + (HEAD_DIM // 2)

        mask = (grp_ofs < n_heads)[:, None] & head_mask[None, :]

        re_x = tl.load(x_ptr + re_ofs, mask=mask).to(tl.float32)
        im_x = tl.load(x_ptr + im_ofs, mask=mask).to(tl.float32)

        re_out = re_x * cosines - im_x * sines
        im_out = im_x * cosines + re_x * sines

        tl.store(x_ptr + re_ofs, re_out, mask=mask)
        tl.store(x_ptr + im_ofs, im_out, mask=mask)


def apply_rope_inplace(
    x: torch.Tensor,
    pos: torch.Tensor,
    freqs: torch.Tensor,
    approx_trigo: bool = False,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must be a 3-D tensor; got x.ndim={x.ndim}.")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous.")
    if not x.is_cuda:
        raise ValueError("x must be on CUDA.")

    n_elem, num_heads, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE; got {head_dim}.")

    if pos.ndim != 1 or pos.numel() != n_elem:
        raise ValueError(
            f"pos must be 1-D with length {n_elem}; got shape {tuple(pos.shape)}."
        )
    if not pos.is_contiguous():
        raise ValueError("pos must be contiguous.")
    if pos.device != x.device:
        raise ValueError("pos and x must be on the same device.")

    if freqs.ndim != 1 or freqs.numel() * 2 != head_dim:
        raise ValueError(
            "freqs must be 1-D with length head_dim // 2; "
            f"got shape {tuple(freqs.shape)} for head_dim={head_dim}."
        )
    if not freqs.is_contiguous():
        raise ValueError("freqs must be contiguous.")
    if freqs.device != x.device:
        raise ValueError("freqs and x must be on the same device.")

    total_heads = n_elem * num_heads
    if total_heads < 2048:
        heads_per_block = 1
        head_group_size = 1
    else:
        head_bytes = head_dim * x.element_size()
        heads_per_block = max(1, 2048 // head_bytes)
        head_group_size = triton.next_power_of_2(max(1, 512 // head_bytes))

    n_blocks = triton.cdiv(total_heads, heads_per_block)
    rope_kernel[(n_blocks,)](
        x,
        pos,
        freqs,
        total_heads,
        head_dim,
        triton.next_power_of_2(head_dim // 2),
        num_heads,
        head_group_size,
        heads_per_block,
        approx_trigo,
    )


class TritonRopeEncoding(nn.Module):
    freqs: torch.Tensor

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        rope_scale: LLaMARoPEScaleConfig | None = None,
        approx_trigo: bool = False,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Triton RoPE requires even head dim, got {dim}.")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rope_scale = rope_scale
        self.approx_trigo = approx_trigo

        self.register_buffer(
            "freqs",
            torch.empty(dim // 2, dtype=torch.float32),
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
        self.freqs.copy_(inv_freq)

    def _flatten_pos(
        self, query: torch.Tensor, input_pos: torch.Tensor | None
    ) -> torch.Tensor:
        if query.ndim == 4:
            batch_size, seqlen = query.size(0), query.size(1)
            if input_pos is None:
                pos = torch.arange(seqlen, device=query.device, dtype=torch.int64)
                return pos.view(1, seqlen).expand(batch_size, seqlen).reshape(-1)

            if input_pos.ndim == 1:
                if input_pos.numel() != seqlen:
                    raise ValueError(
                        "Expected 1-D input_pos length to match seqlen "
                        f"{seqlen}, got {input_pos.numel()}."
                    )
                return input_pos.view(1, seqlen).expand(batch_size, seqlen).reshape(-1)

            if input_pos.ndim == 2:
                expected_shape = (batch_size, seqlen)
                if tuple(input_pos.shape) != expected_shape:
                    raise ValueError(
                        "Expected 2-D input_pos shape "
                        f"{expected_shape}, got {tuple(input_pos.shape)}."
                    )
                return input_pos.reshape(-1)

            raise ValueError(
                "Expected input_pos rank 1 or 2 for rank-4 query, "
                f"got rank {input_pos.ndim}."
            )

        total = query.size(0)
        if input_pos is None:
            return torch.arange(total, device=query.device, dtype=torch.int64)

        if input_pos.ndim != 1 or input_pos.numel() != total:
            raise ValueError(
                "Expected input_pos to be 1-D with length "
                f"{total}, got shape {tuple(input_pos.shape)}."
            )
        return input_pos

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        input_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim not in (3, 4):
            raise ValueError(
                "Expected query to have rank 3 or 4, "
                f"but got shape {tuple(query.shape)}."
            )
        if key.ndim != query.ndim:
            raise ValueError(
                "Expected key rank to match query rank, "
                f"got {key.ndim} and {query.ndim}."
            )
        if query.shape[:-2] != key.shape[:-2]:
            raise ValueError(
                "Expected query and key to share prefix shape up to heads, "
                f"but got {tuple(query.shape[:-2])} and {tuple(key.shape[:-2])}."
            )
        if query.size(-1) != self.dim or key.size(-1) != self.dim:
            raise ValueError(
                f"Expected query/key head_dim={self.dim}, "
                f"got {query.size(-1)} and {key.size(-1)}."
            )

        pos = self._flatten_pos(query, input_pos).to(
            device=query.device, dtype=torch.int64, non_blocking=True
        )
        pos = pos.contiguous()

        if self.freqs.device != query.device:
            raise RuntimeError(
                "TritonRopeEncoding freqs buffer is not on the same device as query."
            )

        query_out = query.contiguous().reshape(-1, query.size(-2), query.size(-1))
        key_out = key.contiguous().reshape(-1, key.size(-2), key.size(-1))

        apply_rope_inplace(query_out, pos, self.freqs, approx_trigo=self.approx_trigo)
        apply_rope_inplace(key_out, pos, self.freqs, approx_trigo=self.approx_trigo)

        return query_out.view_as(query), key_out.view_as(key)
