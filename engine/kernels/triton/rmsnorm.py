import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    out_ptr,
    w_ptr,
    eps,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64)
    x_ptr += idx * DIM
    out_ptr += idx * DIM

    mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, DIM, BLOCK_SIZE):
        ofs = offset + tl.arange(0, BLOCK_SIZE)
        a = tl.load(
            x_ptr + ofs,
            mask=ofs < DIM,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        mean += a * a

    rstd = tl.rsqrt((tl.sum(mean, axis=0) / DIM) + eps)

    for offset in range(0, DIM, BLOCK_SIZE):
        ofs = offset + tl.arange(0, BLOCK_SIZE)
        mask = ofs < DIM
        a = tl.load(
            x_ptr + ofs,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        w = tl.load(w_ptr + ofs, mask=mask)
        tl.store(out_ptr + ofs, a * rstd * w, mask=mask)

def rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"x must be a 2-D tensor; got x.ndim={x.ndim}.")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous.")

    n_rows, dim = x.shape

    if tuple(w.shape) != (dim,):
        raise ValueError(f"w must have shape ({dim},); got w.shape={tuple(w.shape)}.")
    if not w.is_contiguous():
        raise ValueError("w must be contiguous.")

    if not x.is_cuda or not w.is_cuda:
        raise ValueError("x and w must be CUDA tensors for Triton RMSNorm.")
    if x.device != w.device:
        raise ValueError("x and w must be on the same CUDA device.")

    max_fused_size = 65536 // x.element_size()
    block_size = min(max_fused_size, triton.next_power_of_2(dim))
    block_size = min(8192, max(block_size, 128))
    num_warps = min(max(block_size // 256, 1), 8)

    out = torch.empty_like(x)
    rmsnorm_kernel[(n_rows,)](
        x,
        out,
        w,
        eps,
        dim,
        block_size,
        num_warps=num_warps,
    )
    return out
