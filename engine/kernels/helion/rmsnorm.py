from __future__ import annotations

import torch

try:
    import helion
    import helion.language as hl
except ModuleNotFoundError:
    helion = None
    hl = None


def _missing_helion_error() -> ModuleNotFoundError:
    return ModuleNotFoundError(
        "Helion is required for helion RMSNorm kernels. "
        "Install Helion on a supported CUDA/Linux environment."
    )


if helion is not None:

    @helion.kernel(autotune_effort="none")
    def rmsnorm_kernel(
        x: torch.Tensor,
        w: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        n_rows = x.shape[0]
        out = torch.empty_like(x)

        for tile_row in hl.tile(n_rows):
            x_fp32 = x[tile_row, :].to(torch.float32)
            mean = (x_fp32 * x_fp32).mean(dim=1, keepdim=True)
            inv_rms = torch.rsqrt(mean + eps)
            out[tile_row, :] = (x_fp32 * inv_rms * w[:].to(torch.float32)).to(x.dtype)

        return out

else:

    def rmsnorm_kernel(
        x: torch.Tensor,
        w: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        raise _missing_helion_error()


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
    if n_rows < 1:
        raise ValueError("x must have at least one row.")

    if tuple(w.shape) != (dim,):
        raise ValueError(f"w must have shape ({dim},); got w.shape={tuple(w.shape)}.")
    if not w.is_contiguous():
        raise ValueError("w must be contiguous.")

    if not x.is_cuda or not w.is_cuda:
        raise ValueError("x and w must be CUDA tensors for Helion RMSNorm.")
    if x.device != w.device:
        raise ValueError("x and w must be on the same CUDA device.")

    return rmsnorm_kernel(x, w, eps)
