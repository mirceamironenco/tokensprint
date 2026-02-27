from typing import TYPE_CHECKING, Literal, Protocol

import torch
import torch.nn as nn


class Linear(nn.Linear):
    if TYPE_CHECKING:

        def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class RMSNorm(nn.Module):
    class Op(Protocol):
        def __call__(
            self,
            x: torch.Tensor,
            w: torch.Tensor,
            eps: float,
        ) -> torch.Tensor: ...

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        impl: Literal["torch", "triton", "helion"] = "torch",
        norm_op: Op | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim
        if impl not in ("torch", "triton", "helion"):
            raise ValueError(f"Unsupported RMSNorm impl: {impl}.")
        self.impl = impl
        self.weight = nn.Parameter(torch.empty(dim))
        self.norm_op = norm_op or self.resolve_norm_op(impl)

    @staticmethod
    def _torch_norm_op(
        x: torch.Tensor,
        w: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        x_fp32 = x.float()
        output = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
        return output.type_as(x) * w

    @staticmethod
    def resolve_norm_op(impl: str) -> Op:
        if impl == "torch":
            return RMSNorm._torch_norm_op

        if impl == "triton":
            from engine.kernels.triton.rmsnorm import rmsnorm as triton_rmsnorm

            return triton_rmsnorm

        if impl == "helion":
            from engine.kernels.helion.rmsnorm import rmsnorm as helion_rmsnorm

            return helion_rmsnorm

        raise ValueError(f"Unsupported RMSNorm impl: {impl}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.dim:
            raise ValueError(
                f"Expected x.shape[-1] == {self.dim}, got {x.size(-1)}."
            )

        x_shape = x.shape
        x_2d = x.contiguous().view(-1, self.dim)
        out_2d = self.norm_op(x_2d, self.weight, self.eps)
        return out_2d.view(x_shape)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, impl={self.impl}"
