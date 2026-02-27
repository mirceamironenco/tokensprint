import struct
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Callable, Protocol, cast, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
import xxhash

CPU = torch.device("cpu")


@runtime_checkable
class ModuleWithNonPersistentBuffer(Protocol):
    def reset_non_persistent_buffers(self) -> None:
        """Reset the non-persistent buffers of the module."""


def reset_non_persistent_buffers(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, ModuleWithNonPersistentBuffer):
            m.reset_non_persistent_buffers()


def hashints(vals: list[int]) -> int:
    """
    Compute a hash of an int list and return it as
    a non-negative integer less than ``2**31 - 1``.
    """
    vala = np.array(vals, dtype=np.uint32)
    h = xxhash.xxh32_digest(vala.view(np.uint8))
    return struct.unpack("<I", h)[0] & 0x7FFFFFFF


def to_tensor(
    data: int | float | Sequence[int] | Sequence[float],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None or device.type != "cuda":
        return torch.tensor(data, dtype=dtype, device=device)

    t = torch.tensor(data, dtype=dtype, device=CPU, pin_memory=True)
    return t.to(device, non_blocking=True)


def replace_method_signature_with[T, **P](
    other_method: Callable[P, T],
) -> Callable[[Callable[..., Any]], Callable[P, T]]:
    def decorator(method: Callable[..., Any]) -> Callable[P, T]:
        return cast(Callable[P, T], method)

    return decorator


@contextmanager
def default_dtype(dtype: torch.dtype) -> Iterator[None]:
    _dtype = torch.get_default_dtype()

    torch.set_default_dtype(dtype)

    try:
        yield
    finally:
        torch.set_default_dtype(_dtype)
