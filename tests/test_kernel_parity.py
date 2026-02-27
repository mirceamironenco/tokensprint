import pytest
import torch

from engine.nn.base_layers import RMSNorm
from engine.nn.rope import RopeEncoding


def _require_cuda_and_backend(backend: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for kernel parity tests.")

    if backend == "triton":
        pytest.importorskip("triton")
        return
    if backend == "helion":
        pytest.importorskip("helion")
        return

    raise ValueError(f"Unsupported backend: {backend}")


def _build_rope_encoder(
    backend: str,
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
) -> torch.nn.Module:
    if backend == "triton":
        from engine.kernels.triton.rope import TritonRopeEncoding

        return TritonRopeEncoding(head_dim, max_seq_len, rope_theta)

    if backend == "helion":
        from engine.kernels.helion.rope import HelionRopeEncoding

        return HelionRopeEncoding(head_dim, max_seq_len, rope_theta)

    raise ValueError(f"Unsupported backend: {backend}")


def _test_dtype_for_cuda() -> torch.dtype:
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@pytest.mark.parametrize("backend", ["triton", "helion"])
@torch.inference_mode()
def test_rope_backend_matches_table_decode_rank3(backend: str) -> None:
    _require_cuda_and_backend(backend)

    device = torch.device("cuda")
    dtype = _test_dtype_for_cuda()

    total_tokens = 257
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128
    max_seq_len = 4096
    rope_theta = 1_000_000.0

    query = torch.randn(
        total_tokens, num_heads, head_dim, device=device, dtype=dtype
    ).contiguous()
    key = torch.randn(
        total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype
    ).contiguous()
    input_pos = torch.randint(
        0, max_seq_len, (total_tokens,), device=device, dtype=torch.int64
    ).contiguous()

    rope_table = RopeEncoding(head_dim, max_seq_len, rope_theta).to(device=device)
    rope_backend = _build_rope_encoder(
        backend, head_dim, max_seq_len, rope_theta
    ).to(device=device)
    rope_table.reset_non_persistent_buffers()
    rope_backend.reset_non_persistent_buffers()

    query_table, key_table = rope_table(query, key, input_pos)
    query_backend, key_backend = rope_backend(query, key, input_pos)

    torch.testing.assert_close(
        query_table.float(), query_backend.float(), rtol=1e-2, atol=2e-2
    )
    torch.testing.assert_close(
        key_table.float(), key_backend.float(), rtol=1e-2, atol=2e-2
    )


@pytest.mark.parametrize("backend", ["triton", "helion"])
@torch.inference_mode()
def test_rope_backend_matches_table_prefill_rank4(backend: str) -> None:
    _require_cuda_and_backend(backend)

    device = torch.device("cuda")
    dtype = _test_dtype_for_cuda()

    batch_size = 4
    seq_len = 129
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128
    max_seq_len = 4096
    rope_theta = 1_000_000.0

    query = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    ).contiguous()
    key = torch.randn(
        batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype
    ).contiguous()
    input_pos_base = torch.randint(
        0, max_seq_len - seq_len, (batch_size, 1), device=device, dtype=torch.int64
    )
    input_pos = (input_pos_base + torch.arange(seq_len, device=device)).contiguous()

    rope_table = RopeEncoding(head_dim, max_seq_len, rope_theta).to(device=device)
    rope_backend = _build_rope_encoder(
        backend, head_dim, max_seq_len, rope_theta
    ).to(device=device)
    rope_table.reset_non_persistent_buffers()
    rope_backend.reset_non_persistent_buffers()

    query_table, key_table = rope_table(query, key, input_pos)
    query_backend, key_backend = rope_backend(query, key, input_pos)

    torch.testing.assert_close(
        query_table.float(), query_backend.float(), rtol=1e-2, atol=2e-2
    )
    torch.testing.assert_close(
        key_table.float(), key_backend.float(), rtol=1e-2, atol=2e-2
    )


@pytest.mark.parametrize("impl", ["triton", "helion"])
@pytest.mark.parametrize("shape", [(257, 1024), (4, 73, 1024)])
@torch.inference_mode()
def test_gpu_rmsnorm_impl_matches_torch(impl: str, shape: tuple[int, ...]) -> None:
    _require_cuda_and_backend(impl)

    device = torch.device("cuda")
    dtype = _test_dtype_for_cuda()
    eps = 1e-6
    dim = shape[-1]

    x = torch.randn(*shape, device=device, dtype=dtype).contiguous()

    norm_torch = RMSNorm(dim, eps=eps, impl="torch").to(device=device, dtype=dtype)
    norm_gpu = RMSNorm(dim, eps=eps, impl=impl).to(device=device, dtype=dtype)
    with torch.no_grad():
        norm_torch.weight.uniform_(-1.0, 1.0)
        norm_gpu.weight.copy_(norm_torch.weight)

    out_torch = norm_torch(x)
    out_gpu = norm_gpu(x)

    assert out_torch.dtype == x.dtype
    assert out_gpu.dtype == x.dtype
    torch.testing.assert_close(
        out_torch.float(), out_gpu.float(), rtol=5e-3, atol=1e-2
    )
