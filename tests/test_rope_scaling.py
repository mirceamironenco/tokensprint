import math

import torch

from engine.nn.rope import LLaMARoPEScaleConfig, build_rope_inv_freq


def _reference_llama_scaled_inv_freq(
    *,
    dim: int,
    rope_theta: float,
    rope_scale: LLaMARoPEScaleConfig,
    device: torch.device,
) -> torch.Tensor:
    idx = torch.arange(0, dim, step=2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (idx / dim))

    old_context_len = rope_scale.original_context_length
    scale_factor = rope_scale.factor
    low_factor, high_factor = rope_scale.frequency_factors

    low_freq_wavelen = old_context_len / low_factor
    high_freq_wavelen = old_context_len / high_factor

    out = []
    for freq in inv_freq.tolist():
        wavelen = 2.0 * math.pi / freq

        if wavelen < high_freq_wavelen:
            out.append(freq)
            continue

        if wavelen > low_freq_wavelen:
            out.append(freq / scale_factor)
            continue

        smooth = (old_context_len / wavelen - low_factor) / (high_factor - low_factor)
        out.append((1.0 - smooth) * freq / scale_factor + smooth * freq)

    return torch.tensor(out, dtype=inv_freq.dtype, device=device)


def test_build_rope_inv_freq_matches_reference_math() -> None:
    device = torch.device("cpu")
    rope_scale = LLaMARoPEScaleConfig(factor=32.0)

    expected = _reference_llama_scaled_inv_freq(
        dim=128,
        rope_theta=500_000.0,
        rope_scale=rope_scale,
        device=device,
    )
    got = build_rope_inv_freq(
        dim=128,
        rope_theta=500_000.0,
        device=device,
        rope_scale=rope_scale,
    )

    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)


def test_build_rope_inv_freq_without_scaling_matches_base() -> None:
    device = torch.device("cpu")

    base = build_rope_inv_freq(
        dim=128,
        rope_theta=10_000.0,
        device=device,
        rope_scale=None,
    )
    scaled = build_rope_inv_freq(
        dim=128,
        rope_theta=10_000.0,
        device=device,
        rope_scale=LLaMARoPEScaleConfig(factor=1.0),
    )

    torch.testing.assert_close(base, scaled, rtol=1e-6, atol=1e-6)
