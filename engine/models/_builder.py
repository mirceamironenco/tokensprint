from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from engine.kernels.helion.rope import HelionRopeEncoding
from engine.kernels.triton.rope import TritonRopeEncoding
from engine.model import Transformer
from engine.models._registry import ModelConfig
from engine.nn.attention import Attention
from engine.nn.base_layers import RMSNorm
from engine.nn.layers import SwiGLUFFN, TiedProjectionLayer, TransformerBlock
from engine.nn.rope import RopeEncoding
from engine.nn.sdpa import FlashSDPA
from engine.utils import default_dtype


def build_model(
    config: ModelConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Transformer:
    with device, default_dtype(dtype):
        frontend = nn.Embedding(config.vocab_dim, config.model_dim)
        norm_op = RMSNorm.resolve_norm_op(config.rmsnorm_impl)

        if config.model_dim % config.attn_config.num_heads != 0:
            raise ValueError(
                "model_dim must be divisible by num_heads, "
                f"got model_dim={config.model_dim} "
                f"and num_heads={config.attn_config.num_heads}."
            )

        head_dim = config.attn_config.head_dim or (
            config.model_dim // config.attn_config.num_heads
        )

        if config.rope_impl == "table":
            rope_encoder = RopeEncoding(
                head_dim,
                config.max_seq_len,
                config.rope_theta,
                rope_scale=config.rope_scale,
            )
        elif config.rope_impl == "triton":
            rope_encoder = TritonRopeEncoding(
                head_dim,
                config.max_seq_len,
                config.rope_theta,
                rope_scale=config.rope_scale,
                approx_trigo=config.rope_approx_trigo,
            )
        elif config.rope_impl == "helion":
            rope_encoder = HelionRopeEncoding(
                head_dim,
                config.max_seq_len,
                config.rope_theta,
                rope_scale=config.rope_scale,
                approx_trigo=config.rope_approx_trigo,
            )
        else:
            raise ValueError(f"Unsupported rope_impl: {config.rope_impl}.")

        layers: list[TransformerBlock] = []
        for _ in range(config.num_layers):
            attn_config = replace(config.attn_config)
            if attn_config.head_dim is None:
                attn_config.head_dim = head_dim

            attn = Attention(
                attn_config,
                pos_encoder=rope_encoder,
                sdpa=FlashSDPA(),
                norm_impl=config.rmsnorm_impl,
                norm_op=norm_op,
            )

            ffn = SwiGLUFFN(replace(config.ffn_config))

            layers.append(
                TransformerBlock(
                    attn,
                    ffn,
                    config.model_dim,
                    config.norm_order,
                    config.norm_eps,
                    norm_impl=config.rmsnorm_impl,
                    norm_op=norm_op,
                )
            )

        if not config.tie_weights:
            output_proj: nn.Module = nn.Linear(config.model_dim, config.vocab_dim)
        else:
            output_proj = TiedProjectionLayer(frontend)

        return Transformer(
            config.model_dim,
            frontend,
            layers,
            output_proj,
            config.norm_order,
            config.norm_eps,
            norm_impl=config.rmsnorm_impl,
            norm_op=norm_op,
        )
