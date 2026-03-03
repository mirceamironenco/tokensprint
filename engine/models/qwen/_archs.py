from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from engine.models._registry import ModelConfig, get_family_decorator
from engine.nn.attention import AttentionConfig
from engine.nn.layers import FFNConfig, NormOrder


@dataclass(kw_only=True)
class QwenModelConfig(ModelConfig):
    pass


QWEN_ORG: Final = "qwen"


def register_qwen_models() -> None:
    arch = get_family_decorator(family=QWEN_ORG)

    @arch("qwen2.5-7b-instruct")
    def _qwen25_7b_instruct() -> QwenModelConfig:
        attn_cfg = AttentionConfig(
            model_dim=3584,
            num_heads=28,
            num_kv_heads=4,
            head_dim=None,
            bias=True,
            output_bias=False,
            qk_norm=False,
            qk_norm_eps=1e-6,
        )

        ffn_cfg = FFNConfig(
            model_dim=3584,
            inner_dim=18944,
            dim_multiplier=1.0,
            multiple_of=1,
            bias=False,
        )

        return QwenModelConfig(
            attn_config=attn_cfg,
            ffn_config=ffn_cfg,
            vocab_dim=152_064,
            num_layers=28,
            max_seq_len=32768,
            rope_theta=1_000_000.0,
            rope_impl="triton",
            norm_order=NormOrder.PRE,
            norm_eps=1e-6,
            rmsnorm_impl="triton",
            tie_weights=False,
        )

    @arch("qwen2.5-1.5b")
    def _qwen25_1_5b() -> QwenModelConfig:
        attn_cfg = AttentionConfig(
            model_dim=1536,
            num_heads=12,
            num_kv_heads=2,
            head_dim=None,
            bias=True,
            output_bias=False,
            qk_norm=False,
            qk_norm_eps=1e-6,
        )

        ffn_cfg = FFNConfig(
            model_dim=1536,
            inner_dim=8960,
            dim_multiplier=1.0,
            multiple_of=1,
            bias=False,
        )

        return QwenModelConfig(
            attn_config=attn_cfg,
            ffn_config=ffn_cfg,
            vocab_dim=151_936,
            num_layers=28,
            max_seq_len=131072,
            rope_theta=1_000_000.0,
            rope_impl="triton",
            norm_order=NormOrder.PRE,
            norm_eps=1e-6,
            rmsnorm_impl="triton",
            tie_weights=True,
        )

    @arch("qwen3-0.6b")
    def _qwen3_0_6b() -> QwenModelConfig:
        attn_cfg = AttentionConfig(
            model_dim=1024,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            bias=False,
            output_bias=False,
            qk_norm=True,
            qk_norm_eps=1e-6,
        )

        ffn_cfg = FFNConfig(
            model_dim=1024,
            inner_dim=3072,
            dim_multiplier=1.0,
            multiple_of=1,
            bias=False,
        )

        return QwenModelConfig(
            attn_config=attn_cfg,
            ffn_config=ffn_cfg,
            vocab_dim=151_936,
            num_layers=28,
            max_seq_len=40960,
            rope_theta=1_000_000.0,
            rope_impl="triton",
            norm_order=NormOrder.PRE,
            norm_eps=1e-6,
            rmsnorm_impl="triton",
            tie_weights=True,
        )
