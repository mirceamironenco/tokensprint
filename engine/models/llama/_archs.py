from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from engine.models._registry import ModelConfig, get_family_decorator
from engine.nn.attention import AttentionConfig
from engine.nn.layers import FFNConfig, NormOrder
from engine.nn.rope import LLaMARoPEScaleConfig


@dataclass(kw_only=True)
class LlamaModelConfig(ModelConfig):
    pass


META_ORG: Final = "meta-llama"


def register_llama_models() -> None:
    arch = get_family_decorator(family=META_ORG)

    @arch("llama-3-8b")
    def _llama3_8b() -> LlamaModelConfig:
        attn_cfg = AttentionConfig(
            model_dim=4096,
            num_heads=32,
            num_kv_heads=8,
            head_dim=None,
            bias=False,
            output_bias=False,
            qk_norm=False,
            qk_norm_eps=1e-6,
        )

        ffn_cfg = FFNConfig(
            model_dim=4096,
            inner_dim=int(4096 * 4 * 1.3),
            dim_multiplier=2 / 3,
            multiple_of=1024,
            bias=False,
        )

        return LlamaModelConfig(
            attn_config=attn_cfg,
            ffn_config=ffn_cfg,
            vocab_dim=128_256,
            num_layers=32,
            max_seq_len=8192,
            rope_theta=500_000.0,
            rope_impl="triton",
            norm_order=NormOrder.PRE,
            norm_eps=1e-5,
            rmsnorm_impl="triton",
            tie_weights=False,
        )

    @arch("llama-3.1-8b")
    def _llama3_1_8b() -> LlamaModelConfig:
        config = _llama3_8b()
        config.max_seq_len = 131072
        config.rope_scale = LLaMARoPEScaleConfig()
        return config

    @arch("llama-3.1-8b-instruct")
    def _llama3_1_8b_instruct() -> LlamaModelConfig:
        return _llama3_1_8b()

    @arch("llama-3.2-3b")
    def _llama3_2_3b() -> LlamaModelConfig:
        config = _llama3_1_8b()

        config.model_dim = 3072
        config.num_layers = 28
        config.tie_weights = True
        config.ffn_config.inner_dim = int(3072 * 4 * 1.0)
        config.ffn_config.multiple_of = 256
        config.attn_config.num_heads = 24
        config.attn_config.num_kv_heads = 8
        config.rope_scale = LLaMARoPEScaleConfig(factor=32.0)

        return config

    @arch("llama-3.2-3b-instruct")
    def _llama3_2_3b_instruct() -> LlamaModelConfig:
        return _llama3_2_3b()

    @arch("llama-3.2-1b")
    def _llama3_2_1b() -> LlamaModelConfig:
        config = _llama3_1_8b()

        config.model_dim = 2048
        config.num_layers = 16
        config.tie_weights = True
        config.ffn_config.inner_dim = int(2048 * 4 * 1.5)
        config.ffn_config.multiple_of = 256
        config.attn_config.num_heads = 32
        config.attn_config.num_kv_heads = 8
        config.rope_scale = LLaMARoPEScaleConfig(factor=32.0)

        return config

    @arch("llama-3.2-1b-instruct")
    def _llama3_2_1b_instruct() -> LlamaModelConfig:
        return _llama3_2_1b()
