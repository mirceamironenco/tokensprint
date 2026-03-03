from __future__ import annotations

from typing import Any

import torch

from engine.models._loader import convert_model_state_dict
from engine.models._registry import ModelConfig

# fmt: off
key_map = {
    r"^model\.layers\.([0-9]+)\.self_attn\.q_proj\.":        r"layers.\1.attn_layer.q_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.k_proj\.":        r"layers.\1.attn_layer.k_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.v_proj\.":        r"layers.\1.attn_layer.v_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.o_proj\.":        r"layers.\1.attn_layer.o_proj.",
    r"^model\.layers\.([0-9]+)\.post_attention_layernorm\.": r"layers.\1.ffn_norm.",
    r"^model\.layers\.([0-9]+)\.mlp\.gate_proj\.":           r"layers.\1.ffn_layer.gate_proj.",
    r"^model\.layers\.([0-9]+)\.mlp\.down_proj\.":           r"layers.\1.ffn_layer.down_proj.",
    r"^model\.layers\.([0-9]+)\.mlp\.up_proj\.":             r"layers.\1.ffn_layer.up_proj.",
    r"^model\.layers\.([0-9]+)\.input_layernorm\.":          r"layers.\1.attn_norm.",
    r"^model\.norm\.":                                       r"norm.",
    r"^model\.embed_tokens\.":                               r"embedding.",
    r"^lm_head\.":                                           r"output_proj.",
}
# fmt: on

# fmt: off
reverse_key_map = {
    r"^layers\.([0-9]+)\.attn_layer\.q_proj\.":                 r"model.layers.\1.self_attn.q_proj.",
    r"^layers\.([0-9]+)\.attn_layer\.k_proj\.":                 r"model.layers.\1.self_attn.k_proj.",
    r"^layers\.([0-9]+)\.attn_layer\.v_proj\.":                 r"model.layers.\1.self_attn.v_proj.",
    r"^layers\.([0-9]+)\.attn_layer\.o_proj\.":                 r"model.layers.\1.self_attn.o_proj.",
    r"^layers\.([0-9]+)\.ffn_layer\.gate_proj\.":               r"model.layers.\1.mlp.gate_proj.",
    r"^layers\.([0-9]+)\.ffn_layer\.down_proj\.":               r"model.layers.\1.mlp.down_proj.",
    r"^layers\.([0-9]+)\.ffn_layer\.up_proj\.":                 r"model.layers.\1.mlp.up_proj.",
    r"^layers\.([0-9]+)\.attn_norm\.":                          r"model.layers.\1.input_layernorm.",
    r"^layers\.([0-9]+)\.ffn_norm\.":                           r"model.layers.\1.post_attention_layernorm.",
    r"^norm\.":                                                 r"model.norm.",
    r"^embedding\.":                                            r"model.embed_tokens.",
    r"^output_proj\.":                                          r"lm_head.",
}
# fmt: on


def _permute_rotary(
    weight: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    model_dim: int,
) -> torch.Tensor:
    weight = weight.view(num_heads, 2, head_dim // 2, model_dim)
    weight = weight.transpose(1, 2)
    return weight.reshape(-1, model_dim)


def _unpermute_rotary(
    weight: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    model_dim: int,
) -> torch.Tensor:
    weight = weight.view(num_heads, head_dim // 2, 2, model_dim)
    weight = weight.transpose(1, 2)
    return weight.reshape(-1, model_dim)


def convert_llama_hf_checkpoint_to_mini(
    checkpoint: dict[str, Any],
    config: ModelConfig,
) -> dict[str, Any]:
    checkpoint = dict(checkpoint)

    if config.tie_weights and "lm_head.weight" not in checkpoint:
        checkpoint["lm_head.weight"] = checkpoint["model.embed_tokens.weight"]

    head_dim = config.attn_config.head_dim or (config.model_dim // config.attn_config.num_heads)
    attn_heads = config.attn_config.num_heads
    kv_heads = config.attn_config.num_kv_heads or attn_heads

    for index in range(config.num_layers):
        q_key = f"model.layers.{index}.self_attn.q_proj.weight"
        k_key = f"model.layers.{index}.self_attn.k_proj.weight"

        checkpoint[q_key] = _permute_rotary(
            checkpoint[q_key],
            num_heads=attn_heads,
            head_dim=head_dim,
            model_dim=config.model_dim,
        )
        checkpoint[k_key] = _permute_rotary(
            checkpoint[k_key],
            num_heads=kv_heads,
            head_dim=head_dim,
            model_dim=config.model_dim,
        )

    if not config.tie_weights:
        return convert_model_state_dict(checkpoint, key_map)

    local_key_map = {k: v for k, v in key_map.items()}
    local_key_map[r"^lm_head\."] = r"output_proj.embed."

    new_ckpt = convert_model_state_dict(checkpoint, local_key_map)
    new_ckpt["output_proj.embed.weight"] = new_ckpt["embedding.weight"]

    return new_ckpt


def convert_llama_mini_to_hf_checkpoint(
    checkpoint: dict[str, Any],
    config: ModelConfig,
) -> dict[str, Any]:
    local_key_map = {k: v for k, v in reverse_key_map.items()}

    if config.tie_weights:
        local_key_map[r"^output_proj\.embed\."] = local_key_map.pop(r"^output_proj\.")

    new_ckpt = convert_model_state_dict(checkpoint, local_key_map)

    head_dim = config.attn_config.head_dim or (config.model_dim // config.attn_config.num_heads)
    attn_heads = config.attn_config.num_heads
    kv_heads = config.attn_config.num_kv_heads or attn_heads

    for index in range(config.num_layers):
        q_key = f"model.layers.{index}.self_attn.q_proj.weight"
        k_key = f"model.layers.{index}.self_attn.k_proj.weight"

        new_ckpt[q_key] = _unpermute_rotary(
            new_ckpt[q_key],
            num_heads=attn_heads,
            head_dim=head_dim,
            model_dim=config.model_dim,
        )
        new_ckpt[k_key] = _unpermute_rotary(
            new_ckpt[k_key],
            num_heads=kv_heads,
            head_dim=head_dim,
            model_dim=config.model_dim,
        )

    if config.tie_weights and "lm_head.weight" in new_ckpt:
        del new_ckpt["lm_head.weight"]

    return new_ckpt
