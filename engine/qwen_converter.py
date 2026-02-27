import re
from collections.abc import Mapping
from typing import Any

# fmt: off
key_map = {
    r"^model\.layers\.([0-9]+)\.self_attn\.q_proj\.":        r"layers.\1.attn_layer.q_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.k_proj\.":        r"layers.\1.attn_layer.k_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.v_proj\.":        r"layers.\1.attn_layer.v_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.o_proj\.":        r"layers.\1.attn_layer.o_proj.",
    r"^model\.layers\.([0-9]+)\.self_attn\.q_norm\.":        r"layers.\1.attn_layer.q_norm.",
    r"^model\.layers\.([0-9]+)\.self_attn\.k_norm\.":        r"layers.\1.attn_layer.k_norm.",
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
    r"^layers\.([0-9]+)\.attn_layer\.q_proj\.":                  r"model.layers.\1.self_attn.q_proj.",
    r"^layers\.([0-9]+)\.attn_layer\.k_proj\.":                  r"model.layers.\1.self_attn.k_proj.",
    r"^layers\.([0-9]+)\.attn_layer\.v_proj\.":                  r"model.layers.\1.self_attn.v_proj.",
    r"^layers\.([0-9]+)\.attn_layer\.o_proj\.":                  r"model.layers.\1.self_attn.o_proj.",
    r"^layers\.([0-9]+)\.attn_layer\.q_norm\.":                  r"model.layers.\1.self_attn.q_norm.",
    r"^layers\.([0-9]+)\.attn_layer\.k_norm\.":                  r"model.layers.\1.self_attn.k_norm.",
    r"^layers\.([0-9]+)\.ffn_layer\.gate_proj\.":                r"model.layers.\1.mlp.gate_proj.",
    r"^layers\.([0-9]+)\.ffn_layer\.down_proj\.":                r"model.layers.\1.mlp.down_proj.",
    r"^layers\.([0-9]+)\.ffn_layer\.up_proj\.":                  r"model.layers.\1.mlp.up_proj.",
    r"^layers\.([0-9]+)\.attn_norm\.":                           r"model.layers.\1.input_layernorm.",
    r"^layers\.([0-9]+)\.ffn_norm\.":                            r"model.layers.\1.post_attention_layernorm.",
    r"^norm\.":                                                  r"model.norm.",
    r"^embedding\.":                                             r"model.embed_tokens.",
    r"^output_proj\.":                                           r"lm_head.",
}
# fmt: on


def convert_model_state_dict(
    state_dict: dict[str, Any], key_map: Mapping[str, str]
) -> dict[str, Any]:
    new_state_dict = {}

    for old_key in state_dict.keys():
        replacement_key = old_key

        for old_pattern, replacement in key_map.items():
            if (new_key := re.sub(old_pattern, replacement, old_key)) != old_key:
                replacement_key = new_key
                break

        new_state_dict[replacement_key] = state_dict[old_key]

    return new_state_dict


def convert_qwen_hf_checkpoint_to_mini(
    checkpoint: dict[str, Any], tie_weights: bool = False
) -> dict[str, Any]:
    if not tie_weights:
        return convert_model_state_dict(checkpoint, key_map)

    # Note: Assumes we are using a TiedProjectionLayer, which is FSDP2 compatible.

    local_key_map = {k: v for k, v in key_map.items()}

    local_key_map[r"^lm_head\."] = r"output_proj.embed."

    new_ckpt = convert_model_state_dict(checkpoint, local_key_map)

    new_ckpt["output_proj.embed.weight"] = new_ckpt["embedding.weight"]

    return new_ckpt


def convert_qwen_mini_to_hf_checkpoint(
    checkpoint: dict[str, Any], tie_weights: bool = False
) -> dict[str, Any]:
    if not tie_weights:
        return convert_model_state_dict(checkpoint, reverse_key_map)

    # Note: Assumes we are using a TiedProjectionLayer, which is FSDP2 compatible.

    local_key_map = {k: v for k, v in reverse_key_map.items()}

    local_key_map[r"^output_proj\.embed\."] = local_key_map.pop(r"^output_proj\.")

    new_ckpt = convert_model_state_dict(checkpoint, local_key_map)

    if "lm_head.weight" in new_ckpt:
        del new_ckpt["lm_head.weight"]

    return new_ckpt
