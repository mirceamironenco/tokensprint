import torch

from engine.models import (
    ModelConfig,
    convert_hf_sd_to_mini,
    get_model_repo_id,
    model_is_registered,
)


def test_model_config_resolves_short_and_repo_id_names() -> None:
    short_cfg = ModelConfig.from_model_name("qwen3-0.6b")
    repo_cfg = ModelConfig.from_model_name("qwen/qwen3-0.6b")

    assert short_cfg.model == "qwen3-0.6b"
    assert repo_cfg.model == "qwen3-0.6b"
    assert short_cfg.repo_id == "qwen/qwen3-0.6b"
    assert repo_cfg.repo_id == "qwen/qwen3-0.6b"



def test_model_registration_checks_repo_id_aliases() -> None:
    assert model_is_registered("qwen3-0.6b")
    assert model_is_registered("qwen/qwen3-0.6b")
    assert model_is_registered("llama-3.2-1b")
    assert model_is_registered("meta-llama/llama-3.2-1b")



def test_get_model_repo_id_accepts_short_and_repo_names() -> None:
    assert get_model_repo_id("qwen3-0.6b") == "qwen/qwen3-0.6b"
    assert get_model_repo_id("qwen/qwen3-0.6b") == "qwen/qwen3-0.6b"
    assert get_model_repo_id("llama-3.2-1b") == "meta-llama/llama-3.2-1b"
    assert (
        get_model_repo_id("meta-llama/llama-3.2-1b") == "meta-llama/llama-3.2-1b"
    )



def test_convert_hf_sd_to_mini_routes_qwen_converter() -> None:
    cfg = ModelConfig.from_model_name("qwen/qwen2.5-7b-instruct")

    state_dict = {
        "model.embed_tokens.weight": torch.randn(8, 4),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
        "lm_head.weight": torch.randn(8, 4),
    }

    converted = convert_hf_sd_to_mini(state_dict, cfg)

    assert "embedding.weight" in converted
    assert "layers.0.attn_layer.q_proj.weight" in converted
    assert "output_proj.weight" in converted



def test_convert_hf_sd_to_mini_qwen_tied_weights_sets_output_embed() -> None:
    cfg = ModelConfig.from_model_name("qwen/qwen3-0.6b")

    state_dict = {
        "model.embed_tokens.weight": torch.randn(8, 4),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
    }

    converted = convert_hf_sd_to_mini(state_dict, cfg)

    assert "output_proj.embed.weight" in converted
    assert torch.equal(converted["output_proj.embed.weight"], converted["embedding.weight"])


def test_convert_hf_sd_to_mini_routes_llama_converter() -> None:
    cfg = ModelConfig.from_model_name("meta-llama/llama-3.2-1b")
    cfg.model_dim = 8
    cfg.attn_config.num_heads = 2
    cfg.attn_config.num_kv_heads = 1
    cfg.num_layers = 1

    state_dict = {
        "model.embed_tokens.weight": torch.randn(8, 8),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(4, 8),
    }

    converted = convert_hf_sd_to_mini(state_dict, cfg)

    assert "embedding.weight" in converted
    assert "layers.0.attn_layer.q_proj.weight" in converted
    assert "layers.0.attn_layer.k_proj.weight" in converted
    assert "output_proj.embed.weight" in converted


def test_llama_rope_scale_presets() -> None:
    llama3 = ModelConfig.from_model_name("meta-llama/llama-3-8b")
    llama31 = ModelConfig.from_model_name("meta-llama/llama-3.1-8b")
    llama32 = ModelConfig.from_model_name("meta-llama/llama-3.2-1b")

    assert llama3.rope_scale is None
    assert llama31.rope_scale is not None
    assert llama31.rope_scale.factor == 8.0
    assert llama32.rope_scale is not None
    assert llama32.rope_scale.factor == 32.0
