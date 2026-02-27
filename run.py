from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
import logging

from engine.generation.config import Config, SamplingParams
from engine.generation.llm_engine import LLMEngine
from engine.model_loader import load_qwen_model_checkpoint
from engine.nn.layers import NormOrder
from engine.model import Transformer
from engine.tokenizer import (
    PretrainedHFTokenizer,
    load_hf_pretrained_tokenizer,
    log_tokenizer,
)
from engine.utils import default_dtype, reset_non_persistent_buffers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger(__name__)

model_dict = {
    "qwen/qwen2.5-7b-instruct": {
        "model_dim": 3584,
        "vocab_dim": 152_064,
        "num_layers": 28,
        "num_heads": 28,
        "num_kv_heads": 4,
        "max_seq_len": 32768,
        "rope_theta": 1000000.0,
        "rope_impl": "triton",
        "norm_order": NormOrder.PRE,
        "norm_eps": 1e-6,
        "rmsnorm_impl": "triton",
        "tie_weights": False,
        "head_dim": None,
        "attn_bias": True,
        "attn_output_bias": False,
        "attn_qk_norm": False,
        "ffn_inner_dim": 18944,
        "ffn_multiple_of": 1,
        "ffn_multiplier": 1.0,
        "ffn_bias": False,
    },
    "qwen/qwen2.5-1.5b": {
        "model_dim": 1536,
        "vocab_dim": 151_936,
        "num_layers": 28,
        "num_heads": 12,
        "num_kv_heads": 2,
        "max_seq_len": 131072,
        "rope_theta": 1000000.0,
        "rope_impl": "triton",
        "norm_order": NormOrder.PRE,
        "norm_eps": 1e-6,
        "rmsnorm_impl": "triton",
        "tie_weights": True,
        "head_dim": None,
        "attn_bias": True,
        "attn_output_bias": False,
        "attn_qk_norm": False,
        "ffn_inner_dim": 8960,
        "ffn_multiple_of": 1,
        "ffn_multiplier": 1.0,
        "ffn_bias": False,
    },
    "qwen/qwen3-0.6b": {
        "model_dim": 1024,
        "vocab_dim": 151_936,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "max_seq_len": 40960,
        "rope_theta": 1000000.0,
        "rope_impl": "triton",
        "norm_order": NormOrder.PRE,
        "norm_eps": 1e-6,
        "rmsnorm_impl": "triton",
        "tie_weights": True,
        "head_dim": 128,
        "attn_bias": False,
        "attn_output_bias": False,
        "attn_qk_norm": True,
        "attn_qk_norm_eps": 1e-6,
        "ffn_inner_dim": 3072,
        "ffn_multiple_of": 1,
        "ffn_multiplier": 1.0,
        "ffn_bias": False,
    },
}


@dataclass
class Args:
    model_name: str = "qwen/qwen3-0.6b"
    cache_dir: Path = Path("./local_data")
    prompt: str = "Tell me a long story about transformers."
    max_tokens: int = 1024
    temperature: float = 0.8


def load_model(
    *,
    model_name: str,
    cache_dir: Path,
    device: torch.device,
) -> tuple[Transformer, PretrainedHFTokenizer]:
    if model_name not in model_dict:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. "
            f"Available models: {sorted(model_dict.keys())}"
        )

    model_cfg = model_dict[model_name]

    with device, default_dtype(torch.bfloat16):
        model = Transformer.build(**model_cfg)

    state_dict = load_qwen_model_checkpoint(
        model_name,
        cache_dir,
        tie_weights=model_cfg["tie_weights"],
    )
    model = model.to_empty(device=device)
    model.load_state_dict(state_dict, strict=True)

    reset_non_persistent_buffers(model)

    tokenizer = load_hf_pretrained_tokenizer(
        model_name,
        cache_dir=cache_dir,
        force_download=False,
        use_fast=True,
        set_none_pad_token_to_eos=True,
    )

    log_tokenizer(log, tokenizer)

    return model, tokenizer


@torch.inference_mode()
def main(args: Args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to test LLMEngine in this setup.")

    device = torch.device("cuda")
    model, tokenizer = load_model(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=device,
    )

    model_dir = args.cache_dir / args.model_name.split("/")[-1]

    config = Config(
        model=str(model_dir),
        eos=tokenizer.eos_token_id,
        dtype=torch.bfloat16,
    )

    engine = LLMEngine(model=model, config=config, tokenizer=tokenizer)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    outputs = engine.generate([args.prompt], sampling_params)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(decoded[0])


if __name__ == "__main__":
    main(tyro.cli(Args))
