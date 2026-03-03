from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from engine.generation.config import Config, SamplingParams
from engine.generation.llm_engine import LLMEngine
from engine.models import (
    get_local_model_dir,
    get_model_config,
    load_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)


@dataclass
class Args:
    model_name: str = "qwen/qwen3-0.6b"
    cache_dir: Path = Path("./local_data")
    prompt: str = "Tell me a long story about transformers."
    max_tokens: int = 1024
    temperature: float = 0.8
    enforce_eager: bool = False


@torch.inference_mode()
def main(args: Args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to test LLMEngine in this setup.")

    device = torch.device("cuda")

    model, tokenizer, _ = load_model(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=device,
        dtype=torch.bfloat16,
    )
    model_cfg = get_model_config(args.model_name)

    model_dir = get_local_model_dir(args.model_name, cache_dir=args.cache_dir)

    config = Config(
        model=str(model_dir),
        max_model_len=int(model_cfg.max_seq_len),
        eos=tokenizer.eos_token_id,
        dtype=torch.bfloat16,
        enforce_eager=args.enforce_eager,
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
    main(tyro.cli(Args, config=(tyro.conf.FlagConversionOff,)))
