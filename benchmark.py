from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from random import randint, seed
from typing import Literal

import torch
import tyro

from engine.generation.config import Config, SamplingParams
from engine.generation.llm_engine import LLMEngine


@dataclass
class Args:
    backend: Literal["mini", "vllm", "both"] = "mini"
    model_name: str = "qwen/qwen3-0.6b"
    cache_dir: Path = Path("./local_data")
    num_seqs: int = 256
    max_input_len: int = 1024
    max_output_len: int = 1024
    max_model_len: int = 4096
    enforce_eager: bool = False
    vllm_enforce_eager: bool = True
    chunked_prefill: bool = True
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    vllm_async_scheduling: bool = False
    use_tqdm: bool = True
    warmup_prompt: str = "Benchmark: "
    max_prompt_token_id: int = 10000


def _make_workload(args: Args) -> tuple[list[list[int]], list[int]]:
    prompt_token_ids = [
        [
            randint(0, args.max_prompt_token_id)
            for _ in range(randint(100, args.max_input_len))
        ]
        for _ in range(args.num_seqs)
    ]
    output_lens = [randint(100, args.max_output_len) for _ in range(args.num_seqs)]
    return prompt_token_ids, output_lens


def _print_result(backend: str, total_tokens: int, elapsed_s: float) -> None:
    throughput = total_tokens / elapsed_s
    print(
        f"[{backend}] Total: {total_tokens}tok, "
        f"Time: {elapsed_s:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


@torch.inference_mode()
def run_mini(
    args: Args, prompt_token_ids: list[list[int]], output_lens: list[int]
) -> None:
    from run import load_model

    device = torch.device("cuda")
    model, tokenizer = load_model(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=device,
    )

    model_dir = args.cache_dir / args.model_name.split("/")[-1]
    llm = LLMEngine(
        model=model,
        config=Config(
            model=str(model_dir),
            enforce_eager=args.enforce_eager,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            chunked_prefill=args.chunked_prefill,
            eos=tokenizer.eos_token_id,
            dtype=torch.bfloat16,
        ),
        tokenizer=tokenizer,
    )

    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=max_tokens,
        )
        for max_tokens in output_lens
    ]

    llm.generate([args.warmup_prompt], SamplingParams(), use_tqdm=args.use_tqdm)
    torch.cuda.synchronize(device)

    t = time.perf_counter()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=args.use_tqdm)
    torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - t

    total_tokens = sum(output_lens)
    _print_result("mini", total_tokens, elapsed_s)


def _resolve_vllm_model(args: Args) -> str:
    local_model_dir = args.cache_dir / args.model_name.split("/")[-1]
    if local_model_dir.exists():
        return str(local_model_dir)
    return args.model_name


def run_vllm(
    args: Args, prompt_token_ids: list[list[int]], output_lens: list[int]
) -> None:
    try:
        from vllm import LLM, SamplingParams as VLLMSamplingParams
    except ImportError as ex:
        raise RuntimeError(
            "vllm backend requested but vllm is not installed. "
            "Run `uv sync --extra vllm` first."
        ) from ex

    model_ref = _resolve_vllm_model(args)
    llm = LLM(
        model=model_ref,
        enforce_eager=args.vllm_enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=args.chunked_prefill,
        async_scheduling=args.vllm_async_scheduling,
    )

    prompts = [{"prompt_token_ids": p} for p in prompt_token_ids]
    sampling_params = [
        VLLMSamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=max_tokens,
        )
        for max_tokens in output_lens
    ]

    llm.generate([args.warmup_prompt], VLLMSamplingParams(), use_tqdm=args.use_tqdm)

    t = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=args.use_tqdm)
    elapsed_s = time.perf_counter() - t

    total_tokens = sum(output_lens)
    _print_result("vllm", total_tokens, elapsed_s)


def main(args: Args) -> None:
    if args.backend in ("vllm", "both"):
        # vLLM worker processes must avoid CUDA-after-fork failures.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        mp.set_start_method("spawn", force=True)

    if args.backend in ("mini", "both") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run benchmark.py.")

    if args.max_input_len < 100:
        raise ValueError("max_input_len must be >= 100.")

    if args.max_output_len < 100:
        raise ValueError("max_output_len must be >= 100.")

    seed(0)
    prompt_token_ids, output_lens = _make_workload(args)

    if args.backend in ("mini", "both"):
        run_mini(args, prompt_token_ids, output_lens)

    if args.backend in ("vllm", "both"):
        run_vllm(args, prompt_token_ids, output_lens)


if __name__ == "__main__":
    main(tyro.cli(Args, config=(tyro.conf.FlagConversionOff,)))
