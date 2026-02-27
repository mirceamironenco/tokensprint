# tokensprint

Small vllm-style inference engine.

## What this is
`tokensprint` is a compact, inference engine focused on vLLM-style generation flow:
- paged KV-cache block management
- scheduler-driven prefill/decode steps
- FlashAttention-based attention path
- optional Triton/Helion kernel paths (RoPE, RMSNorm)
- benchmark harness for mini backend vs vLLM

## Requirements
- Python `>=3.12`
- torch `==2.9.1`
- CUDA/Linux environment for Triton or Helion kernel backends

## Installation
```bash
uv sync
```

## Quick start
Run generation with the built-in engine:
```bash
uv run run.py \
  --model-name qwen/qwen3-0.6b \
  --prompt "Explain KV cache in one paragraph." \
  --max-tokens 256 \
  --temperature 0.8
```

On first run, model/tokenizer artifacts are downloaded into `./local_data`.

## Supported model configs
Current `run.py` presets:
- `qwen/qwen2.5-7b-instruct`
- `qwen/qwen3-0.6b`
- `qwen/qwen2.5-1.5b`
- more in progress

## Benchmark
Mini backend:
```bash
uv run benchmark.py --backend mini --use-tqdm False
```

vLLM backend (requires extra):
```bash
uv run benchmark.py --backend vllm --use-tqdm False
```

Side-by-side:
```bash
uv run benchmark.py --backend both --use-tqdm False --vllm-async-scheduling False
```

## Layout
- `engine/generation`: config, scheduler, block manager, model runner, engine API
- `engine/kernels`: backend subpackages (`triton/`, `helion/`) with shared kernels.
- `run.py`: simple generation entry point
- `benchmark.py`: performance comparison harness