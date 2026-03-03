# tokensprint

Small vllm-style inference engine.

## What this is
`tokensprint` is a compact, inference engine focused on vLLM-style generation flow:
- paged KV-cache block management
- scheduler-driven prefill/decode steps
- FlashAttention-based attention path
- optional Triton/Helion kernel paths (RoPE, RMSNorm)
- benchmark harness for tokensprint backend vs vLLM

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

## Supported model configs
Models are provided by the runtime registry. List all currently registered models:
```bash
uv run model_registry.py
```

Currently registered:
- `qwen/qwen2.5-7b-instruct`
- `qwen/qwen3-0.6b`
- `qwen/qwen2.5-1.5b`
- `meta-llama/llama-3-8b`
- `meta-llama/llama-3.1-8b`
- `meta-llama/llama-3.1-8b-instruct`
- `meta-llama/llama-3.2-3b`
- `meta-llama/llama-3.2-3b-instruct`
- `meta-llama/llama-3.2-1b`
- `meta-llama/llama-3.2-1b-instruct`

Both short names and repo-style names are accepted, for example:
- `qwen3-0.6b` and `qwen/qwen3-0.6b`
- `llama-3.2-1b` and `meta-llama/llama-3.2-1b`

## Benchmark
tokensprint backend:
```bash
uv run benchmark.py --backend tokensprint --use-tqdm False
```

vLLM backend (requires installing optional dependencies first):
```bash
uv sync --extra vllm
```

Then run:
```bash
uv run benchmark.py --backend vllm --use-tqdm False
```

Side-by-side:
```bash
uv run benchmark.py --backend both --use-tqdm False --vllm-async-scheduling False
```

## Layout
- `engine/generation`: scheduler-driven inference flow (`LLMEngine`, scheduler, block manager, model runner).
- `engine/models`: model registry/runtime, family arch definitions (`qwen/`, `llama/`), HF checkpoint converters, model build/load APIs.
- `engine/nn`: transformer modules (attention, FFN, blocks, RoPE reference implementation).
- `engine/kernels`: backend kernels (`triton/`, `helion/`) for RoPE and RMSNorm.
- `run.py`: single-prompt generation entrypoint.
- `benchmark.py`: throughput comparison harness (tokensprint backend vs vLLM).
- `model_registry.py`: prints all registered models grouped by family.
