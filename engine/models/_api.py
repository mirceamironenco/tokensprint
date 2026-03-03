from __future__ import annotations

import logging
from pathlib import Path

import torch

from engine.model import Transformer
from engine.models._builder import build_model
from engine.models._loader import load_model_checkpoint
from engine.models._registry import ModelConfig
from engine.tokenizer import (
    PretrainedHFTokenizer,
    load_hf_pretrained_tokenizer,
    log_tokenizer,
)
from engine.utils import reset_non_persistent_buffers

log = logging.getLogger(__name__)


def get_model_config(model_name: str) -> ModelConfig:
    return ModelConfig.from_model_name(model_name)


def get_local_model_dir(model_name: str, *, cache_dir: Path) -> Path:
    model_config = get_model_config(model_name)
    return cache_dir / model_config.repo_id.split("/")[-1]


def load_model(
    *,
    model_name: str,
    cache_dir: Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Transformer, PretrainedHFTokenizer, ModelConfig]:
    model_config = get_model_config(model_name)

    model = build_model(model_config, device=device, dtype=dtype)

    state_dict = load_model_checkpoint(model_config, cache_dir=cache_dir)

    model = model.to_empty(device=device)
    model.load_state_dict(state_dict, strict=True)

    reset_non_persistent_buffers(model)

    tokenizer = load_hf_pretrained_tokenizer(
        model_config.repo_id,
        cache_dir=cache_dir,
        force_download=False,
        use_fast=True,
        set_none_pad_token_to_eos=True,
    )

    log_tokenizer(log, tokenizer)

    return model, tokenizer, model_config
