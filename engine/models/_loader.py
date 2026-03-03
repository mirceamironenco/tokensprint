from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from engine.models._download import download_checkpoint
from engine.models._registry import (
    ModelConfig,
    get_from_hf_ckpt_converter,
    get_model_family,
)

log = logging.getLogger(__name__)


def convert_model_state_dict(
    state_dict: dict[str, Any],
    key_map: Mapping[str, str],
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


def load_model_hf_checkpoint(
    *,
    repo_id: str,
    cache_dir: Path,
) -> dict[str, Any]:
    model_name = repo_id.split("/")[-1]

    output_dir = cache_dir / model_name

    download_checkpoint(
        repo_id,
        output_dir=output_dir,
        force=False,
        ignore_patterns="original/consolidated*",
    )

    if not output_dir.exists():
        raise FileNotFoundError(f"Failed to find model state dict at {output_dir}.")

    log.info("Loading checkpoint from %s.", str(output_dir))

    try:
        from safetensors import safe_open
    except ImportError:
        raise RuntimeError("Safetensors not found. Use `pip install safetensors`.")

    files = list(output_dir.glob("*.safetensors"))

    if not files:
        raise RuntimeError(f"No safetensors files found at {output_dir}.")

    state_dict = {}

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return state_dict


def convert_hf_sd_to_mini(
    state_dict: dict[str, Any],
    config: ModelConfig,
) -> dict[str, Any]:
    family = get_model_family(config.model)
    ckpt_converter = get_from_hf_ckpt_converter(family)
    return ckpt_converter(state_dict, config)


def load_model_checkpoint(config: ModelConfig, *, cache_dir: Path) -> dict[str, Any]:
    state_dict = load_model_hf_checkpoint(
        repo_id=config.repo_id,
        cache_dir=cache_dir,
    )

    return convert_hf_sd_to_mini(state_dict, config=config)
