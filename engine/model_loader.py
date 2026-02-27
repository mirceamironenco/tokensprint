import os
import logging
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

from engine.qwen_converter import convert_qwen_hf_checkpoint_to_mini

log = logging.getLogger(__name__)


def download_checkpoint(
    repo_id: str,
    output_dir: Path = Path("./local_data"),
    hf_token: str | None = None,
    ignore_patterns: str | None = None,
    force: bool = False,
    max_workers: int = 4,
) -> Path:
    # repo_id example: meta-llama/llama-3.2-1b

    log.info("Attempting to download %s.", repo_id)

    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN", None)

    log.info(
        "Downloading %s model from HuggingFace. Model will be saved at %s.",
        repo_id,
        str(output_dir),
    )

    try:
        download_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=output_dir,
            force_download=force,
            token=hf_token,
            ignore_patterns=ignore_patterns,
            max_workers=max_workers,
        )
    except GatedRepoError as ex:
        if hf_token is not None:
            raise ValueError(
                f"HuggingFace token is set, but doesn't have access to {repo_id}."
            ) from ex
        raise
    else:
        log.info("Successfully downloaded %s at %s.", repo_id, download_dir)

    return Path(download_dir)


def load_model_hf_checkpoint(
    *,
    repo_id: str,
    cache_dir: Path,
    # machine: Machine,
) -> dict[str, Any]:
    model_name = repo_id.split("/")[-1]

    output_dir = cache_dir / f"{model_name}"

    # if machine.rank == 0:
    download_checkpoint(
        repo_id,
        output_dir=output_dir,
        force=False,
        ignore_patterns="original/consolidated*",
    )

    # machine.barrier()

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
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)

    return state_dict


def load_qwen_model_checkpoint(
    repo_id: str, cache_dir: Path, tie_weights: bool = False
) -> dict[str, Any]:
    state_dict = load_model_hf_checkpoint(
        repo_id=repo_id,
        cache_dir=cache_dir,
    )

    state_dict = convert_qwen_hf_checkpoint_to_mini(state_dict, tie_weights=tie_weights)

    return state_dict
