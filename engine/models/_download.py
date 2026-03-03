from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

log = logging.getLogger(__name__)


def download_checkpoint(
    repo_id: str,
    output_dir: Path = Path("./local_data"),
    hf_token: str | None = None,
    ignore_patterns: str | None = None,
    force: bool = False,
    max_workers: int = 4,
) -> Path:
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
