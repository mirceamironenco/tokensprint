from __future__ import annotations

from engine.models._registry import register_family_checkpoint_converter
from engine.models.llama import (
    META_ORG,
    convert_llama_hf_checkpoint_to_mini,
    convert_llama_mini_to_hf_checkpoint,
    register_llama_models,
)
from engine.models.qwen import (
    QWEN_ORG,
    convert_qwen_hf_checkpoint_to_mini,
    convert_qwen_mini_to_hf_checkpoint,
    register_qwen_models,
)

_models_registered: bool = False


def register_models() -> None:
    global _models_registered

    if _models_registered:
        return

    register_llama_models()
    register_qwen_models()

    register_family_checkpoint_converter(
        META_ORG,
        from_hf_converter=convert_llama_hf_checkpoint_to_mini,
        to_hf_converter=convert_llama_mini_to_hf_checkpoint,
    )

    register_family_checkpoint_converter(
        QWEN_ORG,
        from_hf_converter=convert_qwen_hf_checkpoint_to_mini,
        to_hf_converter=convert_qwen_mini_to_hf_checkpoint,
    )

    _models_registered = True
