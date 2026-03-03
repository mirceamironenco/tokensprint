from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from engine.nn.attention import AttentionConfig
from engine.nn.layers import FFNConfig, NormOrder
from engine.nn.rope import LLaMARoPEScaleConfig


@dataclass(kw_only=True)
class ModelConfig:
    attn_config: AttentionConfig
    ffn_config: FFNConfig

    vocab_dim: int
    num_layers: int

    attn_window_len: int | None = None
    max_seq_len: int = 4096

    rope_theta: float = 10000.0
    rope_impl: Literal["table", "triton", "helion"] = "triton"
    rope_approx_trigo: bool = False
    rope_scale: LLaMARoPEScaleConfig | None = None

    norm_order: NormOrder = NormOrder.PRE
    norm_eps: float = 1e-6
    rmsnorm_impl: Literal["torch", "triton", "helion"] = "torch"

    tie_weights: bool = False

    _model_dim: int = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _family: str = field(init=False, repr=False)

    finetune_repo_id: str | None = None

    def __post_init__(self) -> None:
        if self.attn_config.model_dim != self.ffn_config.model_dim:
            raise ValueError(
                "attn_config.model_dim and ffn_config.model_dim must match, "
                f"got {self.attn_config.model_dim} and {self.ffn_config.model_dim}."
            )

        self._model_dim = self.attn_config.model_dim

    @property
    def model(self) -> str:
        return self._name

    @property
    def family(self) -> str:
        return self._family

    @property
    def model_dim(self) -> int:
        return self._model_dim

    @model_dim.setter
    def model_dim(self, model_dim: int) -> None:
        self._model_dim = model_dim
        self.attn_config.model_dim = model_dim
        self.ffn_config.model_dim = model_dim

    @property
    def base_model_repo_id(self) -> str:
        return get_model_repo_id(self.model)

    @property
    def repo_id(self) -> str:
        if self.finetune_repo_id is not None:
            return self.finetune_repo_id

        return self.base_model_repo_id

    @classmethod
    def from_model_name(
        cls,
        name: str,
        *,
        finetune_repo_id: str | None = None,
    ) -> ModelConfig:
        _ensure_models_registered()

        resolved_name = resolve_model_name(name)

        model_config = MODEL_REGISTRY[resolved_name]()
        model_config._name = resolved_name
        model_config._family = MODEL_FAMILY_REGISTRY[resolved_name]
        model_config.finetune_repo_id = finetune_repo_id

        return model_config


class CheckpointConverter(Protocol):
    def __call__(
        self,
        checkpoint: dict[str, Any],
        config: ModelConfig,
    ) -> dict[str, Any]: ...


MODEL_REGISTRY: dict[str, Callable[..., ModelConfig]] = {}
MODEL_FAMILY_REGISTRY: dict[str, str] = {}
FAMILY_MODEL_REGISTRY: dict[str, set[str]] = defaultdict(set)
FAMILY_CKPT_TO_HF: dict[str, CheckpointConverter] = {}
FAMILY_CKPT_FROM_HF: dict[str, CheckpointConverter] = {}


def _ensure_models_registered() -> None:
    from engine.models.runtime import register_models

    register_models()


class ModelConfigBuilder[ModelConfigT: ModelConfig](Protocol):
    def __call__(self) -> ModelConfigT: ...


class FamilyModelDecorator[ModelConfigT: ModelConfig](Protocol):
    def __call__(
        self,
        builder: ModelConfigBuilder[ModelConfigT],
        /,
    ) -> ModelConfigBuilder[ModelConfigT]: ...


def get_family_decorator[ModelConfigT: ModelConfig](
    family: str,
) -> Callable[[str], FamilyModelDecorator[ModelConfigT]]:
    def family_decorator(name: str) -> FamilyModelDecorator[ModelConfigT]:
        def _register_arch(
            fn: ModelConfigBuilder[ModelConfigT],
        ) -> ModelConfigBuilder[ModelConfigT]:
            if name in MODEL_REGISTRY:
                raise ValueError(f"Model {name} already registered.")

            if name in FAMILY_MODEL_REGISTRY[family]:
                raise ValueError(f"Model {name} already registered to family {family}.")

            MODEL_REGISTRY[name] = fn
            MODEL_FAMILY_REGISTRY[name] = family
            FAMILY_MODEL_REGISTRY[family].add(name)

            return fn

        return _register_arch

    return family_decorator


def register_family_checkpoint_converter(
    family: str,
    *,
    from_hf_converter: CheckpointConverter,
    to_hf_converter: CheckpointConverter,
) -> None:
    if family not in FAMILY_MODEL_REGISTRY:
        raise ValueError(
            f"No registered families named {family}, register a model first."
        )

    FAMILY_CKPT_FROM_HF[family] = from_hf_converter
    FAMILY_CKPT_TO_HF[family] = to_hf_converter


def get_to_hf_checkpoint_converter(family: str) -> CheckpointConverter:
    if family not in FAMILY_MODEL_REGISTRY:
        raise ValueError(f"No registered families named {family}.")

    if family not in FAMILY_CKPT_TO_HF:
        raise ValueError(
            f"No converter to hugging face format registered for model family {family}."
        )

    return FAMILY_CKPT_TO_HF[family]


def get_from_hf_ckpt_converter(family: str) -> CheckpointConverter:
    if family not in FAMILY_MODEL_REGISTRY:
        raise ValueError(f"No registered families named {family}.")

    if family not in FAMILY_CKPT_FROM_HF:
        raise ValueError(
            "No converter from hugging face format registered for "
            f"model family {family}."
        )

    return FAMILY_CKPT_FROM_HF[family]


def resolve_model_name(model_name: str) -> str:
    _ensure_models_registered()

    if model_name in MODEL_REGISTRY:
        return model_name

    if "/" in model_name:
        family, short_name = model_name.split("/", 1)

        if (
            short_name in MODEL_REGISTRY
            and MODEL_FAMILY_REGISTRY.get(short_name) == family
        ):
            return short_name

    available = ", ".join(sorted(all_registered_repo_ids()))

    raise ValueError(
        f"Unsupported model_name '{model_name}'. Available models: [{available}]"
    )


def get_model_family(model: str) -> str:
    model_name = resolve_model_name(model)
    return MODEL_FAMILY_REGISTRY[model_name]


def get_model_repo_id(model: str) -> str:
    model_name = resolve_model_name(model)
    family = MODEL_FAMILY_REGISTRY[model_name]
    return f"{family}/{model_name}"


def get_models_from_family(family: str, newest_first: bool = True) -> list[str]:
    _ensure_models_registered()

    models = list(FAMILY_MODEL_REGISTRY[family])
    if newest_first:
        return models[::-1]

    return models


def all_registered_models(newest_first: bool = True) -> list[str]:
    _ensure_models_registered()

    models = list(MODEL_REGISTRY.keys())
    if newest_first:
        return models[::-1]

    return models


def all_registered_repo_ids(newest_first: bool = True) -> list[str]:
    return [get_model_repo_id(name) for name in all_registered_models(newest_first)]


def model_is_registered(model_name: str) -> bool:
    _ensure_models_registered()

    if model_name in MODEL_REGISTRY:
        return True

    if "/" not in model_name:
        return False

    family, short_name = model_name.split("/", 1)
    return (
        short_name in MODEL_REGISTRY and MODEL_FAMILY_REGISTRY.get(short_name) == family
    )
