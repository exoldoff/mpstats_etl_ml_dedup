from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable


SENTENCE_TRANSFORMER_BACKEND = "sentence_transformer"
CROSS_ENCODER_BACKEND = "cross_encoder"
TRANSFORMERS_AUTO_MODEL_BACKEND = "transformers_auto_model"

DEFAULT_MODEL_CACHE_DIR = Path(__file__).resolve().parent / "models"
MODEL_CACHE_DIR_ENV = "DEDUP_MODEL_CACHE_DIR"
MODEL_LOCAL_ONLY_ENV = "DEDUP_MODEL_LOCAL_ONLY"

SKU_RERANKER_INSTRUCTION = (
    "Decide whether two ecommerce sauce products are the same SKU. "
    "Pay attention to brand, flavor or purpose, unit weight, total weight, and pack count."
)


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_name: str
    backend: str
    method_name: str | None = None
    batch_size: int | None = None
    documents_per_query: int | None = None
    text_prefix: str | None = None
    trust_remote_code: bool = False
    prompts: dict[str, str] | None = None
    default_prompt_name: str | None = None
    fusion_threshold_high: float | None = None
    fusion_threshold_low: float | None = None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "embedding_e5_small": ModelSpec(
        alias="embedding_e5_small",
        model_name="intfloat/multilingual-e5-small",
        backend=SENTENCE_TRANSFORMER_BACKEND,
        batch_size=64,
        text_prefix="passage: ",
    ),
    "bi_encoder_e5_small": ModelSpec(
        alias="bi_encoder_e5_small",
        model_name="intfloat/multilingual-e5-small",
        backend=SENTENCE_TRANSFORMER_BACKEND,
        method_name="bi_encoder_zero_shot",
        batch_size=32,
        text_prefix="passage: ",
        fusion_threshold_high=0.86,
        fusion_threshold_low=0.58,
    ),
    "cross_encoder_mmarco": ModelSpec(
        alias="cross_encoder_mmarco",
        model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        backend=CROSS_ENCODER_BACKEND,
        method_name="cross_encoder_zero_shot",
        batch_size=16,
        fusion_threshold_high=8.0,
        fusion_threshold_low=4.0,
    ),
    "reranker_qwen3_4b": ModelSpec(
        alias="reranker_qwen3_4b",
        model_name="Qwen/Qwen3-Reranker-4B",
        backend=CROSS_ENCODER_BACKEND,
        method_name="reranker_qwen3_4b",
        batch_size=1,
        prompts={"sku_match": SKU_RERANKER_INSTRUCTION},
        default_prompt_name="sku_match",
        fusion_threshold_high=0.0,
        fusion_threshold_low=-5.0,
    ),
    "reranker_jina_v3": ModelSpec(
        alias="reranker_jina_v3",
        model_name="jinaai/jina-reranker-v3",
        backend=TRANSFORMERS_AUTO_MODEL_BACKEND,
        method_name="reranker_jina_v3",
        documents_per_query=8,
        trust_remote_code=True,
        fusion_threshold_high=0.5,
        fusion_threshold_low=0.2,
    ),
}

MODEL_ALIASES: dict[str, str] = {
    "e5_small": "embedding_e5_small",
    "multilingual_e5_small": "embedding_e5_small",
    "qwen3_4b": "reranker_qwen3_4b",
    "jina_v3": "reranker_jina_v3",
    "mmarco": "cross_encoder_mmarco",
}

_MODEL_POOL: dict[tuple[Any, ...], Any] = {}


def _normalise_key(value: str) -> str:
    return value.strip().lower()


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_model_cache_dir(cache_dir: str | Path | None = None) -> Path:
    raw_path = cache_dir or os.environ.get(MODEL_CACHE_DIR_ENV) or DEFAULT_MODEL_CACHE_DIR
    return Path(raw_path).expanduser().resolve()


def get_model_local_only(local_files_only: bool | None = None) -> bool:
    if local_files_only is not None:
        return bool(local_files_only)
    return _env_flag(MODEL_LOCAL_ONLY_ENV)


def resolve_model_spec(alias_or_name: str, *, backend: str | None = None) -> ModelSpec:
    raw_value = alias_or_name.strip()
    key = _normalise_key(raw_value)
    canonical_key = MODEL_ALIASES.get(key, key)
    spec = MODEL_REGISTRY.get(canonical_key)
    if spec is None:
        if backend is None:
            raise KeyError(f"unknown model alias {alias_or_name!r}; pass a backend for custom model ids")
        return ModelSpec(alias=raw_value, model_name=raw_value, backend=backend)
    if backend is not None and spec.backend != backend:
        raise ValueError(f"model alias {alias_or_name!r} uses backend {spec.backend!r}, expected {backend!r}")
    return spec


def list_model_specs() -> list[ModelSpec]:
    return list(MODEL_REGISTRY.values())


def model_text_prefix(alias_or_name: str, *, backend: str = SENTENCE_TRANSFORMER_BACKEND) -> str | None:
    spec = resolve_model_spec(alias_or_name, backend=backend)
    if spec.text_prefix is not None:
        return spec.text_prefix
    if "e5" in spec.model_name.lower():
        return "passage: "
    return None


def _freeze_for_key(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_for_key(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_key(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def _looks_like_unsupported_kwarg(exc: TypeError) -> bool:
    message = str(exc)
    return "unexpected keyword" in message or "got an unexpected" in message


def clear_model_pool() -> None:
    _MODEL_POOL.clear()


def model_pool_size() -> int:
    return len(_MODEL_POOL)


class ModelManager:
    """Central loader for research models and their local cache/pool."""

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        local_files_only: bool | None = None,
        use_pool: bool = True,
    ) -> None:
        self._cache_dir = cache_dir
        self._local_files_only = local_files_only
        self.use_pool = use_pool

    @property
    def cache_dir(self) -> Path:
        return get_model_cache_dir(self._cache_dir)

    @property
    def local_files_only(self) -> bool:
        return get_model_local_only(self._local_files_only)

    def resolve(self, alias_or_name: str, *, backend: str | None = None) -> ModelSpec:
        return resolve_model_spec(alias_or_name, backend=backend)

    def describe_pool(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for spec in list_model_specs():
            rows.append(
                {
                    "alias": spec.alias,
                    "backend": spec.backend,
                    "model_name": spec.model_name,
                    "method_name": spec.method_name or "",
                    "batch_size": spec.batch_size or "",
                    "documents_per_query": spec.documents_per_query or "",
                    "cache_dir": str(self.cache_dir),
                    "local_only": self.local_files_only,
                }
            )
        return rows

    def load_sentence_transformer(self, alias_or_name: str, **overrides: Any) -> Any:
        from sentence_transformers import SentenceTransformer

        spec = self.resolve(alias_or_name, backend=SENTENCE_TRANSFORMER_BACKEND)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "cache_folder": str(self.cache_dir),
            "local_files_only": self.local_files_only,
        }
        if spec.trust_remote_code:
            kwargs["trust_remote_code"] = True
        kwargs.update({key: value for key, value in overrides.items() if value is not None})
        return self._load_from_pool(
            backend=SENTENCE_TRANSFORMER_BACKEND,
            spec=spec,
            kwargs=kwargs,
            loader=lambda: SentenceTransformer(spec.model_name, **kwargs),
        )

    def load_cross_encoder(
        self,
        alias_or_name: str,
        *,
        trust_remote_code: bool | None = None,
        prompts: dict[str, str] | None = None,
        default_prompt_name: str | None = None,
        **overrides: Any,
    ) -> Any:
        from sentence_transformers import CrossEncoder

        spec = self.resolve(alias_or_name, backend=CROSS_ENCODER_BACKEND)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        common_kwargs: dict[str, Any] = {}
        effective_trust_remote_code = trust_remote_code if trust_remote_code is not None else spec.trust_remote_code
        if effective_trust_remote_code:
            common_kwargs["trust_remote_code"] = True
        effective_prompts = prompts if prompts is not None else spec.prompts
        if effective_prompts is not None:
            common_kwargs["prompts"] = effective_prompts
        effective_default_prompt_name = (
            default_prompt_name if default_prompt_name is not None else spec.default_prompt_name
        )
        if effective_default_prompt_name is not None:
            common_kwargs["default_prompt_name"] = effective_default_prompt_name
        common_kwargs.update({key: value for key, value in overrides.items() if value is not None})

        key_kwargs = {
            **common_kwargs,
            "cache_dir": str(self.cache_dir),
            "local_files_only": self.local_files_only,
        }
        return self._load_from_pool(
            backend=CROSS_ENCODER_BACKEND,
            spec=spec,
            kwargs=key_kwargs,
            loader=lambda: self._load_cross_encoder_with_fallbacks(CrossEncoder, spec.model_name, common_kwargs),
        )

    def load_transformers_auto_model(
        self,
        alias_or_name: str,
        *,
        trust_remote_code: bool | None = None,
        **overrides: Any,
    ) -> Any:
        from transformers import AutoModel

        spec = self.resolve(alias_or_name, backend=TRANSFORMERS_AUTO_MODEL_BACKEND)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "cache_dir": str(self.cache_dir),
            "local_files_only": self.local_files_only,
        }
        effective_trust_remote_code = trust_remote_code if trust_remote_code is not None else spec.trust_remote_code
        if effective_trust_remote_code:
            kwargs["trust_remote_code"] = True
        kwargs.update({key: value for key, value in overrides.items() if value is not None})
        model = self._load_from_pool(
            backend=TRANSFORMERS_AUTO_MODEL_BACKEND,
            spec=spec,
            kwargs=kwargs,
            loader=lambda: AutoModel.from_pretrained(spec.model_name, **kwargs),
        )
        if hasattr(model, "eval"):
            model.eval()
        return model

    def _load_cross_encoder_with_fallbacks(
        self,
        cross_encoder_cls: Callable[..., Any],
        model_name: str,
        common_kwargs: dict[str, Any],
    ) -> Any:
        cache_kwargs_variants = [
            {"cache_folder": str(self.cache_dir), "local_files_only": self.local_files_only},
            {"cache_dir": str(self.cache_dir), "local_files_only": self.local_files_only},
            {"local_files_only": self.local_files_only},
        ]
        if not self.local_files_only:
            cache_kwargs_variants.extend(
                [
                    {"cache_folder": str(self.cache_dir)},
                    {"cache_dir": str(self.cache_dir)},
                    {},
                ]
            )

        last_error: TypeError | None = None
        for cache_kwargs in cache_kwargs_variants:
            kwargs = {**cache_kwargs, **common_kwargs}
            try:
                return cross_encoder_cls(model_name, **kwargs)
            except TypeError as exc:
                if not _looks_like_unsupported_kwarg(exc):
                    raise
                last_error = exc
        if last_error is not None:
            raise last_error
        return cross_encoder_cls(model_name, **common_kwargs)

    def _load_from_pool(
        self,
        *,
        backend: str,
        spec: ModelSpec,
        kwargs: dict[str, Any],
        loader: Callable[[], Any],
    ) -> Any:
        key = (backend, spec.model_name, _freeze_for_key(kwargs))
        if self.use_pool and key in _MODEL_POOL:
            return _MODEL_POOL[key]
        model = loader()
        if self.use_pool:
            _MODEL_POOL[key] = model
        return model
