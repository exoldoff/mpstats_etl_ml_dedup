from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from research.dedup.fusion import FusionConfig, decide_label, get_pair_value
from research.dedup.model_registry import ModelManager, model_text_prefix

from .base import MatcherStatus, PairMatcher


ModelFactory = Callable[[str], Any]


@dataclass(frozen=True)
class BiEncoderConfig:
    model_name: str = "intfloat/multilingual-e5-base"
    batch_size: int = 32
    text_prefix: str | None = None
    cache_dir: str | Path | None = None
    local_files_only: bool | None = None
    fusion: FusionConfig = FusionConfig(threshold_high=0.86, threshold_low=0.58)
    uncertain_fallback_label: str = "different_product"


class BiEncoderMatcher(PairMatcher):
    """Baseline B: zero-shot multilingual bi-encoder cosine similarity."""

    name = "bi_encoder_zero_shot"

    def __init__(
        self,
        config: BiEncoderConfig | None = None,
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.config = config or BiEncoderConfig()
        self._model_factory = model_factory
        self._model: Any | None = None
        self._load_error: str | None = None

    def status(self) -> MatcherStatus:
        if self._model is not None:
            return MatcherStatus(available=True, message=f"loaded {self.config.model_name}")
        if self._load_error is not None:
            return MatcherStatus(available=False, message=self._load_error)
        if self._model_factory is not None:
            return MatcherStatus(available=True, message="custom model factory configured")
        if importlib.util.find_spec("sentence_transformers") is None:
            return MatcherStatus(
                available=False,
                message="sentence-transformers is not installed; bi-encoder baseline skipped",
            )
        return MatcherStatus(available=True, message="sentence-transformers available; model loads lazily")

    def _load_model(self) -> Any | None:
        status = self.status()
        if not status.available:
            self._load_error = status.message
            return None
        if self._model is not None:
            return self._model
        try:
            if self._model_factory is not None:
                self._model = self._model_factory(self.config.model_name)
            else:
                manager = ModelManager(
                    cache_dir=self.config.cache_dir,
                    local_files_only=self.config.local_files_only,
                )
                self._model = manager.load_sentence_transformer(self.config.model_name)
        except Exception as exc:  # pragma: no cover - depends on local model/network state
            self._load_error = f"failed to load {self.config.model_name}: {exc}"
            return None
        return self._model

    def _format_text(self, pair: Any, side: str) -> str:
        title = get_pair_value(pair, f"title_{side}", f"name_{side}", f"sku_name_{side}", default="")
        brand = get_pair_value(pair, f"brand_{side}", f"canonical_brand_{side}", default="")
        text = " ".join(part.strip() for part in [str(brand or ""), str(title or "")] if part and str(part).strip())
        prefix = self.config.text_prefix
        if prefix is None:
            prefix = model_text_prefix(self.config.model_name)
        return f"{prefix or ''}{text}".strip()

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def _encode(self, texts: Sequence[str]) -> np.ndarray | None:
        model = self._load_model()
        if model is None:
            return None
        try:
            embeddings = model.encode(
                list(texts),
                batch_size=self.config.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except TypeError:
            embeddings = model.encode(list(texts))
            embeddings = self._normalize(np.asarray(embeddings, dtype=float))
        return np.asarray(embeddings, dtype=float)

    def score(self, pair: Any) -> float:
        scores = self.score_batch([pair])
        return scores[0] if scores else math.nan

    def score_batch(self, pairs: Sequence[Any]) -> list[float]:
        if not pairs:
            return []
        texts: list[str] = []
        for pair in pairs:
            texts.extend([self._format_text(pair, "a"), self._format_text(pair, "b")])
        embeddings = self._encode(texts)
        if embeddings is None:
            return [math.nan for _ in pairs]
        if embeddings.ndim != 2 or len(embeddings) != len(texts):
            self._load_error = "model returned embeddings with unexpected shape"
            return [math.nan for _ in pairs]
        scores: list[float] = []
        for idx in range(0, len(embeddings), 2):
            similarity = float(np.dot(embeddings[idx], embeddings[idx + 1]))
            scores.append(round(max(-1.0, min(1.0, similarity)), 6))
        return scores

    def predict_label(self, pair: Any) -> str:
        score = self.score(pair)
        if math.isnan(score):
            return self.config.uncertain_fallback_label
        label = decide_label(pair, score, self.config.fusion)
        if label == "uncertain":
            return self.config.uncertain_fallback_label
        return label
