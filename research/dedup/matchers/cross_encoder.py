from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from research.dedup.fusion import FusionConfig, decide_label, get_pair_value
from research.dedup.model_registry import ModelManager

from .base import MatcherStatus, PairMatcher


ModelFactory = Callable[[str], Any]


@dataclass(frozen=True)
class CrossEncoderConfig:
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    method_name: str = "cross_encoder_zero_shot"
    batch_size: int = 16
    device: str | None = None
    trust_remote_code: bool = False
    prompts: dict[str, str] | None = None
    default_prompt_name: str | None = None
    cache_dir: str | Path | None = None
    local_files_only: bool | None = None
    fusion: FusionConfig = FusionConfig(threshold_high=8.0, threshold_low=4.0)
    uncertain_fallback_label: str = "different_product"


class CrossEncoderMatcher(PairMatcher):
    """Baseline D0: ready-made cross-encoder reranker, without fine-tuning."""

    @property
    def name(self) -> str:
        return self.config.method_name

    def __init__(
        self,
        config: CrossEncoderConfig | None = None,
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.config = config or CrossEncoderConfig()
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
                message="sentence-transformers is not installed; cross-encoder baseline skipped",
            )
        return MatcherStatus(available=True, message="sentence-transformers available; cross-encoder loads lazily")

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
                self._model = manager.load_cross_encoder(
                    self.config.model_name,
                    device=self.config.device,
                    trust_remote_code=self.config.trust_remote_code or None,
                    prompts=self.config.prompts,
                    default_prompt_name=self.config.default_prompt_name,
                )
        except Exception as exc:  # pragma: no cover - depends on local model/network state
            self._load_error = f"failed to load {self.config.model_name}: {exc}"
            return None
        return self._model

    def _format_text(self, pair: Any, side: str) -> str:
        title = get_pair_value(pair, f"title_{side}", f"name_{side}", f"sku_name_{side}", default="")
        brand = get_pair_value(pair, f"brand_{side}", f"canonical_brand_{side}", default="")
        unit = get_pair_value(pair, f"unit_amount_{side}", f"unit_weight_{side}", default="")
        total = get_pair_value(pair, f"total_amount_{side}", f"total_weight_{side}", default="")
        multipack = get_pair_value(pair, f"multipack_count_{side}", f"pack_count_{side}", default="")
        parts = [
            f"бренд: {brand}" if brand else "",
            f"название: {title}" if title else "",
            f"вес единицы: {unit}" if unit not in (None, "") else "",
            f"общий вес: {total}" if total not in (None, "") else "",
            f"штук в наборе: {multipack}" if multipack not in (None, "") else "",
        ]
        return " | ".join(part for part in parts if str(part).strip())

    def score(self, pair: Any) -> float:
        scores = self.score_batch([pair])
        return scores[0] if scores else math.nan

    def score_batch(self, pairs: Sequence[Any]) -> list[float]:
        if not pairs:
            return []
        model = self._load_model()
        if model is None:
            return [math.nan for _ in pairs]
        text_pairs = [(self._format_text(pair, "a"), self._format_text(pair, "b")) for pair in pairs]
        try:
            raw_scores = model.predict(text_pairs, batch_size=self.config.batch_size)
        except TypeError:
            raw_scores = model.predict(text_pairs)
        scores = np.asarray(raw_scores, dtype=float).reshape(-1)
        if len(scores) != len(pairs):
            self._load_error = "model returned scores with unexpected shape"
            return [math.nan for _ in pairs]
        return [round(float(score), 6) for score in scores]

    def predict_label(self, pair: Any) -> str:
        score = self.score(pair)
        if math.isnan(score):
            return self.config.uncertain_fallback_label
        label = decide_label(pair, score, self.config.fusion)
        if label == "uncertain":
            return self.config.uncertain_fallback_label
        return label
