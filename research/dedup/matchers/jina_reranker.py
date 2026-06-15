from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from research.dedup.fusion import FusionConfig, decide_label, get_pair_value
from research.dedup.model_registry import ModelManager

from .base import MatcherStatus, PairMatcher


ModelFactory = Callable[[str], Any]


@dataclass(frozen=True)
class JinaRerankerConfig:
    model_name: str = "jinaai/jina-reranker-v3"
    method_name: str = "reranker_jina_v3"
    documents_per_query: int = 16
    trust_remote_code: bool = True
    cache_dir: str | Path | None = None
    local_files_only: bool | None = None
    fusion: FusionConfig = FusionConfig(threshold_high=0.5, threshold_low=0.2)
    uncertain_fallback_label: str = "different_product"


class JinaRerankerMatcher(PairMatcher):
    """Research wrapper for Jina rerankers that expose ``model.rerank``."""

    def __init__(
        self,
        config: JinaRerankerConfig | None = None,
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.config = config or JinaRerankerConfig()
        self._model_factory = model_factory
        self._model: Any | None = None
        self._load_error: str | None = None

    @property
    def name(self) -> str:
        return self.config.method_name

    def status(self) -> MatcherStatus:
        if self._model is not None:
            return MatcherStatus(available=True, message=f"loaded {self.config.model_name}")
        if self._load_error is not None:
            return MatcherStatus(available=False, message=self._load_error)
        if self._model_factory is not None:
            return MatcherStatus(available=True, message="custom model factory configured")
        if importlib.util.find_spec("transformers") is None:
            return MatcherStatus(
                available=False,
                message="transformers is not installed; Jina reranker benchmark skipped",
            )
        return MatcherStatus(available=True, message="transformers available; Jina reranker loads lazily")

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
                self._model = manager.load_transformers_auto_model(
                    self.config.model_name,
                    dtype="auto",
                    trust_remote_code=self.config.trust_remote_code,
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

        grouped_documents: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for idx, pair in enumerate(pairs):
            grouped_documents[self._format_text(pair, "a")].append((idx, self._format_text(pair, "b")))

        scores = [math.nan for _ in pairs]
        chunk_size = max(1, int(self.config.documents_per_query))
        try:
            for query, indexed_documents in grouped_documents.items():
                for offset in range(0, len(indexed_documents), chunk_size):
                    chunk = indexed_documents[offset : offset + chunk_size]
                    documents = [document for _, document in chunk]
                    try:
                        results = model.rerank(query, documents, top_n=None)
                    except TypeError:
                        results = model.rerank(query, documents)
                    for result in results:
                        local_index = int(result.get("index", 0))
                        if local_index >= len(chunk):
                            continue
                        pair_index, _ = chunk[local_index]
                        raw_score = result.get("relevance_score", result.get("score", math.nan))
                        scores[pair_index] = round(float(raw_score), 6)
        except Exception as exc:  # pragma: no cover - depends on local model/runtime state
            self._load_error = f"failed to score {self.config.model_name}: {exc}"
            return [math.nan for _ in pairs]
        return scores

    def predict_label(self, pair: Any) -> str:
        score = self.score(pair)
        if math.isnan(score):
            return self.config.uncertain_fallback_label
        label = decide_label(pair, score, self.config.fusion)
        if label == "uncertain":
            return self.config.uncertain_fallback_label
        return label
