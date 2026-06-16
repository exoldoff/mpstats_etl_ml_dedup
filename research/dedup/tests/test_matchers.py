from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from research.dedup import (
    BiEncoderMatcher,
    CrossEncoderMatcher,
    FusionConfig,
    JinaRerankerMatcher,
    RuleBasedMatcher,
    decide_label,
)
from research.dedup.matchers.bi_encoder import BiEncoderConfig
from research.dedup.matchers.cross_encoder import CrossEncoderConfig
from research.dedup.matchers.jina_reranker import JinaRerankerConfig


def _pair(**overrides: object) -> dict[str, object]:
    pair = {
        "title_a": "Соус томатный острый 200 г",
        "title_b": "Соус томатный острый 200 г",
        "brand_a": "Kinto",
        "brand_b": "kinto",
        "unit_amount_a": 0.2,
        "unit_amount_b": 0.2,
        "total_amount_a": 0.2,
        "total_amount_b": 0.2,
        "multipack_count_a": 1,
        "multipack_count_b": 1,
    }
    pair.update(overrides)
    return pair


def test_fusion_treats_same_product_different_multipack_as_positive_class() -> None:
    label = decide_label(
        _pair(total_amount_b=0.6, multipack_count_b=3),
        rerank_score=0.95,
        config=FusionConfig(),
    )

    assert label == "exact_duplicate"


def test_fusion_does_not_block_when_brand_is_missing() -> None:
    label = decide_label(_pair(brand_a="", brand_b="Kinto"), rerank_score=0.95)

    assert label == "exact_duplicate"


def test_rule_based_matcher_predicts_exact_duplicate_and_brand_mismatch() -> None:
    matcher = RuleBasedMatcher()

    assert matcher.predict_label(_pair()) == "exact_duplicate"
    assert matcher.predict_label(_pair(brand_b="Other Brand")) == "different_product"


def test_rule_based_matcher_collapses_pack_variant_into_exact_duplicate() -> None:
    matcher = RuleBasedMatcher()

    assert matcher.predict_label(_pair(title_b="Соус томатный острый 3 x 200 г", total_amount_b=0.6, multipack_count_b=3)) == (
        "exact_duplicate"
    )


def test_bi_encoder_gracefully_skips_when_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "sentence_transformers" else object())
    matcher = BiEncoderMatcher()

    assert matcher.status().available is False
    assert math.isnan(matcher.score(_pair()))
    assert matcher.predict_label(_pair()) == "different_product"


def test_bi_encoder_polza_backend_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLZA_API_KEY", raising=False)
    monkeypatch.delenv("POLZA_AI_API_KEY", raising=False)
    matcher = BiEncoderMatcher(BiEncoderConfig(model_name="polza_embedding_3_small"))

    status = matcher.status()

    assert status.available is False
    assert "Polza.ai API key is missing" in status.message


def test_bi_encoder_polza_backend_status_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLZA_API_KEY", "secret")
    matcher = BiEncoderMatcher(BiEncoderConfig(model_name="polza_embedding_3_small"))

    status = matcher.status()

    assert status.available is True
    assert "openai/text-embedding-3-small" in status.message


def test_bi_encoder_scores_with_injected_model() -> None:
    class FakeModel:
        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            vectors = []
            for text in texts:
                if "острый" in text:
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return np.asarray(vectors)

    matcher = BiEncoderMatcher(
        BiEncoderConfig(model_name="fake-model", text_prefix=""),
        model_factory=lambda _: FakeModel(),
    )

    assert matcher.score(_pair()) == 1.0
    assert matcher.score(_pair(title_b="Соус сливочный 200 г")) == 0.0


def test_cross_encoder_gracefully_skips_when_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "sentence_transformers" else object())
    matcher = CrossEncoderMatcher()

    assert matcher.status().available is False
    assert math.isnan(matcher.score(_pair()))
    assert matcher.predict_label(_pair()) == "different_product"


def test_cross_encoder_scores_with_injected_model() -> None:
    class FakeModel:
        def predict(self, pairs: list[tuple[str, str]], **_: object) -> np.ndarray:
            scores = []
            for left, right in pairs:
                scores.append(0.95 if "острый" in left and "острый" in right else 0.1)
            return np.asarray(scores)

    matcher = CrossEncoderMatcher(
        CrossEncoderConfig(model_name="fake-cross-encoder"),
        model_factory=lambda _: FakeModel(),
    )

    assert matcher.score(_pair()) == 0.95
    assert matcher.score(_pair(title_b="Соус сливочный 200 г")) == 0.1


def test_cross_encoder_uses_configured_method_name() -> None:
    matcher = CrossEncoderMatcher(CrossEncoderConfig(model_name="fake-cross-encoder", method_name="reranker_qwen3_4b"))

    assert matcher.name == "reranker_qwen3_4b"


def test_jina_reranker_gracefully_skips_when_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "transformers" else object())
    matcher = JinaRerankerMatcher()

    assert matcher.status().available is False
    assert math.isnan(matcher.score(_pair()))
    assert matcher.predict_label(_pair()) == "different_product"


def test_jina_reranker_scores_with_injected_model() -> None:
    class FakeModel:
        def rerank(self, query: str, documents: list[str], **_: object) -> list[dict[str, object]]:
            return [
                {
                    "index": idx,
                    "relevance_score": 0.9 if "острый" in query and "острый" in document else 0.05,
                    "document": document,
                }
                for idx, document in enumerate(documents)
            ]

    matcher = JinaRerankerMatcher(
        JinaRerankerConfig(model_name="fake-jina-reranker", documents_per_query=1),
        model_factory=lambda _: FakeModel(),
    )

    assert matcher.score(_pair()) == 0.9
    assert matcher.score(_pair(title_b="Соус сливочный 200 г")) == 0.05
