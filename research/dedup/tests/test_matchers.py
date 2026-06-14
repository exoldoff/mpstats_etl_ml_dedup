from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from research.dedup import BiEncoderMatcher, FusionConfig, RuleBasedMatcher, decide_label
from research.dedup.matchers.bi_encoder import BiEncoderConfig


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


def test_fusion_marks_same_unit_different_multipack_as_pack_variant() -> None:
    label = decide_label(
        _pair(total_amount_b=0.6, multipack_count_b=3),
        rerank_score=0.95,
        config=FusionConfig(),
    )

    assert label == "same_product_different_pack"


def test_fusion_does_not_block_when_brand_is_missing() -> None:
    label = decide_label(_pair(brand_a="", brand_b="Kinto"), rerank_score=0.95)

    assert label == "exact_duplicate"


def test_rule_based_matcher_predicts_exact_duplicate_and_brand_mismatch() -> None:
    matcher = RuleBasedMatcher()

    assert matcher.predict_label(_pair()) == "exact_duplicate"
    assert matcher.predict_label(_pair(brand_b="Other Brand")) == "different_product"


def test_rule_based_matcher_predicts_pack_variant() -> None:
    matcher = RuleBasedMatcher()

    assert matcher.predict_label(_pair(title_b="Соус томатный острый 3 x 200 г", total_amount_b=0.6, multipack_count_b=3)) == (
        "same_product_different_pack"
    )


def test_bi_encoder_gracefully_skips_when_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "sentence_transformers" else object())
    matcher = BiEncoderMatcher()

    assert matcher.status().available is False
    assert math.isnan(matcher.score(_pair()))
    assert matcher.predict_label(_pair()) == "different_product"


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
