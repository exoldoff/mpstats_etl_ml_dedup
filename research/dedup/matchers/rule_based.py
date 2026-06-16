from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from research.dedup.fusion import FusionConfig, decide_label, get_pair_value
from research.dedup.normalization import normalize_title, title_similarity

from .base import MatcherStatus, PairMatcher

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - exercised only in lean environments
    fuzz = None


@dataclass(frozen=True)
class RuleBasedConfig:
    fusion: FusionConfig = FusionConfig()
    uncertain_fallback_label: str = "different_product"


class RuleBasedMatcher(PairMatcher):
    """Baseline A: fuzzy title similarity plus deterministic binary fusion."""

    name = "rule_based_fuzzy"

    def __init__(self, config: RuleBasedConfig | None = None) -> None:
        self.config = config or RuleBasedConfig()

    def status(self) -> MatcherStatus:
        if fuzz is None:
            return MatcherStatus(
                available=True,
                message="rapidfuzz is not installed; falling back to difflib SequenceMatcher",
            )
        return MatcherStatus(available=True, message="rapidfuzz available")

    def score(self, pair: Any) -> float:
        title_a = get_pair_value(pair, "title_a", "name_a", "sku_name_a", default="")
        title_b = get_pair_value(pair, "title_b", "name_b", "sku_name_b", default="")
        norm_a = normalize_title(title_a)
        norm_b = normalize_title(title_b)
        if not norm_a or not norm_b:
            return 0.0
        if fuzz is not None:
            fuzzy_score = fuzz.WRatio(norm_a, norm_b) / 100.0
        else:
            fuzzy_score = SequenceMatcher(None, norm_a, norm_b).ratio()
        token_score = title_similarity(title_a, title_b)
        return round(max(fuzzy_score, token_score), 6)

    def predict_label(self, pair: Any) -> str:
        label = decide_label(pair, self.score(pair), self.config.fusion)
        if label == "uncertain":
            return self.config.uncertain_fallback_label
        return label
