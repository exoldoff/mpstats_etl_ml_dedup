from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .normalization import normalize_brand


@dataclass(frozen=True)
class FusionConfig:
    threshold_high: float = 0.82
    threshold_low: float = 0.55


def get_pair_value(pair: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(pair, dict) and name in pair:
            return pair[name]
        getter = getattr(pair, "get", None)
        if getter is not None:
            try:
                value = getter(name, default)
            except TypeError:
                value = default
            if value is not default:
                return value
        if hasattr(pair, name):
            return getattr(pair, name)
    return default


def _brand(pair: Any, side: str) -> str:
    return normalize_brand(
        get_pair_value(pair, f"canonical_brand_{side}", f"brand_{side}", f"brand_norm_{side}", default="")
    )


def decide_label(
    pair: Any,
    rerank_score: float,
    config: FusionConfig | None = None,
) -> str:
    """Apply binary pairwise fusion: same base product vs different product."""
    cfg = config or FusionConfig()
    brand_a = _brand(pair, "a")
    brand_b = _brand(pair, "b")
    if brand_a and brand_b and brand_a != brand_b:
        return "different_product"

    if rerank_score < cfg.threshold_low:
        return "different_product"
    if rerank_score >= cfg.threshold_high:
        return "exact_duplicate"
    return "uncertain"
