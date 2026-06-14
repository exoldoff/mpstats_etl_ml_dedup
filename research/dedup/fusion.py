from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .normalization import normalize_brand


@dataclass(frozen=True)
class FusionConfig:
    threshold_high: float = 0.82
    threshold_low: float = 0.55
    unit_abs_tolerance: float = 0.02
    unit_rel_tolerance: float = 0.05
    pack_abs_tolerance: float = 0.25


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


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if bool(value != value):
            return None
    except TypeError:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _numbers_close(left: float | None, right: float | None, *, abs_tol: float, rel_tol: float) -> bool:
    if left is None or right is None:
        return False
    return abs(left - right) <= max(abs_tol, rel_tol * max(abs(left), abs(right)))


def _brand(pair: Any, side: str) -> str:
    return normalize_brand(
        get_pair_value(pair, f"canonical_brand_{side}", f"brand_{side}", f"brand_norm_{side}", default="")
    )


def _unit_amount(pair: Any, side: str) -> float | None:
    return _to_number(get_pair_value(pair, f"unit_amount_{side}", f"unit_weight_{side}", default=None))


def _total_amount(pair: Any, side: str) -> float | None:
    return _to_number(get_pair_value(pair, f"total_amount_{side}", f"total_weight_{side}", default=None))


def _multipack_count(pair: Any, side: str) -> float | None:
    explicit = _to_number(get_pair_value(pair, f"multipack_count_{side}", f"pack_count_{side}", default=None))
    if explicit is not None:
        return explicit
    unit = _unit_amount(pair, side)
    total = _total_amount(pair, side)
    if unit is None or total is None:
        return None
    return max(1.0, float(round(total / unit)))


def decide_label(
    pair: Any,
    rerank_score: float,
    config: FusionConfig | None = None,
) -> str:
    """Apply research fusion from ARCHITECTURE.md section 5."""
    cfg = config or FusionConfig()
    brand_a = _brand(pair, "a")
    brand_b = _brand(pair, "b")
    if brand_a and brand_b and brand_a != brand_b:
        return "different_product"

    unit_a = _unit_amount(pair, "a")
    unit_b = _unit_amount(pair, "b")
    multipack_a = _multipack_count(pair, "a")
    multipack_b = _multipack_count(pair, "b")
    unit_same = _numbers_close(
        unit_a,
        unit_b,
        abs_tol=cfg.unit_abs_tolerance,
        rel_tol=cfg.unit_rel_tolerance,
    )
    multipack_same = _numbers_close(
        multipack_a,
        multipack_b,
        abs_tol=cfg.pack_abs_tolerance,
        rel_tol=0.0,
    )
    multipack_known = multipack_a is not None and multipack_b is not None

    if unit_same and multipack_known and not multipack_same:
        return "same_product_different_pack"

    if unit_same and multipack_same:
        return "exact_duplicate" if rerank_score >= cfg.threshold_high else "different_product"

    if rerank_score < cfg.threshold_low:
        return "different_product"
    if rerank_score >= cfg.threshold_high:
        return "exact_duplicate"
    return "uncertain"
