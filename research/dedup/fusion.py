from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd

from .clustering import same_pack_signature_mask
from .normalization import normalize_brand


@dataclass(frozen=True)
class FusionConfig:
    threshold_high: float = 0.82
    threshold_low: float = 0.55


@dataclass(frozen=True)
class FusionRun:
    method: str
    threshold_strategy: str
    threshold_same: float
    selection_split: str = "dev"


DEFAULT_PREFERRED_STRATEGIES: tuple[str, ...] = (
    "threshold_weighted_cost",
    "threshold_cost_sensitive",
)


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


def _first_present_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"Missing any of required columns: {list(candidates)}")


def _to_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric threshold_same, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Expected finite threshold_same, got {value!r}")
    return number


def _fill_numeric_column(frame: pd.DataFrame, column: str, default: float) -> None:
    if column not in frame.columns:
        frame[column] = default
    frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _select_default_strategy(
    candidates: pd.DataFrame,
    strategy_col: str,
    preferred_strategy: str | None,
) -> tuple[pd.DataFrame, str | None]:
    preferred_strategies = (
        (preferred_strategy,) if preferred_strategy is not None else DEFAULT_PREFERRED_STRATEGIES
    )
    for strategy in preferred_strategies:
        preferred = candidates[candidates[strategy_col].astype(str).eq(strategy)].copy()
        if not preferred.empty:
            return preferred, strategy
    return candidates, None


def _rank_fusion_candidates(
    candidates: pd.DataFrame,
    *,
    strategy_col: str,
    threshold_col: str,
) -> pd.DataFrame:
    ranked = candidates.copy()
    for column, default in [
        ("cost", math.inf),
        ("false_merge_count", math.inf),
        ("false_split_count", math.inf),
        ("f1", -math.inf),
        (threshold_col, math.inf),
    ]:
        _fill_numeric_column(ranked, column, default)

    return ranked.sort_values(
        ["cost", "false_merge_count", "false_split_count", "f1", threshold_col],
        ascending=[True, True, True, False, True],
    )


def select_fusion_run(
    summary: pd.DataFrame,
    *,
    method: str | None = None,
    threshold_strategy: str | None = None,
    split: str = "dev",
    preferred_strategy: str | None = None,
) -> FusionRun:
    """Select one method/threshold strategy using only the requested split.

    By default the downstream fusion step uses sales-volume weighted cost when
    it is available. If the benchmark did not produce weighted rows, selection
    falls back to the unweighted cost-sensitive strategy.
    """
    if summary.empty:
        raise ValueError("Cannot select fusion run from empty threshold summary")

    method_col = _first_present_column(summary, ("method",))
    split_col = _first_present_column(summary, ("split", "eval_split"))
    strategy_col = _first_present_column(summary, ("threshold_strategy",))
    threshold_col = _first_present_column(summary, ("threshold_same",))

    candidates = summary[summary[split_col].astype(str).eq(split)].copy()
    if candidates.empty:
        raise ValueError(f"No threshold summary rows for split={split!r}")

    if method:
        candidates = candidates[candidates[method_col].astype(str).eq(method)].copy()
        if candidates.empty:
            raise ValueError(f"No threshold summary rows for method={method!r} on split={split!r}")

    if threshold_strategy:
        candidates = candidates[candidates[strategy_col].astype(str).eq(threshold_strategy)].copy()
        if candidates.empty:
            raise ValueError(
                f"No threshold summary rows for strategy={threshold_strategy!r} on split={split!r}"
            )
    else:
        candidates, _ = _select_default_strategy(candidates, strategy_col, preferred_strategy)

    ranked = _rank_fusion_candidates(candidates, strategy_col=strategy_col, threshold_col=threshold_col)
    selected = ranked.iloc[0]
    return FusionRun(
        method=str(selected[method_col]),
        threshold_strategy=str(selected[strategy_col]),
        threshold_same=_to_float(selected[threshold_col]),
        selection_split=split,
    )


def prepare_fusion_pair_edges(
    predictions: pd.DataFrame,
    run: FusionRun,
    *,
    eval_split: str | None = None,
) -> pd.DataFrame:
    """Filter threshold predictions and add family/pack edge flags for graph steps."""
    if predictions.empty:
        raise ValueError("Cannot prepare fusion edges from empty predictions")

    required = {"method", "threshold_strategy", "predicted_binary"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing columns for fusion pair edges: {missing}")

    output = predictions[
        predictions["method"].astype(str).eq(run.method)
        & predictions["threshold_strategy"].astype(str).eq(run.threshold_strategy)
    ].copy()
    if eval_split is not None:
        split_col = _first_present_column(output, ("split", "eval_split"))
        output = output[output[split_col].astype(str).eq(eval_split)].copy()
    if output.empty:
        split_note = f" and eval_split={eval_split!r}" if eval_split is not None else ""
        raise ValueError(
            f"No predictions for method={run.method!r}, strategy={run.threshold_strategy!r}{split_note}"
        )

    predicted_binary = pd.to_numeric(output["predicted_binary"], errors="coerce").fillna(0).astype(int)
    output["fusion_method"] = run.method
    output["fusion_threshold_strategy"] = run.threshold_strategy
    output["fusion_threshold_same"] = run.threshold_same
    output["fusion_selection_split"] = run.selection_split
    output["fusion_family_edge"] = predicted_binary.eq(1)
    output["same_pack_signature"] = same_pack_signature_mask(output)
    output["fusion_pack_edge"] = output["fusion_family_edge"] & output["same_pack_signature"]
    output["predicted_family_label"] = output["fusion_family_edge"].map(
        {True: "exact_duplicate", False: "different_product"}
    )
    output["predicted_pack_label"] = output["fusion_pack_edge"].map(
        {True: "exact_duplicate", False: "different_product"}
    )

    if "same_base_product" in output.columns:
        target = pd.to_numeric(output["same_base_product"], errors="coerce").fillna(0).astype(int)
        output["true_family_edge"] = target.eq(1)
        output["true_pack_edge"] = output["true_family_edge"] & output["same_pack_signature"]
        output["true_family_label"] = output["true_family_edge"].map(
            {True: "exact_duplicate", False: "different_product"}
        )
        output["true_pack_label"] = output["true_pack_edge"].map(
            {True: "exact_duplicate", False: "different_product"}
        )

    return output.reset_index(drop=True)
