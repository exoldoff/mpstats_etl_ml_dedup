from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


POSITIVE_LABELS = frozenset({"exact_duplicate", "same_product_different_pack"})
NEGATIVE_LABELS = frozenset({"different_product"})
FP_COST = 5
FN_COST = 1
UNIT_WEIGHT_FALLBACK_SOURCE = "unit_weight_fallback"
WEIGHTED_STRATEGIES = frozenset({"threshold_max_weighted_f1", "threshold_weighted_cost"})

SUMMARY_COLUMNS = [
    "method",
    "split",
    "threshold_strategy",
    "threshold_same",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "false_merge_count",
    "false_split_count",
    "false_merge_rate",
    "false_split_rate",
    "cost",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "weighted_false_merge_cost",
    "weighted_false_split_cost",
    "weighted_total_cost",
    "weight_source",
]

PREDICTION_CORE_COLUMNS = [
    "method",
    "split",
    "threshold_strategy",
    "threshold_same",
    "score",
    "same_base_product",
    "predicted_binary",
    "true_label",
    "predicted_label",
    "false_merge",
    "false_split",
    "pair_importance",
    "pair_weight",
    "weight_source",
]

PAIR_CONTEXT_COLUMNS = [
    "label",
    "title_a",
    "title_b",
    "brand_a",
    "brand_b",
    "raw_record_id_a",
    "raw_record_id_b",
    "marketplace_a",
    "marketplace_b",
    "sku_a",
    "sku_b",
    "unit_amount_a",
    "unit_amount_b",
    "total_amount_a",
    "total_amount_b",
    "multipack_count_a",
    "multipack_count_b",
    "benchmark_pair_key",
    "benchmark_source",
]


@dataclass(frozen=True)
class BinaryThresholdConfig:
    fp_cost: float = FP_COST
    fn_cost: float = FN_COST


@dataclass(frozen=True)
class PairWeightInfo:
    frame: pd.DataFrame
    weights_available: bool
    weight_source: str
    warning: str | None = None


def _safe_div(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def _coerce_bool_target(value: object) -> int | None:
    if value is None:
        return None
    try:
        if bool(value != value):
            return None
    except TypeError:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return int(value)
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "t", "yes", "y", "same", "same_base_product", "exact_duplicate"}:
        return 1
    if normalized in {"0", "false", "f", "no", "n", "different", "different_product"}:
        return 0
    return None


def same_base_product_from_labels(labels: pd.Series) -> pd.Series:
    normalized = labels.fillna("").astype(str).str.strip()
    target = pd.Series(pd.NA, index=labels.index, dtype="Int64")
    target[normalized.isin(POSITIVE_LABELS)] = 1
    target[normalized.isin(NEGATIVE_LABELS)] = 0
    return target


def same_base_product_target(
    frame: pd.DataFrame,
    *,
    label_col: str = "label",
    target_col: str = "same_base_product",
) -> pd.Series:
    if target_col in frame.columns:
        return frame[target_col].map(_coerce_bool_target).astype("Int64")
    if label_col not in frame.columns:
        raise ValueError(f"Missing either {target_col!r} or {label_col!r}")
    return same_base_product_from_labels(frame[label_col])


def prepare_calibration_frame(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    label_col: str = "label",
    target_col: str = "same_base_product",
) -> pd.DataFrame:
    """Attach binary same-base target and keep rows usable for threshold evaluation."""
    if score_col not in frame.columns:
        raise ValueError(f"Missing score column: {score_col}")

    output = frame.copy()
    output[score_col] = pd.to_numeric(output[score_col], errors="coerce")
    target = same_base_product_target(output, label_col=label_col, target_col=target_col)

    output[target_col] = target
    output = output[output[score_col].notna() & output[target_col].notna()].copy()
    output[target_col] = output[target_col].astype(int)
    if score_col != "score":
        output["score"] = output[score_col]
    return output.reset_index(drop=True)


def threshold_above_max(scores: Iterable[float]) -> float:
    numeric = pd.to_numeric(pd.Series(list(scores)), errors="coerce").dropna()
    if numeric.empty:
        return math.inf
    return math.nextafter(float(numeric.max()), math.inf)


def threshold_candidates(scores: Iterable[float]) -> list[float]:
    numeric = pd.to_numeric(pd.Series(list(scores)), errors="coerce").dropna()
    if numeric.empty:
        return []
    candidates = {float(value) for value in numeric.unique()}
    candidates.add(threshold_above_max(numeric))
    return sorted(candidates)


def detect_sales_volume_columns(frame: pd.DataFrame) -> tuple[str, str] | None:
    """Find explicit A/B sales-volume columns and ignore revenue/GMV-like fields."""
    explicit_pairs = [
        ("sales_volume_a", "sales_volume_b"),
        ("sales_units_a", "sales_units_b"),
        ("sales_qty_a", "sales_qty_b"),
        ("quantity_sold_a", "quantity_sold_b"),
        ("units_sold_a", "units_sold_b"),
        ("Продажи, шт_a", "Продажи, шт_b"),
        ("Продажи шт_a", "Продажи шт_b"),
        ("Продажи_a", "Продажи_b"),
    ]
    for left_col, right_col in explicit_pairs:
        if left_col in frame.columns and right_col in frame.columns:
            return left_col, right_col
    return None


def add_pair_weights(
    frame: pd.DataFrame,
    *,
    sales_volume_cols: tuple[str, str] | None = None,
) -> PairWeightInfo:
    output = frame.copy()
    volume_cols = sales_volume_cols or detect_sales_volume_columns(output)
    if volume_cols is None:
        output["pair_importance"] = 1.0
        output["pair_weight"] = 1.0
        output["weight_source"] = UNIT_WEIGHT_FALLBACK_SOURCE
        return PairWeightInfo(
            frame=output,
            weights_available=False,
            weight_source=UNIT_WEIGHT_FALLBACK_SOURCE,
            warning="weighted metrics disabled: no reliable A/B sales-volume columns or join source were available",
        )

    left_col, right_col = volume_cols
    left = pd.to_numeric(output[left_col], errors="coerce").clip(lower=0)
    right = pd.to_numeric(output[right_col], errors="coerce").clip(lower=0)
    if left.notna().sum() == 0 and right.notna().sum() == 0:
        output["pair_importance"] = 1.0
        output["pair_weight"] = 1.0
        output["weight_source"] = UNIT_WEIGHT_FALLBACK_SOURCE
        return PairWeightInfo(
            frame=output,
            weights_available=False,
            weight_source=UNIT_WEIGHT_FALLBACK_SOURCE,
            warning=(
                "weighted metrics disabled: detected sales-volume columns "
                f"{left_col!r}/{right_col!r}, but they contain no numeric values"
            ),
        )

    output["sales_volume_a"] = left.fillna(0.0)
    output["sales_volume_b"] = right.fillna(0.0)
    output["pair_importance"] = output[["sales_volume_a", "sales_volume_b"]].max(axis=1)
    output["pair_weight"] = output["pair_importance"].map(math.log1p)
    source = f"sales_volume:{left_col},{right_col}"
    output["weight_source"] = source
    return PairWeightInfo(frame=output, weights_available=True, weight_source=source)


def apply_binary_predictions(
    frame: pd.DataFrame,
    *,
    threshold_same: float,
    threshold_strategy: str,
) -> pd.DataFrame:
    output = prepare_calibration_frame(frame)
    output["threshold_strategy"] = threshold_strategy
    output["threshold_same"] = float(threshold_same)
    output["predicted_binary"] = output["score"].ge(threshold_same).astype(int)
    output["true_label"] = output["same_base_product"].map({1: "same_base_product", 0: "different_product"})
    output["predicted_label"] = output["predicted_binary"].map({1: "same_base_product", 0: "different_product"})
    output["false_merge"] = output["same_base_product"].eq(0) & output["predicted_binary"].eq(1)
    output["false_split"] = output["same_base_product"].eq(1) & output["predicted_binary"].eq(0)
    return output


def _binary_counts(frame: pd.DataFrame) -> dict[str, int]:
    target = frame["same_base_product"].astype(int)
    pred = frame["predicted_binary"].astype(int)
    tp = int((target.eq(1) & pred.eq(1)).sum())
    fp = int((target.eq(0) & pred.eq(1)).sum())
    tn = int((target.eq(0) & pred.eq(0)).sum())
    fn = int((target.eq(1) & pred.eq(0)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def summarize_binary_predictions(
    frame: pd.DataFrame,
    *,
    method: str,
    split_name: str,
    threshold_strategy: str,
    threshold_same: float,
    config: BinaryThresholdConfig,
    weights_available: bool,
    weight_source: str,
) -> dict[str, float | int | str]:
    counts = _binary_counts(frame)
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    total = len(frame)
    actual_positive = tp + fn
    actual_negative = tn + fp
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    false_merge_count = fp
    false_split_count = fn

    row: dict[str, float | int | str] = {
        "method": method,
        "split": split_name,
        "threshold_strategy": threshold_strategy,
        "threshold_same": float(threshold_same),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": _safe_div(tp + tn, total),
        "false_merge_count": false_merge_count,
        "false_split_count": false_split_count,
        "false_merge_rate": _safe_div(false_merge_count, actual_negative),
        "false_split_rate": _safe_div(false_split_count, actual_positive),
        "cost": config.fp_cost * false_merge_count + config.fn_cost * false_split_count,
        "weighted_precision": math.nan,
        "weighted_recall": math.nan,
        "weighted_f1": math.nan,
        "weighted_false_merge_cost": math.nan,
        "weighted_false_split_cost": math.nan,
        "weighted_total_cost": math.nan,
        "weight_source": weight_source,
    }

    if weights_available:
        weights = pd.to_numeric(frame["pair_weight"], errors="coerce").fillna(0.0)
        target = frame["same_base_product"].astype(int)
        pred = frame["predicted_binary"].astype(int)
        weighted_tp = float(weights[target.eq(1) & pred.eq(1)].sum())
        weighted_fp = float(weights[target.eq(0) & pred.eq(1)].sum())
        weighted_fn = float(weights[target.eq(1) & pred.eq(0)].sum())
        weighted_precision = _safe_div(weighted_tp, weighted_tp + weighted_fp)
        weighted_recall = _safe_div(weighted_tp, weighted_tp + weighted_fn)
        row["weighted_precision"] = weighted_precision
        row["weighted_recall"] = weighted_recall
        row["weighted_f1"] = _safe_div(
            2 * weighted_precision * weighted_recall,
            weighted_precision + weighted_recall,
        )
        row["weighted_false_merge_cost"] = weighted_fp
        row["weighted_false_split_cost"] = weighted_fn
        row["weighted_total_cost"] = config.fp_cost * weighted_fp + config.fn_cost * weighted_fn

    return row


def threshold_metric_grid(
    dev_frame: pd.DataFrame,
    *,
    config: BinaryThresholdConfig,
    weights_available: bool,
    weight_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for threshold in threshold_candidates(dev_frame["score"]):
        predictions = apply_binary_predictions(
            dev_frame,
            threshold_same=threshold,
            threshold_strategy="threshold_grid",
        )
        if "pair_weight" not in predictions.columns and "pair_weight" in dev_frame.columns:
            predictions["pair_weight"] = dev_frame["pair_weight"].to_numpy()
        rows.append(
            summarize_binary_predictions(
                predictions,
                method=str(dev_frame["method"].iloc[0]) if "method" in dev_frame.columns and not dev_frame.empty else "",
                split_name="dev",
                threshold_strategy="threshold_grid",
                threshold_same=threshold,
                config=config,
                weights_available=weights_available,
                weight_source=weight_source,
            )
        )
    return pd.DataFrame(rows)


def _select_threshold(grid: pd.DataFrame, strategy: str, weights_available: bool) -> float | None:
    if grid.empty:
        return None

    if strategy == "threshold_max_f1":
        sorted_grid = grid.sort_values(
            ["f1", "cost", "false_merge_count", "threshold_same"],
            ascending=[False, True, True, False],
        )
    elif strategy == "threshold_cost_sensitive":
        sorted_grid = grid.sort_values(
            ["cost", "f1", "false_merge_count", "threshold_same"],
            ascending=[True, False, True, False],
        )
    elif strategy == "threshold_max_weighted_f1" and weights_available:
        sorted_grid = grid.sort_values(
            ["weighted_f1", "weighted_total_cost", "weighted_false_merge_cost", "threshold_same"],
            ascending=[False, True, True, False],
        )
    elif strategy == "threshold_weighted_cost" and weights_available:
        sorted_grid = grid.sort_values(
            ["weighted_total_cost", "weighted_f1", "weighted_false_merge_cost", "threshold_same"],
            ascending=[True, False, True, False],
        )
    else:
        return None

    selected = sorted_grid.iloc[0]
    return float(selected["threshold_same"])


def select_thresholds_for_method(
    dev_frame: pd.DataFrame,
    *,
    config: BinaryThresholdConfig,
    weights_available: bool,
    weight_source: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    grid = threshold_metric_grid(
        dev_frame,
        config=config,
        weights_available=weights_available,
        weight_source=weight_source,
    )
    thresholds: dict[str, float] = {}
    for strategy in ["threshold_max_f1", "threshold_cost_sensitive", "threshold_max_weighted_f1", "threshold_weighted_cost"]:
        threshold = _select_threshold(grid, strategy, weights_available)
        if threshold is not None:
            thresholds[strategy] = threshold
    return thresholds, grid


def _ensure_pair_weights(
    frame: pd.DataFrame,
    *,
    sales_volume_cols: tuple[str, str] | None,
) -> PairWeightInfo:
    if {"pair_importance", "pair_weight", "weight_source"}.issubset(frame.columns):
        weight_source = str(frame["weight_source"].dropna().iloc[0]) if frame["weight_source"].notna().any() else UNIT_WEIGHT_FALLBACK_SOURCE
        return PairWeightInfo(
            frame=frame.copy(),
            weights_available=weight_source != UNIT_WEIGHT_FALLBACK_SOURCE,
            weight_source=weight_source,
        )
    return add_pair_weights(frame, sales_volume_cols=sales_volume_cols)


def calibrate_and_evaluate_methods(
    predictions: pd.DataFrame,
    *,
    config: BinaryThresholdConfig | None = None,
    sales_volume_cols: tuple[str, str] | None = None,
) -> dict[str, pd.DataFrame | bool | str | None]:
    cfg = config or BinaryThresholdConfig()
    prepared = prepare_calibration_frame(predictions)
    weight_info = _ensure_pair_weights(prepared, sales_volume_cols=sales_volume_cols)
    prepared = weight_info.frame

    summary_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    grid_frames: list[pd.DataFrame] = []

    for method, method_frame in prepared.groupby("method", sort=False):
        dev_frame = method_frame[method_frame["eval_split"].eq("dev")].copy()
        if dev_frame.empty:
            continue

        thresholds, grid = select_thresholds_for_method(
            dev_frame,
            config=cfg,
            weights_available=weight_info.weights_available,
            weight_source=weight_info.weight_source,
        )
        if not grid.empty:
            grid_frames.append(grid.assign(method=str(method)))

        for split_name in ["dev", "test"]:
            split_frame = method_frame[method_frame["eval_split"].eq(split_name)].copy()
            if split_frame.empty:
                continue
            for strategy, threshold in thresholds.items():
                split_predictions = apply_binary_predictions(
                    split_frame,
                    threshold_same=threshold,
                    threshold_strategy=strategy,
                )
                for column in ["pair_importance", "pair_weight", "weight_source", "sales_volume_a", "sales_volume_b"]:
                    if column in split_frame.columns:
                        split_predictions[column] = split_frame[column].to_numpy()
                split_predictions["split"] = split_name
                prediction_frames.append(split_predictions)
                summary_rows.append(
                    summarize_binary_predictions(
                        split_predictions,
                        method=str(method),
                        split_name=split_name,
                        threshold_strategy=strategy,
                        threshold_same=threshold,
                        config=cfg,
                        weights_available=weight_info.weights_available,
                        weight_source=weight_info.weight_source,
                    )
                )

    return {
        "summary": pd.DataFrame(summary_rows),
        "predictions": pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
        "threshold_grid_dev": pd.concat(grid_frames, ignore_index=True) if grid_frames else pd.DataFrame(),
        "weights_available": weight_info.weights_available,
        "weight_source": weight_info.weight_source,
        "weight_warning": weight_info.warning,
    }


def volume_bucket_cutoffs(dev_importance: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(dev_importance, errors="coerce").fillna(0.0).clip(lower=0)
    nonzero = numeric[numeric.gt(0)]
    if nonzero.empty:
        return {"low_max": 0.0, "medium_max": 0.0}
    return {
        "low_max": float(nonzero.quantile(1 / 3)),
        "medium_max": float(nonzero.quantile(2 / 3)),
    }


def assign_volume_bucket(pair_importance: pd.Series, cutoffs: dict[str, float]) -> pd.Series:
    numeric = pd.to_numeric(pair_importance, errors="coerce").fillna(0.0).clip(lower=0)
    low_max = cutoffs["low_max"]
    medium_max = cutoffs["medium_max"]
    bucket = pd.Series("high", index=numeric.index, dtype="object")
    bucket[numeric.le(0)] = "zero"
    bucket[numeric.gt(0) & numeric.le(low_max)] = "low"
    bucket[numeric.gt(low_max) & numeric.le(medium_max)] = "medium"
    return bucket


def summarize_by_volume_bucket(
    predictions: pd.DataFrame,
    *,
    config: BinaryThresholdConfig | None = None,
    weight_source: str,
) -> pd.DataFrame:
    if predictions.empty or "pair_importance" not in predictions.columns:
        return pd.DataFrame()
    cfg = config or BinaryThresholdConfig()
    dev_rows = predictions[predictions["split"].eq("dev")]
    if dev_rows.empty:
        return pd.DataFrame()

    cutoffs = volume_bucket_cutoffs(dev_rows["pair_importance"])
    output = predictions.copy()
    output["volume_bucket"] = assign_volume_bucket(output["pair_importance"], cutoffs)

    rows: list[dict[str, object]] = []
    group_cols = ["method", "split", "threshold_strategy", "threshold_same", "volume_bucket"]
    for keys, group in output.groupby(group_cols, dropna=False, sort=False):
        method, split_name, strategy, threshold, bucket = keys
        row = summarize_binary_predictions(
            group,
            method=str(method),
            split_name=str(split_name),
            threshold_strategy=str(strategy),
            threshold_same=float(threshold),
            config=cfg,
            weights_available=True,
            weight_source=weight_source,
        )
        row["volume_bucket"] = bucket
        row["pair_count"] = len(group)
        row["bucket_low_max_dev"] = cutoffs["low_max"]
        row["bucket_medium_max_dev"] = cutoffs["medium_max"]
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_export_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=PREDICTION_CORE_COLUMNS)
    columns = [
        column
        for column in [*PREDICTION_CORE_COLUMNS, *PAIR_CONTEXT_COLUMNS, "sales_volume_a", "sales_volume_b"]
        if column in predictions.columns
    ]
    return predictions[columns].copy()


def write_binary_threshold_reports(
    results: dict[str, pd.DataFrame | bool | str | None],
    reports_dir: str | Path,
    *,
    summary_filename: str = "binary_threshold_summary.csv",
    predictions_filename: str = "binary_threshold_predictions.csv",
    by_volume_bucket_filename: str = "binary_threshold_by_volume_bucket.csv",
) -> dict[str, Path]:
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    summary = results.get("summary", pd.DataFrame())
    predictions = results.get("predictions", pd.DataFrame())
    weights_available = bool(results.get("weights_available", False))
    weight_source = str(results.get("weight_source") or UNIT_WEIGHT_FALLBACK_SOURCE)

    summary_path = reports_path / summary_filename
    predictions_path = reports_path / predictions_filename
    summary_frame = summary if isinstance(summary, pd.DataFrame) else pd.DataFrame()
    prediction_frame = prediction_export_frame(predictions if isinstance(predictions, pd.DataFrame) else pd.DataFrame())
    summary_frame.reindex(columns=SUMMARY_COLUMNS).to_csv(summary_path, index=False)
    prediction_frame.to_csv(predictions_path, index=False)

    paths = {"summary": summary_path, "predictions": predictions_path}
    if weights_available and isinstance(predictions, pd.DataFrame):
        bucket_frame = summarize_by_volume_bucket(predictions, weight_source=weight_source)
        bucket_path = reports_path / by_volume_bucket_filename
        bucket_frame.to_csv(bucket_path, index=False)
        paths["by_volume_bucket"] = bucket_path
    return paths


def write_threshold_reports(results: dict[str, pd.DataFrame | bool | str | None], reports_dir: str | Path) -> None:
    write_binary_threshold_reports(results, reports_dir)
