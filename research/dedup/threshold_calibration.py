from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


POSITIVE_LABELS = frozenset({"exact_duplicate", "same_product_different_pack"})
NEGATIVE_LABELS = frozenset({"different_product"})
TRIAGE_LABELS = ["auto_same", "auto_different", "manual_review"]

ERROR_REPORT_COLUMNS = [
    "method",
    "score",
    "threshold_auto_same",
    "threshold_auto_diff",
    "label",
    "same_base_product",
    "predicted_triage_label",
    "title_a",
    "title_b",
    "brand_a",
    "brand_b",
    "unit_amount_a",
    "unit_amount_b",
    "total_amount_a",
    "total_amount_b",
    "multipack_count_a",
    "multipack_count_b",
    "benchmark_pair_key",
]


@dataclass(frozen=True)
class ThresholdCalibrationConfig:
    target_auto_same_precision: float = 0.97
    max_false_merges_on_dev: int = 0
    target_auto_diff_precision: float = 0.95


@dataclass(frozen=True)
class CalibratedThresholds:
    method: str
    threshold_auto_same: float
    threshold_auto_diff: float
    passed_auto_same_constraints: bool
    passed_auto_diff_constraints: bool


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
    """Attach binary same-base target and keep rows usable for threshold calibration."""
    if score_col not in frame.columns:
        raise ValueError(f"Missing score column: {score_col}")

    output = frame.copy()
    output[score_col] = pd.to_numeric(output[score_col], errors="coerce")
    target = same_base_product_target(output, label_col=label_col, target_col=target_col)

    output[target_col] = target
    output = output[output[score_col].notna() & output[target_col].notna()].copy()
    output[target_col] = output[target_col].astype(int)
    return output.reset_index(drop=True)


def threshold_candidates(scores: Iterable[float]) -> list[float]:
    numeric = pd.to_numeric(pd.Series(list(scores)), errors="coerce").dropna()
    if numeric.empty:
        return []
    return sorted(float(value) for value in numeric.unique())


def threshold_above_max(scores: Iterable[float]) -> float:
    numeric = pd.to_numeric(pd.Series(list(scores)), errors="coerce").dropna()
    if numeric.empty:
        return math.inf
    return math.nextafter(float(numeric.max()), math.inf)


def threshold_below_min(scores: Iterable[float]) -> float:
    numeric = pd.to_numeric(pd.Series(list(scores)), errors="coerce").dropna()
    if numeric.empty:
        return -math.inf
    return math.nextafter(float(numeric.min()), -math.inf)


def auto_same_metrics_at_threshold(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    target = frame["same_base_product"].astype(int)
    auto_same = frame["score"].ge(threshold)
    tp = int((auto_same & target.eq(1)).sum())
    fp = int((auto_same & target.eq(0)).sum())
    actual_positive = int(target.eq(1).sum())
    auto_count = int(auto_same.sum())
    total = len(frame)
    return {
        "threshold": float(threshold),
        "auto_same_precision": _safe_div(tp, auto_count, default=1.0),
        "auto_same_recall": _safe_div(tp, actual_positive),
        "false_merge_count": fp,
        "false_merge_rate": _safe_div(fp, total),
        "auto_same_coverage": _safe_div(auto_count, total),
    }


def auto_diff_metrics_at_threshold(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    target = frame["same_base_product"].astype(int)
    auto_diff = frame["score"].le(threshold)
    tn = int((auto_diff & target.eq(0)).sum())
    fn = int((auto_diff & target.eq(1)).sum())
    actual_negative = int(target.eq(0).sum())
    auto_count = int(auto_diff.sum())
    total = len(frame)
    return {
        "threshold": float(threshold),
        "auto_diff_precision": _safe_div(tn, auto_count, default=1.0),
        "auto_diff_recall": _safe_div(tn, actual_negative),
        "false_reject_count": fn,
        "auto_diff_coverage": _safe_div(auto_count, total),
    }


def auto_same_threshold_grid(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [auto_same_metrics_at_threshold(frame, threshold) for threshold in threshold_candidates(frame["score"])]
    return pd.DataFrame(rows)


def auto_diff_threshold_grid(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [auto_diff_metrics_at_threshold(frame, threshold) for threshold in threshold_candidates(frame["score"])]
    return pd.DataFrame(rows)


def select_auto_same_threshold(
    dev_frame: pd.DataFrame,
    config: ThresholdCalibrationConfig,
) -> tuple[float, bool, pd.DataFrame]:
    grid = auto_same_threshold_grid(dev_frame)
    if grid.empty:
        return math.inf, False, grid

    eligible = grid[
        grid["auto_same_precision"].ge(config.target_auto_same_precision)
        & grid["false_merge_count"].le(config.max_false_merges_on_dev)
    ].copy()
    if eligible.empty:
        return threshold_above_max(dev_frame["score"]), False, grid

    selected = eligible.sort_values(
        ["auto_same_recall", "auto_same_coverage", "auto_same_precision", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return float(selected["threshold"]), True, grid


def select_auto_diff_threshold(
    dev_frame: pd.DataFrame,
    config: ThresholdCalibrationConfig,
) -> tuple[float, bool, pd.DataFrame]:
    grid = auto_diff_threshold_grid(dev_frame)
    if grid.empty:
        return -math.inf, False, grid

    eligible = grid[grid["auto_diff_precision"].ge(config.target_auto_diff_precision)].copy()
    if eligible.empty:
        return threshold_below_min(dev_frame["score"]), False, grid

    selected = eligible.sort_values(
        ["auto_diff_coverage", "auto_diff_recall", "auto_diff_precision", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]
    return float(selected["threshold"]), True, grid


def apply_triage_predictions(
    frame: pd.DataFrame,
    thresholds: CalibratedThresholds,
) -> pd.DataFrame:
    output = prepare_calibration_frame(frame)
    output["threshold_auto_same"] = thresholds.threshold_auto_same
    output["threshold_auto_diff"] = thresholds.threshold_auto_diff
    output["predicted_triage_label"] = "manual_review"

    auto_same = output["score"].ge(thresholds.threshold_auto_same)
    auto_diff = output["score"].le(thresholds.threshold_auto_diff) & ~auto_same
    output.loc[auto_same, "predicted_triage_label"] = "auto_same"
    output.loc[auto_diff, "predicted_triage_label"] = "auto_different"

    output["predicted_label"] = "manual_review"
    output.loc[auto_same, "predicted_label"] = "exact_duplicate"
    output.loc[auto_diff, "predicted_label"] = "different_product"
    output["forced_predicted_same_base_product"] = output["score"].ge(thresholds.threshold_auto_same).astype(int)
    output["forced_predicted_label"] = output["forced_predicted_same_base_product"].map(
        {1: "same_base_product", 0: "different_product"}
    )
    output["false_merge"] = output["predicted_triage_label"].eq("auto_same") & output["same_base_product"].eq(0)
    output["false_reject"] = output["predicted_triage_label"].eq("auto_different") & output["same_base_product"].eq(1)
    output["manual_review"] = output["predicted_triage_label"].eq("manual_review")
    return output


def forced_binary_f1(frame: pd.DataFrame) -> float:
    target = frame["same_base_product"].astype(int)
    pred = frame["forced_predicted_same_base_product"].astype(int)
    tp = int((target.eq(1) & pred.eq(1)).sum())
    fp = int((target.eq(0) & pred.eq(1)).sum())
    fn = int((target.eq(1) & pred.eq(0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return _safe_div(2 * precision * recall, precision + recall)


def summarize_triage_predictions(
    frame: pd.DataFrame,
    *,
    method: str,
    split_name: str,
    thresholds: CalibratedThresholds,
) -> dict[str, float | int | str | bool]:
    target = frame["same_base_product"].astype(int)
    auto_same = frame["predicted_triage_label"].eq("auto_same")
    auto_diff = frame["predicted_triage_label"].eq("auto_different")
    manual = frame["predicted_triage_label"].eq("manual_review")

    tp = int((auto_same & target.eq(1)).sum())
    fp = int((auto_same & target.eq(0)).sum())
    tn = int((auto_diff & target.eq(0)).sum())
    fn = int((auto_diff & target.eq(1)).sum())
    actual_positive = int(target.eq(1).sum())
    actual_negative = int(target.eq(0).sum())
    total = len(frame)

    return {
        "method": method,
        "eval_split": split_name,
        "threshold_auto_same": thresholds.threshold_auto_same,
        "threshold_auto_diff": thresholds.threshold_auto_diff,
        "auto_same_precision": _safe_div(tp, tp + fp, default=1.0),
        "auto_same_recall": _safe_div(tp, actual_positive),
        "false_merge_count": fp,
        "false_merge_rate": _safe_div(fp, total),
        "auto_same_coverage": _safe_div(tp + fp, total),
        "auto_diff_precision": _safe_div(tn, tn + fn, default=1.0),
        "auto_diff_recall": _safe_div(tn, actual_negative),
        "false_reject_count": fn,
        "auto_diff_coverage": _safe_div(tn + fn, total),
        "manual_review_rate": _safe_div(int(manual.sum()), total),
        "auto_coverage": _safe_div(int(auto_same.sum() + auto_diff.sum()), total),
        "binary_f1_if_forced": forced_binary_f1(frame),
        "passed_auto_same_constraints": thresholds.passed_auto_same_constraints,
        "passed_auto_diff_constraints": thresholds.passed_auto_diff_constraints,
        "pairs": total,
        "same_base_pairs": actual_positive,
        "different_product_pairs": actual_negative,
    }


def calibrate_method_thresholds(
    method: str,
    dev_frame: pd.DataFrame,
    config: ThresholdCalibrationConfig | None = None,
) -> tuple[CalibratedThresholds, pd.DataFrame, pd.DataFrame]:
    cfg = config or ThresholdCalibrationConfig()
    prepared = prepare_calibration_frame(dev_frame)
    threshold_auto_same, passed_same, same_grid = select_auto_same_threshold(prepared, cfg)
    threshold_auto_diff, passed_diff, diff_grid = select_auto_diff_threshold(prepared, cfg)
    thresholds = CalibratedThresholds(
        method=method,
        threshold_auto_same=threshold_auto_same,
        threshold_auto_diff=threshold_auto_diff,
        passed_auto_same_constraints=passed_same,
        passed_auto_diff_constraints=passed_diff,
    )
    same_grid = same_grid.assign(method=method, target_auto_same_precision=cfg.target_auto_same_precision)
    diff_grid = diff_grid.assign(method=method, target_auto_diff_precision=cfg.target_auto_diff_precision)
    return thresholds, same_grid, diff_grid


def calibrate_and_evaluate_methods(
    predictions: pd.DataFrame,
    *,
    config: ThresholdCalibrationConfig | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or ThresholdCalibrationConfig()
    calibration_rows: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    same_grid_frames: list[pd.DataFrame] = []
    diff_grid_frames: list[pd.DataFrame] = []

    prepared = prepare_calibration_frame(predictions)
    for method, method_frame in prepared.groupby("method", sort=False):
        dev_frame = method_frame[method_frame["eval_split"].eq("dev")].copy()
        test_frame = method_frame[method_frame["eval_split"].eq("test")].copy()
        if dev_frame.empty:
            continue

        thresholds, same_grid, diff_grid = calibrate_method_thresholds(str(method), dev_frame, cfg)
        same_grid_frames.append(same_grid)
        diff_grid_frames.append(diff_grid)

        dev_predictions = apply_triage_predictions(dev_frame, thresholds)
        dev_predictions["eval_split"] = "dev"
        calibration_rows.append(
            summarize_triage_predictions(dev_predictions, method=str(method), split_name="dev", thresholds=thresholds)
        )
        prediction_frames.append(dev_predictions)

        if not test_frame.empty:
            test_predictions = apply_triage_predictions(test_frame, thresholds)
            test_predictions["eval_split"] = "test"
            evaluation_rows.append(
                summarize_triage_predictions(
                    test_predictions,
                    method=str(method),
                    split_name="test",
                    thresholds=thresholds,
                )
            )
            prediction_frames.append(test_predictions)

    return {
        "calibration_on_dev": pd.DataFrame(calibration_rows),
        "evaluation_on_test": pd.DataFrame(evaluation_rows),
        "predictions": pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
        "auto_same_threshold_grid": pd.concat(same_grid_frames, ignore_index=True) if same_grid_frames else pd.DataFrame(),
        "auto_diff_threshold_grid": pd.concat(diff_grid_frames, ignore_index=True) if diff_grid_frames else pd.DataFrame(),
    }


def forced_confusion_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_calibration_frame(frame)
    matrix = pd.crosstab(
        prepared["same_base_product"].map({1: "same_base_product", 0: "different_product"}),
        prepared["forced_predicted_label"],
        dropna=False,
    )
    return matrix.reindex(
        index=["same_base_product", "different_product"],
        columns=["same_base_product", "different_product"],
        fill_value=0,
    )


def triage_confusion_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_calibration_frame(frame)
    matrix = pd.crosstab(
        prepared["same_base_product"].map({1: "same_base_product", 0: "different_product"}),
        prepared["predicted_triage_label"],
        dropna=False,
    )
    return matrix.reindex(
        index=["same_base_product", "different_product"],
        columns=TRIAGE_LABELS,
        fill_value=0,
    )


def error_report(frame: pd.DataFrame, mask_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=ERROR_REPORT_COLUMNS)
    rows = frame[frame[mask_col].astype(bool)].copy()
    return rows[[column for column in ERROR_REPORT_COLUMNS if column in rows.columns]]


def threshold_pair_review(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    predictions = results.get("predictions", pd.DataFrame())
    frames: list[pd.DataFrame] = []

    for split_name in ["dev", "test"]:
        split = (
            predictions[predictions["eval_split"].eq(split_name)].copy()
            if not predictions.empty
            else pd.DataFrame()
        )
        for issue_type, mask_col in [
            ("false_merge", "false_merge"),
            ("false_reject", "false_reject"),
            ("manual_review", "manual_review"),
        ]:
            report = error_report(split, mask_col)
            if report.empty:
                continue
            report = report.copy()
            report.insert(0, "issue_type", issue_type)
            report.insert(1, "eval_split", split_name)
            frames.append(report)

    if not frames:
        return pd.DataFrame(columns=["issue_type", "eval_split", *ERROR_REPORT_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def _threshold_model_ranking(calibration: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    ranking = calibration.copy()
    if ranking.empty:
        return ranking

    if not evaluation.empty:
        test_columns = [
            "method",
            "auto_same_precision",
            "auto_same_recall",
            "false_merge_count",
            "false_merge_rate",
            "auto_diff_precision",
            "auto_diff_recall",
            "false_reject_count",
            "manual_review_rate",
            "auto_coverage",
            "binary_f1_if_forced",
            "pairs",
            "same_base_pairs",
            "different_product_pairs",
        ]
        test_view = evaluation[[column for column in test_columns if column in evaluation.columns]].copy()
        test_view = test_view.rename(
            columns={column: f"test_{column}" for column in test_view.columns if column != "method"}
        )
        ranking = ranking.merge(test_view, on="method", how="left")

    sort_columns = [
        column
        for column in ["passed_auto_same_constraints", "auto_coverage", "auto_same_recall", "manual_review_rate"]
        if column in ranking.columns
    ]
    if sort_columns:
        ranking = ranking.sort_values(
            sort_columns,
            ascending=[False, False, False, True][: len(sort_columns)],
        ).reset_index(drop=True)
    return ranking


def write_compact_threshold_report(
    results: dict[str, pd.DataFrame],
    reports_dir: str | Path,
    *,
    summary_filename: str = "threshold_summary.xlsx",
    pair_review_filename: str = "threshold_pair_review.csv",
) -> dict[str, Path]:
    """Write the SKU threshold benchmark as one workbook plus one pair-review CSV."""
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    summary_path = reports_path / summary_filename
    pair_review_path = reports_path / pair_review_filename

    calibration = results.get("calibration_on_dev", pd.DataFrame())
    evaluation = results.get("evaluation_on_test", pd.DataFrame())
    confusion = results.get("confusion_matrices", pd.DataFrame())
    pair_review = threshold_pair_review(results)
    ranking = _threshold_model_ranking(calibration, evaluation)

    readme = pd.DataFrame(
        [
            {"field": "generated_at", "value": pd.Timestamp.now().isoformat()},
            {"field": "source", "value": "notebooks/03_matching_comparison.ipynb"},
            {
                "field": "auto_merge_rule",
                "value": "Choose threshold_auto_same on dev by max recall with precision target and false_merge_count limit.",
            },
            {
                "field": "test_rule",
                "value": "evaluation_test is held-out validation only; do not choose thresholds or models on test.",
            },
            {
                "field": "pair_review",
                "value": "threshold_pair_review.csv contains false_merge, false_reject and manual_review rows for dev/test.",
            },
        ]
    )

    with pd.ExcelWriter(summary_path) as writer:
        readme.to_excel(writer, sheet_name="readme", index=False)
        ranking.to_excel(writer, sheet_name="model_ranking", index=False)
        calibration.to_excel(writer, sheet_name="calibration_dev", index=False)
        evaluation.to_excel(writer, sheet_name="evaluation_test", index=False)
        confusion.to_excel(writer, sheet_name="confusion_matrices", index=False)
        pair_review.to_excel(writer, sheet_name="pair_review", index=False)

    pair_review.to_csv(pair_review_path, index=False)
    return {"summary": summary_path, "pair_review": pair_review_path}


def write_threshold_reports(results: dict[str, pd.DataFrame], reports_dir: str | Path) -> None:
    write_compact_threshold_report(results, reports_dir)
