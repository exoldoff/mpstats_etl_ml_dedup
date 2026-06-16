from __future__ import annotations

import math

import pandas as pd

from research.dedup.threshold_calibration import (
    BinaryThresholdConfig,
    add_pair_weights,
    calibrate_and_evaluate_methods,
    prepare_calibration_frame,
    summarize_by_volume_bucket,
    write_binary_threshold_reports,
)


def _row(method: str, split: str, score: float, label: str, idx: int) -> dict[str, object]:
    return {
        "method": method,
        "eval_split": split,
        "score": score,
        "label": label,
        "title_a": f"A {idx}",
        "title_b": f"B {idx}",
        "brand_a": "Brand",
        "brand_b": "Brand",
        "raw_record_id_a": f"wb::{idx}",
        "raw_record_id_b": f"ozon::{idx}",
        "sku_a": str(idx),
        "sku_b": str(idx + 100),
        "benchmark_pair_key": f"pair-{idx}",
    }


def test_prepare_calibration_frame_maps_legacy_pack_label_to_positive() -> None:
    frame = pd.DataFrame(
        [
            {"score": 0.9, "label": "exact_duplicate"},
            {"score": 0.8, "label": "same_product_different_pack"},
            {"score": 0.1, "label": "different_product"},
            {"score": 0.5, "label": "uncertain"},
        ]
    )

    prepared = prepare_calibration_frame(frame)

    assert prepared["same_base_product"].tolist() == [1, 1, 0]


def test_existing_same_base_product_column_overrides_label_mapping() -> None:
    frame = pd.DataFrame(
        [
            {"score": 0.9, "label": "different_product", "same_base_product": 1},
            {"score": 0.1, "label": "exact_duplicate", "same_base_product": 0},
        ]
    )

    prepared = prepare_calibration_frame(frame)

    assert prepared["same_base_product"].tolist() == [1, 0]


def test_binary_threshold_strategies_are_selected_on_dev_only() -> None:
    frame = pd.DataFrame(
        [
            _row("m", "dev", 0.95, "exact_duplicate", 1),
            _row("m", "dev", 0.90, "same_product_different_pack", 2),
            _row("m", "dev", 0.80, "different_product", 3),
            _row("m", "dev", 0.20, "different_product", 4),
            _row("m", "test", 0.96, "exact_duplicate", 5),
            _row("m", "test", 0.50, "different_product", 6),
            _row("m", "test", 0.05, "same_product_different_pack", 7),
        ]
    )

    results = calibrate_and_evaluate_methods(frame)
    summary = results["summary"]
    predictions = results["predictions"]

    max_f1 = summary[
        summary["split"].eq("dev") & summary["threshold_strategy"].eq("threshold_max_f1")
    ].iloc[0]
    cost_sensitive_test = summary[
        summary["split"].eq("test") & summary["threshold_strategy"].eq("threshold_cost_sensitive")
    ].iloc[0]

    assert max_f1["threshold_same"] == 0.90
    assert cost_sensitive_test["threshold_same"] == 0.90
    assert cost_sensitive_test["false_merge_count"] == 0
    assert cost_sensitive_test["false_split_count"] == 1
    assert math.isclose(cost_sensitive_test["cost"], 1.0)
    assert set(predictions["threshold_strategy"]) == {"threshold_max_f1", "threshold_cost_sensitive"}
    assert "manual_review" not in predictions.columns


def test_cost_sensitive_threshold_penalizes_false_merges() -> None:
    frame = pd.DataFrame(
        [
            _row("m", "dev", 0.95, "exact_duplicate", 1),
            _row("m", "dev", 0.90, "different_product", 2),
            _row("m", "dev", 0.85, "exact_duplicate", 3),
            _row("m", "dev", 0.10, "different_product", 4),
        ]
    )

    results = calibrate_and_evaluate_methods(frame, config=BinaryThresholdConfig(fp_cost=5, fn_cost=1))
    summary = results["summary"]

    max_f1 = summary[
        summary["split"].eq("dev") & summary["threshold_strategy"].eq("threshold_max_f1")
    ].iloc[0]
    cost_sensitive = summary[
        summary["split"].eq("dev") & summary["threshold_strategy"].eq("threshold_cost_sensitive")
    ].iloc[0]

    assert max_f1["threshold_same"] == 0.85
    assert max_f1["false_merge_count"] == 1
    assert cost_sensitive["threshold_same"] == 0.95
    assert cost_sensitive["false_merge_count"] == 0
    assert cost_sensitive["false_split_count"] == 1


def test_sales_volume_weights_enable_weighted_strategies() -> None:
    frame = pd.DataFrame(
        [
            {**_row("m", "dev", 0.95, "exact_duplicate", 1), "sales_volume_a": 100, "sales_volume_b": 10},
            {**_row("m", "dev", 0.90, "different_product", 2), "sales_volume_a": 1000, "sales_volume_b": 5},
            {**_row("m", "dev", 0.85, "exact_duplicate", 3), "sales_volume_a": 1, "sales_volume_b": 1},
            {**_row("m", "dev", 0.10, "different_product", 4), "sales_volume_a": 0, "sales_volume_b": 0},
            {**_row("m", "test", 0.94, "different_product", 5), "sales_volume_a": 500, "sales_volume_b": 2},
        ]
    )

    weighted = add_pair_weights(frame)
    results = calibrate_and_evaluate_methods(weighted.frame)
    summary = results["summary"]

    assert results["weights_available"] is True
    assert {
        "threshold_max_f1",
        "threshold_cost_sensitive",
        "threshold_max_weighted_f1",
        "threshold_weighted_cost",
    }.issubset(set(summary["threshold_strategy"]))
    assert summary["weighted_f1"].notna().any()
    weighted_cost = summary[
        summary["split"].eq("dev") & summary["threshold_strategy"].eq("threshold_weighted_cost")
    ].iloc[0]
    assert weighted_cost["threshold_same"] == 0.95
    assert weighted_cost["weighted_total_cost"] < summary[
        summary["split"].eq("dev") & summary["threshold_strategy"].eq("threshold_max_f1")
    ].iloc[0]["weighted_total_cost"]


def test_unit_weight_fallback_disables_weighted_metrics() -> None:
    frame = pd.DataFrame(
        [
            _row("m", "dev", 0.90, "exact_duplicate", 1),
            _row("m", "dev", 0.10, "different_product", 2),
        ]
    )

    results = calibrate_and_evaluate_methods(frame)
    summary = results["summary"]

    assert results["weights_available"] is False
    assert results["weight_source"] == "unit_weight_fallback"
    assert set(summary["threshold_strategy"]) == {"threshold_max_f1", "threshold_cost_sensitive"}
    assert summary["weighted_f1"].isna().all()


def test_binary_threshold_reports_write_only_compact_csvs(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {**_row("m", "dev", 0.90, "exact_duplicate", 1), "sales_volume_a": 10, "sales_volume_b": 20},
            {**_row("m", "dev", 0.10, "different_product", 2), "sales_volume_a": 0, "sales_volume_b": 0},
            {**_row("m", "test", 0.80, "different_product", 3), "sales_volume_a": 5, "sales_volume_b": 4},
        ]
    )

    results = calibrate_and_evaluate_methods(add_pair_weights(frame).frame)
    paths = write_binary_threshold_reports(results, tmp_path)

    assert paths["summary"].name == "binary_threshold_summary.csv"
    assert paths["predictions"].name == "binary_threshold_predictions.csv"
    assert paths["by_volume_bucket"].name == "binary_threshold_by_volume_bucket.csv"
    assert paths["summary"].exists()
    assert paths["predictions"].exists()
    assert paths["by_volume_bucket"].exists()

    summary = pd.read_csv(paths["summary"])
    predictions = pd.read_csv(paths["predictions"])
    by_bucket = summarize_by_volume_bucket(results["predictions"], weight_source=str(results["weight_source"]))

    assert {"method", "split", "threshold_strategy", "threshold_same", "weighted_total_cost"}.issubset(summary.columns)
    assert {"predicted_binary", "false_merge", "false_split", "pair_weight"}.issubset(predictions.columns)
    assert not by_bucket.empty
