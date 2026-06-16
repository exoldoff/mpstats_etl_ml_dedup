from __future__ import annotations

import math

import pandas as pd

from research.dedup.threshold_calibration import (
    ThresholdCalibrationConfig,
    calibrate_and_evaluate_methods,
    calibrate_method_thresholds,
    error_report,
    prepare_calibration_frame,
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
        "unit_amount_a": 0.2,
        "unit_amount_b": 0.2,
        "total_amount_a": 0.2,
        "total_amount_b": 0.2,
        "multipack_count_a": 1,
        "multipack_count_b": 1,
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


def test_auto_same_threshold_chooses_highest_recall_without_false_merges() -> None:
    frame = pd.DataFrame(
        [
            _row("m", "dev", 0.95, "exact_duplicate", 1),
            _row("m", "dev", 0.90, "same_product_different_pack", 2),
            _row("m", "dev", 0.80, "different_product", 3),
            _row("m", "dev", 0.20, "different_product", 4),
        ]
    )

    thresholds, same_grid, _ = calibrate_method_thresholds(
        "m",
        frame,
        ThresholdCalibrationConfig(target_auto_same_precision=0.97, max_false_merges_on_dev=0),
    )

    assert thresholds.threshold_auto_same == 0.90
    assert thresholds.passed_auto_same_constraints is True
    selected = same_grid[same_grid["threshold"].eq(0.90)].iloc[0]
    assert selected["auto_same_recall"] == 1.0
    assert selected["false_merge_count"] == 0


def test_auto_same_threshold_falls_back_above_max_when_constraints_fail() -> None:
    frame = pd.DataFrame(
        [
            _row("m", "dev", 0.90, "different_product", 1),
            _row("m", "dev", 0.80, "exact_duplicate", 2),
            _row("m", "dev", 0.70, "different_product", 3),
        ]
    )

    thresholds, _, _ = calibrate_method_thresholds(
        "m",
        frame,
        ThresholdCalibrationConfig(target_auto_same_precision=0.97, max_false_merges_on_dev=0),
    )

    assert thresholds.passed_auto_same_constraints is False
    assert thresholds.threshold_auto_same > 0.90


def test_calibrate_and_evaluate_outputs_dev_and_test_triage_metrics() -> None:
    frame = pd.DataFrame(
        [
            _row("m", "dev", 0.95, "exact_duplicate", 1),
            _row("m", "dev", 0.90, "same_product_different_pack", 2),
            _row("m", "dev", 0.20, "different_product", 3),
            _row("m", "dev", 0.10, "different_product", 4),
            _row("m", "test", 0.96, "exact_duplicate", 5),
            _row("m", "test", 0.50, "different_product", 6),
            _row("m", "test", 0.05, "same_product_different_pack", 7),
        ]
    )

    results = calibrate_and_evaluate_methods(frame)
    calibration = results["calibration_on_dev"].iloc[0]
    evaluation = results["evaluation_on_test"].iloc[0]
    predictions = results["predictions"]

    assert calibration["threshold_auto_same"] == 0.90
    assert calibration["threshold_auto_diff"] == 0.20
    assert calibration["auto_coverage"] == 1.0
    assert evaluation["false_reject_count"] == 1
    assert math.isclose(evaluation["manual_review_rate"], 1 / 3)

    test_predictions = predictions[predictions["eval_split"].eq("test")]
    rejects = error_report(test_predictions, "false_reject")
    manual = error_report(test_predictions, "manual_review")
    assert rejects["benchmark_pair_key"].tolist() == ["pair-7"]
    assert manual["benchmark_pair_key"].tolist() == ["pair-6"]
