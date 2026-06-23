from __future__ import annotations

import pandas as pd

from research.dedup.training.calibrate_scores import main as calibrate_scores_main
from research.dedup.training.sales_weights import attach_sales_volumes


def _score_row(split: str, score: float, label: str, idx: int) -> dict[str, object]:
    return {
        "split": split,
        "score_model": score,
        "label": label,
        "raw_record_id_a": f"wb::{idx}",
        "raw_record_id_b": f"ozon::{idx}",
        "title_a": f"A {idx}",
        "title_b": f"B {idx}",
        "category_run": "sauces",
    }


def test_attach_sales_volumes_from_lookup_csv(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            _score_row("dev", 0.9, "exact_duplicate", 1),
            _score_row("dev", 0.1, "different_product", 2),
        ]
    )
    lookup_path = tmp_path / "sales_lookup.csv"
    pd.DataFrame(
        [
            {"raw_record_id": "wb::1", "sales_volume": 100},
            {"raw_record_id": "ozon::1", "sales_volume": 10},
            {"raw_record_id": "wb::2", "sales_volume": 5},
            {"raw_record_id": "ozon::2", "sales_volume": 50},
        ]
    ).to_csv(lookup_path, index=False)

    attached, status = attach_sales_volumes(
        frame,
        sales_lookup_path=lookup_path,
        enable_duckdb_join=False,
    )

    assert status.status == "joined_sales_volume"
    assert status.matched_left_rows == 2
    assert status.matched_right_rows == 2
    assert attached["sales_volume_a"].tolist() == [100, 5]
    assert attached["sales_volume_b"].tolist() == [10, 50]


def test_calibrate_scores_requires_weighted_metrics(tmp_path) -> None:
    score_path = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            _score_row("dev", 0.9, "exact_duplicate", 1),
            _score_row("dev", 0.1, "different_product", 2),
        ]
    ).to_csv(score_path, index=False)

    reports_dir = tmp_path / "reports"
    code = calibrate_scores_main(
        [
            "--score-path",
            str(score_path),
            "--score-column",
            "score_model",
            "--method",
            "model",
            "--reports-dir",
            str(reports_dir),
            "--disable-sales-duckdb-join",
            "--require-weighted",
        ]
    )

    assert code == 2
    assert not (reports_dir / "model_binary_threshold_summary.csv").exists()


def test_calibrate_scores_with_sales_lookup_writes_weighted_strategies(tmp_path) -> None:
    score_path = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            _score_row("dev", 0.95, "exact_duplicate", 1),
            _score_row("dev", 0.90, "different_product", 2),
            _score_row("dev", 0.85, "exact_duplicate", 3),
            _score_row("dev", 0.10, "different_product", 4),
            _score_row("test", 0.94, "different_product", 5),
        ]
    ).to_csv(score_path, index=False)

    lookup_path = tmp_path / "sales_lookup.csv"
    pd.DataFrame(
        [
            {"raw_record_id": "wb::1", "sales_volume": 100},
            {"raw_record_id": "ozon::1", "sales_volume": 10},
            {"raw_record_id": "wb::2", "sales_volume": 1000},
            {"raw_record_id": "ozon::2", "sales_volume": 5},
            {"raw_record_id": "wb::3", "sales_volume": 1},
            {"raw_record_id": "ozon::3", "sales_volume": 1},
            {"raw_record_id": "wb::4", "sales_volume": 0},
            {"raw_record_id": "ozon::4", "sales_volume": 0},
            {"raw_record_id": "wb::5", "sales_volume": 500},
            {"raw_record_id": "ozon::5", "sales_volume": 2},
        ]
    ).to_csv(lookup_path, index=False)

    reports_dir = tmp_path / "reports"
    code = calibrate_scores_main(
        [
            "--score-path",
            str(score_path),
            "--score-column",
            "score_model",
            "--method",
            "model",
            "--reports-dir",
            str(reports_dir),
            "--sales-lookup-path",
            str(lookup_path),
            "--require-weighted",
        ]
    )

    assert code == 0
    summary = pd.read_csv(reports_dir / "model_binary_threshold_summary.csv")
    predictions = pd.read_csv(reports_dir / "model_binary_threshold_predictions.csv")
    assert "threshold_weighted_cost" in set(summary["threshold_strategy"])
    assert "threshold_max_weighted_f1" in set(summary["threshold_strategy"])
    assert {"sales_volume_a", "sales_volume_b", "pair_weight"}.issubset(predictions.columns)
