from __future__ import annotations

import pandas as pd

from tools.dedup_labeling_bot.csv_repository import LabelingCsvRepository


def test_set_label_preserves_existing_columns(tmp_path) -> None:
    csv_path = tmp_path / "labeling.csv"
    pd.DataFrame(
        {
            "label": ["", ""],
            "notes": ["keep", ""],
            "title_a": ["a1", "a2"],
            "title_b": ["b1", "b2"],
            "custom_column": ["x", "y"],
        }
    ).to_csv(csv_path, index=False)

    repository = LabelingCsvRepository(csv_path)
    repository.set_label(1, "different_product")

    frame = pd.read_csv(csv_path)
    assert list(frame.columns) == ["label", "notes", "title_a", "title_b", "custom_column"]
    assert frame.at[1, "label"] == "different_product"
    assert frame.at[0, "custom_column"] == "x"


def test_missing_label_and_notes_columns_are_added(tmp_path) -> None:
    csv_path = tmp_path / "labeling.csv"
    pd.DataFrame({"title_a": ["a"], "title_b": ["b"]}).to_csv(csv_path, index=False)

    repository = LabelingCsvRepository(csv_path)
    stats = repository.stats()

    assert stats.total_rows == 1
    assert stats.unlabeled_rows == 1
    assert {"label", "notes"}.issubset(repository.load().columns)
