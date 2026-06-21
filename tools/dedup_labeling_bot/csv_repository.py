from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from research.dedup.annotation import (
    VALID_LABELS,
    is_labeled,
    load_labeling_file,
    save_labeling_file,
)


@dataclass(frozen=True)
class CsvStats:
    total_rows: int
    labeled_rows: int
    unlabeled_rows: int


class LabelingCsvRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        return load_labeling_file(self.path)

    def get_row(self, row_index: int) -> pd.Series:
        frame = self.load()
        self._ensure_row_index(frame, row_index)
        return frame.iloc[row_index]

    def available_row_indices(self) -> list[int]:
        frame = self.load()
        return [int(idx) for idx, value in frame["label"].items() if not is_labeled(value)]

    def is_row_available(self, row_index: int) -> bool:
        frame = self.load()
        self._ensure_row_index(frame, row_index)
        return not is_labeled(frame.at[row_index, "label"])

    def set_label(self, row_index: int, label: str) -> None:
        if label not in VALID_LABELS:
            raise ValueError(f"Unsupported label: {label!r}")
        frame = self.load()
        self._ensure_row_index(frame, row_index)
        frame.at[row_index, "label"] = label
        save_labeling_file(frame, self.path)

    def stats(self) -> CsvStats:
        frame = self.load()
        total = len(frame)
        labeled = int(frame["label"].map(is_labeled).sum())
        return CsvStats(total_rows=total, labeled_rows=labeled, unlabeled_rows=total - labeled)

    @staticmethod
    def _ensure_row_index(frame: pd.DataFrame, row_index: int) -> None:
        if row_index < 0 or row_index >= len(frame):
            raise IndexError(f"row_index out of range: {row_index}")
