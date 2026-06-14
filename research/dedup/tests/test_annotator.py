from __future__ import annotations

import pandas as pd

from research.dedup.annotator import apply_label, clear_label, labeled_count, next_unlabeled_index


def test_apply_label_and_count() -> None:
    frame = pd.DataFrame({"label": ["", ""], "title_a": ["a", "b"], "title_b": ["c", "d"]})

    label = apply_label(frame, 0, "w")

    assert label == "exact_duplicate"
    assert frame.at[0, "label"] == "exact_duplicate"
    assert labeled_count(frame) == 1


def test_next_unlabeled_wraps_from_start() -> None:
    frame = pd.DataFrame({"label": ["exact_duplicate", "", "different_product", ""]})

    assert next_unlabeled_index(frame, 2) == 3
    assert next_unlabeled_index(frame, 4) == 3
    assert next_unlabeled_index(frame, 0) == 1


def test_clear_label() -> None:
    frame = pd.DataFrame({"label": ["uncertain"]})

    clear_label(frame, 0)

    assert frame.at[0, "label"] == ""
