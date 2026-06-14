from __future__ import annotations

import pandas as pd

from research.dedup.annotator import (
    _item_summary,
    _signal_summary,
    apply_label,
    clear_label,
    labeled_count,
    next_unlabeled_index,
)


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


def test_item_summary_includes_marketplace_and_pack_fields() -> None:
    row = pd.Series(
        {
            "sku_a": "123",
            "marketplace_a": "Ozon",
            "brand_a": "Brand",
            "unit_amount_a": 0.5,
            "total_amount_a": 3.0,
            "multipack_count_a": 6,
        }
    )

    summary = _item_summary(row, "a")

    assert summary == "Ozon | sku 123 | brand Brand | unit 0.5 | total 3.0 | x6"


def test_signal_summary_uses_russian_flag_labels() -> None:
    row = pd.Series(
        {
            "candidate_source": "faiss_embedding_topk",
            "candidate_rank": 3,
            "embedding_similarity_score": 0.98,
            "labeling_stratum": "pack_variant_candidate",
            "is_cross_marketplace_pair": True,
            "is_hard_negative_candidate": False,
            "is_pack_variant_candidate": True,
        }
    )

    summary = _signal_summary(row)

    assert "источник=faiss_embedding_topk" in summary
    assert "межмаркетплейс=да" in summary
    assert "сложный негатив=нет" in summary
    assert "вариант упаковки=да" in summary
