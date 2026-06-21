from __future__ import annotations

import pandas as pd

from research.dedup.labeling import (
    LabelingSamplingConfig,
    labeling_score_strata_masks,
    merge_preserved_labeling_rows,
    select_preserved_labeling_rows,
    split_labeling_target_size,
    stratified_labeling_sample,
)


def _candidate(
    idx: int,
    *,
    cross_marketplace: bool,
    score: float = 0.8,
    brand_a: str = "A",
    brand_b: str = "A",
) -> dict[str, object]:
    left_marketplace = "WB"
    right_marketplace = "Ozon" if cross_marketplace else "WB"
    return {
        "raw_record_id_a": f"{left_marketplace.casefold()}::{idx}",
        "raw_record_id_b": f"{right_marketplace.casefold()}::{idx}",
        "marketplace_a": left_marketplace,
        "marketplace_b": right_marketplace,
        "sku_a": str(idx),
        "sku_b": str(idx),
        "title_a": f"Соус томатный острый {idx} 200 г",
        "title_b": f"Соус томатный острый {idx} 200 г",
        "brand_a": brand_a,
        "brand_b": brand_b,
        "unit_amount_a": 0.2,
        "unit_amount_b": 0.2,
        "total_amount_a": 0.2,
        "total_amount_b": 0.2,
        "multipack_count_a": 1,
        "multipack_count_b": 1,
        "baseline_similarity_score": score,
        "is_hard_negative_candidate": False,
        "is_cross_marketplace_pair": cross_marketplace,
    }


def test_stratified_labeling_sample_has_cross_marketplace_stratum() -> None:
    candidates = pd.DataFrame(
        [_candidate(idx, cross_marketplace=idx < 6) for idx in range(20)]
    )

    labeling = stratified_labeling_sample(
        candidates,
        LabelingSamplingConfig(target_size=10, random_state=7),
    )

    cross_rows = labeling[labeling["labeling_stratum"] == "cross_marketplace_candidate"]
    assert len(cross_rows) == 2
    assert cross_rows["is_cross_marketplace_pair"].all()
    assert labeling["label"].fillna("").eq("").all()


def test_score_strata_use_relative_buckets_for_embedding_cosine_scores() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(idx, cross_marketplace=False, score=0.90 + idx * 0.001)
            for idx in range(100)
        ]
    )

    masks = labeling_score_strata_masks(candidates, LabelingSamplingConfig())

    assert masks["high_similarity"].sum() == 25
    assert masks["medium_similarity"].sum() == 50
    assert masks["random_easy_negative"].sum() == 25

    high_scores = candidates.loc[masks["high_similarity"], "baseline_similarity_score"]
    medium_scores = candidates.loc[masks["medium_similarity"], "baseline_similarity_score"]
    easy_scores = candidates.loc[masks["random_easy_negative"], "baseline_similarity_score"]
    assert high_scores.min() > medium_scores.max()
    assert medium_scores.min() > easy_scores.max()


def test_stratified_labeling_sample_keeps_medium_and_low_score_controls() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(idx, cross_marketplace=False, score=0.90 + idx * 0.001)
            for idx in range(100)
        ]
    )

    labeling = stratified_labeling_sample(
        candidates,
        LabelingSamplingConfig(target_size=20, random_state=7),
    )

    assert "medium_similarity" in set(labeling["labeling_stratum"])
    assert "random_easy_negative" in set(labeling["labeling_stratum"])


def test_stratified_labeling_sample_caps_different_brand_pairs() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(
                idx,
                cross_marketplace=False,
                score=0.90 + idx * 0.001,
                brand_a="A",
                brand_b="B" if idx < 60 else "A",
            )
            for idx in range(100)
        ]
    )

    labeling = stratified_labeling_sample(
        candidates,
        LabelingSamplingConfig(target_size=50, random_state=7, max_different_brand_share=0.20),
    )

    assert len(labeling) == 50
    assert labeling["is_different_brand_pair"].sum() <= 10
    assert set(labeling["brand_relation"]).issubset({"same_brand", "different_brand", "unknown_brand"})


def test_merge_preserved_labeling_rows_keeps_existing_labels_and_avoids_duplicates() -> None:
    existing = pd.DataFrame(
        [
            _candidate(1, cross_marketplace=True) | {"label": "exact_duplicate", "notes": "checked"},
            _candidate(2, cross_marketplace=True) | {"label": "", "notes": ""},
        ]
    )
    fresh = pd.DataFrame(
        [
            _candidate(1, cross_marketplace=True) | {"label": "", "notes": ""},
            _candidate(3, cross_marketplace=True) | {"label": "", "notes": ""},
            _candidate(4, cross_marketplace=True) | {"label": "", "notes": ""},
        ]
    )

    preserved = select_preserved_labeling_rows(existing)
    merged = merge_preserved_labeling_rows(
        preserved,
        fresh,
        target_size=3,
        random_state=11,
    )

    assert "exact_duplicate" in set(merged["label"])
    assert len(merged) == 3
    assert merged["raw_record_id_a"].eq("wb::1").sum() == 1


def test_split_labeling_target_size_balances_category_runs() -> None:
    assert split_labeling_target_size(3000, 3) == [1000, 1000, 1000]
    assert split_labeling_target_size(3002, 3) == [1001, 1001, 1000]
