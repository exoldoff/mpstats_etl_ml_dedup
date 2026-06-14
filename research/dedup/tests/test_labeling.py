from __future__ import annotations

import pandas as pd

from research.dedup.labeling import LabelingSamplingConfig, stratified_labeling_sample


def _candidate(idx: int, *, cross_marketplace: bool, score: float = 0.8) -> dict[str, object]:
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
        "brand_a": "A",
        "brand_b": "A",
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
