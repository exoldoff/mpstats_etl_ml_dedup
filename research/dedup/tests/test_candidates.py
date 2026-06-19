from __future__ import annotations

import pandas as pd

from research.dedup import (
    CandidateGenerationConfig,
    add_hard_negative_flags,
    classification_report_df,
    generate_candidate_pairs,
    normalize_title,
    prepare_product_records,
)


def test_normalize_title_removes_punctuation_and_lowercases() -> None:
    assert normalize_title("Соус TABASCO PEPPER / Красный, 60мл") == "соус tabasco pepper красный 60мл"


def test_candidate_generation_keeps_brand_as_feature_not_filter() -> None:
    df = pd.DataFrame(
        [
            {"Артикул": 1, "SKU": "Соус томатный острый 200 г", "Бренд": "A", "Вес, кг (ед.)": 0.2, "Вес, кг": 0.2},
            {"Артикул": 2, "SKU": "Соус томатный острый 200 г", "Бренд": "B", "Вес, кг (ед.)": 0.2, "Вес, кг": 0.2},
            {"Артикул": 3, "SKU": "Уксус бальзамический 250 мл", "Бренд": "A", "Вес, кг (ед.)": 0.25, "Вес, кг": 0.25},
        ]
    )
    pairs = generate_candidate_pairs(df, CandidateGenerationConfig(min_similarity=0.4, max_candidates=None))
    matching_pair = pairs[(pairs["sku_a"] == "1") & (pairs["sku_b"] == "2")]
    assert not matching_pair.empty
    assert matching_pair.iloc[0]["brand_a"] == "A"
    assert matching_pair.iloc[0]["brand_b"] == "B"


def test_product_records_use_marketplace_article_key_not_article_only() -> None:
    df = pd.DataFrame(
        [
            {
                "Маркетплейс": "WB",
                "Артикул": "100",
                "месяц": "2026-01",
                "SKU": "Соус томатный острый 200 г",
                "Бренд": "A",
                "Вес, кг (ед.)": 0.2,
                "Вес, кг": 0.2,
            },
            {
                "Маркетплейс": "WB",
                "Артикул": "100",
                "месяц": "2026-02",
                "SKU": "Соус томатный острый 200 г",
                "Бренд": "A",
                "Вес, кг (ед.)": 0.2,
                "Вес, кг": 0.2,
            },
            {
                "Маркетплейс": "Ozon",
                "Артикул": "100",
                "месяц": "2026-01",
                "SKU": "Соус томатный острый 200 г",
                "Бренд": "A",
                "Вес, кг (ед.)": 0.2,
                "Вес, кг": 0.2,
            },
        ]
    )

    config = CandidateGenerationConfig(
        min_similarity=0.4,
        max_candidates=None,
        collapse_empty_brand_exact_titles=False,
        collapse_exact_title_same_brand=False,
    )
    records = prepare_product_records(df, config)
    pairs = generate_candidate_pairs(df, config)

    assert len(records) == 2
    assert set(records["raw_record_id"]) == {"wb::100", "ozon::100"}
    assert not pairs.empty
    assert bool(pairs.iloc[0]["is_cross_marketplace_pair"]) is True
    assert pairs.iloc[0]["sku_a"] == pairs.iloc[0]["sku_b"] == "100"


def test_empty_brand_exact_title_records_collapse_before_embeddings() -> None:
    df = pd.DataFrame(
        [
            {"Маркетплейс": "WB", "Артикул": "100", "SKU": "Мыло детское 90 г", "Бренд": ""},
            {"Маркетплейс": "Ozon", "Артикул": "200", "SKU": "мыло  детское 90 г", "Бренд": None},
            {"Маркетплейс": "WB", "Артикул": "300", "SKU": "Мыло хозяйственное 180 г", "Бренд": ""},
        ]
    )

    records = prepare_product_records(df)

    assert len(records) == 2
    collapsed = records[records["title_norm"].eq("мыло детское 90 г")].iloc[0]
    assert collapsed["raw_record_id"].startswith("exact_title::")
    assert collapsed["collapsed_record_count"] == 2
    assert set(collapsed["source_raw_record_ids"]) == {"wb::100", "ozon::200"}
    assert set(collapsed["source_skus"]) == {"100", "200"}
    assert set(collapsed["marketplaces"]) == {"WB", "Ozon"}


def test_empty_brand_exact_title_collapse_can_be_disabled() -> None:
    df = pd.DataFrame(
        [
            {"Маркетплейс": "WB", "Артикул": "100", "SKU": "Мыло детское 90 г", "Бренд": ""},
            {"Маркетплейс": "Ozon", "Артикул": "200", "SKU": "мыло  детское 90 г", "Бренд": ""},
        ]
    )

    records = prepare_product_records(
        df,
        CandidateGenerationConfig(
            collapse_empty_brand_exact_titles=False,
            collapse_exact_title_same_brand=False,
        ),
    )

    assert len(records) == 2
    assert set(records["raw_record_id"]) == {"wb::100", "ozon::200"}


def test_exact_title_records_with_same_brand_collapse_before_embeddings() -> None:
    df = pd.DataFrame(
        [
            {"Маркетплейс": "WB", "Артикул": "100", "SKU": "Мыло детское 90 г", "Бренд": "A"},
            {"Маркетплейс": "Ozon", "Артикул": "200", "SKU": "мыло  детское 90 г", "Бренд": "A"},
            {"Маркетплейс": "WB", "Артикул": "300", "SKU": "Мыло хозяйственное 180 г", "Бренд": "A"},
        ]
    )

    records = prepare_product_records(df)

    assert len(records) == 2
    collapsed = records[records["title_norm"].eq("мыло детское 90 г")].iloc[0]
    assert collapsed["raw_record_id"].startswith("exact_title::")
    assert collapsed["collapsed_record_count"] == 2
    assert set(collapsed["source_raw_record_ids"]) == {"wb::100", "ozon::200"}


def test_exact_title_records_with_different_brands_stay_separate() -> None:
    df = pd.DataFrame(
        [
            {"Маркетплейс": "WB", "Артикул": "100", "SKU": "Мыло детское 90 г", "Бренд": "A"},
            {"Маркетплейс": "Ozon", "Артикул": "200", "SKU": "мыло  детское 90 г", "Бренд": "B"},
        ]
    )

    records = prepare_product_records(df)

    assert len(records) == 2
    assert set(records["raw_record_id"]) == {"wb::100", "ozon::200"}


def test_hard_negative_same_brand_weight_different_flavor() -> None:
    pairs = pd.DataFrame(
        [
            {
                "sku_a": "1",
                "sku_b": "2",
                "title_a": "Соус соевый натуральный 250 мл",
                "title_b": "Соус чили сладкий 250 мл",
                "brand_a": "Sen Soy",
                "brand_b": "sen soy",
                "unit_amount_a": 0.25,
                "unit_amount_b": 0.25,
                "total_amount_a": 0.25,
                "total_amount_b": 0.25,
                "multipack_count_a": 1,
                "multipack_count_b": 1,
                "baseline_similarity_score": 0.5,
            }
        ]
    )
    marked = add_hard_negative_flags(pairs)
    assert bool(marked.loc[0, "is_hard_negative_candidate"]) is True


def test_classification_report_scaffold() -> None:
    report = classification_report_df(
        ["exact_duplicate", "different_product"],
        ["exact_duplicate", "exact_duplicate"],
    )
    exact_row = report[report["label"] == "exact_duplicate"].iloc[0]
    assert exact_row["precision"] == 0.5
    assert exact_row["recall"] == 1.0
