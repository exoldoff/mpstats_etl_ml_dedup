from __future__ import annotations

import pandas as pd
import pytest

from research.dedup.fusion import FusionRun, prepare_fusion_pair_edges, select_fusion_run


def test_select_fusion_run_prefers_weighted_strategy_then_ranks_methods_by_cost() -> None:
    summary = pd.DataFrame(
        [
            {
                "method": "rule_based_fuzzy",
                "split": "dev",
                "threshold_strategy": "threshold_cost_sensitive",
                "threshold_same": 0.91,
                "cost": 2,
                "false_merge_count": 1,
                "false_split_count": 1,
                "f1": 0.90,
                "weighted_total_cost": 500.0,
                "weighted_false_merge_cost": 10.0,
                "weighted_f1": 0.90,
            },
            {
                "method": "reranker_bge_v2_m3",
                "split": "dev",
                "threshold_strategy": "threshold_weighted_cost",
                "threshold_same": 0.42,
                "cost": 1,
                "false_merge_count": 0,
                "false_split_count": 1,
                "f1": 0.72,
                "weighted_total_cost": 520.0,
                "weighted_false_merge_cost": 20.0,
                "weighted_f1": 0.80,
            },
            {
                "method": "rule_based_fuzzy",
                "split": "dev",
                "threshold_strategy": "threshold_weighted_cost",
                "threshold_same": 0.91,
                "cost": 5,
                "false_merge_count": 1,
                "false_split_count": 0,
                "f1": 0.60,
                "weighted_total_cost": 20.0,
                "weighted_false_merge_cost": 5.0,
                "weighted_f1": 0.60,
            },
            {
                "method": "reranker_bge_v2_m3",
                "split": "test",
                "threshold_strategy": "threshold_weighted_cost",
                "threshold_same": 0.01,
                "cost": 0,
                "false_merge_count": 0,
                "false_split_count": 0,
                "f1": 1.0,
                "weighted_total_cost": 0.0,
                "weighted_false_merge_cost": 0.0,
                "weighted_f1": 1.0,
            },
        ]
    )

    run = select_fusion_run(summary)

    assert run == FusionRun(
        method="reranker_bge_v2_m3",
        threshold_strategy="threshold_weighted_cost",
        threshold_same=0.42,
        selection_split="dev",
    )


def test_select_fusion_run_falls_back_to_cost_sensitive_without_weighted_rows() -> None:
    summary = pd.DataFrame(
        [
            {
                "method": "rule_based_fuzzy",
                "split": "dev",
                "threshold_strategy": "threshold_max_f1",
                "threshold_same": 0.50,
                "cost": 20,
                "false_merge_count": 4,
                "false_split_count": 0,
                "f1": 0.90,
            },
            {
                "method": "reranker_bge_v2_m3",
                "split": "dev",
                "threshold_strategy": "threshold_cost_sensitive",
                "threshold_same": 0.88,
                "cost": 12,
                "false_merge_count": 0,
                "false_split_count": 12,
                "f1": 0.60,
            },
        ]
    )

    run = select_fusion_run(summary)

    assert run == FusionRun(
        method="reranker_bge_v2_m3",
        threshold_strategy="threshold_cost_sensitive",
        threshold_same=0.88,
        selection_split="dev",
    )


def test_select_fusion_run_supports_method_and_strategy_override() -> None:
    summary = pd.DataFrame(
        [
            {
                "method": "rule_based_fuzzy",
                "split": "dev",
                "threshold_strategy": "threshold_max_f1",
                "threshold_same": 0.80,
                "cost": 100,
                "false_merge_count": 20,
                "false_split_count": 0,
                "f1": 0.90,
            },
            {
                "method": "rule_based_fuzzy",
                "split": "dev",
                "threshold_strategy": "threshold_cost_sensitive",
                "threshold_same": 0.95,
                "cost": 10,
                "false_merge_count": 0,
                "false_split_count": 10,
                "f1": 0.40,
            },
        ]
    )

    run = select_fusion_run(summary, method="rule_based_fuzzy", threshold_strategy="threshold_max_f1")

    assert run.method == "rule_based_fuzzy"
    assert run.threshold_strategy == "threshold_max_f1"
    assert run.threshold_same == 0.80


def test_select_fusion_run_errors_when_override_is_missing() -> None:
    summary = pd.DataFrame(
        [
            {
                "method": "rule_based_fuzzy",
                "split": "dev",
                "threshold_strategy": "threshold_cost_sensitive",
                "threshold_same": 0.90,
            }
        ]
    )

    with pytest.raises(ValueError, match="method='missing'"):
        select_fusion_run(summary, method="missing")


def test_prepare_fusion_pair_edges_keeps_family_and_pack_separate() -> None:
    predictions = pd.DataFrame(
        [
            {
                "method": "m",
                "split": "test",
                "threshold_strategy": "threshold_cost_sensitive",
                "threshold_same": 0.7,
                "predicted_binary": 1,
                "same_base_product": 1,
                "raw_record_id_a": "wb::1",
                "raw_record_id_b": "ozon::2",
                "unit_amount_a": 0.2,
                "unit_amount_b": 0.2,
                "total_amount_a": 0.2,
                "total_amount_b": 0.6,
                "multipack_count_a": 1,
                "multipack_count_b": 3,
            },
            {
                "method": "m",
                "split": "test",
                "threshold_strategy": "threshold_cost_sensitive",
                "threshold_same": 0.7,
                "predicted_binary": 1,
                "same_base_product": 1,
                "raw_record_id_a": "wb::3",
                "raw_record_id_b": "ozon::4",
                "unit_amount_a": 0.2,
                "unit_amount_b": 0.2,
                "total_amount_a": 0.2,
                "total_amount_b": 0.2,
                "multipack_count_a": 1,
                "multipack_count_b": 1,
            },
            {
                "method": "m",
                "split": "test",
                "threshold_strategy": "threshold_cost_sensitive",
                "threshold_same": 0.7,
                "predicted_binary": 0,
                "same_base_product": 0,
                "raw_record_id_a": "wb::5",
                "raw_record_id_b": "ozon::6",
                "unit_amount_a": 0.2,
                "unit_amount_b": 0.2,
                "total_amount_a": 0.2,
                "total_amount_b": 0.2,
                "multipack_count_a": 1,
                "multipack_count_b": 1,
            },
        ]
    )

    edges = prepare_fusion_pair_edges(
        predictions,
        FusionRun(method="m", threshold_strategy="threshold_cost_sensitive", threshold_same=0.7),
    )

    assert edges["fusion_family_edge"].tolist() == [True, True, False]
    assert edges["same_pack_signature"].tolist() == [False, True, True]
    assert edges["fusion_pack_edge"].tolist() == [False, True, False]
    assert edges["true_pack_edge"].tolist() == [False, True, False]
    assert edges["predicted_family_label"].tolist() == [
        "exact_duplicate",
        "exact_duplicate",
        "different_product",
    ]
    assert edges["predicted_pack_label"].tolist() == [
        "different_product",
        "exact_duplicate",
        "different_product",
    ]
