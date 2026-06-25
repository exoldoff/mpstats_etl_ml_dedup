from __future__ import annotations

import pandas as pd

from research.dedup.clustering import (
    FAMILY_EDGE_LABELS,
    PACK_EDGE_LABELS,
    ComponentConfig,
    GraphGroupingConfig,
    add_component_flags,
    build_components,
    build_graph_groups,
    component_size_summary,
    same_pack_signature_mask,
)


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "left": "a",
                "right": "b",
                "label": "exact_duplicate",
                "unit_amount_a": 0.2,
                "unit_amount_b": 0.2,
                "total_amount_a": 0.2,
                "total_amount_b": 0.2,
                "multipack_count_a": 1,
                "multipack_count_b": 1,
            },
            {
                "left": "b",
                "right": "c",
                "label": "exact_duplicate",
                "unit_amount_a": 0.2,
                "unit_amount_b": 0.2,
                "total_amount_a": 0.2,
                "total_amount_b": 0.6,
                "multipack_count_a": 1,
                "multipack_count_b": 3,
            },
            {
                "left": "d",
                "right": "e",
                "label": "different_product",
                "unit_amount_a": 0.2,
                "unit_amount_b": 0.2,
                "total_amount_a": 0.2,
                "total_amount_b": 0.2,
                "multipack_count_a": 1,
                "multipack_count_b": 1,
            },
        ]
    )


def test_family_components_use_binary_positive_edges() -> None:
    config = ComponentConfig(left_id_col="left", right_id_col="right", label_col="label")

    components = build_components(_pairs(), edge_labels=FAMILY_EDGE_LABELS, config=config)
    component_by_node = components.set_index("node_id")["component_id"].to_dict()

    assert component_by_node["a"] == component_by_node["b"] == component_by_node["c"]
    assert component_by_node["d"] != component_by_node["e"]


def test_pack_components_use_positive_edges_with_same_pack_signature() -> None:
    config = ComponentConfig(left_id_col="left", right_id_col="right", label_col="label")

    pairs = _pairs()
    components = build_components(
        pairs,
        edge_labels=PACK_EDGE_LABELS,
        config=config,
        edge_mask=same_pack_signature_mask(pairs),
    )
    flagged = add_component_flags(pairs, components, config=config)

    assert bool(flagged.loc[0, "same_component"]) is True
    assert bool(flagged.loc[1, "same_component"]) is False
    assert bool(flagged.loc[2, "same_component"]) is False


def test_component_size_summary_sorts_largest_first() -> None:
    config = ComponentConfig(left_id_col="left", right_id_col="right", label_col="label")

    components = build_components(_pairs(), edge_labels=FAMILY_EDGE_LABELS, config=config)
    summary = component_size_summary(components)

    assert summary.iloc[0]["nodes"] == 3


def test_graph_grouping_connected_components_preserves_old_behavior() -> None:
    config = ComponentConfig(left_id_col="left", right_id_col="right", label_col="label")

    old_components = build_components(_pairs(), edge_labels=FAMILY_EDGE_LABELS, config=config)
    new_components = build_graph_groups(
        _pairs(),
        edge_labels=FAMILY_EDGE_LABELS,
        config=config,
        grouping_config=GraphGroupingConfig(algorithm="connected_components"),
    )

    assert new_components.equals(old_components)


def _weak_bridge_pairs() -> pd.DataFrame:
    rows = []
    for left, right, score in [
        ("a", "b", 0.99),
        ("a", "c", 0.99),
        ("b", "c", 0.99),
        ("d", "e", 0.99),
        ("d", "f", 0.99),
        ("e", "f", 0.99),
        ("c", "d", 0.05),
    ]:
        rows.append(
            {
                "left": left,
                "right": right,
                "label": "exact_duplicate",
                "score": score,
                "unit_amount_a": 0.2,
                "unit_amount_b": 0.2,
                "total_amount_a": 0.2,
                "total_amount_b": 0.2,
                "multipack_count_a": 1,
                "multipack_count_b": 1,
            }
        )
    return pd.DataFrame(rows)


def test_louvain_splits_dense_groups_connected_by_weak_bridge() -> None:
    config = ComponentConfig(left_id_col="left", right_id_col="right", label_col="label")
    pairs = _weak_bridge_pairs()

    connected = build_graph_groups(
        pairs,
        edge_labels=FAMILY_EDGE_LABELS,
        config=config,
        grouping_config=GraphGroupingConfig(algorithm="connected_components"),
    )
    louvain = build_graph_groups(
        pairs,
        edge_labels=FAMILY_EDGE_LABELS,
        config=config,
        grouping_config=GraphGroupingConfig(
            algorithm="louvain",
            edge_weight_col="score",
            resolution=1.0,
            seed=42,
        ),
    )

    connected_by_node = connected.set_index("node_id")["component_id"].to_dict()
    louvain_by_node = louvain.set_index("node_id")["component_id"].to_dict()

    assert len(set(connected_by_node.values())) == 1
    assert louvain_by_node["a"] == louvain_by_node["b"] == louvain_by_node["c"]
    assert louvain_by_node["d"] == louvain_by_node["e"] == louvain_by_node["f"]
    assert louvain_by_node["a"] != louvain_by_node["d"]


def test_pack_groups_do_not_cross_final_family_groups() -> None:
    family_config = ComponentConfig(left_id_col="left", right_id_col="right", label_col="label", component_col="family_id")
    pack_config = ComponentConfig(left_id_col="left", right_id_col="right", label_col="label", component_col="pack_id")
    pairs = _weak_bridge_pairs()
    family_groups = build_graph_groups(
        pairs,
        edge_labels=FAMILY_EDGE_LABELS,
        config=family_config,
        grouping_config=GraphGroupingConfig(algorithm="louvain", edge_weight_col="score", seed=42),
    )
    flagged_pairs = add_component_flags(
        pairs,
        family_groups,
        config=family_config,
        same_component_col="same_final_family",
    )
    pack_groups = build_graph_groups(
        flagged_pairs,
        edge_labels=PACK_EDGE_LABELS,
        config=pack_config,
        grouping_config=GraphGroupingConfig(algorithm="connected_components"),
        edge_mask=same_pack_signature_mask(flagged_pairs) & flagged_pairs["same_final_family"],
    )

    pack_by_node = pack_groups.set_index("node_id")["pack_id"].to_dict()

    assert pack_by_node["a"] == pack_by_node["b"] == pack_by_node["c"]
    assert pack_by_node["d"] == pack_by_node["e"] == pack_by_node["f"]
    assert pack_by_node["a"] != pack_by_node["d"]
