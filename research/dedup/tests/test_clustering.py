from __future__ import annotations

import pandas as pd

from research.dedup.clustering import (
    FAMILY_EDGE_LABELS,
    PACK_EDGE_LABELS,
    ComponentConfig,
    add_component_flags,
    build_components,
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
