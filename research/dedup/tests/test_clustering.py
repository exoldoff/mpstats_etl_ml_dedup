from __future__ import annotations

import pandas as pd

from research.dedup.clustering import (
    FAMILY_EDGE_LABELS,
    PACK_EDGE_LABELS,
    ComponentConfig,
    add_component_flags,
    build_components,
    component_size_summary,
)


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"left": "a", "right": "b", "label": "exact_duplicate"},
            {"left": "b", "right": "c", "label": "same_product_different_pack"},
            {"left": "d", "right": "e", "label": "different_product"},
        ]
    )


def test_family_components_use_exact_and_pack_variant_edges() -> None:
    config = ComponentConfig(left_id_col="left", right_id_col="right", label_col="label")

    components = build_components(_pairs(), edge_labels=FAMILY_EDGE_LABELS, config=config)
    component_by_node = components.set_index("node_id")["component_id"].to_dict()

    assert component_by_node["a"] == component_by_node["b"] == component_by_node["c"]
    assert component_by_node["d"] != component_by_node["e"]


def test_pack_components_use_only_exact_edges() -> None:
    config = ComponentConfig(left_id_col="left", right_id_col="right", label_col="label")

    components = build_components(_pairs(), edge_labels=PACK_EDGE_LABELS, config=config)
    flagged = add_component_flags(_pairs(), components, config=config)

    assert bool(flagged.loc[0, "same_component"]) is True
    assert bool(flagged.loc[1, "same_component"]) is False
    assert bool(flagged.loc[2, "same_component"]) is False


def test_component_size_summary_sorts_largest_first() -> None:
    config = ComponentConfig(left_id_col="left", right_id_col="right", label_col="label")

    components = build_components(_pairs(), edge_labels=FAMILY_EDGE_LABELS, config=config)
    summary = component_size_summary(components)

    assert summary.iloc[0]["nodes"] == 3
