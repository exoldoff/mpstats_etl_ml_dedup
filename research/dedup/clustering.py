from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


FAMILY_EDGE_LABELS = frozenset({"exact_duplicate", "same_product_different_pack"})
PACK_EDGE_LABELS = frozenset({"exact_duplicate"})


@dataclass(frozen=True)
class ComponentConfig:
    left_id_col: str = "raw_record_id_a"
    right_id_col: str = "raw_record_id_b"
    label_col: str = "predicted_label"
    component_col: str = "component_id"


class _UnionFind:
    def __init__(self, nodes: Iterable[str]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        if node not in self.parent:
            self.parent[node] = node
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


def build_components(
    pairs: pd.DataFrame,
    *,
    edge_labels: Iterable[str],
    config: ComponentConfig | None = None,
) -> pd.DataFrame:
    """Build connected components from pair rows whose label is an edge."""
    cfg = config or ComponentConfig()
    required = {cfg.left_id_col, cfg.right_id_col, cfg.label_col}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Missing columns for component build: {missing}")

    labels = set(edge_labels)
    left = pairs[cfg.left_id_col].astype("string")
    right = pairs[cfg.right_id_col].astype("string")
    nodes = sorted(set(left.dropna()) | set(right.dropna()))
    union_find = _UnionFind(nodes)

    edge_rows = pairs[pairs[cfg.label_col].isin(labels)]
    for row in edge_rows[[cfg.left_id_col, cfg.right_id_col]].itertuples(index=False):
        if pd.isna(row[0]) or pd.isna(row[1]):
            continue
        union_find.union(str(row[0]), str(row[1]))

    root_to_id: dict[str, int] = {}
    records: list[dict[str, object]] = []
    for node in nodes:
        root = union_find.find(node)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id) + 1
        records.append({"node_id": node, cfg.component_col: root_to_id[root]})
    return pd.DataFrame(records)


def add_component_flags(
    pairs: pd.DataFrame,
    components: pd.DataFrame,
    *,
    config: ComponentConfig | None = None,
    same_component_col: str = "same_component",
) -> pd.DataFrame:
    """Attach component ids for both sides and mark whether a pair is linked."""
    cfg = config or ComponentConfig()
    if "node_id" not in components.columns or cfg.component_col not in components.columns:
        raise ValueError("components must contain node_id and the configured component column")

    mapping = components.set_index("node_id")[cfg.component_col]
    output = pairs.copy()
    left_component_col = f"{cfg.component_col}_a"
    right_component_col = f"{cfg.component_col}_b"
    output[left_component_col] = output[cfg.left_id_col].astype("string").map(mapping)
    output[right_component_col] = output[cfg.right_id_col].astype("string").map(mapping)
    output[same_component_col] = output[left_component_col].eq(output[right_component_col])
    return output


def component_size_summary(
    components: pd.DataFrame,
    *,
    component_col: str = "component_id",
) -> pd.DataFrame:
    """Return compact component-size distribution for notebook reporting."""
    if components.empty:
        return pd.DataFrame(columns=[component_col, "nodes"])
    return (
        components.groupby(component_col, as_index=False)
        .size()
        .rename(columns={"size": "nodes"})
        .sort_values(["nodes", component_col], ascending=[False, True])
        .reset_index(drop=True)
    )
