from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import pandas as pd


FAMILY_EDGE_LABELS = frozenset({"exact_duplicate"})
PACK_EDGE_LABELS = frozenset({"exact_duplicate"})


@dataclass(frozen=True)
class ComponentConfig:
    left_id_col: str = "raw_record_id_a"
    right_id_col: str = "raw_record_id_b"
    label_col: str = "predicted_label"
    component_col: str = "component_id"


@dataclass(frozen=True)
class PackSignatureConfig:
    unit_abs_tolerance: float = 0.02
    unit_rel_tolerance: float = 0.05
    total_abs_tolerance: float = 0.02
    total_rel_tolerance: float = 0.05
    pack_abs_tolerance: float = 0.25


@dataclass(frozen=True)
class GraphGroupingConfig:
    algorithm: str = "connected_components"
    edge_weight_col: str = "score"
    resolution: float = 1.0
    seed: int | None = 42
    default_edge_weight: float = 1.0


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
    edge_mask: pd.Series | None = None,
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
    if edge_mask is not None:
        aligned_mask = edge_mask.reindex(pairs.index, fill_value=False).astype(bool)
        edge_rows = edge_rows[aligned_mask.loc[edge_rows.index]]
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


def _to_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        if bool(value != value):
            return None
    except TypeError:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _valid_node(value: object) -> str | None:
    if pd.isna(value):
        return None
    node = str(value)
    return node if node else None


def _component_records_from_groups(
    groups: Iterable[Iterable[str]],
    *,
    component_col: str,
) -> pd.DataFrame:
    sorted_groups = sorted(
        (sorted(set(group)) for group in groups if group),
        key=lambda group: (group[0], len(group)),
    )
    records: list[dict[str, object]] = []
    for component_id, nodes in enumerate(sorted_groups, start=1):
        for node in nodes:
            records.append({"node_id": node, component_col: component_id})
    if not records:
        return pd.DataFrame(columns=["node_id", component_col])
    return pd.DataFrame(records).sort_values("node_id").reset_index(drop=True)


def _edge_rows(
    pairs: pd.DataFrame,
    *,
    labels: set[str],
    cfg: ComponentConfig,
    edge_mask: pd.Series | None,
) -> pd.DataFrame:
    edge_rows = pairs[pairs[cfg.label_col].isin(labels)]
    if edge_mask is not None:
        aligned_mask = edge_mask.reindex(pairs.index, fill_value=False).astype(bool)
        edge_rows = edge_rows[aligned_mask.loc[edge_rows.index]]
    return edge_rows


def _edge_weight(value: object, default: float) -> float:
    number = _to_number(value)
    return number if number is not None else default


def _weighted_edges(
    edge_rows: pd.DataFrame,
    *,
    cfg: ComponentConfig,
    grouping_cfg: GraphGroupingConfig,
) -> dict[tuple[str, str], float]:
    edges: dict[tuple[str, str], float] = {}
    has_weight_col = grouping_cfg.edge_weight_col in edge_rows.columns
    for row in edge_rows.itertuples(index=False):
        values = row._asdict()
        left = _valid_node(values[cfg.left_id_col])
        right = _valid_node(values[cfg.right_id_col])
        if left is None or right is None or left == right:
            continue
        edge = tuple(sorted((left, right)))
        weight = (
            _edge_weight(values[grouping_cfg.edge_weight_col], grouping_cfg.default_edge_weight)
            if has_weight_col
            else grouping_cfg.default_edge_weight
        )
        edges[edge] = max(edges.get(edge, 0.0), weight)
    return edges


def _louvain_components(
    pairs: pd.DataFrame,
    *,
    edge_labels: Iterable[str],
    config: ComponentConfig,
    grouping_config: GraphGroupingConfig,
    edge_mask: pd.Series | None,
) -> pd.DataFrame:
    if not math.isfinite(grouping_config.resolution) or grouping_config.resolution <= 0:
        raise ValueError("Louvain resolution must be a positive finite number")

    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError(
            "NetworkX is required for graph grouping algorithm='louvain'. "
            "Install research dependencies from requirements-research.txt."
        ) from exc

    labels = set(edge_labels)
    left = pairs[config.left_id_col].astype("string")
    right = pairs[config.right_id_col].astype("string")
    nodes = sorted(set(left.dropna()) | set(right.dropna()))
    edge_rows = _edge_rows(pairs, labels=labels, cfg=config, edge_mask=edge_mask)
    weighted_edges = _weighted_edges(edge_rows, cfg=config, grouping_cfg=grouping_config)

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for (left_node, right_node), weight in weighted_edges.items():
        graph.add_edge(left_node, right_node, weight=weight)

    if not weighted_edges:
        return _component_records_from_groups(({node} for node in nodes), component_col=config.component_col)

    communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        resolution=grouping_config.resolution,
        seed=grouping_config.seed,
    )
    covered = set().union(*(set(community) for community in communities)) if communities else set()
    missing_nodes = [node for node in nodes if node not in covered]
    groups = list(communities) + [{node} for node in missing_nodes]
    return _component_records_from_groups(groups, component_col=config.component_col)


def build_graph_groups(
    pairs: pd.DataFrame,
    *,
    edge_labels: Iterable[str],
    config: ComponentConfig | None = None,
    grouping_config: GraphGroupingConfig | None = None,
    edge_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Build graph groups from positive pair edges.

    ``connected_components`` preserves the historical union-find behavior.
    ``louvain`` can split a chained positive component into denser communities.
    """
    cfg = config or ComponentConfig()
    graph_cfg = grouping_config or GraphGroupingConfig()
    required = {cfg.left_id_col, cfg.right_id_col, cfg.label_col}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Missing columns for graph grouping: {missing}")

    algorithm = graph_cfg.algorithm.strip().lower()
    if algorithm in {"connected_components", "components"}:
        return build_components(pairs, edge_labels=edge_labels, config=cfg, edge_mask=edge_mask)
    if algorithm == "louvain":
        return _louvain_components(
            pairs,
            edge_labels=edge_labels,
            config=cfg,
            grouping_config=graph_cfg,
            edge_mask=edge_mask,
        )
    raise ValueError(f"Unknown graph grouping algorithm: {graph_cfg.algorithm!r}")


def _numbers_close(left: float | None, right: float | None, *, abs_tol: float, rel_tol: float) -> bool:
    if left is None or right is None:
        return False
    return abs(left - right) <= max(abs_tol, rel_tol * max(abs(left), abs(right)))


def _row_same_pack_signature(row: pd.Series, cfg: PackSignatureConfig) -> bool:
    unit_same = _numbers_close(
        _to_number(row.get("unit_amount_a", row.get("unit_weight_a"))),
        _to_number(row.get("unit_amount_b", row.get("unit_weight_b"))),
        abs_tol=cfg.unit_abs_tolerance,
        rel_tol=cfg.unit_rel_tolerance,
    )
    total_same = _numbers_close(
        _to_number(row.get("total_amount_a", row.get("total_weight_a"))),
        _to_number(row.get("total_amount_b", row.get("total_weight_b"))),
        abs_tol=cfg.total_abs_tolerance,
        rel_tol=cfg.total_rel_tolerance,
    )
    multipack_same = _numbers_close(
        _to_number(row.get("multipack_count_a", row.get("pack_count_a"))),
        _to_number(row.get("multipack_count_b", row.get("pack_count_b"))),
        abs_tol=cfg.pack_abs_tolerance,
        rel_tol=0.0,
    )
    return unit_same and (multipack_same or total_same)


def same_pack_signature_mask(
    pairs: pd.DataFrame,
    *,
    config: PackSignatureConfig | None = None,
) -> pd.Series:
    """Return rows where deterministic unit/total/multipack signatures match."""
    cfg = config or PackSignatureConfig()
    if pairs.empty:
        return pd.Series(dtype=bool, index=pairs.index)
    return pairs.apply(lambda row: _row_same_pack_signature(row, cfg), axis=1).astype(bool)


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
