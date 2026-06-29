from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class GraphGroupingConfig:
    algorithm: str = "leiden"
    edge_weight_col: str = "score"
    resolution: float = 0.1
    seed: int | None = 42
    default_edge_weight: float = 1.0


class _UnionFind:
    def __init__(self, nodes: Sequence[str]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        parent = self.parent.setdefault(node, node)
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


def normalize_graph_algorithm(value: object) -> str:
    algorithm = str(value or "").strip().casefold()
    return algorithm if algorithm in {"leiden", "connected_components"} else "leiden"


def build_family_components(
    *,
    node_ids: Sequence[str],
    edges: pd.DataFrame,
    previous_assignments: dict[str, dict[str, Any]],
    manual_singleton_nodes: Iterable[str],
    config: GraphGroupingConfig,
) -> dict[str, str]:
    clean_node_ids = [str(node_id) for node_id in node_ids if str(node_id)]
    manual_nodes = {str(node_id) for node_id in manual_singleton_nodes}
    active_nodes = [node_id for node_id in clean_node_ids if node_id not in manual_nodes]
    if not active_nodes:
        return {node_id: node_id for node_id in clean_node_ids}

    algorithm = normalize_graph_algorithm(config.algorithm)
    if algorithm == "connected_components":
        active_components = _connected_components(
            active_nodes,
            edges,
            previous_assignments=previous_assignments,
            config=config,
        )
    else:
        active_components = _leiden_components(
            active_nodes,
            edges,
            previous_assignments=previous_assignments,
            config=config,
        )

    components = {node_id: active_components.get(node_id, node_id) for node_id in active_nodes}
    components.update({node_id: node_id for node_id in clean_node_ids if node_id in manual_nodes})
    return components


def _connected_components(
    node_ids: Sequence[str],
    edges: pd.DataFrame,
    *,
    previous_assignments: dict[str, dict[str, Any]],
    config: GraphGroupingConfig,
) -> dict[str, str]:
    union_find = _UnionFind(node_ids)
    for left, right, _ in _iter_previous_family_edges(node_ids, previous_assignments, config=config):
        union_find.union(left, right)
    for left, right, _ in _iter_positive_edges(node_ids, edges, config=config):
        union_find.union(left, right)
    return {node_id: union_find.find(node_id) for node_id in node_ids}


def _leiden_components(
    node_ids: Sequence[str],
    edges: pd.DataFrame,
    *,
    previous_assignments: dict[str, dict[str, Any]],
    config: GraphGroupingConfig,
) -> dict[str, str]:
    if not math.isfinite(config.resolution) or config.resolution <= 0:
        raise ValueError("Leiden resolution must be a positive finite number")

    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise RuntimeError(
            "python-igraph and leidenalg are required for production ML-dedup graph grouping. "
            "Install dependencies from requirements.txt."
        ) from exc

    weighted_edges: dict[tuple[str, str], float] = {}
    for left, right, weight in (
        *_iter_previous_family_edges(node_ids, previous_assignments, config=config),
        *_iter_positive_edges(node_ids, edges, config=config),
    ):
        edge = tuple(sorted((left, right)))
        weighted_edges[edge] = max(weighted_edges.get(edge, 0.0), weight)

    if not weighted_edges:
        return {node_id: node_id for node_id in node_ids}

    ordered_nodes = sorted(set(node_ids))
    node_to_index = {node_id: index for index, node_id in enumerate(ordered_nodes)}
    edge_pairs = [(node_to_index[left], node_to_index[right]) for left, right in weighted_edges]
    weights = [weighted_edges[edge] for edge in weighted_edges]
    graph = ig.Graph(n=len(ordered_nodes), edges=edge_pairs, directed=False)
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=config.resolution,
        n_iterations=-1,
        seed=config.seed,
    )

    components: dict[str, str] = {}
    for community in partition:
        members = sorted(ordered_nodes[index] for index in community)
        if not members:
            continue
        component_key = members[0]
        for member in members:
            components[member] = component_key
    for node_id in ordered_nodes:
        components.setdefault(node_id, node_id)
    return components


def _iter_previous_family_edges(
    node_ids: Sequence[str],
    previous_assignments: dict[str, dict[str, Any]],
    *,
    config: GraphGroupingConfig,
) -> list[tuple[str, str, float]]:
    node_set = set(node_ids)
    family_members: dict[str, list[str]] = {}
    for node_id in node_ids:
        previous = previous_assignments.get(node_id)
        family_id = _clean_text(previous.get("ml_family_id")) if previous else ""
        if family_id:
            family_members.setdefault(family_id, []).append(node_id)

    rows: list[tuple[str, str, float]] = []
    for members in family_members.values():
        clean_members = [member for member in members if member in node_set]
        if len(clean_members) < 2:
            continue
        anchor = clean_members[0]
        for member in clean_members[1:]:
            rows.append((anchor, member, config.default_edge_weight))
    return rows


def _iter_positive_edges(
    node_ids: Sequence[str],
    edges: pd.DataFrame,
    *,
    config: GraphGroupingConfig,
) -> list[tuple[str, str, float]]:
    required = {"node_id_a", "node_id_b", "predicted_binary"}
    if edges.empty or not required.issubset(edges.columns):
        return []
    node_set = set(node_ids)
    has_weight_col = config.edge_weight_col in edges.columns
    rows: list[tuple[str, str, float]] = []
    positive_edges = edges[edges["predicted_binary"].astype(bool)]
    for row in positive_edges.itertuples(index=False):
        values = row._asdict()
        left = _valid_node(values.get("node_id_a"))
        right = _valid_node(values.get("node_id_b"))
        if left is None or right is None or left == right:
            continue
        if left not in node_set or right not in node_set:
            continue
        weight = _edge_weight(values.get(config.edge_weight_col), config.default_edge_weight) if has_weight_col else config.default_edge_weight
        rows.append((left, right, weight))
    return rows


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(value != value):
            return ""
    except TypeError:
        return ""
    return " ".join(str(value).strip().split())


def _valid_node(value: object) -> str | None:
    text = _clean_text(value)
    return text or None


def _edge_weight(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number <= 0:
        return default
    return number
