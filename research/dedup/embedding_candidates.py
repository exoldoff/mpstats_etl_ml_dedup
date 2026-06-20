from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .candidates import (
    CandidateGenerationConfig,
    add_cross_marketplace_flags,
    add_hard_negative_flags,
    add_pack_variant_flags,
    subcategory_relation,
)


FAISS_CANDIDATE_OUTPUT_COLUMNS = [
    "raw_record_id_a",
    "raw_record_id_b",
    "marketplace_a",
    "marketplace_b",
    "marketplaces_a",
    "marketplaces_b",
    "sku_a",
    "sku_b",
    "title_a",
    "title_b",
    "brand_a",
    "brand_b",
    "subcategory_a",
    "subcategory_b",
    "subcategory_relation",
    "unit_amount_a",
    "unit_amount_b",
    "total_amount_a",
    "total_amount_b",
    "multipack_count_a",
    "multipack_count_b",
    "embedding_similarity_score",
    "candidate_rank",
    "candidate_source",
    "blocking_scope",
    "baseline_similarity_score",
    "is_cross_marketplace_pair",
    "is_hard_negative_candidate",
    "is_pack_variant_candidate",
]


@dataclass(frozen=True)
class FaissCandidateGenerationConfig:
    """Configuration for embedding top-k blocking with FAISS."""

    top_k: int = 30
    max_candidates: int | None = 60_000
    min_similarity: float | None = None
    normalize_vectors: bool = True
    subcategory_blocking: bool = True
    global_safety_top_k: int = 5
    unknown_subcategory_top_k: int = 30
    candidate_features: CandidateGenerationConfig = CandidateGenerationConfig()


def _load_faiss(faiss_module: Any | None = None) -> Any:
    if faiss_module is not None:
        return faiss_module
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - depends on local research env
        raise ImportError(
            "FAISS is required for embedding candidate generation. "
            "Install research dependencies with: python3 -m pip install -r requirements-research.txt"
        ) from exc
    return faiss


def _normalize_l2(vectors: np.ndarray, faiss_module: Any) -> np.ndarray:
    normalized = np.ascontiguousarray(vectors, dtype="float32").copy()
    if hasattr(faiss_module, "normalize_L2"):
        faiss_module.normalize_L2(normalized)
        return normalized

    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return normalized / norms


def _marketplaces_value(row: pd.Series) -> object:
    value = row.get("marketplaces")
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return value


def _candidate_row(
    left: pd.Series,
    right: pd.Series,
    *,
    score: float,
    rank: int,
    blocking_scope: str,
) -> dict[str, object]:
    return {
        "raw_record_id_a": left["raw_record_id"],
        "raw_record_id_b": right["raw_record_id"],
        "marketplace_a": left["marketplace"],
        "marketplace_b": right["marketplace"],
        "marketplaces_a": _marketplaces_value(left),
        "marketplaces_b": _marketplaces_value(right),
        "sku_a": left["sku"],
        "sku_b": right["sku"],
        "title_a": left["title"],
        "title_b": right["title"],
        "brand_a": left["brand"],
        "brand_b": right["brand"],
        "subcategory_a": left.get("subcategory", ""),
        "subcategory_b": right.get("subcategory", ""),
        "subcategory_relation": subcategory_relation(
            left.get("subcategory_norm", left.get("subcategory", "")),
            right.get("subcategory_norm", right.get("subcategory", "")),
        ),
        "unit_amount_a": left["unit_amount"],
        "unit_amount_b": right["unit_amount"],
        "total_amount_a": left["total_amount"],
        "total_amount_b": right["total_amount"],
        "multipack_count_a": left["multipack_count"],
        "multipack_count_b": right["multipack_count"],
        "embedding_similarity_score": round(score, 6),
        "candidate_rank": rank,
        "candidate_source": "faiss_embedding_topk",
        "blocking_scope": blocking_scope,
        "baseline_similarity_score": round(score, 6),
    }


def _scope_priority(scope: str) -> int:
    return {
        "same_subcategory": 4,
        "unknown_subcategory": 3,
        "global_safety": 2,
        "global": 1,
    }.get(scope, 0)


def _should_replace_pair(
    existing: dict[str, object] | None,
    *,
    score: float,
    rank: int,
    blocking_scope: str,
) -> bool:
    if existing is None:
        return True
    existing_score = float(existing["embedding_similarity_score"])
    rounded_score = round(score, 6)
    if rounded_score > existing_score:
        return True
    if rounded_score < existing_score:
        return False
    existing_priority = _scope_priority(str(existing.get("blocking_scope", "")))
    new_priority = _scope_priority(blocking_scope)
    if new_priority > existing_priority:
        return True
    if new_priority < existing_priority:
        return False
    return rank < int(existing["candidate_rank"])


def _add_scope_pairs(
    *,
    pair_rows: dict[tuple[int, int], dict[str, object]],
    records: pd.DataFrame,
    index_vectors: np.ndarray,
    query_indices: list[int],
    scope_indices: list[int],
    search_top_k: int,
    blocking_scope: str,
    min_similarity: float | None,
    faiss_module: Any,
) -> None:
    if search_top_k <= 0 or len(scope_indices) < 2 or not query_indices:
        return

    dimension = index_vectors.shape[1]
    index = faiss_module.IndexFlatIP(dimension)
    scope_array = np.asarray(scope_indices, dtype="int64")
    index.add(np.ascontiguousarray(index_vectors[scope_array], dtype="float32"))

    query_array = np.asarray(query_indices, dtype="int64")
    has_self_hits = bool(set(query_indices) & set(scope_indices))
    search_k = min(len(scope_indices), search_top_k + (1 if has_self_hits else 0))
    scores, local_indices = index.search(
        np.ascontiguousarray(index_vectors[query_array], dtype="float32"),
        search_k,
    )

    for left_idx, row_scores, row_local_indices in zip(query_indices, scores, local_indices, strict=True):
        neighbor_rank = 0
        for score, local_right_idx in zip(row_scores, row_local_indices, strict=True):
            local_right_idx = int(local_right_idx)
            if local_right_idx < 0:
                continue
            right_idx = int(scope_array[local_right_idx])
            if right_idx == left_idx:
                continue
            score_value = float(score)
            if min_similarity is not None and score_value < min_similarity:
                continue
            neighbor_rank += 1
            if neighbor_rank > search_top_k:
                break
            pair_key = tuple(sorted((left_idx, right_idx)))
            existing = pair_rows.get(pair_key)
            if not _should_replace_pair(
                existing,
                score=score_value,
                rank=neighbor_rank,
                blocking_scope=blocking_scope,
            ):
                continue
            left = records.iloc[pair_key[0]]
            right = records.iloc[pair_key[1]]
            pair_rows[pair_key] = _candidate_row(
                left,
                right,
                score=score_value,
                rank=neighbor_rank,
                blocking_scope=blocking_scope,
            )


def generate_faiss_candidate_pairs(
    product_records: pd.DataFrame,
    embeddings: np.ndarray,
    config: FaissCandidateGenerationConfig | None = None,
    *,
    faiss_module: Any | None = None,
) -> pd.DataFrame:
    """Generate candidate pairs by searching top-k embedding neighbors in FAISS.

    ``product_records`` should come from ``prepare_product_records`` and
    ``embeddings`` must be aligned row-by-row with that frame. The returned
    ``baseline_similarity_score`` intentionally mirrors the FAISS cosine score
    so downstream labeling notebooks can keep their existing score contract.
    """
    cfg = config or FaissCandidateGenerationConfig()
    records = product_records.reset_index(drop=True).copy()
    vectors = np.asarray(embeddings, dtype="float32")

    if len(records) != len(vectors):
        raise ValueError(
            f"records/embeddings length mismatch: {len(records)} records, {len(vectors)} embeddings"
        )
    if vectors.ndim != 2:
        raise ValueError(f"embeddings must be a 2D array, got shape {vectors.shape}")
    if len(records) < 2:
        return pd.DataFrame(columns=FAISS_CANDIDATE_OUTPUT_COLUMNS)
    if cfg.top_k < 1:
        raise ValueError("top_k must be >= 1")
    if cfg.global_safety_top_k < 0:
        raise ValueError("global_safety_top_k must be >= 0")
    if cfg.unknown_subcategory_top_k < 0:
        raise ValueError("unknown_subcategory_top_k must be >= 0")

    faiss = _load_faiss(faiss_module)
    index_vectors = _normalize_l2(vectors, faiss) if cfg.normalize_vectors else np.ascontiguousarray(vectors)

    pair_rows: dict[tuple[int, int], dict[str, object]] = {}
    all_indices = list(range(len(records)))
    has_subcategory = (
        cfg.subcategory_blocking
        and "subcategory_norm" in records.columns
        and records["subcategory_norm"].fillna("").astype(str).str.strip().ne("").any()
    )

    if has_subcategory:
        known_mask = records["subcategory_norm"].fillna("").astype(str).str.strip().ne("")
        known_records = records[known_mask]
        for _, group in known_records.groupby("subcategory_norm", sort=False):
            scope_indices = group.index.tolist()
            _add_scope_pairs(
                pair_rows=pair_rows,
                records=records,
                index_vectors=index_vectors,
                query_indices=scope_indices,
                scope_indices=scope_indices,
                search_top_k=cfg.top_k,
                blocking_scope="same_subcategory",
                min_similarity=cfg.min_similarity,
                faiss_module=faiss,
            )

        unknown_indices = records[~known_mask].index.tolist()
        _add_scope_pairs(
            pair_rows=pair_rows,
            records=records,
            index_vectors=index_vectors,
            query_indices=unknown_indices,
            scope_indices=all_indices,
            search_top_k=cfg.unknown_subcategory_top_k,
            blocking_scope="unknown_subcategory",
            min_similarity=cfg.min_similarity,
            faiss_module=faiss,
        )
        _add_scope_pairs(
            pair_rows=pair_rows,
            records=records,
            index_vectors=index_vectors,
            query_indices=all_indices,
            scope_indices=all_indices,
            search_top_k=cfg.global_safety_top_k,
            blocking_scope="global_safety",
            min_similarity=cfg.min_similarity,
            faiss_module=faiss,
        )
    else:
        _add_scope_pairs(
            pair_rows=pair_rows,
            records=records,
            index_vectors=index_vectors,
            query_indices=all_indices,
            scope_indices=all_indices,
            search_top_k=cfg.top_k,
            blocking_scope="global",
            min_similarity=cfg.min_similarity,
            faiss_module=faiss,
        )

    pairs = pd.DataFrame(pair_rows.values())
    if pairs.empty:
        return pd.DataFrame(columns=FAISS_CANDIDATE_OUTPUT_COLUMNS)

    pairs = add_cross_marketplace_flags(pairs)
    pairs = add_hard_negative_flags(pairs, cfg.candidate_features)
    pairs = add_pack_variant_flags(pairs, cfg.candidate_features)
    pairs = pairs.sort_values(
        ["embedding_similarity_score", "candidate_rank", "marketplace_a", "sku_a", "marketplace_b", "sku_b"],
        ascending=[False, True, True, True, True, True],
    ).reset_index(drop=True)
    if cfg.max_candidates is not None:
        pairs = pairs.head(cfg.max_candidates).copy()
    return pairs[FAISS_CANDIDATE_OUTPUT_COLUMNS]
