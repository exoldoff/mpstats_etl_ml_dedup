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
    "unit_amount_a",
    "unit_amount_b",
    "total_amount_a",
    "total_amount_b",
    "multipack_count_a",
    "multipack_count_b",
    "embedding_similarity_score",
    "candidate_rank",
    "candidate_source",
    "baseline_similarity_score",
    "is_cross_marketplace_pair",
    "is_hard_negative_candidate",
    "is_pack_variant_candidate",
]


@dataclass(frozen=True)
class FaissCandidateGenerationConfig:
    """Configuration for embedding top-k blocking with FAISS."""

    top_k: int = 20
    max_candidates: int | None = 60_000
    min_similarity: float | None = None
    normalize_vectors: bool = True
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
        "unit_amount_a": left["unit_amount"],
        "unit_amount_b": right["unit_amount"],
        "total_amount_a": left["total_amount"],
        "total_amount_b": right["total_amount"],
        "multipack_count_a": left["multipack_count"],
        "multipack_count_b": right["multipack_count"],
        "embedding_similarity_score": round(score, 6),
        "candidate_rank": rank,
        "candidate_source": "faiss_embedding_topk",
        "baseline_similarity_score": round(score, 6),
    }


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

    faiss = _load_faiss(faiss_module)
    index_vectors = _normalize_l2(vectors, faiss) if cfg.normalize_vectors else np.ascontiguousarray(vectors)
    dimension = index_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(index_vectors)

    search_k = min(len(records), cfg.top_k + 1)
    scores, indices = index.search(index_vectors, search_k)

    pair_rows: dict[tuple[int, int], dict[str, object]] = {}
    for left_idx, (row_scores, row_indices) in enumerate(zip(scores, indices, strict=True)):
        neighbor_rank = 0
        for score, right_idx in zip(row_scores, row_indices, strict=True):
            right_idx = int(right_idx)
            if right_idx < 0 or right_idx == left_idx:
                continue
            score_value = float(score)
            if cfg.min_similarity is not None and score_value < cfg.min_similarity:
                continue
            neighbor_rank += 1
            pair_key = tuple(sorted((left_idx, right_idx)))
            existing = pair_rows.get(pair_key)
            if existing is not None and float(existing["embedding_similarity_score"]) >= score_value:
                continue
            left = records.iloc[pair_key[0]]
            right = records.iloc[pair_key[1]]
            pair_rows[pair_key] = _candidate_row(left, right, score=score_value, rank=neighbor_rank)

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
