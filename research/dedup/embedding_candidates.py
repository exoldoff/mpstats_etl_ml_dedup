from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
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
from .normalization import title_similarity


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
    supplemental_lexical_pairs: int = 8_000
    supplemental_same_brand_pack_pairs: int = 6_000
    supplemental_cross_marketplace_random_pairs: int = 3_000
    supplemental_random_pairs: int = 3_000
    supplemental_random_state: int = 42


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
    candidate_source: str = "faiss_embedding_topk",
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
        "candidate_source": candidate_source,
        "blocking_scope": blocking_scope,
        "baseline_similarity_score": round(score, 6),
    }


def _pair_key(left_idx: int, right_idx: int) -> tuple[int, int]:
    return tuple(sorted((int(left_idx), int(right_idx))))


def _cosine_pair_score(vectors: np.ndarray, left_idx: int, right_idx: int) -> float:
    return float(np.dot(vectors[int(left_idx)], vectors[int(right_idx)]))


def _add_pair_if_new(
    *,
    pair_rows: dict[tuple[int, int], dict[str, object]],
    records: pd.DataFrame,
    index_vectors: np.ndarray,
    left_idx: int,
    right_idx: int,
    candidate_source: str,
    blocking_scope: str,
    rank: int,
) -> bool:
    pair_key = _pair_key(left_idx, right_idx)
    if pair_key in pair_rows:
        return False
    left = records.iloc[pair_key[0]]
    right = records.iloc[pair_key[1]]
    pair_rows[pair_key] = _candidate_row(
        left,
        right,
        score=_cosine_pair_score(index_vectors, pair_key[0], pair_key[1]),
        rank=rank,
        blocking_scope=blocking_scope,
        candidate_source=candidate_source,
    )
    return True


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


def _numbers_close(left: object, right: object, *, abs_tol: float, rel_tol: float) -> bool:
    left_num = pd.to_numeric(pd.Series([left]), errors="coerce").iloc[0]
    right_num = pd.to_numeric(pd.Series([right]), errors="coerce").iloc[0]
    if pd.isna(left_num) or pd.isna(right_num) or left_num <= 0 or right_num <= 0:
        return False
    return abs(float(left_num) - float(right_num)) <= max(
        abs_tol,
        rel_tol * max(abs(float(left_num)), abs(float(right_num))),
    )


def _same_weight_pack(row: pd.Series, cfg: CandidateGenerationConfig) -> bool:
    unit_close = _numbers_close(
        row.get("unit_amount"),
        row.get("unit_amount_other"),
        abs_tol=cfg.weight_abs_tolerance,
        rel_tol=cfg.weight_rel_tolerance,
    )
    total_close = _numbers_close(
        row.get("total_amount"),
        row.get("total_amount_other"),
        abs_tol=cfg.weight_abs_tolerance,
        rel_tol=cfg.weight_rel_tolerance,
    )
    pack_close = _numbers_close(
        row.get("multipack_count"),
        row.get("multipack_count_other"),
        abs_tol=0.25,
        rel_tol=0.0,
    )
    return (unit_close and total_close) or (unit_close and pack_close) or (total_close and pack_close)


def _random_distinct_pair(indices: np.ndarray, rng: np.random.Generator) -> tuple[int, int] | None:
    if len(indices) < 2:
        return None
    left, right = rng.choice(indices, size=2, replace=False)
    return int(left), int(right)


def _sample_random_pairs(
    *,
    pair_rows: dict[tuple[int, int], dict[str, object]],
    records: pd.DataFrame,
    index_vectors: np.ndarray,
    budget: int,
    candidate_source: str,
    blocking_scope: str,
    rng: np.random.Generator,
    left_indices: np.ndarray,
    right_indices: np.ndarray | None = None,
    max_attempt_multiplier: int = 20,
) -> None:
    if budget <= 0 or len(left_indices) == 0:
        return
    right_pool = left_indices if right_indices is None else right_indices
    if len(right_pool) == 0:
        return

    added = 0
    attempts = 0
    max_attempts = max(budget * max_attempt_multiplier, 100)
    while added < budget and attempts < max_attempts:
        attempts += 1
        if right_indices is None:
            sampled = _random_distinct_pair(left_indices, rng)
            if sampled is None:
                return
            left_idx, right_idx = sampled
        else:
            left_idx = int(rng.choice(left_indices))
            right_idx = int(rng.choice(right_pool))
            if left_idx == right_idx:
                continue
        if _add_pair_if_new(
            pair_rows=pair_rows,
            records=records,
            index_vectors=index_vectors,
            left_idx=left_idx,
            right_idx=right_idx,
            candidate_source=candidate_source,
            blocking_scope=blocking_scope,
            rank=added + 1,
        ):
            added += 1


def _add_lexical_supplemental_pairs(
    *,
    pair_rows: dict[tuple[int, int], dict[str, object]],
    records: pd.DataFrame,
    index_vectors: np.ndarray,
    budget: int,
    cfg: CandidateGenerationConfig,
) -> None:
    if budget <= 0 or "title_tokens" not in records.columns:
        return

    candidate_keys: set[tuple[int, int]] = set()
    token_index: dict[str, list[int]] = {}
    for row_idx, tokens in enumerate(records["title_tokens"]):
        for token in set(tokens):
            token_index.setdefault(token, []).append(row_idx)

    token_blocks = sorted(token_index.values(), key=lambda row_ids: (len(row_ids), row_ids[0] if row_ids else -1))
    max_candidates_for_scoring = max(budget * 10, budget)
    for row_ids in token_blocks:
        if len(row_ids) < 2 or len(row_ids) > cfg.max_block_size:
            continue
        for left_idx, right_idx in combinations(sorted(row_ids), 2):
            key = _pair_key(left_idx, right_idx)
            if key in pair_rows:
                continue
            candidate_keys.add(key)
            if len(candidate_keys) >= max_candidates_for_scoring:
                break
        if len(candidate_keys) >= max_candidates_for_scoring:
            break

    scored_rows: list[tuple[float, tuple[int, int]]] = []
    for key in candidate_keys:
        left = records.iloc[key[0]]
        right = records.iloc[key[1]]
        score = title_similarity(
            left["title"],
            right["title"],
            tokens_a=left.get("title_tokens"),
            tokens_b=right.get("title_tokens"),
        )
        if score >= cfg.min_similarity:
            scored_rows.append((score, key))

    scored_rows.sort(reverse=True, key=lambda item: item[0])
    for rank, (_, key) in enumerate(scored_rows[:budget], start=1):
        _add_pair_if_new(
            pair_rows=pair_rows,
            records=records,
            index_vectors=index_vectors,
            left_idx=key[0],
            right_idx=key[1],
            candidate_source="supplemental_lexical_overlap",
            blocking_scope="supplemental_lexical",
            rank=rank,
        )


def _add_same_brand_pack_supplemental_pairs(
    *,
    pair_rows: dict[tuple[int, int], dict[str, object]],
    records: pd.DataFrame,
    index_vectors: np.ndarray,
    budget: int,
    cfg: CandidateGenerationConfig,
    rng: np.random.Generator,
) -> None:
    if budget <= 0 or "brand_norm" not in records.columns:
        return

    added = 0
    for _, group in records[records["brand_norm"].fillna("").ne("")].groupby("brand_norm", sort=False):
        if added >= budget:
            break
        group_indices = group.index.to_numpy(dtype=int)
        if len(group_indices) < 2:
            continue
        attempts = 0
        while added < budget and attempts < min(len(group_indices) * 20, 1000):
            attempts += 1
            sampled = _random_distinct_pair(group_indices, rng)
            if sampled is None:
                break
            left_idx, right_idx = sampled
            key = _pair_key(left_idx, right_idx)
            if key in pair_rows:
                continue
            left = records.iloc[key[0]]
            right = records.iloc[key[1]]
            check_row = left.copy()
            for column in ["unit_amount", "total_amount", "multipack_count"]:
                check_row[f"{column}_other"] = right.get(column)
            if not _same_weight_pack(check_row, cfg):
                continue
            if _add_pair_if_new(
                pair_rows=pair_rows,
                records=records,
                index_vectors=index_vectors,
                left_idx=key[0],
                right_idx=key[1],
                candidate_source="supplemental_same_brand_pack",
                blocking_scope="supplemental_same_brand_pack",
                rank=added + 1,
            ):
                added += 1


def _add_supplemental_pairs(
    *,
    pair_rows: dict[tuple[int, int], dict[str, object]],
    records: pd.DataFrame,
    index_vectors: np.ndarray,
    cfg: FaissCandidateGenerationConfig,
) -> None:
    feature_cfg = cfg.candidate_features
    rng = np.random.default_rng(cfg.supplemental_random_state)
    all_indices = records.index.to_numpy(dtype=int)

    _add_lexical_supplemental_pairs(
        pair_rows=pair_rows,
        records=records,
        index_vectors=index_vectors,
        budget=cfg.supplemental_lexical_pairs,
        cfg=feature_cfg,
    )
    _add_same_brand_pack_supplemental_pairs(
        pair_rows=pair_rows,
        records=records,
        index_vectors=index_vectors,
        budget=cfg.supplemental_same_brand_pack_pairs,
        cfg=feature_cfg,
        rng=rng,
    )

    if cfg.supplemental_cross_marketplace_random_pairs > 0 and "marketplace_key" in records.columns:
        grouped = {
            marketplace: group.index.to_numpy(dtype=int)
            for marketplace, group in records.groupby("marketplace_key", sort=False)
        }
        marketplaces = list(grouped)
        added = 0
        attempts = 0
        max_attempts = max(cfg.supplemental_cross_marketplace_random_pairs * 20, 100)
        while added < cfg.supplemental_cross_marketplace_random_pairs and attempts < max_attempts:
            attempts += 1
            if len(marketplaces) < 2:
                break
            left_marketplace, right_marketplace = rng.choice(marketplaces, size=2, replace=False)
            if _add_pair_if_new(
                pair_rows=pair_rows,
                records=records,
                index_vectors=index_vectors,
                left_idx=int(rng.choice(grouped[left_marketplace])),
                right_idx=int(rng.choice(grouped[right_marketplace])),
                candidate_source="supplemental_cross_marketplace_random",
                blocking_scope="supplemental_cross_marketplace_random",
                rank=added + 1,
            ):
                added += 1

    _sample_random_pairs(
        pair_rows=pair_rows,
        records=records,
        index_vectors=index_vectors,
        budget=cfg.supplemental_random_pairs,
        candidate_source="supplemental_random_control",
        blocking_scope="supplemental_random",
        rng=rng,
        left_indices=all_indices,
    )


def _source_order(source: object) -> int:
    return {
        "faiss_embedding_topk": 0,
        "supplemental_lexical_overlap": 1,
        "supplemental_same_brand_pack": 2,
        "supplemental_cross_marketplace_random": 3,
        "supplemental_random_control": 4,
    }.get(str(source), 99)


def _sort_candidate_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    return (
        pairs.assign(_source_order=pairs["candidate_source"].map(_source_order))
        .sort_values(
            [
                "_source_order",
                "embedding_similarity_score",
                "candidate_rank",
                "marketplace_a",
                "sku_a",
                "marketplace_b",
                "sku_b",
            ],
            ascending=[True, False, True, True, True, True, True],
        )
        .drop(columns=["_source_order"])
        .reset_index(drop=True)
    )


def _limit_candidate_pairs(pairs: pd.DataFrame, max_candidates: int | None) -> pd.DataFrame:
    if max_candidates is None or len(pairs) <= max_candidates:
        return pairs
    faiss_pairs = pairs[pairs["candidate_source"].eq("faiss_embedding_topk")]
    supplemental_pairs = pairs[~pairs["candidate_source"].eq("faiss_embedding_topk")]
    supplemental_cap = min(len(supplemental_pairs), max_candidates // 3)
    faiss_cap = max_candidates - supplemental_cap
    limited = pd.concat(
        [faiss_pairs.head(faiss_cap), supplemental_pairs.head(supplemental_cap)],
        ignore_index=True,
    )
    return _sort_candidate_pairs(limited)


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
            pair_key = _pair_key(left_idx, right_idx)
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

    _add_supplemental_pairs(
        pair_rows=pair_rows,
        records=records,
        index_vectors=index_vectors,
        cfg=cfg,
    )

    pairs = pd.DataFrame(pair_rows.values())
    if pairs.empty:
        return pd.DataFrame(columns=FAISS_CANDIDATE_OUTPUT_COLUMNS)

    pairs = add_cross_marketplace_flags(pairs)
    pairs = add_hard_negative_flags(pairs, cfg.candidate_features)
    pairs = add_pack_variant_flags(pairs, cfg.candidate_features)
    pairs = _sort_candidate_pairs(pairs)
    pairs = _limit_candidate_pairs(pairs, cfg.max_candidates)
    return pairs[FAISS_CANDIDATE_OUTPUT_COLUMNS]
