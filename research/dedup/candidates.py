from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import pandas as pd

from .normalization import (
    DEFAULT_FLAVOR_TOKENS,
    flavor_token_set,
    meaningful_title_tokens,
    normalize_brand,
    normalize_title,
    title_similarity,
)


CANDIDATE_OUTPUT_COLUMNS = [
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
    "baseline_similarity_score",
    "is_hard_negative_candidate",
]


@dataclass(frozen=True)
class CandidateGenerationConfig:
    title_col: str | None = None
    sku_col: str | None = None
    brand_col: str | None = None
    unit_amount_col: str | None = None
    total_amount_col: str | None = None
    multipack_count_col: str | None = None
    min_similarity: float = 0.46
    min_shared_tokens: int = 1
    max_block_size: int = 120
    max_exact_title_group_size: int = 80
    max_pair_candidates_for_scoring: int | None = 250_000
    max_candidates: int | None = 50_000
    weight_abs_tolerance: float = 0.02
    weight_rel_tolerance: float = 0.05


def _resolve_column(df: pd.DataFrame, explicit: str | None, aliases: Iterable[str], *, required: bool) -> str | None:
    if explicit:
        if explicit not in df.columns and required:
            raise KeyError(f"Колонка {explicit!r} не найдена.")
        return explicit if explicit in df.columns else None
    for alias in aliases:
        if alias in df.columns:
            return alias
    if required:
        raise KeyError(f"Не найдена ни одна из колонок: {', '.join(aliases)}")
    return None


def _first_present(series: pd.Series) -> object:
    non_empty = series.dropna()
    if non_empty.empty:
        return pd.NA
    text_non_empty = non_empty[non_empty.astype(str).str.strip().ne("")]
    if text_non_empty.empty:
        return non_empty.iloc[0]
    return text_non_empty.iloc[0]


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
    return number if number > 0 else None


def _derive_multipack(unit_amount: object, total_amount: object) -> float | None:
    unit = _to_number(unit_amount)
    total = _to_number(total_amount)
    if unit is None or total is None:
        return None
    ratio = total / unit
    if ratio <= 0:
        return None
    return max(1.0, float(round(ratio)))


def prepare_product_records(
    products_df: pd.DataFrame,
    config: CandidateGenerationConfig | None = None,
) -> pd.DataFrame:
    """Normalize the classified mpstats product slice into one row per SKU."""
    cfg = config or CandidateGenerationConfig()
    title_col = _resolve_column(products_df, cfg.title_col, ["SKU", "Название", "title", "name"], required=True)
    sku_col = _resolve_column(products_df, cfg.sku_col, ["Артикул", "sku_id", "id", "article"], required=False)
    brand_col = _resolve_column(products_df, cfg.brand_col, ["Бренд", "brand"], required=False)
    unit_col = _resolve_column(
        products_df,
        cfg.unit_amount_col,
        ["Вес, кг (ед.)", "Вес кг (ед.)", "unit_amount", "unit_weight"],
        required=False,
    )
    total_col = _resolve_column(
        products_df,
        cfg.total_amount_col,
        ["Вес, кг", "Вес кг", "total_amount", "total_weight"],
        required=False,
    )
    multipack_col = _resolve_column(
        products_df,
        cfg.multipack_count_col,
        ["multipack_count", "pack_count", "multipack"],
        required=False,
    )

    records = pd.DataFrame(
        {
            "sku": products_df[sku_col].astype(str) if sku_col else [f"row_{idx}" for idx in products_df.index],
            "title": products_df[title_col],
            "brand": products_df[brand_col] if brand_col else "",
            "unit_amount": pd.to_numeric(products_df[unit_col], errors="coerce") if unit_col else pd.NA,
            "total_amount": pd.to_numeric(products_df[total_col], errors="coerce") if total_col else pd.NA,
        }
    )
    if multipack_col:
        records["multipack_count"] = pd.to_numeric(products_df[multipack_col], errors="coerce")
    else:
        records["multipack_count"] = [
            _derive_multipack(unit, total)
            for unit, total in zip(records["unit_amount"], records["total_amount"], strict=False)
        ]

    records["title"] = records["title"].fillna("").astype(str).str.strip()
    records = records[records["title"].ne("")].copy()
    records["sku"] = records["sku"].fillna("").astype(str).str.strip()
    records.loc[records["sku"].eq(""), "sku"] = [f"row_{idx}" for idx in records.index[records["sku"].eq("")]]
    records["brand"] = records["brand"].fillna("").astype(str).str.strip()

    grouped = (
        records.groupby("sku", as_index=False)
        .agg(
            title=("title", _first_present),
            brand=("brand", _first_present),
            unit_amount=("unit_amount", _first_present),
            total_amount=("total_amount", _first_present),
            multipack_count=("multipack_count", _first_present),
        )
        .reset_index(drop=True)
    )
    grouped["brand_norm"] = grouped["brand"].map(normalize_brand)
    grouped["title_norm"] = grouped["title"].map(normalize_title)
    grouped["title_tokens"] = grouped["title"].map(meaningful_title_tokens)
    grouped["flavor_tokens"] = grouped["title"].map(flavor_token_set)
    return grouped


def _candidate_pair_rows(records: pd.DataFrame, cfg: CandidateGenerationConfig) -> list[dict[str, object]]:
    pair_shared_counts: Counter[tuple[int, int]] = Counter()

    for _, group in records[records["title_norm"].ne("")].groupby("title_norm", sort=False):
        if len(group) < 2 or len(group) > cfg.max_exact_title_group_size:
            continue
        row_ids = sorted(group.index.tolist())
        for left_idx, right_idx in combinations(row_ids, 2):
            pair_shared_counts[(left_idx, right_idx)] += max(1, cfg.min_shared_tokens)

    token_index: dict[str, list[int]] = defaultdict(list)
    for row_idx, tokens in enumerate(records["title_tokens"]):
        for token in set(tokens):
            token_index[token].append(row_idx)

    token_blocks = sorted(token_index.values(), key=lambda row_ids: (len(row_ids), row_ids[0] if row_ids else -1))
    for row_ids in token_blocks:
        if len(row_ids) < 2 or len(row_ids) > cfg.max_block_size:
            continue
        for left_idx, right_idx in combinations(sorted(row_ids), 2):
            pair_shared_counts[(left_idx, right_idx)] += 1
        if cfg.max_pair_candidates_for_scoring is not None and len(pair_shared_counts) >= cfg.max_pair_candidates_for_scoring:
            break

    rows: list[dict[str, object]] = []
    for (left_idx, right_idx), shared_count in pair_shared_counts.items():
        if shared_count < cfg.min_shared_tokens:
            continue
        left = records.iloc[left_idx]
        right = records.iloc[right_idx]
        score = title_similarity(
            left["title"],
            right["title"],
            tokens_a=left["title_tokens"],
            tokens_b=right["title_tokens"],
        )
        if score < cfg.min_similarity:
            continue
        rows.append(
            {
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
                "baseline_similarity_score": score,
                "shared_title_tokens_count": shared_count,
            }
        )
    return rows


def _numbers_close(left: object, right: object, *, abs_tol: float, rel_tol: float) -> bool:
    left_num = _to_number(left)
    right_num = _to_number(right)
    if left_num is None or right_num is None:
        return False
    return abs(left_num - right_num) <= max(abs_tol, rel_tol * max(abs(left_num), abs(right_num)))


def _same_weight_pack(row: pd.Series, cfg: CandidateGenerationConfig) -> bool:
    unit_close = _numbers_close(
        row.get("unit_amount_a"),
        row.get("unit_amount_b"),
        abs_tol=cfg.weight_abs_tolerance,
        rel_tol=cfg.weight_rel_tolerance,
    )
    total_close = _numbers_close(
        row.get("total_amount_a"),
        row.get("total_amount_b"),
        abs_tol=cfg.weight_abs_tolerance,
        rel_tol=cfg.weight_rel_tolerance,
    )
    pack_close = _numbers_close(
        row.get("multipack_count_a"),
        row.get("multipack_count_b"),
        abs_tol=0.25,
        rel_tol=0.0,
    )
    return (unit_close and total_close) or (unit_close and pack_close) or (total_close and pack_close)


def _different_pack(row: pd.Series, cfg: CandidateGenerationConfig) -> bool:
    unit_close = _numbers_close(
        row.get("unit_amount_a"),
        row.get("unit_amount_b"),
        abs_tol=cfg.weight_abs_tolerance,
        rel_tol=cfg.weight_rel_tolerance,
    )
    total_close = _numbers_close(
        row.get("total_amount_a"),
        row.get("total_amount_b"),
        abs_tol=cfg.weight_abs_tolerance,
        rel_tol=cfg.weight_rel_tolerance,
    )
    pack_close = _numbers_close(
        row.get("multipack_count_a"),
        row.get("multipack_count_b"),
        abs_tol=0.25,
        rel_tol=0.0,
    )
    return unit_close and (not total_close or not pack_close)


def add_hard_negative_flags(
    pairs_df: pd.DataFrame,
    config: CandidateGenerationConfig | None = None,
    flavor_vocabulary: Iterable[str] = DEFAULT_FLAVOR_TOKENS,
) -> pd.DataFrame:
    """Mark same-brand/same-pack pairs with different flavor/type tokens."""
    cfg = config or CandidateGenerationConfig()
    if pairs_df.empty:
        result = pairs_df.copy()
        result["is_hard_negative_candidate"] = pd.Series(dtype=bool)
        return result

    def is_hard_negative(row: pd.Series) -> bool:
        brand_a = normalize_brand(row.get("brand_a"))
        brand_b = normalize_brand(row.get("brand_b"))
        if not brand_a or brand_a != brand_b:
            return False
        if not _same_weight_pack(row, cfg):
            return False
        flavors_a = flavor_token_set(row.get("title_a"), flavor_vocabulary)
        flavors_b = flavor_token_set(row.get("title_b"), flavor_vocabulary)
        return bool(flavors_a or flavors_b) and flavors_a != flavors_b

    result = pairs_df.copy()
    result["is_hard_negative_candidate"] = result.apply(is_hard_negative, axis=1)
    return result


def add_pack_variant_flags(
    pairs_df: pd.DataFrame,
    config: CandidateGenerationConfig | None = None,
    *,
    min_similarity: float = 0.52,
) -> pd.DataFrame:
    """Mark likely same product family with different pack/total amount."""
    cfg = config or CandidateGenerationConfig()
    if pairs_df.empty:
        result = pairs_df.copy()
        result["is_pack_variant_candidate"] = pd.Series(dtype=bool)
        return result

    def is_pack_variant(row: pd.Series) -> bool:
        brand_a = normalize_brand(row.get("brand_a"))
        brand_b = normalize_brand(row.get("brand_b"))
        if not brand_a or brand_a != brand_b:
            return False
        if (_to_number(row.get("baseline_similarity_score")) or 0.0) < min_similarity:
            return False
        if not _different_pack(row, cfg):
            return False
        flavors_a = flavor_token_set(row.get("title_a"))
        flavors_b = flavor_token_set(row.get("title_b"))
        if flavors_a and flavors_b and flavors_a != flavors_b:
            return False
        return True

    result = pairs_df.copy()
    result["is_pack_variant_candidate"] = result.apply(is_pack_variant, axis=1)
    return result


def generate_candidate_pairs(
    products_df: pd.DataFrame,
    config: CandidateGenerationConfig | None = None,
) -> pd.DataFrame:
    """Generate candidate SKU pairs inside one already-filtered category."""
    cfg = config or CandidateGenerationConfig()
    records = prepare_product_records(products_df, cfg)
    rows = _candidate_pair_rows(records, cfg)
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pd.DataFrame(columns=CANDIDATE_OUTPUT_COLUMNS)

    pairs = add_hard_negative_flags(pairs, cfg)
    pairs = pairs.sort_values(
        ["baseline_similarity_score", "shared_title_tokens_count", "sku_a", "sku_b"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    if cfg.max_candidates is not None:
        pairs = pairs.head(cfg.max_candidates).copy()
    return pairs
