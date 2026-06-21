from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .candidates import add_cross_marketplace_flags, add_pack_variant_flags
from .normalization import normalize_brand


@dataclass(frozen=True)
class LabelingSamplingConfig:
    target_size: int = 400
    random_state: int = 42
    score_stratification: str = "quantile"
    high_similarity_top_share: float = 0.25
    easy_negative_bottom_share: float = 0.25
    high_similarity_threshold: float = 0.72
    medium_similarity_lower: float = 0.50
    medium_similarity_upper: float = 0.72
    easy_negative_upper: float = 0.50
    max_different_brand_share: float = 0.20
    max_different_brand_count: int | None = None
    shuffle_output: bool = True


DEFAULT_STRATA_SHARES = {
    "cross_marketplace_candidate": 0.20,
    "hard_negative_candidate": 0.20,
    "pack_variant_candidate": 0.18,
    "high_similarity": 0.22,
    "medium_similarity": 0.15,
    "random_easy_negative": 0.05,
}


def _pair_key(row: pd.Series) -> str:
    left = _clean_pair_key(row.get("raw_record_id_a")) or _clean_pair_key(row.get("sku_a"))
    right = _clean_pair_key(row.get("raw_record_id_b")) or _clean_pair_key(row.get("sku_b"))
    return "||".join(sorted([left, right]))


def _clean_pair_key(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(value != value):
            return ""
    except TypeError:
        return ""
    return str(value).strip()


def _brand_relation(row: pd.Series) -> str:
    brand_a = normalize_brand(row.get("brand_a"))
    brand_b = normalize_brand(row.get("brand_b"))
    if brand_a and brand_b:
        if brand_a == brand_b:
            return "same_brand"
        return "different_brand"
    return "unknown_brand"


def add_brand_relation_flags(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized brand-bucket columns for sampling diagnostics."""
    result = pairs_df.copy()
    if result.empty:
        result["brand_relation"] = pd.Series(dtype=str)
        result["is_different_brand_pair"] = pd.Series(dtype=bool)
        return result
    result["brand_relation"] = result.apply(_brand_relation, axis=1)
    result["is_different_brand_pair"] = result["brand_relation"].eq("different_brand")
    return result


def labeling_pair_keys(labeling_df: pd.DataFrame) -> set[str]:
    """Return stable pair keys for a labeling/candidate frame."""
    if labeling_df.empty:
        return set()
    return set(labeling_df.apply(_pair_key, axis=1).tolist())


def labeled_row_mask(labeling_df: pd.DataFrame) -> pd.Series:
    """Rows with a human label that must survive regeneration."""
    if "label" not in labeling_df.columns:
        return pd.Series(False, index=labeling_df.index)
    return labeling_df["label"].fillna("").astype(str).str.strip().ne("")


def select_preserved_labeling_rows(labeling_df: pd.DataFrame) -> pd.DataFrame:
    """Keep already labeled rows and refresh brand diagnostic columns."""
    preserved = labeling_df[labeled_row_mask(labeling_df)].copy()
    if preserved.empty:
        return preserved
    if "brand_relation" not in preserved.columns or "is_different_brand_pair" not in preserved.columns:
        preserved = add_brand_relation_flags(preserved)
    return preserved


def merge_preserved_labeling_rows(
    preserved_df: pd.DataFrame,
    fresh_df: pd.DataFrame,
    *,
    target_size: int,
    random_state: int,
) -> pd.DataFrame:
    """Keep human labels, then fill the remaining target with fresh rows."""
    if target_size < 0:
        raise ValueError("target_size must be >= 0")
    preserved = preserved_df.copy()
    fresh = fresh_df.copy()

    preserved_keys = labeling_pair_keys(preserved)
    if preserved_keys and not fresh.empty:
        fresh = fresh[~fresh.apply(_pair_key, axis=1).isin(preserved_keys)].copy()

    if len(preserved) >= target_size:
        combined = preserved
    else:
        combined = pd.concat([preserved, fresh.head(target_size - len(preserved))], ignore_index=True, sort=False)

    if not combined.empty:
        combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return combined.drop(columns=["_pair_key"], errors="ignore")


def _different_brand_limit(cfg: LabelingSamplingConfig) -> int:
    if cfg.max_different_brand_count is not None:
        if cfg.max_different_brand_count < 0:
            raise ValueError("max_different_brand_count must be >= 0")
        return cfg.max_different_brand_count
    if not 0 <= cfg.max_different_brand_share <= 1:
        raise ValueError("max_different_brand_share must be between 0 and 1")
    return int(cfg.target_size * cfg.max_different_brand_share)


def _quota_counts(target_size: int) -> dict[str, int]:
    quotas = {name: int(round(target_size * share)) for name, share in DEFAULT_STRATA_SHARES.items()}
    delta = target_size - sum(quotas.values())
    quotas["random_easy_negative"] += delta
    return quotas


def split_labeling_target_size(target_size: int, group_count: int) -> list[int]:
    """Split a total labeling size across category-runs as evenly as possible."""
    if target_size < 0:
        raise ValueError("target_size must be >= 0")
    if group_count < 1:
        raise ValueError("group_count must be >= 1")
    base, remainder = divmod(target_size, group_count)
    return [base + (1 if idx < remainder else 0) for idx in range(group_count)]


def _sample_pool(
    pool: pd.DataFrame,
    *,
    n: int,
    random_state: int,
    used_keys: set[str],
    stratum: str,
    brand_budget: dict[str, int] | None = None,
) -> pd.DataFrame:
    if n <= 0 or pool.empty:
        return pool.head(0).copy()
    available = pool[~pool["_pair_key"].isin(used_keys)].copy()
    if available.empty:
        return available

    if brand_budget is None:
        sampled = available.sample(n=min(n, len(available)), random_state=random_state).copy()
    else:
        shuffled = available.sample(frac=1, random_state=random_state)
        selected_positions: list[int] = []
        for position, (_, row) in enumerate(shuffled.iterrows()):
            is_different_brand = bool(row.get("is_different_brand_pair", False))
            if is_different_brand:
                if brand_budget["remaining"] <= 0:
                    continue
                brand_budget["remaining"] -= 1
            selected_positions.append(position)
            if len(selected_positions) >= n:
                break
        sampled = shuffled.iloc[selected_positions].copy()

    sampled["labeling_stratum"] = stratum
    used_keys.update(sampled["_pair_key"].tolist())
    return sampled


def _empty_mask(index: pd.Index) -> pd.Series:
    return pd.Series(False, index=index)


def _position_score_masks(score: pd.Series, cfg: LabelingSamplingConfig) -> dict[str, pd.Series]:
    if not 0 <= cfg.high_similarity_top_share <= 1:
        raise ValueError("high_similarity_top_share must be between 0 and 1")
    if not 0 <= cfg.easy_negative_bottom_share <= 1:
        raise ValueError("easy_negative_bottom_share must be between 0 and 1")
    if cfg.high_similarity_top_share + cfg.easy_negative_bottom_share > 1:
        raise ValueError("high/easy score shares must not overlap")

    high_mask = _empty_mask(score.index)
    medium_mask = _empty_mask(score.index)
    easy_mask = _empty_mask(score.index)
    if score.empty:
        return {
            "high_similarity": high_mask,
            "medium_similarity": medium_mask,
            "random_easy_negative": easy_mask,
        }

    ordered_index = score.sort_values(ascending=False, kind="mergesort").index.tolist()
    row_count = len(ordered_index)
    high_count = round(row_count * cfg.high_similarity_top_share)
    easy_count = round(row_count * cfg.easy_negative_bottom_share)
    if cfg.high_similarity_top_share > 0:
        high_count = max(1, high_count)
    if cfg.easy_negative_bottom_share > 0:
        easy_count = max(1, easy_count)

    high_count = min(high_count, row_count)
    easy_count = min(easy_count, row_count - high_count)
    medium_start = high_count
    medium_end = row_count - easy_count

    high_mask.loc[ordered_index[:high_count]] = True
    medium_mask.loc[ordered_index[medium_start:medium_end]] = True
    if easy_count:
        easy_mask.loc[ordered_index[medium_end:]] = True
    return {
        "high_similarity": high_mask,
        "medium_similarity": medium_mask,
        "random_easy_negative": easy_mask,
    }


def labeling_score_strata_masks(
    candidates_df: pd.DataFrame,
    config: LabelingSamplingConfig | None = None,
) -> dict[str, pd.Series]:
    """Return score-bucket masks used by the labeling sampler."""
    cfg = config or LabelingSamplingConfig()
    if "baseline_similarity_score" not in candidates_df.columns:
        raise KeyError("baseline_similarity_score is required for labeling score strata")

    score = pd.to_numeric(candidates_df["baseline_similarity_score"], errors="coerce").fillna(0.0)
    if cfg.score_stratification == "quantile":
        return _position_score_masks(score, cfg)
    if cfg.score_stratification == "absolute":
        return {
            "high_similarity": score >= cfg.high_similarity_threshold,
            "medium_similarity": (score >= cfg.medium_similarity_lower) & (score < cfg.medium_similarity_upper),
            "random_easy_negative": score < cfg.easy_negative_upper,
        }
    raise ValueError("score_stratification must be 'quantile' or 'absolute'")


def stratified_labeling_sample(
    candidates_df: pd.DataFrame,
    config: LabelingSamplingConfig | None = None,
    *,
    excluded_pair_keys: set[str] | None = None,
) -> pd.DataFrame:
    """Build a manual-labeling batch without auto-generating labels."""
    cfg = config or LabelingSamplingConfig()
    candidates = candidates_df.copy()
    if "is_pack_variant_candidate" not in candidates.columns:
        candidates = add_pack_variant_flags(candidates)
    if "is_cross_marketplace_pair" not in candidates.columns:
        candidates = add_cross_marketplace_flags(candidates)
    if "is_different_brand_pair" not in candidates.columns:
        candidates = add_brand_relation_flags(candidates)
    candidates["_pair_key"] = candidates.apply(_pair_key, axis=1)
    if excluded_pair_keys:
        candidates = candidates[~candidates["_pair_key"].isin(excluded_pair_keys)].copy()

    score_masks = labeling_score_strata_masks(candidates, cfg)
    quotas = _quota_counts(cfg.target_size)
    pools = {
        "cross_marketplace_candidate": candidates[candidates["is_cross_marketplace_pair"].fillna(False)],
        "hard_negative_candidate": candidates[candidates["is_hard_negative_candidate"].fillna(False)],
        "pack_variant_candidate": candidates[candidates["is_pack_variant_candidate"].fillna(False)],
        "high_similarity": candidates[score_masks["high_similarity"]],
        "medium_similarity": candidates[score_masks["medium_similarity"]],
        "random_easy_negative": candidates[score_masks["random_easy_negative"]],
    }

    used_keys: set[str] = set()
    brand_budget = {"remaining": _different_brand_limit(cfg)}
    samples: list[pd.DataFrame] = []
    for offset, (stratum, quota) in enumerate(quotas.items()):
        samples.append(
            _sample_pool(
                pools[stratum],
                n=quota,
                random_state=cfg.random_state + offset,
                used_keys=used_keys,
                stratum=stratum,
                brand_budget=brand_budget,
            )
        )

    labeling = pd.concat(samples, ignore_index=True) if samples else candidates.head(0).copy()
    if len(labeling) < cfg.target_size:
        backfill = _sample_pool(
            candidates,
            n=cfg.target_size - len(labeling),
            random_state=cfg.random_state + 99,
            used_keys=used_keys,
            stratum="backfill_other_candidate",
            brand_budget=brand_budget,
        )
        labeling = pd.concat([labeling, backfill], ignore_index=True)

    if cfg.shuffle_output and not labeling.empty:
        labeling = labeling.sample(frac=1, random_state=cfg.random_state + 10_000).reset_index(drop=True)
    else:
        labeling = labeling.sort_values(
            ["labeling_stratum", "baseline_similarity_score", "sku_a", "sku_b"],
            ascending=[True, False, True, True],
        ).reset_index(drop=True)
    labeling = labeling.drop(columns=["label", "notes"], errors="ignore")
    labeling.insert(0, "label", "")
    labeling.insert(1, "notes", "")
    return labeling.drop(columns=["_pair_key"], errors="ignore")
