from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .candidates import add_pack_variant_flags


@dataclass(frozen=True)
class LabelingSamplingConfig:
    target_size: int = 400
    random_state: int = 42
    high_similarity_threshold: float = 0.72
    medium_similarity_lower: float = 0.50
    medium_similarity_upper: float = 0.72
    easy_negative_upper: float = 0.50


DEFAULT_STRATA_SHARES = {
    "hard_negative_candidate": 0.25,
    "pack_variant_candidate": 0.20,
    "high_similarity": 0.25,
    "medium_similarity": 0.20,
    "random_easy_negative": 0.10,
}


def _pair_key(row: pd.Series) -> str:
    left = str(row.get("sku_a"))
    right = str(row.get("sku_b"))
    return "||".join(sorted([left, right]))


def _quota_counts(target_size: int) -> dict[str, int]:
    quotas = {name: int(round(target_size * share)) for name, share in DEFAULT_STRATA_SHARES.items()}
    delta = target_size - sum(quotas.values())
    quotas["random_easy_negative"] += delta
    return quotas


def _sample_pool(
    pool: pd.DataFrame,
    *,
    n: int,
    random_state: int,
    used_keys: set[str],
    stratum: str,
) -> pd.DataFrame:
    if n <= 0 or pool.empty:
        return pool.head(0).copy()
    available = pool[~pool["_pair_key"].isin(used_keys)].copy()
    if available.empty:
        return available
    sampled = available.sample(n=min(n, len(available)), random_state=random_state).copy()
    sampled["labeling_stratum"] = stratum
    used_keys.update(sampled["_pair_key"].tolist())
    return sampled


def stratified_labeling_sample(
    candidates_df: pd.DataFrame,
    config: LabelingSamplingConfig | None = None,
) -> pd.DataFrame:
    """Build a manual-labeling batch without auto-generating labels."""
    cfg = config or LabelingSamplingConfig()
    candidates = candidates_df.copy()
    if "is_pack_variant_candidate" not in candidates.columns:
        candidates = add_pack_variant_flags(candidates)
    candidates["_pair_key"] = candidates.apply(_pair_key, axis=1)

    score = pd.to_numeric(candidates["baseline_similarity_score"], errors="coerce").fillna(0.0)
    quotas = _quota_counts(cfg.target_size)
    pools = {
        "hard_negative_candidate": candidates[candidates["is_hard_negative_candidate"].fillna(False)],
        "pack_variant_candidate": candidates[candidates["is_pack_variant_candidate"].fillna(False)],
        "high_similarity": candidates[score >= cfg.high_similarity_threshold],
        "medium_similarity": candidates[
            (score >= cfg.medium_similarity_lower) & (score < cfg.medium_similarity_upper)
        ],
        "random_easy_negative": candidates[score < cfg.easy_negative_upper],
    }

    used_keys: set[str] = set()
    samples: list[pd.DataFrame] = []
    for offset, (stratum, quota) in enumerate(quotas.items()):
        samples.append(
            _sample_pool(
                pools[stratum],
                n=quota,
                random_state=cfg.random_state + offset,
                used_keys=used_keys,
                stratum=stratum,
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
        )
        labeling = pd.concat([labeling, backfill], ignore_index=True)

    labeling = labeling.sort_values(
        ["labeling_stratum", "baseline_similarity_score", "sku_a", "sku_b"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    labeling.insert(0, "label", "")
    labeling.insert(1, "notes", "")
    return labeling.drop(columns=["_pair_key"], errors="ignore")
