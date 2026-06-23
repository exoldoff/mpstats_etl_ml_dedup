from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

import pandas as pd


POSITIVE_LABELS = frozenset({"exact_duplicate", "same_product_different_pack"})
NEGATIVE_LABELS = frozenset({"different_product"})
UNCERTAIN_LABELS = frozenset({"uncertain"})

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_DEV_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

CLEAN_PAIR_COLUMNS = [
    "pair_id",
    "pair_key",
    "source_dataset",
    "category_run",
    "label",
    "same_base_product",
    "sentence_A",
    "sentence_B",
    "raw_record_id_a",
    "raw_record_id_b",
    "marketplace_a",
    "marketplace_b",
    "sku_a",
    "sku_b",
    "title_a",
    "title_b",
    "brand_a",
    "brand_b",
    "subcategory_a",
    "subcategory_b",
    "unit_amount_a",
    "unit_amount_b",
    "total_amount_a",
    "total_amount_b",
    "multipack_count_a",
    "multipack_count_b",
]

CLEAN_SPLIT_COLUMNS = ["split", *CLEAN_PAIR_COLUMNS]
CLEAN_DROPPED_COLUMNS = ["drop_reason", *CLEAN_PAIR_COLUMNS]
CLEAN_CONFLICT_COLUMNS = [
    "pair_id",
    "pair_key",
    "conflict_type",
    "category_run",
    "source_dataset",
    "raw_record_id_a",
    "raw_record_id_b",
    "title_a",
    "title_b",
    "brand_a",
    "brand_b",
    "csv_label",
    "sqlite_label",
    "label",
]
CLEAN_EXCLUDED_COLUMNS = [
    "pair_id",
    "pair_key",
    "exclude_reason",
    "source_dataset",
    "category_run",
    "label",
    "sentence_A",
    "sentence_B",
    "raw_record_id_a",
    "raw_record_id_b",
    "title_a",
    "title_b",
    "brand_a",
    "brand_b",
]


def _empty_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(["" for _ in range(len(frame))], index=frame.index, dtype=object)


def _column_or_empty(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return _empty_series(frame)


@dataclass(frozen=True)
class DatasetFreezeResult:
    pairs: pd.DataFrame
    conflicts: pd.DataFrame
    excluded: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SplitResult:
    pairs: pd.DataFrame
    manifest: dict[str, Any]
    dropped_pairs: pd.DataFrame | None = None


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(value != value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _normalise_label(value: object) -> str:
    return _clean_text(value).casefold()


def binary_target_from_label(label: object) -> int | None:
    normalized = _normalise_label(label)
    if normalized in POSITIVE_LABELS:
        return 1
    if normalized in NEGATIVE_LABELS:
        return 0
    return None


def pair_key(left: object, right: object) -> str:
    left_text = _clean_text(left)
    right_text = _clean_text(right)
    if left_text <= right_text:
        return f"{left_text} || {right_text}"
    return f"{right_text} || {left_text}"


def pair_id_from_key(value: object) -> str:
    digest = hashlib.sha256(_clean_text(value).encode("utf-8")).hexdigest()
    return f"pair_{digest[:16]}"


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_telegram_row_states(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["row_index", "status", "sqlite_label", "sqlite_updated_at"])
    with sqlite3.connect(path) as connection:
        states = pd.read_sql_query(
            "select row_index, status, final_label, updated_at from row_states order by row_index",
            connection,
        )
    states = states.rename(columns={"final_label": "sqlite_label", "updated_at": "sqlite_updated_at"})
    states["sqlite_label"] = states["sqlite_label"].map(_normalise_label)
    return states


def _ensure_category_columns(frame: pd.DataFrame, *, category_run: str, source_dataset: str) -> pd.DataFrame:
    output = frame.copy()
    if "category_run" not in output.columns:
        output["category_run"] = category_run
    else:
        output["category_run"] = output["category_run"].fillna("").astype(str)
        output.loc[output["category_run"].str.strip().eq(""), "category_run"] = category_run
    if "source_dataset" not in output.columns:
        output["source_dataset"] = source_dataset
    if "category_name" not in output.columns:
        output["category_name"] = ""
    if "project_name" not in output.columns:
        output["project_name"] = ""
    return output


def _prepare_multi_rows(labeling_path: Path, telegram_state_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_frame = _read_csv(labeling_path)
    if csv_frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    csv_frame = _ensure_category_columns(csv_frame, category_run="unknown", source_dataset="multi_category")
    csv_frame = csv_frame.reset_index(names="row_index")
    csv_frame["csv_label"] = _column_or_empty(csv_frame, "label").map(_normalise_label)

    states = _read_telegram_row_states(telegram_state_path)
    merged = csv_frame.merge(
        states[["row_index", "status", "sqlite_label", "sqlite_updated_at"]],
        on="row_index",
        how="left",
    )
    merged["sqlite_label"] = merged["sqlite_label"].fillna("").map(_normalise_label)
    merged["csv_label"] = merged["csv_label"].fillna("").map(_normalise_label)

    both_labeled = merged["csv_label"].ne("") & merged["sqlite_label"].ne("")
    conflict_mask = both_labeled & merged["csv_label"].ne(merged["sqlite_label"])
    conflicts = merged[conflict_mask].copy()
    conflicts["conflict_type"] = "csv_vs_telegram"

    output = merged[~conflict_mask].copy()
    output["label"] = output["sqlite_label"].where(output["sqlite_label"].ne(""), output["csv_label"])
    output["label_source"] = "unlabeled"
    output.loc[output["csv_label"].ne("") & output["sqlite_label"].eq(""), "label_source"] = "csv"
    output.loc[output["csv_label"].eq("") & output["sqlite_label"].ne(""), "label_source"] = "telegram_sqlite"
    output.loc[output["csv_label"].ne("") & output["sqlite_label"].ne(""), "label_source"] = "csv_and_telegram"
    return output, conflicts


def _prepare_old_rows(old_sauces_path: Path) -> pd.DataFrame:
    old = _read_csv(old_sauces_path)
    if old.empty:
        return old
    old = _ensure_category_columns(old, category_run="sauces", source_dataset="old_sauces")
    old = old.copy()
    old["row_index"] = pd.NA
    old["csv_label"] = _column_or_empty(old, "label").map(_normalise_label)
    old["sqlite_label"] = ""
    old["label"] = old["csv_label"]
    old["label_source"] = "old_sauces_csv"
    old["sqlite_updated_at"] = ""
    return old


def _attach_pair_keys(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    missing = {"raw_record_id_a", "raw_record_id_b"} - set(output.columns)
    if missing:
        raise ValueError(f"missing pair id columns: {sorted(missing)}")
    output["pair_key"] = [
        pair_key(left, right)
        for left, right in zip(output["raw_record_id_a"], output["raw_record_id_b"], strict=False)
    ]
    return output


def _deduplicate_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame, pd.DataFrame()

    priority = {"multi_category": 0, "old_sauces": 1}
    work = frame.copy()
    work["source_priority"] = work["source_dataset"].map(priority).fillna(9).astype(int)
    conflict_groups: list[pd.DataFrame] = []
    kept_rows: list[pd.Series] = []

    for _, group in work.sort_values(["pair_key", "source_priority"]).groupby("pair_key", sort=False):
        labels = sorted(set(group["label"].map(_normalise_label)))
        if len(labels) > 1:
            conflict = group.copy()
            conflict["conflict_type"] = "duplicate_pair_label_conflict"
            conflict_groups.append(conflict)
            continue
        kept_rows.append(group.iloc[0])

    deduped = pd.DataFrame(kept_rows).reset_index(drop=True) if kept_rows else pd.DataFrame(columns=work.columns)
    conflicts = pd.concat(conflict_groups, ignore_index=True) if conflict_groups else pd.DataFrame()
    return deduped, conflicts


def freeze_labeling_dataset(
    *,
    labeling_path: Path,
    telegram_state_path: Path,
    old_sauces_path: Path,
    created_by: str = "research.dedup.training.freeze_labeling_dataset",
) -> DatasetFreezeResult:
    multi_rows, csv_sqlite_conflicts = _prepare_multi_rows(labeling_path, telegram_state_path)
    old_rows = _prepare_old_rows(old_sauces_path)
    combined = pd.concat([multi_rows, old_rows], ignore_index=True, sort=False)

    if combined.empty:
        raise ValueError("no labeling rows found")

    combined = _attach_pair_keys(combined)
    combined["label"] = combined["label"].map(_normalise_label)
    combined = combined[combined["label"].ne("")].copy()

    deduped, duplicate_conflicts = _deduplicate_pairs(combined)
    conflicts = pd.concat([csv_sqlite_conflicts, duplicate_conflicts], ignore_index=True, sort=False)
    if not conflicts.empty:
        conflicts = _attach_pair_keys(conflicts) if "pair_key" not in conflicts.columns else conflicts

    deduped["same_base_product"] = deduped["label"].map(binary_target_from_label)
    binary_mask = deduped["same_base_product"].notna()
    pairs = deduped[binary_mask].copy()
    pairs["same_base_product"] = pairs["same_base_product"].astype(int)
    pairs = add_pair_text_columns(pairs)

    excluded = deduped[~binary_mask].copy()
    if not excluded.empty:
        excluded["exclude_reason"] = excluded["label"].map(
            lambda value: "uncertain_label" if _normalise_label(value) in UNCERTAIN_LABELS else "unsupported_label"
        )

    manifest = {
        "created_at": _now_utc(),
        "created_by": created_by,
        "input_paths": {
            "labeling_path": str(labeling_path),
            "telegram_state_path": str(telegram_state_path),
            "old_sauces_path": str(old_sauces_path),
        },
        "input_sha256": {
            "labeling_path": _sha256_file(labeling_path),
            "telegram_state_path": _sha256_file(telegram_state_path),
            "old_sauces_path": _sha256_file(old_sauces_path),
        },
        "rows": {
            "multi_rows": int(len(multi_rows)),
            "old_rows": int(len(old_rows)),
            "combined_labeled_rows": int(len(combined)),
            "training_pairs": int(len(pairs)),
            "excluded_rows": int(len(excluded)),
            "conflict_rows": int(len(conflicts)),
            "unique_pair_keys": int(pairs["pair_key"].nunique()) if not pairs.empty else 0,
        },
        "label_counts": pairs["label"].value_counts(dropna=False).to_dict() if not pairs.empty else {},
        "target_counts": pairs["same_base_product"].value_counts(dropna=False).to_dict() if not pairs.empty else {},
        "category_counts": pairs["category_run"].value_counts(dropna=False).to_dict() if "category_run" in pairs else {},
    }
    return DatasetFreezeResult(
        pairs=pairs.reset_index(drop=True),
        conflicts=conflicts.reset_index(drop=True),
        excluded=excluded.reset_index(drop=True),
        manifest=manifest,
    )


def _format_number(value: object) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return f"{number:.4g}"


def build_pair_text(row: pd.Series | dict[str, object], side: str) -> str:
    def get(name: str) -> object:
        if isinstance(row, pd.Series):
            return row.get(name, "")
        return row.get(name, "")

    parts = []
    field_labels = [
        (f"brand_{side}", "brand", _clean_text),
        (f"title_{side}", "title", _clean_text),
        (f"subcategory_{side}", "subcategory", _clean_text),
        (f"marketplace_{side}", "marketplace", _clean_text),
        (f"unit_amount_{side}", "unit_weight_kg", _format_number),
        (f"total_amount_{side}", "total_weight_kg", _format_number),
        (f"multipack_count_{side}", "pack_count", _format_number),
    ]
    for field, label, formatter in field_labels:
        value = formatter(get(field))
        if value:
            parts.append(f"{label}: {value}")
    return " | ".join(parts)


def add_pair_text_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["sentence_A"] = [build_pair_text(row, "a") for _, row in output.iterrows()]
    output["sentence_B"] = [build_pair_text(row, "b") for _, row in output.iterrows()]
    return output


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _component_ids(frame: pd.DataFrame) -> pd.Series:
    union_find = _UnionFind()
    for left, right in zip(frame["raw_record_id_a"], frame["raw_record_id_b"], strict=False):
        union_find.union(_clean_text(left), _clean_text(right))
    roots = [union_find.find(_clean_text(left)) for left in frame["raw_record_id_a"]]
    root_to_id = {root: f"component_{idx:06d}" for idx, root in enumerate(sorted(set(roots)))}
    return pd.Series([root_to_id[root] for root in roots], index=frame.index)


def _component_feature_counts(frame: pd.DataFrame) -> dict[str, float]:
    counts: dict[str, float] = {
        "rows": float(len(frame)),
        "positive": float(frame["same_base_product"].eq(1).sum()),
        "negative": float(frame["same_base_product"].eq(0).sum()),
    }
    if "category_run" in frame.columns:
        for key, value in frame["category_run"].fillna("unknown").astype(str).value_counts().items():
            counts[f"category:{key}"] = float(value)
    return counts


def _split_nested_counts(frame: pd.DataFrame, value_column: str) -> dict[str, dict[str, int]]:
    if value_column not in frame.columns:
        return {}
    counts = frame.groupby("split")[value_column].value_counts(dropna=False).unstack(fill_value=0)
    return {
        str(split_name): {str(column): int(value) for column, value in row.items()}
        for split_name, row in counts.iterrows()
    }


def _hash_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _score_assignment(
    current: dict[str, dict[str, float]],
    targets: dict[str, dict[str, float]],
    split_name: str,
    component_counts: dict[str, float],
) -> float:
    projected = {key: dict(value) for key, value in current.items()}
    for feature, count in component_counts.items():
        projected[split_name][feature] = projected[split_name].get(feature, 0.0) + count
    score = 0.0
    for candidate_split, feature_counts in projected.items():
        for feature, target_value in targets[candidate_split].items():
            weight = 2.0 if feature in {"positive", "negative"} else 1.0
            if feature.startswith("category:"):
                weight = 0.5
            score += weight * abs(feature_counts.get(feature, 0.0) - target_value)
    return score


def component_aware_split(
    frame: pd.DataFrame,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    dev_ratio: float = DEFAULT_DEV_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 42,
    max_strict_component_share: float = 0.25,
    large_component_strategy: str = "positive_record_holdout",
) -> SplitResult:
    strict = _strict_component_aware_split(
        frame,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    largest_component_rows = int(strict.pairs.groupby("component_id").size().max())
    largest_component_share = largest_component_rows / len(strict.pairs)
    strict.manifest["splitter"] = "strict_all_edges"
    strict.manifest["largest_component_rows"] = largest_component_rows
    strict.manifest["largest_component_share"] = largest_component_share
    if largest_component_share <= max_strict_component_share or large_component_strategy == "strict":
        return strict
    if large_component_strategy != "positive_record_holdout":
        raise ValueError(f"unknown large component strategy: {large_component_strategy!r}")

    fallback = _positive_record_holdout_split(
        frame,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
        strict_largest_component_rows=largest_component_rows,
        strict_largest_component_share=largest_component_share,
        max_strict_component_share=max_strict_component_share,
    )
    return fallback


def _strict_component_aware_split(
    frame: pd.DataFrame,
    *,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitResult:
    ratio_sum = train_ratio + dev_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/dev/test ratios must sum to 1.0")
    if frame.empty:
        raise ValueError("cannot split empty frame")

    output = frame.copy()
    output["component_id"] = _component_ids(output)
    split_ratios = {"train": train_ratio, "dev": dev_ratio, "test": test_ratio}
    total_counts = _component_feature_counts(output)
    targets = {
        split_name: {feature: count * ratio for feature, count in total_counts.items()}
        for split_name, ratio in split_ratios.items()
    }
    current = {split_name: {} for split_name in split_ratios}
    assignments: dict[str, str] = {}
    components = []
    for component_id, component_frame in output.groupby("component_id", sort=False):
        counts = _component_feature_counts(component_frame)
        components.append((component_id, counts, len(component_frame), _hash_fraction(component_id, seed)))
    components.sort(key=lambda item: (-item[2], item[3], item[0]))

    for component_id, counts, _, _ in components:
        best_split = min(
            split_ratios,
            key=lambda split_name: (
                _score_assignment(current, targets, split_name, counts),
                current[split_name].get("rows", 0.0),
                split_name,
            ),
        )
        assignments[component_id] = best_split
        for feature, count in counts.items():
            current[best_split][feature] = current[best_split].get(feature, 0.0) + count

    output["split"] = output["component_id"].map(assignments)
    leak_rows = _split_leakage_rows(output)
    if leak_rows:
        raise RuntimeError(f"component split leakage detected for {len(leak_rows)} raw_record_ids")

    manifest = {
        "created_at": _now_utc(),
        "seed": seed,
        "ratios": split_ratios,
        "rows": int(len(output)),
        "components": int(output["component_id"].nunique()),
        "split_counts": output["split"].value_counts().to_dict(),
        "split_target_counts": _split_nested_counts(output, "same_base_product"),
        "category_split_counts": _split_nested_counts(output, "category_run"),
    }
    return SplitResult(pairs=output.reset_index(drop=True), manifest=manifest, dropped_pairs=pd.DataFrame())


def _positive_record_units(frame: pd.DataFrame) -> dict[str, str]:
    union_find = _UnionFind()
    raw_ids: set[str] = set()
    for _, row in frame.iterrows():
        left = _clean_text(row.get("raw_record_id_a"))
        right = _clean_text(row.get("raw_record_id_b"))
        if left:
            raw_ids.add(left)
            union_find.find(left)
        if right:
            raw_ids.add(right)
            union_find.find(right)
        if int(row.get("same_base_product", 0)) == 1 and left and right:
            union_find.union(left, right)

    roots = {raw_id: union_find.find(raw_id) for raw_id in raw_ids}
    root_to_id = {root: f"positive_component_{idx:06d}" for idx, root in enumerate(sorted(set(roots.values())))}
    return {raw_id: root_to_id[root] for raw_id, root in roots.items()}


def _unit_feature_counts(frame: pd.DataFrame, unit_id: str) -> dict[str, float]:
    incident = frame[frame["record_component_a"].eq(unit_id) | frame["record_component_b"].eq(unit_id)]
    counts = _component_feature_counts(incident)
    counts["units"] = 1.0
    return counts


def _positive_record_holdout_split(
    frame: pd.DataFrame,
    *,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
    strict_largest_component_rows: int,
    strict_largest_component_share: float,
    max_strict_component_share: float,
) -> SplitResult:
    ratio_sum = train_ratio + dev_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/dev/test ratios must sum to 1.0")
    if frame.empty:
        raise ValueError("cannot split empty frame")

    raw_to_unit = _positive_record_units(frame)
    output = frame.copy()
    output["record_component_a"] = output["raw_record_id_a"].map(lambda value: raw_to_unit.get(_clean_text(value), ""))
    output["record_component_b"] = output["raw_record_id_b"].map(lambda value: raw_to_unit.get(_clean_text(value), ""))
    if output["record_component_a"].eq("").any() or output["record_component_b"].eq("").any():
        raise ValueError("cannot build record-aware split: empty raw_record_id found")

    split_ratios = {"train": train_ratio, "dev": dev_ratio, "test": test_ratio}
    unit_ids = sorted(set(output["record_component_a"]) | set(output["record_component_b"]))
    unit_counts = {unit_id: _unit_feature_counts(output, unit_id) for unit_id in unit_ids}
    total_counts: dict[str, float] = {}
    for counts in unit_counts.values():
        for feature, count in counts.items():
            total_counts[feature] = total_counts.get(feature, 0.0) + count
    targets = {
        split_name: {feature: count * ratio for feature, count in total_counts.items()}
        for split_name, ratio in split_ratios.items()
    }
    current = {split_name: {} for split_name in split_ratios}
    assignments: dict[str, str] = {}
    units = [
        (unit_id, counts, counts.get("rows", 0.0), _hash_fraction(unit_id, seed))
        for unit_id, counts in unit_counts.items()
    ]
    units.sort(key=lambda item: (-item[2], item[3], item[0]))

    for unit_id, counts, _, _ in units:
        best_split = min(
            split_ratios,
            key=lambda split_name: (
                _score_assignment(current, targets, split_name, counts),
                current[split_name].get("rows", 0.0),
                split_name,
            ),
        )
        assignments[unit_id] = best_split
        for feature, count in counts.items():
            current[best_split][feature] = current[best_split].get(feature, 0.0) + count

    output["record_split_a"] = output["record_component_a"].map(assignments)
    output["record_split_b"] = output["record_component_b"].map(assignments)
    keep_mask = output["record_split_a"].eq(output["record_split_b"])
    kept = output[keep_mask].copy()
    kept["split"] = kept["record_split_a"]
    kept["component_id"] = [
        f"{left}__{right}" if left != right else left
        for left, right in zip(kept["record_component_a"], kept["record_component_b"], strict=False)
    ]
    dropped = output[~keep_mask].copy()
    if not dropped.empty:
        dropped["drop_reason"] = "cross_split_pair_after_positive_record_holdout"

    leak_rows = _split_leakage_rows(kept)
    if leak_rows:
        raise RuntimeError(f"record-aware split leakage detected for {len(leak_rows)} raw_record_ids")

    manifest = {
        "created_at": _now_utc(),
        "splitter": "positive_record_holdout",
        "fallback_from": "strict_all_edges",
        "fallback_reason": "strict component exceeded max_strict_component_share",
        "strict_largest_component_rows": strict_largest_component_rows,
        "strict_largest_component_share": strict_largest_component_share,
        "max_strict_component_share": max_strict_component_share,
        "seed": seed,
        "ratios": split_ratios,
        "input_rows": int(len(output)),
        "rows": int(len(kept)),
        "dropped_rows": int(len(dropped)),
        "record_components": int(len(unit_ids)),
        "components": int(kept["component_id"].nunique()),
        "split_counts": kept["split"].value_counts().to_dict(),
        "dropped_target_counts": (
            dropped["same_base_product"].value_counts(dropna=False).to_dict() if not dropped.empty else {}
        ),
        "split_target_counts": _split_nested_counts(kept, "same_base_product"),
        "category_split_counts": _split_nested_counts(kept, "category_run"),
    }
    return SplitResult(
        pairs=kept.reset_index(drop=True),
        manifest=manifest,
        dropped_pairs=dropped.reset_index(drop=True),
    )


def _split_leakage_rows(frame: pd.DataFrame) -> list[str]:
    seen: dict[str, set[str]] = {}
    for _, row in frame.iterrows():
        split_name = _clean_text(row.get("split"))
        for column in ("raw_record_id_a", "raw_record_id_b"):
            raw_id = _clean_text(row.get(column))
            if raw_id:
                seen.setdefault(raw_id, set()).add(split_name)
    return sorted(raw_id for raw_id, splits in seen.items() if len(splits) > 1)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_output_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    if "pair_id" not in output.columns:
        if "pair_key" in output.columns:
            output["pair_id"] = output["pair_key"].map(pair_id_from_key)
        elif {"raw_record_id_a", "raw_record_id_b"}.issubset(output.columns):
            keys = [
                pair_key(left, right)
                for left, right in zip(output["raw_record_id_a"], output["raw_record_id_b"], strict=False)
            ]
            output["pair_id"] = [pair_id_from_key(value) for value in keys]
    existing_columns = [column for column in columns if column in output.columns]
    return output.reindex(columns=existing_columns).copy()


def write_freeze_outputs(
    result: DatasetFreezeResult,
    output_dir: Path,
    *,
    prefix: str = "dedup_pairs_final",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pairs": output_dir / f"{prefix}.csv",
        "conflicts": output_dir / f"{prefix}_conflicts.csv",
        "excluded": output_dir / f"{prefix}_excluded.csv",
        "manifest": output_dir / f"{prefix}_manifest.json",
    }
    clean_output_frame(result.pairs, CLEAN_PAIR_COLUMNS).to_csv(paths["pairs"], index=False)
    clean_output_frame(result.conflicts, CLEAN_CONFLICT_COLUMNS).to_csv(paths["conflicts"], index=False)
    clean_output_frame(result.excluded, CLEAN_EXCLUDED_COLUMNS).to_csv(paths["excluded"], index=False)
    manifest = dict(result.manifest)
    manifest["csv_schema"] = "dedup_training_clean_v1"
    manifest["output_paths"] = {key: str(path) for key, path in paths.items()}
    write_json(paths["manifest"], manifest)
    return paths


def write_split_outputs(result: SplitResult, output_dir: Path, *, prefix: str = "dedup_pairs_final") -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "split_pairs": output_dir / f"{prefix}_split.csv",
        "split_manifest": output_dir / f"{prefix}_split_manifest.json",
    }
    if result.dropped_pairs is not None and not result.dropped_pairs.empty:
        paths["split_dropped"] = output_dir / f"{prefix}_split_dropped.csv"
    clean_output_frame(result.pairs, CLEAN_SPLIT_COLUMNS).to_csv(paths["split_pairs"], index=False)
    if "split_dropped" in paths and result.dropped_pairs is not None:
        clean_output_frame(result.dropped_pairs, CLEAN_DROPPED_COLUMNS).to_csv(paths["split_dropped"], index=False)
    manifest = dict(result.manifest)
    manifest["csv_schema"] = "dedup_training_clean_v1"
    manifest["output_paths"] = {key: str(path) for key, path in paths.items()}
    write_json(paths["split_manifest"], manifest)
    return paths
