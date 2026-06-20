from __future__ import annotations

from pathlib import Path

import pandas as pd


LABEL_EXACT_DUPLICATE = "exact_duplicate"
LABEL_DIFFERENT_PRODUCT = "different_product"
LABEL_UNCERTAIN = "uncertain"

VALID_LABELS = (
    LABEL_EXACT_DUPLICATE,
    LABEL_DIFFERENT_PRODUCT,
    LABEL_UNCERTAIN,
)

LABEL_BY_KEY = {
    "w": LABEL_EXACT_DUPLICATE,
    "s": LABEL_DIFFERENT_PRODUCT,
    "d": LABEL_UNCERTAIN,
}

LABEL_HINTS = {
    "w": "same base product",
    "s": "different product",
    "d": "uncertain",
}


def normalize_label_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(value != value):
            return ""
    except TypeError:
        return ""
    return str(value).strip()


def ensure_labeling_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("label", "notes"):
        if column not in frame.columns:
            frame.insert(0 if column == "label" else 1, column, "")
    frame["label"] = frame["label"].map(normalize_label_value)
    frame["notes"] = frame["notes"].map(normalize_label_value)
    return frame


def load_labeling_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Labeling CSV not found: {path}")
    return ensure_labeling_columns(pd.read_csv(path))


def save_labeling_file(frame: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(path)


def is_labeled(value: object) -> bool:
    return normalize_label_value(value) != ""


def labeled_count(frame: pd.DataFrame) -> int:
    return int(frame["label"].map(is_labeled).sum())


def next_unlabeled_index(frame: pd.DataFrame, start: int = 0) -> int | None:
    if frame.empty:
        return None
    start = max(0, min(start, len(frame) - 1))
    for idx in range(start, len(frame)):
        if not is_labeled(frame.at[idx, "label"]):
            return idx
    for idx in range(0, start):
        if not is_labeled(frame.at[idx, "label"]):
            return idx
    return None


def apply_label(frame: pd.DataFrame, row_index: int, key: str) -> str:
    label = LABEL_BY_KEY[key.lower()]
    frame.at[row_index, "label"] = label
    return label


def clear_label(frame: pd.DataFrame, row_index: int) -> None:
    frame.at[row_index, "label"] = ""


def cell(row: pd.Series, column: str) -> str:
    return normalize_label_value(row.get(column, ""))


def first_cell(row: pd.Series, *columns: str) -> str:
    for column in columns:
        value = cell(row, column)
        if value:
            return value
    return "-"


def yes_no(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return "да"
    if normalized in {"false", "0", "no", "n"}:
        return "нет"
    return value or "-"


def item_summary(row: pd.Series, side: str) -> str:
    marketplace = first_cell(row, f"marketplace_{side}")
    sku = first_cell(row, f"sku_{side}")
    brand = first_cell(row, f"brand_{side}")
    subcategory = first_cell(row, f"subcategory_{side}")
    unit = first_cell(row, f"unit_amount_{side}")
    total = first_cell(row, f"total_amount_{side}")
    pack = first_cell(row, f"multipack_count_{side}")
    return f"{marketplace} | sku {sku} | brand {brand} | subcat {subcategory} | unit {unit} | total {total} | x{pack}"


def signal_summary(row: pd.Series) -> str:
    source = first_cell(row, "candidate_source")
    rank = first_cell(row, "candidate_rank")
    score = first_cell(row, "embedding_similarity_score", "baseline_similarity_score")
    blocking_scope = first_cell(row, "blocking_scope")
    subcategory = first_cell(row, "subcategory_relation")
    stratum = first_cell(row, "labeling_stratum")
    cross = yes_no(cell(row, "is_cross_marketplace_pair"))
    hard_negative = yes_no(cell(row, "is_hard_negative_candidate"))
    pack_variant = yes_no(cell(row, "is_pack_variant_candidate"))
    return (
        f"источник={source} | scope={blocking_scope} | subcat={subcategory} | "
        f"rank={rank} | score={score} | стратегия={stratum} | "
        f"межмаркетплейс={cross} | сложный негатив={hard_negative} | вариант упаковки={pack_variant}"
    )
