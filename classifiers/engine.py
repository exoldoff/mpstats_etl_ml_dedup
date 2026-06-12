from __future__ import annotations

from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Iterable, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_RULE_COLUMNS: Tuple[str, ...] = (
    "active",
    "priority",
    "category",
    "target_column",
    "match_field",
    "match_type",
    "pattern",
    "set_value",
    "mode",
)

NUMERIC_MATCH_TYPES = {"gt", "gte", "lt", "lte"}
ALLOWED_MATCH_TYPES = {
    "contains",
    "not_contains",
    "regex",
    "equals",
    "startswith",
    "otherwise",
    *NUMERIC_MATCH_TYPES,
}
ALLOWED_MODES = {"fill_empty", "overwrite"}
ALLOWED_LOGIC_OPERATORS = {"and", "or"}
CONDITIONS_COLUMN = "conditions_json"
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
CATEGORY_FILTER_SEPARATOR = re.compile(r"\s*(?:\||;)\s*")
FULL_RULE_TIMING_ENV = "MPSTATS_CLASSIFIER_VERBOSE_RULES"
SLOW_RULE_THRESHOLD_ENV = "MPSTATS_CLASSIFIER_SLOW_RULE_SEC"
DEFAULT_SLOW_RULE_THRESHOLD_SEC = 0.5


def default_rules_path(base_dir: str | Path | None = None) -> Path:
    """Return default rules file path."""
    if base_dir is None:
        return Path(__file__).resolve().parent / "rules.csv"
    return Path(base_dir) / "rules.csv"


def load_rules(rules_path: str | Path) -> pd.DataFrame:
    """Load classifier rules from CSV or Excel."""
    path = Path(rules_path)
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")

    if path.suffix.lower() == ".csv":
        # Prefer deterministic separators first. `sep=None` can mis-detect when
        # conditions_json contains commas inside JSON payload.
        for sep in (";", ","):
            candidate = pd.read_csv(path, sep=sep, dtype=str).fillna("")
            if all(col in candidate.columns for col in REQUIRED_RULE_COLUMNS):
                return candidate
        return pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str).fillna("")
    raise ValueError(f"Unsupported rules format: {path.suffix}. Use .csv/.xlsx/.xls")


def _rules_file_signature(rules_path: str | Path) -> tuple[str, int, int]:
    path = Path(rules_path)
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=16)
def _load_prepared_rules_cached(path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    del mtime_ns, size
    return _validate_and_prepare_rules(load_rules(path))


def load_prepared_rules(rules_path: str | Path) -> pd.DataFrame:
    path, mtime_ns, size = _rules_file_signature(rules_path)
    return _load_prepared_rules_cached(path, mtime_ns, size)


def _to_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on", "да"}


def _normalize_string_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = out[col].astype(str).str.strip()
    return out


def _parse_conditions_json(raw_value: object, row_num: int) -> list[dict[str, str]]:
    text = str(raw_value).strip()
    if text == "":
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Rule row {row_num} has invalid {CONDITIONS_COLUMN}: {exc}") from exc

    if isinstance(parsed, dict):
        parsed = parsed.get("conditions")

    if not isinstance(parsed, list) or not parsed:
        raise ValueError(
            f"Rule row {row_num} must provide a non-empty JSON array in {CONDITIONS_COLUMN}."
        )

    normalized: list[dict[str, str]] = []
    for idx, condition in enumerate(parsed, start=1):
        if not isinstance(condition, dict):
            raise ValueError(f"Rule row {row_num} condition #{idx} must be a JSON object.")

        join_with_prev = str(condition.get("join_with_prev", "and")).strip().lower() or "and"
        if join_with_prev not in ALLOWED_LOGIC_OPERATORS:
            raise ValueError(
                f"Rule row {row_num} condition #{idx} has invalid join_with_prev "
                f"'{join_with_prev}'. Use 'and' or 'or'."
            )

        match_field = str(condition.get("match_field", "")).strip()
        match_type = str(condition.get("match_type", "")).strip().lower()
        pattern = str(condition.get("pattern", "")).strip()

        if match_type != "otherwise" and match_field == "":
            raise ValueError(f"Rule row {row_num} condition #{idx} has empty match_field.")
        if match_type not in ALLOWED_MATCH_TYPES:
            raise ValueError(
                f"Rule row {row_num} condition #{idx} has invalid match_type '{match_type}'."
            )
        if match_type != "otherwise" and pattern == "":
            raise ValueError(f"Rule row {row_num} condition #{idx} has empty pattern.")

        normalized.append(
            {
                "join_with_prev": join_with_prev,
                "match_field": match_field,
                "match_type": match_type,
                "pattern": pattern,
            }
        )
    return normalized


def _validate_and_prepare_rules(rules_df: pd.DataFrame) -> pd.DataFrame:
    out = rules_df.copy()
    missing = [c for c in REQUIRED_RULE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Rules file is missing required columns: {missing}")

    if CONDITIONS_COLUMN not in out.columns:
        out[CONDITIONS_COLUMN] = ""

    out["row_num"] = range(2, len(out) + 2)  # +2 because header is row 1.
    out = _normalize_string_columns(
        out,
        [
            "active",
            "priority",
            "category",
            "target_column",
            "match_field",
            "match_type",
            "pattern",
            "set_value",
            "mode",
            CONDITIONS_COLUMN,
        ],
    )

    out["active"] = out["active"].map(_to_bool)
    out["priority"] = pd.to_numeric(out["priority"], errors="coerce").fillna(9999).astype(int)
    out["match_type"] = out["match_type"].str.lower()
    out["mode"] = out["mode"].str.lower().replace("", "fill_empty")
    has_conditions = out[CONDITIONS_COLUMN].ne("")
    bad_match = out.loc[
        (~has_conditions) & (~out["match_type"].isin(ALLOWED_MATCH_TYPES)),
        ["row_num", "match_type"],
    ]
    if not bad_match.empty:
        raise ValueError(
            "Invalid match_type in rules rows: "
            + ", ".join(f"{int(r.row_num)}='{r.match_type}'" for r in bad_match.itertuples())
        )

    bad_mode = out.loc[~out["mode"].isin(ALLOWED_MODES), ["row_num", "mode"]]
    if not bad_mode.empty:
        raise ValueError(
            "Invalid mode in rules rows: "
            + ", ".join(f"{int(r.row_num)}='{r.mode}'" for r in bad_mode.itertuples())
        )

    active_rules = out[out["active"]]
    bad_target = active_rules.loc[active_rules["target_column"].eq(""), "row_num"].tolist()
    if bad_target:
        raise ValueError(f"Column 'target_column' must be non-empty in active rules rows: {bad_target}")
    bad_set_value = active_rules.loc[active_rules["set_value"].eq(""), "row_num"].tolist()
    if bad_set_value:
        raise ValueError(f"Column 'set_value' must be non-empty in active rules rows: {bad_set_value}")

    parsed_conditions: list[list[dict[str, str]]] = []
    bad_match_field_rows: list[int] = []
    bad_pattern_rows: list[int] = []
    for row in out.itertuples(index=False):
        row_num = int(row.row_num)
        conditions = _parse_conditions_json(getattr(row, CONDITIONS_COLUMN, ""), row_num)
        parsed_conditions.append(conditions)

        if row.active and not conditions and str(row.match_type).strip().lower() != "otherwise":
            if str(row.match_field).strip() == "":
                bad_match_field_rows.append(row_num)
            if str(row.pattern).strip() == "":
                bad_pattern_rows.append(row_num)

    if bad_match_field_rows:
        raise ValueError(
            "Column 'match_field' must be non-empty in active rules rows without conditions_json: "
            f"{bad_match_field_rows}"
        )
    if bad_pattern_rows:
        raise ValueError(
            "Column 'pattern' must be non-empty in active rules rows without conditions_json: "
            f"{bad_pattern_rows}"
        )

    out["parsed_conditions"] = parsed_conditions
    return out.sort_values(["priority", "row_num"], kind="stable").reset_index(drop=True)


def _is_empty_series(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return series.isna() | text.str.strip().fillna("").eq("")


def _normalize_fill_unclassified(
    fill_unclassified: dict[str, object] | None,
) -> dict[str, str]:
    if fill_unclassified is None:
        return {}
    if not isinstance(fill_unclassified, dict):
        raise ValueError("fill_unclassified must be a dict {column_name: fill_value}.")

    normalized: dict[str, str] = {}
    for raw_column, raw_value in fill_unclassified.items():
        column_name = str(raw_column).strip()
        if column_name == "":
            raise ValueError("fill_unclassified contains an empty column name.")
        fill_value = str(raw_value).strip()
        if fill_value == "":
            raise ValueError(
                f"fill_unclassified for column '{column_name}' has empty fill_value."
            )
        normalized[column_name] = fill_value
    return normalized


def _numeric_series(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    text = text.str.replace("\u00a0", "", regex=False)
    text = text.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    extracted = text.str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _numeric_pattern(pattern: str, match_type: str) -> float:
    text = str(pattern).strip().replace("\u00a0", "").replace(" ", "")
    match = NUMBER_PATTERN.search(text)
    if match is None:
        raise ValueError(
            f"Numeric match_type '{match_type}' requires a numeric pattern, got '{pattern}'."
        )
    return float(match.group(0).replace(",", "."))


def _build_numeric_match_mask(series: pd.Series, match_type: str, pattern: str) -> pd.Series:
    values = _numeric_series(series)
    threshold = _numeric_pattern(pattern, match_type)
    if match_type == "gt":
        return values.gt(threshold).fillna(False)
    if match_type == "gte":
        return values.ge(threshold).fillna(False)
    if match_type == "lt":
        return values.lt(threshold).fillna(False)
    if match_type == "lte":
        return values.le(threshold).fillna(False)
    raise ValueError(f"Unsupported numeric match_type: {match_type}")


def _build_match_mask(series: pd.Series, match_type: str, pattern: str) -> pd.Series:
    if match_type in NUMERIC_MATCH_TYPES:
        return _build_numeric_match_mask(series, match_type, pattern)

    text = series.fillna("").astype(str)
    if match_type == "contains":
        return text.str.contains(pattern, case=False, regex=False, na=False)
    if match_type == "not_contains":
        return ~text.str.contains(pattern, case=False, regex=False, na=False)
    if match_type == "regex":
        try:
            return text.str.contains(pattern, case=False, regex=True, na=False)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern '{pattern}': {exc}") from exc
    if match_type == "equals":
        return text.str.strip().str.casefold().eq(pattern.strip().casefold())
    if match_type == "startswith":
        return text.str.startswith(pattern, na=False)
    raise ValueError(f"Unsupported match_type: {match_type}")


def _build_condition_mask(
    df: pd.DataFrame,
    *,
    row_num: int,
    match_field: str,
    match_type: str,
    pattern: str,
) -> pd.Series:
    if match_type == "otherwise":
        return pd.Series(True, index=df.index)
    if match_field not in df.columns:
        raise ValueError(f"Rule row {row_num} refers to missing column '{match_field}'")
    return _build_match_mask(df[match_field], match_type, pattern)


def _build_rule_mask(
    df: pd.DataFrame,
    *,
    row_num: int,
    conditions: list[dict[str, str]],
    match_field: str,
    match_type: str,
    pattern: str,
) -> pd.Series:
    has_primary_condition = match_type == "otherwise" or all(
        str(value).strip() != "" for value in (match_field, match_type, pattern)
    )

    if has_primary_condition:
        mask = _build_condition_mask(
            df,
            row_num=row_num,
            match_field=match_field,
            match_type=match_type,
            pattern=pattern,
        )
        for condition in conditions:
            cond_mask = _build_condition_mask(
                df,
                row_num=row_num,
                match_field=condition["match_field"],
                match_type=condition["match_type"],
                pattern=condition["pattern"],
            )
            if condition["join_with_prev"] == "or":
                mask = mask | cond_mask
            else:
                mask = mask & cond_mask
        return mask

    if conditions:
        first = conditions[0]
        mask = _build_condition_mask(
            df,
            row_num=row_num,
            match_field=first["match_field"],
            match_type=first["match_type"],
            pattern=first["pattern"],
        )
        for condition in conditions[1:]:
            cond_mask = _build_condition_mask(
                df,
                row_num=row_num,
                match_field=condition["match_field"],
                match_type=condition["match_type"],
                pattern=condition["pattern"],
            )
            if condition["join_with_prev"] == "or":
                mask = mask | cond_mask
            else:
                mask = mask & cond_mask
        return mask

    return _build_condition_mask(
        df,
        row_num=row_num,
        match_field=match_field,
        match_type=match_type,
        pattern=pattern,
    )


def _rule_uses_otherwise(match_type: str, conditions: list[dict[str, str]]) -> bool:
    return match_type == "otherwise" or any(condition["match_type"] == "otherwise" for condition in conditions)


def _split_rule_categories(rule_category: object) -> list[str]:
    text = str(rule_category).strip()
    if not text or text == "*":
        return []
    parts = CATEGORY_FILTER_SEPARATOR.split(text) if "|" in text or ";" in text else [text]
    categories = [part.strip() for part in parts if part.strip()]
    return [] if "*" in categories else categories


def _build_category_mask_from_normalized(
    normalized_category: pd.Series,
    rule_categories: list[str],
) -> pd.Series:
    normalized_categories = {category.casefold() for category in rule_categories}
    return normalized_category.isin(normalized_categories)


def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "да", "all", "full"}


def _slow_rule_threshold_seconds() -> float:
    raw_value = os.getenv(SLOW_RULE_THRESHOLD_ENV, "").strip()
    if not raw_value:
        return DEFAULT_SLOW_RULE_THRESHOLD_SEC
    try:
        return max(0.0, float(raw_value.replace(",", ".")))
    except ValueError:
        LOGGER.warning("Ignoring invalid %s=%r: expected seconds as a number.", SLOW_RULE_THRESHOLD_ENV, raw_value)
        return DEFAULT_SLOW_RULE_THRESHOLD_SEC


def _log_classifier_payload(message: str, payload: dict[str, object]) -> None:
    LOGGER.info("%s: %s", message, json.dumps(payload, ensure_ascii=False, default=str))


def _rule_matches_category_set(rule_category: object, normalized_categories: set[str]) -> bool:
    """Return True when a rule can be relevant for at least one category in the slice.

    Empty category and "*" mean a global rule. Multiple categories in one cell
    are split the same way as the runtime row-level category filter.
    """
    rule_categories = _split_rule_categories(rule_category)
    if not rule_categories:
        return True
    return any(category.casefold() in normalized_categories for category in rule_categories)


def _prefilter_rules_for_slice_category(
    rules: pd.DataFrame,
    *,
    slice_category: str | None,
    category_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Filter rules before applying them to a single category slice.

    The web pipeline processes one category per file, so rules for unrelated
    categories cannot affect the result. To keep older dynamic behavior safe,
    the filter also includes categories that can be reached by active rules
    writing into the category column.
    """
    clean_category = str(slice_category or "").strip()
    if clean_category == "":
        return rules, []

    normalized_categories = {clean_category.casefold()}
    display_categories = {clean_category}
    changed = True
    while changed:
        changed = False
        for rule in rules.itertuples(index=False):
            if not bool(rule.active):
                continue
            if str(rule.target_column).strip() != category_column:
                continue
            next_category = str(rule.set_value).strip()
            if next_category == "":
                continue
            if not _rule_matches_category_set(rule.category, normalized_categories):
                continue
            normalized_next = next_category.casefold()
            if normalized_next not in normalized_categories:
                normalized_categories.add(normalized_next)
                display_categories.add(next_category)
                changed = True

    keep_mask = rules["category"].map(lambda value: _rule_matches_category_set(value, normalized_categories))
    return rules.loc[keep_mask].reset_index(drop=True), sorted(display_categories, key=str.casefold)


def _empty_share_by_target(out: pd.DataFrame, target_columns: Iterable[str]) -> dict[str, float]:
    shares: dict[str, float] = {}
    total_rows = len(out)
    if total_rows == 0:
        return {column: 0.0 for column in target_columns if column in out.columns}
    for column in target_columns:
        if column in out.columns:
            shares[column] = round(float(_is_empty_series(out[column]).mean()), 4)
    return shares


def apply_classifiers(
    df: pd.DataFrame,
    rules_path: str | Path | None = None,
    *,
    category_column: str = "Категория",
    fill_unclassified: dict[str, object] | None = None,
    slice_category: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply CSV/Excel rules to dataframe and return:
      1) transformed dataframe
      2) per-rule execution report

    Optional:
      - fill_unclassified: dict {target_column: fill_value}
        Applied after all rules; fills only empty values in selected columns.
      - slice_category: category name of the current processed file. When set,
        rules for other categories are skipped before the rule loop.
    """
    if rules_path is None:
        rules_path = default_rules_path()

    total_started = time.perf_counter()
    out = df.copy()
    initial_columns = set(out.columns)
    all_rules = load_prepared_rules(rules_path)
    rules, prefilter_categories = _prefilter_rules_for_slice_category(
        all_rules,
        slice_category=slice_category,
        category_column=category_column,
    )
    fill_unclassified_map = _normalize_fill_unclassified(fill_unclassified)
    report_rows: list[dict[str, object]] = []
    rule_timings: list[dict[str, object]] = []
    normalized_category_cache: pd.Series | None = None
    target_empty_cache: dict[str, pd.Series] = {}
    created_columns: set[str] = set()
    log_all_rules = _truthy_env(os.getenv(FULL_RULE_TIMING_ENV))
    slow_rule_threshold_sec = _slow_rule_threshold_seconds()

    def normalized_category(row_num: int, rule_categories: list[str]) -> pd.Series:
        nonlocal normalized_category_cache
        if category_column not in out.columns:
            category_filter = " | ".join(rule_categories)
            raise ValueError(
                f"Rule row {row_num} has category filter '{category_filter}', "
                f"but dataframe has no '{category_column}' column."
            )
        if normalized_category_cache is None:
            normalized_category_cache = (
                out[category_column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.casefold()
            )
        return normalized_category_cache

    def target_empty(column_name: str) -> pd.Series:
        if column_name not in target_empty_cache:
            target_empty_cache[column_name] = _is_empty_series(out[column_name])
        return target_empty_cache[column_name]

    for rule in rules.itertuples(index=False):
        rule_started = time.perf_counter()
        conditions = list(getattr(rule, "parsed_conditions", []))
        row_num = int(rule.row_num)
        rule_id = f"row:{row_num}"
        priority = int(rule.priority) if str(getattr(rule, "priority", "")).strip() else None
        target_column = str(rule.target_column)
        candidate_rows = 0
        applied_rows = 0
        reason = ""

        if not rule.active:
            duration_sec = time.perf_counter() - rule_started
            timing_row = {
                "rule_id": rule_id,
                "row_num": row_num,
                "priority": priority,
                "category": rule.category,
                "target_column": target_column,
                "match_type": rule.match_type,
                "mode": rule.mode,
                "candidate_rows": 0,
                "applied_rows": 0,
                "duration_sec": round(duration_sec, 6),
                "active": False,
            }
            rule_timings.append(timing_row)
            if log_all_rules:
                _log_classifier_payload("Classifier rule timing", timing_row)
            report_rows.append(
                {
                    "rule_id": rule_id,
                    "row_num": row_num,
                    "active": False,
                    "applied_rows": 0,
                    "candidate_rows": 0,
                    "duration_sec": round(duration_sec, 6),
                    "reason": "inactive",
                    "target_column": rule.target_column,
                    "comment": getattr(rule, "comment", ""),
                    "conditions_count": len(conditions),
                    CONDITIONS_COLUMN: getattr(rule, CONDITIONS_COLUMN, ""),
                }
            )
            continue

        if target_column not in out.columns:
            out[target_column] = pd.NA
            created_columns.add(target_column)

        rule_categories = _split_rule_categories(rule.category)
        base_mask = pd.Series(True, index=out.index)
        if rule_categories:
            base_mask = _build_category_mask_from_normalized(
                normalized_category(row_num, rule_categories),
                rule_categories,
            )

        write_only_empty = rule.mode == "fill_empty" or _rule_uses_otherwise(rule.match_type, conditions)
        if write_only_empty:
            base_mask = base_mask & target_empty(target_column)

        if base_mask.any():
            match_mask = _build_rule_mask(
                out.loc[base_mask],
                row_num=row_num,
                conditions=conditions,
                match_field=rule.match_field,
                match_type=rule.match_type,
                pattern=rule.pattern,
            )
            mask = pd.Series(False, index=out.index)
            mask.loc[base_mask] = match_mask.to_numpy(dtype=bool)
        else:
            mask = pd.Series(False, index=out.index)
            reason = "no_candidate_rows"

        candidate_rows = int(mask.sum())
        write_mask = mask
        applied_rows = candidate_rows
        if applied_rows > 0:
            out.loc[write_mask, target_column] = rule.set_value
            if target_column in target_empty_cache:
                target_empty_cache[target_column] = target_empty_cache[target_column] & ~write_mask
            if target_column == category_column:
                normalized_category_cache = None

        duration_sec = time.perf_counter() - rule_started
        timing_row = {
            "rule_id": rule_id,
            "row_num": row_num,
            "priority": priority,
            "category": rule.category,
            "target_column": target_column,
            "match_type": rule.match_type,
            "mode": rule.mode,
            "candidate_rows": candidate_rows,
            "applied_rows": applied_rows,
            "duration_sec": round(duration_sec, 6),
            "active": True,
        }
        rule_timings.append(timing_row)
        if log_all_rules or duration_sec >= slow_rule_threshold_sec:
            _log_classifier_payload("Classifier rule timing", timing_row)

        report_rows.append(
            {
                "rule_id": rule_id,
                "row_num": row_num,
                "active": True,
                "priority": int(rule.priority),
                "category": rule.category,
                "target_column": target_column,
                "match_field": rule.match_field,
                "match_type": rule.match_type,
                "pattern": rule.pattern,
                "set_value": rule.set_value,
                "mode": rule.mode,
                "candidate_rows": candidate_rows,
                "applied_rows": applied_rows,
                "duration_sec": round(duration_sec, 6),
                "reason": reason,
                "comment": getattr(rule, "comment", ""),
                "conditions_count": len(conditions),
                CONDITIONS_COLUMN: getattr(rule, CONDITIONS_COLUMN, ""),
            }
        )

    for column_name, fill_value in fill_unclassified_map.items():
        if column_name not in out.columns:
            out[column_name] = pd.NA
            created_columns.add(column_name)
        empty_mask = _is_empty_series(out[column_name])
        if empty_mask.any():
            out.loc[empty_mask, column_name] = fill_value

    report = pd.DataFrame(report_rows)
    active_rules_count = int(rules["active"].sum()) if "active" in rules.columns else len(rules)
    inactive_rules_count = int(len(rules) - active_rules_count)
    if report.empty:
        rows_per_target: dict[str, int] = {}
    else:
        active_report = report[report["active"] == True].copy() if "active" in report.columns else report.copy()
        rows_per_target = {
            str(target): int(rows["applied_rows"].sum())
            for target, rows in active_report.groupby("target_column", dropna=False)
        }
    rules_per_target = {
        str(target): int(count)
        for target, count in rules.loc[rules["active"], "target_column"].value_counts(dropna=False).items()
    }
    seconds_per_target: dict[str, float] = {}
    for timing in rule_timings:
        target = str(timing["target_column"])
        seconds_per_target[target] = round(seconds_per_target.get(target, 0.0) + float(timing["duration_sec"]), 6)
    target_columns = set(rows_per_target) | set(rules_per_target) | created_columns
    summary = {
        "total_rules_loaded": int(len(all_rules)),
        "rules_after_category_prefilter": int(len(rules)),
        "category_prefilter_values": prefilter_categories,
        "active_rules_count": active_rules_count,
        "inactive_rules_count": inactive_rules_count,
        "total_classification_time_sec": round(time.perf_counter() - total_started, 6),
        "top_10_slowest_rules": sorted(rule_timings, key=lambda item: float(item["duration_sec"]), reverse=True)[:10],
        "columns_created_by_classifier": sorted(created_columns - initial_columns),
        "rows_classified_per_target_column": rows_per_target,
        "rules_per_target_column": rules_per_target,
        "seconds_per_target_column": seconds_per_target,
        "empty_share_per_target_column": _empty_share_by_target(out, sorted(target_columns)),
    }
    _log_classifier_payload("Classifier summary", summary)
    return out, report
