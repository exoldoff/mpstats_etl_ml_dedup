from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


DEFAULT_SALES_MIN_QUANTILE: float | None = None
DEFAULT_SALES_MIN_UNITS = 15.0
SALES_COLUMN_CANDIDATES = ("Продажи, шт", "Продажи", "sales")
DEFAULT_SALES_FILTER_GROUP_COLUMNS = ("__project_name", "__marketplace_code", "Категория", "__year", "__month")


def coerce_sales_series(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    text = text.str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").fillna(0.0).astype("float64")


def resolve_sales_column(columns: Iterable[object], *, candidates: Iterable[str] = SALES_COLUMN_CANDIDATES) -> str | None:
    available = {str(column): str(column) for column in columns}
    for candidate in candidates:
        if candidate in available:
            return available[candidate]
    return None


def filter_sales_by_quantile(
    df: pd.DataFrame,
    *,
    quantile: float | None = DEFAULT_SALES_MIN_QUANTILE,
    min_sales: float = DEFAULT_SALES_MIN_UNITS,
    sales_column: str | None = None,
    group_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    if quantile is not None and not 0 <= quantile < 1:
        raise ValueError(f"sales quantile must be >= 0 and < 1, got {quantile!r}")
    if min_sales < 0:
        raise ValueError(f"minimum sales must be >= 0, got {min_sales!r}")

    column = sales_column or resolve_sales_column(df.columns)
    if column is None:
        raise ValueError("В данных нет колонки продаж: 'Продажи, шт', 'Продажи' или 'sales'.")

    out = df.copy()
    out[column] = coerce_sales_series(out[column])
    positive_mask = out[column] > 0
    if not positive_mask.any():
        return out.iloc[0:0].copy()

    min_sales_mask = positive_mask & (out[column] >= float(min_sales))
    if quantile is None:
        return out.loc[min_sales_mask].copy()

    groups = [group for group in (group_columns or []) if group in out.columns]
    keep_mask = pd.Series(False, index=out.index)
    if groups:
        positive = out.loc[positive_mask, groups + [column]].copy()
        thresholds = positive.groupby(groups, dropna=False)[column].transform(lambda values: values.quantile(quantile))
        effective_thresholds = thresholds.clip(lower=float(min_sales))
        keep_mask.loc[positive.index] = positive[column] >= effective_thresholds
    else:
        threshold = out.loc[positive_mask, column].quantile(quantile)
        effective_threshold = max(float(threshold), float(min_sales))
        keep_mask = positive_mask & (out[column] >= effective_threshold)

    return out.loc[keep_mask].copy()
