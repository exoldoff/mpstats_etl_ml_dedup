from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from research.dedup.category_runs import KNOWN_CATEGORY_RUNS, resolve_category_run
from research.dedup.env import find_project_root
from research.dedup.threshold_calibration import detect_sales_volume_columns


DEFAULT_PRODUCTS_TABLE = "mpstats_products"
DEFAULT_SALES_COLUMN = "Продажи, шт"
DEFAULT_MIN_SALES = 15.0
SALES_COLUMN_CANDIDATES = (
    DEFAULT_SALES_COLUMN,
    "Продажи",
    "sales",
    "sales_volume",
    "sales_units",
    "units_sold",
)


@dataclass(frozen=True)
class SalesVolumeStatus:
    status: str
    source: str | None = None
    warning: str | None = None
    matched_left_rows: int = 0
    matched_right_rows: int = 0
    total_rows: int = 0
    lookup_rows: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "source": self.source or "",
            "warning": self.warning or "",
            "matched_left_rows": self.matched_left_rows,
            "matched_right_rows": self.matched_right_rows,
            "total_rows": self.total_rows,
            "lookup_rows": self.lookup_rows,
        }


def marketplace_key(value: object) -> str:
    if value is None:
        return "unknown_marketplace"
    try:
        if bool(value != value):
            return "unknown_marketplace"
    except TypeError:
        return "unknown_marketplace"
    text = str(value).casefold().strip()
    if not text:
        return "unknown_marketplace"
    return " ".join(text.split())


def coerce_sales_series(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    text = text.str.replace("\u00a0", "", regex=False)
    text = text.str.replace(" ", "", regex=False)
    text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").fillna(0.0).astype("float64")


def resolve_sales_column(columns: Iterable[object], requested: str | None = None) -> str | None:
    available = {str(column): str(column) for column in columns}
    if requested and requested in available:
        return available[requested]
    for candidate in SALES_COLUMN_CANDIDATES:
        if candidate in available:
            return available[candidate]
    return None


def parse_category_runs(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in value if str(part).strip()]

    slugs: list[str] = []
    for part in parts:
        slugs.append(resolve_category_run(part).slug)
    return sorted(set(slugs))


def category_runs_from_frame(frame: pd.DataFrame, explicit: str | Sequence[str] | None = None) -> list[str]:
    requested = parse_category_runs(explicit)
    if requested:
        return requested

    if "category_run" in frame.columns:
        slugs: list[str] = []
        for value in frame["category_run"].dropna().astype(str).str.strip().unique():
            if not value:
                continue
            try:
                slugs.append(resolve_category_run(value).slug)
            except ValueError:
                continue
        if slugs:
            return sorted(set(slugs))

    return sorted(KNOWN_CATEGORY_RUNS)


def filter_terms_for_runs(category_runs: Sequence[str]) -> tuple[list[str], list[str]]:
    aliases: list[str] = []
    projects: list[str] = []
    for slug in category_runs:
        run = KNOWN_CATEGORY_RUNS.get(slug)
        if run is None:
            continue
        aliases.extend(run.category_aliases)
        if run.project_name:
            projects.append(run.project_name)
    return list(dict.fromkeys(aliases)), list(dict.fromkeys(projects))


def resolve_duckdb_path(path: str | Path | None = None, *, project_root: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    env_path = os.environ.get("MPSTATS_DUCKDB_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    root = project_root or find_project_root()
    candidates.append(root / "mpstats.duckdb")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _where_in_clause(column: str, values: Sequence[str], params: list[object]) -> str:
    placeholders = ", ".join(["?"] * len(values))
    params.extend(values)
    return f'"{column}" IN ({placeholders})'


def load_sales_lookup_from_duckdb(
    *,
    duckdb_path: str | Path | None = None,
    category_runs: str | Sequence[str] | None = None,
    products_table: str = DEFAULT_PRODUCTS_TABLE,
    sales_column: str | None = DEFAULT_SALES_COLUMN,
    min_sales: float = DEFAULT_MIN_SALES,
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, SalesVolumeStatus]:
    db_path = resolve_duckdb_path(duckdb_path, project_root=project_root)
    if db_path is None:
        return (
            pd.DataFrame(columns=["raw_record_id", "sales_volume"]),
            SalesVolumeStatus(
                status="unit_weight_fallback",
                warning="sales volume lookup disabled: mpstats.duckdb was not found and MPSTATS_DUCKDB_PATH is not set",
            ),
        )

    try:
        import duckdb
    except ImportError:
        return (
            pd.DataFrame(columns=["raw_record_id", "sales_volume"]),
            SalesVolumeStatus(
                status="unit_weight_fallback",
                warning="sales volume lookup disabled: duckdb package is not installed",
            ),
        )

    slugs = parse_category_runs(category_runs) or sorted(KNOWN_CATEGORY_RUNS)
    category_aliases, project_names = filter_terms_for_runs(slugs)

    with duckdb.connect(str(db_path), read_only=True) as connection:
        tables = connection.execute("SHOW TABLES").fetchdf().iloc[:, 0].astype(str).tolist()
        if products_table not in tables:
            return (
                pd.DataFrame(columns=["raw_record_id", "sales_volume"]),
                SalesVolumeStatus(
                    status="unit_weight_fallback",
                    source=str(db_path),
                    warning=f"sales volume lookup disabled: table {products_table!r} not found",
                ),
            )

        columns = set(connection.execute(f"DESCRIBE {products_table}").fetchdf()["column_name"].astype(str))
        missing_base = sorted({"Маркетплейс", "Артикул"} - columns)
        resolved_sales_column = resolve_sales_column(columns, sales_column)
        if missing_base or resolved_sales_column is None:
            missing = missing_base + ([] if resolved_sales_column is not None else [sales_column or "sales volume"])
            return (
                pd.DataFrame(columns=["raw_record_id", "sales_volume"]),
                SalesVolumeStatus(
                    status="unit_weight_fallback",
                    source=str(db_path),
                    warning=f"sales volume lookup disabled: missing columns in {products_table}: {missing}",
                ),
            )

        where_clauses: list[str] = []
        query_params: list[object] = []
        if category_aliases and "Категория" in columns:
            where_clauses.append(_where_in_clause("Категория", category_aliases, query_params))
        if project_names:
            if "__project_name" not in columns:
                return (
                    pd.DataFrame(columns=["raw_record_id", "sales_volume"]),
                    SalesVolumeStatus(
                        status="unit_weight_fallback",
                        source=str(db_path),
                        warning="sales volume lookup disabled: __project_name is required for multi-run filtering but is missing",
                    ),
                )
            where_clauses.append(_where_in_clause("__project_name", project_names, query_params))

        select_sql = ", ".join(f'"{column}"' for column in ["Маркетплейс", "Артикул", resolved_sales_column])
        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        raw = connection.execute(
            f'SELECT {select_sql} FROM "{products_table}"{where_sql}',
            query_params,
        ).fetchdf()

    if raw.empty:
        return (
            pd.DataFrame(columns=["raw_record_id", "sales_volume"]),
            SalesVolumeStatus(
                status="unit_weight_fallback",
                source=str(db_path),
                warning="sales volume lookup disabled: filtered product table is empty",
            ),
        )

    raw["sales_volume"] = coerce_sales_series(raw[resolved_sales_column]).clip(lower=0)
    raw = raw[raw["sales_volume"].ge(float(min_sales))].copy()
    if raw.empty:
        return (
            pd.DataFrame(columns=["raw_record_id", "sales_volume"]),
            SalesVolumeStatus(
                status="unit_weight_fallback",
                source=str(db_path),
                warning="sales volume lookup disabled: all rows are below min_sales",
            ),
        )

    raw["raw_record_id"] = raw["Маркетплейс"].map(marketplace_key) + "::" + raw["Артикул"].astype(str).str.strip()
    lookup = raw.groupby("raw_record_id", as_index=False)["sales_volume"].sum()
    return (
        lookup,
        SalesVolumeStatus(
            status="loaded_sales_volume_lookup",
            source=str(db_path),
            lookup_rows=int(len(lookup)),
        ),
    )


def load_sales_lookup_from_csv(
    path: str | Path,
    *,
    sales_column: str | None = None,
) -> tuple[pd.DataFrame, SalesVolumeStatus]:
    lookup_path = Path(path).expanduser()
    raw = pd.read_csv(lookup_path)
    resolved_sales_column = resolve_sales_column(raw.columns, sales_column)
    if resolved_sales_column is None:
        raise ValueError(f"sales lookup is missing a sales column: {lookup_path}")

    output = raw.copy()
    if "raw_record_id" not in output.columns:
        required = {"Маркетплейс", "Артикул"}
        missing = sorted(required - set(output.columns))
        if missing:
            raise ValueError(f"sales lookup is missing raw_record_id and source columns: {missing}")
        output["raw_record_id"] = output["Маркетплейс"].map(marketplace_key) + "::" + output["Артикул"].astype(str).str.strip()

    output["sales_volume"] = coerce_sales_series(output[resolved_sales_column]).clip(lower=0)
    lookup = output[["raw_record_id", "sales_volume"]].dropna(subset=["raw_record_id"]).copy()
    lookup["raw_record_id"] = lookup["raw_record_id"].astype(str).str.strip()
    lookup = lookup[lookup["raw_record_id"].ne("")]
    lookup = lookup.groupby("raw_record_id", as_index=False)["sales_volume"].sum()
    return (
        lookup,
        SalesVolumeStatus(
            status="loaded_sales_volume_lookup",
            source=str(lookup_path),
            lookup_rows=int(len(lookup)),
        ),
    )


def attach_sales_volumes_from_lookup(
    frame: pd.DataFrame,
    lookup: pd.DataFrame,
    *,
    source: str | None = None,
) -> tuple[pd.DataFrame, SalesVolumeStatus]:
    if not {"raw_record_id_a", "raw_record_id_b"}.issubset(frame.columns):
        return (
            frame,
            SalesVolumeStatus(
                status="unit_weight_fallback",
                source=source,
                total_rows=int(len(frame)),
                warning="raw_record_id_a/raw_record_id_b are missing; no reliable sales volume join",
            ),
        )

    output = frame.copy()
    left_lookup = lookup.rename(columns={"raw_record_id": "raw_record_id_a", "sales_volume": "sales_volume_a"})
    right_lookup = lookup.rename(columns={"raw_record_id": "raw_record_id_b", "sales_volume": "sales_volume_b"})
    output = output.merge(left_lookup, on="raw_record_id_a", how="left")
    output = output.merge(right_lookup, on="raw_record_id_b", how="left")

    matched_left = int(output["sales_volume_a"].notna().sum())
    matched_right = int(output["sales_volume_b"].notna().sum())
    if matched_left == 0 and matched_right == 0:
        return (
            output,
            SalesVolumeStatus(
                status="unit_weight_fallback",
                source=source,
                total_rows=int(len(output)),
                lookup_rows=int(len(lookup)),
                warning="sales volume join found no matching raw_record_id keys",
            ),
        )

    return (
        output,
        SalesVolumeStatus(
            status="joined_sales_volume",
            source=source,
            matched_left_rows=matched_left,
            matched_right_rows=matched_right,
            total_rows=int(len(output)),
            lookup_rows=int(len(lookup)),
        ),
    )


def attach_sales_volumes(
    frame: pd.DataFrame,
    *,
    sales_lookup_path: str | Path | None = None,
    duckdb_path: str | Path | None = None,
    category_runs: str | Sequence[str] | None = None,
    products_table: str = DEFAULT_PRODUCTS_TABLE,
    sales_column: str | None = DEFAULT_SALES_COLUMN,
    min_sales: float = DEFAULT_MIN_SALES,
    enable_duckdb_join: bool = True,
) -> tuple[pd.DataFrame, SalesVolumeStatus]:
    if frame.empty:
        return frame, SalesVolumeStatus(status="skipped_empty_scores")
    if detect_sales_volume_columns(frame) is not None:
        return (
            frame,
            SalesVolumeStatus(
                status="using_existing_sales_volume_columns",
                total_rows=int(len(frame)),
            ),
        )

    if sales_lookup_path is not None:
        lookup, lookup_status = load_sales_lookup_from_csv(sales_lookup_path, sales_column=sales_column)
        return attach_sales_volumes_from_lookup(frame, lookup, source=lookup_status.source)

    if not enable_duckdb_join:
        return (
            frame,
            SalesVolumeStatus(
                status="unit_weight_fallback",
                total_rows=int(len(frame)),
                warning="sales volume join is disabled",
            ),
        )

    slugs = category_runs_from_frame(frame, category_runs)
    lookup, lookup_status = load_sales_lookup_from_duckdb(
        duckdb_path=duckdb_path,
        category_runs=slugs,
        products_table=products_table,
        sales_column=sales_column,
        min_sales=min_sales,
    )
    if lookup.empty:
        return frame, lookup_status
    return attach_sales_volumes_from_lookup(frame, lookup, source=lookup_status.source)


def write_sales_lookup_manifest(
    path: str | Path,
    *,
    status: SalesVolumeStatus,
    category_runs: Sequence[str],
    products_table: str,
    sales_column: str | None,
    min_sales: float,
) -> Path:
    manifest_path = Path(path).with_suffix(Path(path).suffix + ".manifest.json")
    manifest = {
        "sales_lookup_path": str(path),
        "status": status.as_dict(),
        "category_runs": list(category_runs),
        "products_table": products_table,
        "sales_column": sales_column,
        "min_sales": min_sales,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
