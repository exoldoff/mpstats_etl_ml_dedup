from __future__ import annotations

import argparse
from pathlib import Path

from .sales_weights import (
    DEFAULT_MIN_SALES,
    DEFAULT_PRODUCTS_TABLE,
    DEFAULT_SALES_COLUMN,
    load_sales_lookup_from_duckdb,
    parse_category_runs,
    write_sales_lookup_manifest,
)


DEFAULT_OUTPUT_PATH = Path("research/dedup/data/training/sales_volume_lookup.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export raw_record_id -> sales_volume lookup for weighted calibration.")
    parser.add_argument("--duckdb-path", type=Path)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--category-runs", default="sauces,coconut_oil,soap")
    parser.add_argument("--products-table", default=DEFAULT_PRODUCTS_TABLE)
    parser.add_argument("--sales-column", default=DEFAULT_SALES_COLUMN)
    parser.add_argument("--min-sales", type=float, default=DEFAULT_MIN_SALES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    category_runs = parse_category_runs(args.category_runs)
    lookup, status = load_sales_lookup_from_duckdb(
        duckdb_path=args.duckdb_path,
        category_runs=category_runs,
        products_table=args.products_table,
        sales_column=args.sales_column,
        min_sales=args.min_sales,
    )
    if lookup.empty:
        print(f"ERROR: sales lookup was not created: {status.as_dict()}")
        return 2

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(args.output_path, index=False)
    manifest_path = write_sales_lookup_manifest(
        args.output_path,
        status=status,
        category_runs=category_runs,
        products_table=args.products_table,
        sales_column=args.sales_column,
        min_sales=args.min_sales,
    )
    print(f"sales_lookup: {args.output_path}")
    print(f"manifest: {manifest_path}")
    print(f"rows: {len(lookup)}")
    print(f"status: {status.as_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
