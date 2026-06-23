from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from research.dedup.threshold_calibration import calibrate_and_evaluate_methods, write_binary_threshold_reports

from .common import safe_slug
from .sales_weights import (
    DEFAULT_MIN_SALES,
    DEFAULT_PRODUCTS_TABLE,
    DEFAULT_SALES_COLUMN,
    attach_sales_volumes,
)


DEFAULT_REPORTS_DIR = Path("artifacts/reports/fine_tuning")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run existing dev-threshold calibration for a fine-tuned score CSV.")
    parser.add_argument("--score-path", type=Path, required=True)
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--method")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--report-prefix")
    parser.add_argument(
        "--sales-lookup-path",
        type=Path,
        help="Optional CSV with raw_record_id,sales_volume or marketplace/article/sales columns.",
    )
    parser.add_argument("--duckdb-path", type=Path, help="Optional mpstats.duckdb path for sales-volume join.")
    parser.add_argument(
        "--category-runs",
        default="auto",
        help="Comma-separated run slugs for DuckDB filtering; default auto uses score CSV category_run values.",
    )
    parser.add_argument("--products-table", default=DEFAULT_PRODUCTS_TABLE)
    parser.add_argument("--sales-column", default=DEFAULT_SALES_COLUMN)
    parser.add_argument("--min-sales", type=float, default=DEFAULT_MIN_SALES)
    parser.add_argument(
        "--disable-sales-duckdb-join",
        action="store_true",
        help="Do not try to read mpstats.duckdb when score CSV has no sales columns and no lookup CSV is provided.",
    )
    parser.add_argument(
        "--require-weighted",
        action="store_true",
        help="Fail instead of silently falling back to unweighted calibration when sales weights are unavailable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = pd.read_csv(args.score_path)
    if args.score_column not in frame.columns:
        raise ValueError(f"score column not found: {args.score_column}")
    if "split" not in frame.columns:
        raise ValueError("score CSV must contain split column")

    method = args.method or args.score_column
    prepared = frame.copy()
    prepared["method"] = method
    prepared["score"] = pd.to_numeric(prepared[args.score_column], errors="coerce")
    prepared["eval_split"] = prepared["split"]

    category_runs = None if args.category_runs == "auto" else args.category_runs
    prepared, sales_status = attach_sales_volumes(
        prepared,
        sales_lookup_path=args.sales_lookup_path,
        duckdb_path=args.duckdb_path,
        category_runs=category_runs,
        products_table=args.products_table,
        sales_column=args.sales_column,
        min_sales=args.min_sales,
        enable_duckdb_join=not args.disable_sales_duckdb_join,
    )
    print(f"sales_volume_status: {sales_status.as_dict()}")

    results = calibrate_and_evaluate_methods(prepared)
    if args.require_weighted and not bool(results.get("weights_available", False)):
        warning = results.get("weight_warning") or sales_status.warning or "sales weights are unavailable"
        print(f"ERROR: weighted calibration required but unavailable: {warning}")
        return 2

    prefix = args.report_prefix or safe_slug(method)
    paths = write_binary_threshold_reports(
        results,
        args.reports_dir,
        summary_filename=f"{prefix}_binary_threshold_summary.csv",
        predictions_filename=f"{prefix}_binary_threshold_predictions.csv",
        by_volume_bucket_filename=f"{prefix}_binary_threshold_by_volume_bucket.csv",
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"weights_available: {bool(results.get('weights_available', False))}")
    print(f"weight_source: {results.get('weight_source')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
