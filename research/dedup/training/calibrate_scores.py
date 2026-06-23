from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from research.dedup.threshold_calibration import calibrate_and_evaluate_methods, write_binary_threshold_reports

from .common import safe_slug


DEFAULT_REPORTS_DIR = Path("artifacts/reports/fine_tuning")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run existing dev-threshold calibration for a fine-tuned score CSV.")
    parser.add_argument("--score-path", type=Path, required=True)
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--method")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--report-prefix")
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

    results = calibrate_and_evaluate_methods(prepared)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
