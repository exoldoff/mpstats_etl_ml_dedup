from __future__ import annotations

import argparse
from pathlib import Path

from .data import (
    DEFAULT_DEV_RATIO,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    component_aware_split,
    freeze_labeling_dataset,
    write_freeze_outputs,
    write_split_outputs,
)


DEFAULT_LABELING_PATH = Path("research/dedup/data/labeling_sauces_coconut_oil_soap.csv")
DEFAULT_TELEGRAM_STATE_PATH = Path("research/dedup/data/labeling_sauces_coconut_oil_soap.csv.telegram_state.sqlite")
DEFAULT_OLD_SAUCES_PATH = Path("research/dedup/data/labeling_sauces.csv")
DEFAULT_OUTPUT_DIR = Path("research/dedup/data/training")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze dedup labels and create a component-aware train/dev/test split.")
    parser.add_argument("--labeling-path", type=Path, default=DEFAULT_LABELING_PATH)
    parser.add_argument("--telegram-state-path", type=Path, default=DEFAULT_TELEGRAM_STATE_PATH)
    parser.add_argument("--old-sauces-path", type=Path, default=DEFAULT_OLD_SAUCES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="dedup_pairs_final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--dev-ratio", type=float, default=DEFAULT_DEV_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--max-strict-component-share", type=float, default=0.25)
    parser.add_argument(
        "--large-component-strategy",
        choices=["positive_record_holdout", "strict", "pair_stratified"],
        default="positive_record_holdout",
    )
    parser.add_argument(
        "--fail-on-conflicts",
        action="store_true",
        help="Write conflict files and exit non-zero when label conflicts are present.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    freeze_result = freeze_labeling_dataset(
        labeling_path=args.labeling_path,
        telegram_state_path=args.telegram_state_path,
        old_sauces_path=args.old_sauces_path,
    )
    freeze_paths = write_freeze_outputs(freeze_result, args.output_dir, prefix=args.prefix)

    print("Freeze outputs:")
    for name, path in freeze_paths.items():
        print(f"  {name}: {path}")
    print(
        "Rows: "
        f"training_pairs={len(freeze_result.pairs)}, "
        f"conflicts={len(freeze_result.conflicts)}, "
        f"excluded={len(freeze_result.excluded)}"
    )
    if args.fail_on_conflicts and not freeze_result.conflicts.empty:
        print("Label conflicts found; inspect the conflicts CSV before training.")
        return 2

    split_result = component_aware_split(
        freeze_result.pairs,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_strict_component_share=args.max_strict_component_share,
        large_component_strategy=args.large_component_strategy,
    )
    split_paths = write_split_outputs(split_result, args.output_dir, prefix=args.prefix)

    print("Split outputs:")
    for name, path in split_paths.items():
        print(f"  {name}: {path}")
    print(f"Split counts: {split_result.pairs['split'].value_counts().to_dict()}")
    if split_result.dropped_pairs is not None and not split_result.dropped_pairs.empty:
        print(f"Dropped crossing pairs: {len(split_result.dropped_pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
