"""Training utilities for dedup research experiments.

This package intentionally keeps data preparation separate from notebook code
and from production pipeline modules.
"""

from .data import (
    DEFAULT_DEV_RATIO,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    NEGATIVE_LABELS,
    POSITIVE_LABELS,
    DatasetFreezeResult,
    SplitResult,
    add_pair_text_columns,
    binary_target_from_label,
    build_pair_text,
    component_aware_split,
    freeze_labeling_dataset,
    pair_key,
    write_freeze_outputs,
    write_split_outputs,
)

__all__ = [
    "DEFAULT_DEV_RATIO",
    "DEFAULT_TEST_RATIO",
    "DEFAULT_TRAIN_RATIO",
    "NEGATIVE_LABELS",
    "POSITIVE_LABELS",
    "DatasetFreezeResult",
    "SplitResult",
    "add_pair_text_columns",
    "binary_target_from_label",
    "build_pair_text",
    "component_aware_split",
    "freeze_labeling_dataset",
    "pair_key",
    "write_freeze_outputs",
    "write_split_outputs",
]
