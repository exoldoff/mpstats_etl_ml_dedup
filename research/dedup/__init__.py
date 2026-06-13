"""Research helpers for SKU deduplication experiments.

This package is intentionally independent from ``pipeline/`` and
``mpstats_app/``.  It is imported by research notebooks only.
"""

from .candidates import (
    CandidateGenerationConfig,
    add_hard_negative_flags,
    add_pack_variant_flags,
    generate_candidate_pairs,
    prepare_product_records,
)
from .labeling import LabelingSamplingConfig, stratified_labeling_sample
from .metrics import classification_report_df, confusion_matrix_df
from .normalization import (
    DEFAULT_FLAVOR_TOKENS,
    meaningful_title_tokens,
    normalize_brand,
    normalize_title,
    title_similarity,
    tokenize_title,
)

__all__ = [
    "CandidateGenerationConfig",
    "DEFAULT_FLAVOR_TOKENS",
    "LabelingSamplingConfig",
    "add_hard_negative_flags",
    "add_pack_variant_flags",
    "classification_report_df",
    "confusion_matrix_df",
    "generate_candidate_pairs",
    "meaningful_title_tokens",
    "normalize_brand",
    "normalize_title",
    "prepare_product_records",
    "stratified_labeling_sample",
    "title_similarity",
    "tokenize_title",
]
