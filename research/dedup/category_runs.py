from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


DEDUP_CATEGORY_RUN_ENV = "DEDUP_CATEGORY_RUN"


@dataclass(frozen=True)
class DedupCategoryRun:
    slug: str
    display_name: str
    project_name: str
    category_aliases: tuple[str, ...]
    artifact_suffix: str
    uses_legacy_paths: bool = False


@dataclass(frozen=True)
class DedupRunPaths:
    data_dir: Path
    reports_dir: Path
    candidates_path: Path
    labeling_path: Path
    binary_summary_path: Path
    binary_predictions_path: Path
    binary_volume_bucket_path: Path
    fusion_components_path: Path
    fusion_pair_eval_path: Path
    grouped_sku_demo_path: Path


KNOWN_CATEGORY_RUNS: dict[str, DedupCategoryRun] = {
    "sauces": DedupCategoryRun(
        slug="sauces",
        display_name="Соусы",
        project_name="Соусы_тест",
        category_aliases=("Соусы", "Соус"),
        artifact_suffix="sauces",
        uses_legacy_paths=True,
    ),
    "coconut_oil": DedupCategoryRun(
        slug="coconut_oil",
        display_name="Кокосовое масло",
        project_name="кокос_тест",
        category_aliases=("Кокосовое масло",),
        artifact_suffix="coconut_oil",
    ),
    "soap": DedupCategoryRun(
        slug="soap",
        display_name="Мыло",
        project_name="мыло_тест",
        category_aliases=("Мыло",),
        artifact_suffix="soap",
    ),
}


RUN_ALIASES: dict[str, str] = {
    "sauce": "sauces",
    "соус": "sauces",
    "соусы": "sauces",
    "coconut": "coconut_oil",
    "coconut-oil": "coconut_oil",
    "кокос": "coconut_oil",
    "кокосовое масло": "coconut_oil",
    "soap": "soap",
    "мыло": "soap",
}


def _normalize_run_name(value: str) -> str:
    return " ".join(value.strip().casefold().replace("-", "_").split())


def available_category_run_names() -> str:
    return ", ".join(sorted(KNOWN_CATEGORY_RUNS))


def resolve_category_run(value: str | None = None) -> DedupCategoryRun:
    requested = value or os.environ.get(DEDUP_CATEGORY_RUN_ENV) or "sauces"
    normalized = _normalize_run_name(requested)
    canonical = RUN_ALIASES.get(normalized, normalized)
    if canonical in KNOWN_CATEGORY_RUNS:
        return KNOWN_CATEGORY_RUNS[canonical]
    raise ValueError(
        f"Unknown {DEDUP_CATEGORY_RUN_ENV}={requested!r}. "
        f"Available runs: {available_category_run_names()}"
    )


def resolve_run_paths(project_root: Path, run: DedupCategoryRun) -> DedupRunPaths:
    data_root = project_root / "research" / "dedup" / "data"
    reports_root = project_root / "artifacts" / "reports"

    if run.uses_legacy_paths:
        data_dir = data_root
        reports_dir = reports_root
        demo_name = "dedup_grouped_sku_demo.csv"
    else:
        data_dir = data_root / run.slug
        reports_dir = reports_root / run.slug
        demo_name = f"dedup_grouped_sku_demo_{run.artifact_suffix}.csv"

    suffix = run.artifact_suffix
    return DedupRunPaths(
        data_dir=data_dir,
        reports_dir=reports_dir,
        candidates_path=data_dir / f"candidates_{suffix}.csv",
        labeling_path=data_dir / f"labeling_{suffix}.csv",
        binary_summary_path=reports_dir / "binary_threshold_summary.csv",
        binary_predictions_path=reports_dir / "binary_threshold_predictions.csv",
        binary_volume_bucket_path=reports_dir / "binary_threshold_by_volume_bucket.csv",
        fusion_components_path=data_dir / f"fusion_components_{suffix}.csv",
        fusion_pair_eval_path=data_dir / f"fusion_pair_eval_{suffix}.csv",
        grouped_sku_demo_path=reports_dir / demo_name,
    )
