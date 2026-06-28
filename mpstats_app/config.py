from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    project_root: Path
    workdir: Path
    db_path: Path
    config_path: Path
    rules_path: Path
    manual_overrides_path: Path
    products_table: str = "mpstats_products"
    scheduler_poll_seconds: float = 10.0
    static_dir: Path | None = None
    category_catalog_path: Path | None = None

    @classmethod
    def create(
        cls,
        *,
        project_root: str | Path | None = None,
        workdir: str | Path | None = None,
        db_path: str | Path | None = None,
        config_path: str | Path | None = None,
        rules_path: str | Path | None = None,
        manual_overrides_path: str | Path | None = None,
        products_table: str | None = None,
        scheduler_poll_seconds: float | None = None,
        static_dir: str | Path | None = None,
        category_catalog_path: str | Path | None = None,
    ) -> "AppSettings":
        root = (
            _resolve_path(project_root or os.environ.get("MPSTATS_PROJECT_ROOT"))
            or Path(__file__).resolve().parent.parent
        )
        wd = _resolve_path(workdir or os.environ.get("MPSTATS_WORKDIR")) or root / "pipeline"
        scheduler_seconds = (
            scheduler_poll_seconds
            if scheduler_poll_seconds is not None
            else _float_env("MPSTATS_SCHEDULER_POLL_SECONDS", 10.0)
        )
        resolved_products_table = products_table or os.environ.get("MPSTATS_PRODUCTS_TABLE", "mpstats_products")
        return cls(
            project_root=root,
            workdir=wd,
            db_path=_resolve_path(db_path or os.environ.get("MPSTATS_DB_PATH")) or root / "mpstats.duckdb",
            config_path=_resolve_path(config_path or os.environ.get("MPSTATS_CONFIG_PATH"))
            or wd / "step1_export_config.json",
            rules_path=_resolve_path(rules_path or os.environ.get("MPSTATS_RULES_PATH"))
            or root / "classifiers" / "rules.csv",
            manual_overrides_path=_resolve_path(
                manual_overrides_path or os.environ.get("MPSTATS_MANUAL_OVERRIDES_PATH")
            )
            or root / "classifiers" / "manual_overrides.csv",
            products_table=resolved_products_table,
            scheduler_poll_seconds=scheduler_seconds,
            static_dir=_resolve_path(static_dir or os.environ.get("MPSTATS_STATIC_DIR"))
            or root / "web" / "dist",
            category_catalog_path=_resolve_path(
                category_catalog_path or os.environ.get("MPSTATS_CATEGORY_CATALOG_PATH")
            ),
        )


def _resolve_path(value: str | Path | None) -> Path | None:
    if value in ("", None):
        return None
    return Path(value).expanduser().resolve()


def _float_env(name: str, fallback: float) -> float:
    value = os.environ.get(name)
    if value in ("", None):
        return fallback
    try:
        return float(value)
    except ValueError:
        return fallback
