from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mpstats_app.config import AppSettings


class AppSettingsEnvTest(unittest.TestCase):
    def test_create_reads_runtime_paths_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env = {
                "MPSTATS_PROJECT_ROOT": str(root / "app"),
                "MPSTATS_WORKDIR": str(root / "runtime" / "pipeline"),
                "MPSTATS_DB_PATH": str(root / "runtime" / "mpstats.duckdb"),
                "MPSTATS_CONFIG_PATH": str(root / "runtime" / "pipeline" / "step1_export_config.json"),
                "MPSTATS_RULES_PATH": str(root / "runtime" / "classifiers" / "rules.csv"),
                "MPSTATS_MANUAL_OVERRIDES_PATH": str(
                    root / "runtime" / "classifiers" / "manual_overrides.csv"
                ),
                "MPSTATS_CATEGORY_CATALOG_PATH": str(root / "runtime" / "catalog" / "category_catalog.csv"),
                "MPSTATS_STATIC_DIR": str(root / "app" / "web" / "dist"),
                "MPSTATS_PRODUCTS_TABLE": "products_from_env",
                "MPSTATS_SCHEDULER_POLL_SECONDS": "2.5",
            }

            with patch.dict(os.environ, env, clear=False):
                settings = AppSettings.create()

            self.assertEqual(settings.project_root, (root / "app").resolve())
            self.assertEqual(settings.workdir, (root / "runtime" / "pipeline").resolve())
            self.assertEqual(settings.db_path, (root / "runtime" / "mpstats.duckdb").resolve())
            self.assertEqual(
                settings.config_path,
                (root / "runtime" / "pipeline" / "step1_export_config.json").resolve(),
            )
            self.assertEqual(settings.rules_path, (root / "runtime" / "classifiers" / "rules.csv").resolve())
            self.assertEqual(
                settings.manual_overrides_path,
                (root / "runtime" / "classifiers" / "manual_overrides.csv").resolve(),
            )
            self.assertEqual(
                settings.category_catalog_path,
                (root / "runtime" / "catalog" / "category_catalog.csv").resolve(),
            )
            self.assertEqual(settings.static_dir, (root / "app" / "web" / "dist").resolve())
            self.assertEqual(settings.products_table, "products_from_env")
            self.assertEqual(settings.scheduler_poll_seconds, 2.5)

    def test_explicit_paths_override_environment_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env = {
                "MPSTATS_DB_PATH": str(root / "env.duckdb"),
                "MPSTATS_RULES_PATH": str(root / "env-rules.csv"),
            }

            with patch.dict(os.environ, env, clear=False):
                settings = AppSettings.create(
                    project_root=root,
                    db_path=root / "explicit.duckdb",
                    rules_path=root / "explicit-rules.csv",
                    products_table="explicit_products",
                    scheduler_poll_seconds=0.25,
                )

            self.assertEqual(settings.db_path, (root / "explicit.duckdb").resolve())
            self.assertEqual(settings.rules_path, (root / "explicit-rules.csv").resolve())
            self.assertEqual(settings.products_table, "explicit_products")
            self.assertEqual(settings.scheduler_poll_seconds, 0.25)


if __name__ == "__main__":
    unittest.main()
