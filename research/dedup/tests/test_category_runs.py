from __future__ import annotations

from pathlib import Path

import pytest

from research.dedup.category_runs import resolve_category_run, resolve_run_paths


def test_resolve_known_category_runs_and_aliases() -> None:
    assert resolve_category_run("sauces").project_name == "Соусы_тест"
    assert resolve_category_run("мыло").slug == "soap"
    assert resolve_category_run("кокосовое масло").slug == "coconut_oil"


def test_new_category_runs_use_isolated_paths() -> None:
    paths = resolve_run_paths(Path("/repo"), resolve_category_run("soap"))

    assert paths.data_dir == Path("/repo/research/dedup/data/soap")
    assert paths.reports_dir == Path("/repo/artifacts/reports/soap")
    assert paths.candidates_path.name == "candidates_soap.csv"
    assert paths.labeling_path.name == "labeling_soap.csv"
    assert paths.fusion_components_path.name == "fusion_components_soap.csv"


def test_sauces_keeps_legacy_paths() -> None:
    paths = resolve_run_paths(Path("/repo"), resolve_category_run("sauces"))

    assert paths.data_dir == Path("/repo/research/dedup/data")
    assert paths.reports_dir == Path("/repo/artifacts/reports")
    assert paths.labeling_path == Path("/repo/research/dedup/data/labeling_sauces.csv")
    assert paths.binary_summary_path == Path("/repo/artifacts/reports/binary_threshold_summary.csv")


def test_unknown_category_run_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Available runs"):
        resolve_category_run("unknown")
