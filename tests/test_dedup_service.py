from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

from mpstats_app.config import AppSettings
from mpstats_app.repositories.duckdb_repository import DuckDbAppRepository
from pipeline.repositories.file_repository import write_semicolon_csv
from pipeline.repositories.sql_repository import connect
from pipeline.services.dedup import DedupProfile, DedupRuntimeError, DedupService


def make_settings(root: Path) -> AppSettings:
    (root / "pipeline").mkdir(parents=True, exist_ok=True)
    (root / "classifiers").mkdir(parents=True, exist_ok=True)
    (root / "web" / "dist").mkdir(parents=True, exist_ok=True)
    (root / "pipeline" / "step1_export_config.json").write_text("{}", encoding="utf-8")
    (root / "classifiers" / "rules.csv").write_text(
        "active;priority;category;target_column;match_field;match_type;pattern;set_value;mode;comment;conditions_json\n",
        encoding="utf-8",
    )
    return AppSettings.create(
        project_root=root,
        workdir=root / "pipeline",
        db_path=root / "mpstats.duckdb",
        config_path=root / "pipeline" / "step1_export_config.json",
        rules_path=root / "classifiers" / "rules.csv",
        scheduler_poll_seconds=0.1,
        static_dir=root / "web" / "dist",
    )


def seed_dedup_cube(repository: DuckDbAppRepository, settings: AppSettings, root: Path, *, rows_count: int = 3) -> None:
    rows = []
    for index in range(rows_count):
        rows.append(
            {
                "Маркетплейс": "Ozon" if index % 2 == 0 else "WB",
                "Категория": "Соус",
                "Артикул": f"SKU-{index + 1}",
                "SKU": f"Томатный соус {index + 1} 500 г",
                "Бренд": "DedupBrand",
                "Подкатегория": "Томатные соусы",
                "Продажи, шт": str(20 + index),
                "Выручка, руб": str(1000 + index),
                "Вес, кг": "0.5",
                "Вес, кг (ед.)": "0.5",
            }
        )
    source_file = root / f"dedup-source-{rows_count}.csv"
    write_semicolon_csv(pd.DataFrame(rows), source_file)
    inserted = repository.import_products_file_idempotent(
        run_id=f"run-dedup-source-{rows_count}",
        csv_path=source_file,
        table_name=settings.products_table,
        project_name="unit",
        year=2026,
        month=5,
        marketplace_code="oz",
        category_key="sauce",
        category_name="Соус",
    )
    repository.upsert_cube_entry(
        {
            "project_name": "unit",
            "year": 2026,
            "month": 5,
            "marketplace": "Ozon",
            "marketplace_code": "oz",
            "category_key": "sauce",
            "category_name": "Соус",
            "rows_count": inserted,
            "source_processed_file_path": str(source_file),
            "file_hash": f"dedup-source-{rows_count}",
        }
    )


class FakeEmbeddingModel:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        return np.eye(len(texts), dtype="float32")


class FakeCrossEncoder:
    def __init__(self, score: float = 0.9) -> None:
        self.score = score

    def predict(self, pairs: list[tuple[str, str]], **_: object) -> np.ndarray:
        return np.full(len(pairs), self.score, dtype=float)


class FakeFaissModule:
    def __init__(self) -> None:
        self.search_ks: list[int] = []

        outer = self

        class IndexFlatIP:
            def __init__(self, dim: int) -> None:
                self.dim = dim
                self.vectors: np.ndarray | None = None

            def add(self, vectors: np.ndarray) -> None:
                self.vectors = np.asarray(vectors, dtype="float32")

            def search(self, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
                outer.search_ks.append(k)
                if self.vectors is None:
                    raise AssertionError("FAISS index used before add().")
                scores = np.asarray(vectors, dtype="float32") @ self.vectors.T
                order = np.argsort(-scores, axis=1)[:, :k]
                sorted_scores = np.take_along_axis(scores, order, axis=1)
                return sorted_scores.astype("float32"), order.astype("int64")

        self.IndexFlatIP = IndexFlatIP


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_nn = types.SimpleNamespace(Sigmoid=lambda: object())
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(nn=fake_nn))


def make_service(root: Path, *, cross_encoder_factory=None) -> tuple[DedupService, DuckDbAppRepository, FakeFaissModule, AppSettings]:
    settings = make_settings(root)
    repository = DuckDbAppRepository(settings)
    repository.ensure_ready()
    fake_faiss = FakeFaissModule()
    service = DedupService(
        settings=settings,
        repository=repository,
        embedding_model_factory=lambda _: FakeEmbeddingModel(),
        cross_encoder_factory=cross_encoder_factory or (lambda _: FakeCrossEncoder()),
        faiss_module=fake_faiss,
    )
    return service, repository, fake_faiss, settings


def test_profile_locks_weighted_cost_threshold_and_faiss_k() -> None:
    profile = DedupProfile.from_settings(
        {
            "threshold_strategy": "threshold_cost_sensitive",
            "threshold_same": 0.1,
            "faiss_top_k": 99,
            "model_method": "zero_shot",
        }
    )

    assert profile.model_method == "ft_bge_reranker_v2_m3"
    assert profile.threshold_strategy == "threshold_weighted_cost"
    assert profile.threshold_same == pytest.approx(0.872321)
    assert profile.category_thresholds["sauces"] == pytest.approx(0.917444)
    assert profile.category_thresholds["coconut_oil"] == pytest.approx(0.872321)
    assert profile.category_thresholds["soap"] == pytest.approx(0.930329)
    assert profile.threshold_for_category(category_key="sauce", category_name="Соус") == pytest.approx(0.917444)
    assert profile.threshold_for_category(category_key="soap", category_name="Мыло") == pytest.approx(0.930329)
    assert profile.threshold_for_category(category_key="unknown", category_name="Другая") == pytest.approx(0.872321)
    assert profile.faiss_top_k == 30


def test_faiss_candidate_retrieval_uses_k_30(tmp_path: Path) -> None:
    service, _, fake_faiss, _ = make_service(tmp_path)
    nodes = pd.DataFrame(
        [
            {
                "node_id": f"sku_{index:03d}",
                "category_key": "sauce",
                "subcategory": "Томатные соусы",
                "sku": f"Соус {index}",
            }
            for index in range(35)
        ]
    )
    embeddings = np.eye(len(nodes), dtype="float32")

    candidates = service._generate_candidates(nodes, embeddings, DedupProfile())

    assert not candidates.empty
    assert fake_faiss.search_ks == [31]


def test_run_fails_if_fine_tuned_model_unavailable(tmp_path: Path) -> None:
    def missing_model(_: str) -> FakeCrossEncoder:
        raise DedupRuntimeError("fine-tuned model unavailable")

    service, repository, _, settings = make_service(tmp_path, cross_encoder_factory=missing_model)
    seed_dedup_cube(repository, settings, tmp_path, rows_count=2)

    result = service.start_runs(project_name="unit", category_keys=["sauce"], wait=True)

    run = result["runs"][0]
    assert run["status"] == "failed"
    assert "fine-tuned model unavailable" in str(run["error"])


def test_success_run_writes_identity_tables_and_export_join_preserves_rows(tmp_path: Path) -> None:
    service, repository, _, settings = make_service(tmp_path)
    seed_dedup_cube(repository, settings, tmp_path, rows_count=3)
    with connect(settings.db_path) as con:
        before_count = con.execute(f"SELECT COUNT(*) FROM {settings.products_table}").fetchone()[0]

    result = service.start_runs(project_name="unit", category_keys=["sauce"], wait=True)
    run = result["runs"][0]

    assert run["status"] == "success"
    assert run["threshold_strategy"] == "threshold_weighted_cost"
    assert run["threshold_same"] == pytest.approx(0.917444)
    assert run["faiss_top_k"] == 30

    groups = repository.fetch_dedup_artifact(run_id=run["run_id"], artifact="groups")
    edges = repository.fetch_dedup_artifact(run_id=run["run_id"], artifact="edges")
    assert len(groups) == 3
    assert edges
    assert {edge["threshold_strategy"] for edge in edges} == {"threshold_weighted_cost"}
    assert {round(float(edge["threshold_same"]), 6) for edge in edges} == {0.917444}

    visible_columns = repository.export_visible_columns(table_name=settings.products_table, project_name="unit")
    assert "ML-группа товара" in visible_columns
    exported = repository.fetch_export_products_dataframe(
        table_name=settings.products_table,
        project_name="unit",
        output_columns=["SKU", "ML-группа товара", "ML-группа фасовки", "ML-канонический SKU", "ML-dedup статус"],
        category_keys=["sauce"],
        limit=10,
    )
    assert len(exported) == before_count
    assert exported["ML-группа товара"].notna().all()

    with connect(settings.db_path) as con:
        after_count = con.execute(f"SELECT COUNT(*) FROM {settings.products_table}").fetchone()[0]
    assert after_count == before_count
