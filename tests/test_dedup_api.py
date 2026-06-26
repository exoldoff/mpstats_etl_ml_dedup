from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mpstats_app.config import AppSettings
from mpstats_app.main import create_app
from mpstats_app.repositories.duckdb_repository import DuckDbAppRepository
from pipeline.repositories.file_repository import write_semicolon_csv
from pipeline.repositories.sql_repository import connect
from pipeline.services.dedup import DedupService


def make_settings(root: Path) -> AppSettings:
    (root / "pipeline").mkdir(parents=True, exist_ok=True)
    (root / "classifiers").mkdir(parents=True, exist_ok=True)
    (root / "web" / "dist").mkdir(parents=True, exist_ok=True)
    (root / "pipeline" / "step1_export_config.json").write_text("{}", encoding="utf-8")
    (root / "classifiers" / "rules.csv").write_text(
        "active;priority;category;target_column;match_field;match_type;pattern;set_value;mode;comment;conditions_json\n",
        encoding="utf-8",
    )
    (root / "Справочник категорий MP STATS.csv").write_text(
        "Чек;Категория;МП;FBS;Тип выгрузки;От;До;Комментарий;Путь;Фильтр;Путь2;Фильтр2;Актуализация\n",
        encoding="utf-8-sig",
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


def seed_cube(repository: DuckDbAppRepository, settings: AppSettings, root: Path) -> None:
    source_file = root / "api-dedup-source.csv"
    write_semicolon_csv(
        pd.DataFrame(
            [
                {
                    "Маркетплейс": "Ozon",
                    "Категория": "Соус",
                    "Артикул": "OZ-1",
                    "SKU": "Томатный соус 500 г",
                    "Бренд": "Brand",
                    "Подкатегория": "Томатные соусы",
                    "Продажи, шт": "20",
                    "Выручка, руб": "1000",
                    "Вес, кг": "0.5",
                    "Вес, кг (ед.)": "0.5",
                },
                {
                    "Маркетплейс": "Ozon",
                    "Категория": "Соус",
                    "Артикул": "OZ-2",
                    "SKU": "Соус томатный 0.5 кг",
                    "Бренд": "Brand",
                    "Подкатегория": "Томатные соусы",
                    "Продажи, шт": "19",
                    "Выручка, руб": "900",
                    "Вес, кг": "0.5",
                    "Вес, кг (ед.)": "0.5",
                },
            ]
        ),
        source_file,
    )
    inserted = repository.import_products_file_idempotent(
        run_id="api-dedup-run-source",
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
            "file_hash": "api-dedup-source",
        }
    )


class FakeEmbeddingModel:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        return np.eye(len(texts), dtype="float32")


class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]], **_: object) -> np.ndarray:
        return np.full(len(pairs), 0.99, dtype=float)


class FakeFaissModule:
    class IndexFlatIP:
        def __init__(self, dim: int) -> None:
            self.vectors: np.ndarray | None = None

        def add(self, vectors: np.ndarray) -> None:
            self.vectors = np.asarray(vectors, dtype="float32")

        def search(self, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            if self.vectors is None:
                raise AssertionError("FAISS index used before add().")
            scores = np.asarray(vectors, dtype="float32") @ self.vectors.T
            order = np.argsort(-scores, axis=1)[:, :k]
            return np.take_along_axis(scores, order, axis=1).astype("float32"), order.astype("int64")


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_nn = types.SimpleNamespace(Sigmoid=lambda: object())
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(nn=fake_nn))


def test_dedup_api_settings_lifecycle_and_export(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings, start_workers=False)
    app.state.dedup_service = DedupService(
        settings=settings,
        repository=app.state.repository,
        embedding_model_factory=lambda _: FakeEmbeddingModel(),
        cross_encoder_factory=lambda _: FakeCrossEncoder(),
        faiss_module=FakeFaissModule(),
    )

    with TestClient(app) as client:
        repository: DuckDbAppRepository = app.state.repository
        seed_cube(repository, settings, tmp_path)

        settings_response = client.get("/api/dedup/settings")
        assert settings_response.status_code == 200
        settings_payload = settings_response.json()
        assert settings_payload["threshold_strategy"] == "threshold_weighted_cost"
        assert settings_payload["threshold_same"] == pytest.approx(0.872321)
        assert settings_payload["category_thresholds"]["sauces"] == pytest.approx(0.917444)
        assert settings_payload["category_thresholds"]["coconut_oil"] == pytest.approx(0.872321)
        assert settings_payload["category_thresholds"]["soap"] == pytest.approx(0.930329)
        assert settings_payload["faiss_top_k"] == 30
        assert settings_payload["retrieval_cache_enabled"] is True
        assert settings_payload["retrieval_cache_schema_version"] == "dedup_retrieval_cache_v1"

        saved_response = client.put(
            "/api/dedup/settings",
            json={
                "model_path": "",
                "hf_model_id": "exoldoff/bge-reranker-v2-m3-cross-encoder-marketplaces-rus",
                "embedding_model_name": "intfloat/multilingual-e5-small",
                "threshold_strategy": "threshold_cost_sensitive",
                "threshold_same": 0.1,
                "faiss_top_k": 99,
            },
        )
        assert saved_response.status_code == 200
        saved_payload = saved_response.json()
        assert saved_payload["threshold_strategy"] == "threshold_weighted_cost"
        assert saved_payload["threshold_same"] == pytest.approx(0.872321)
        assert saved_payload["category_thresholds"]["sauces"] == pytest.approx(0.917444)
        assert saved_payload["faiss_top_k"] == 30

        eligible_response = client.get("/api/dedup/eligible-categories", params={"project_name": "unit"})
        assert eligible_response.status_code == 200
        categories = eligible_response.json()["categories"]
        assert [category["category_key"] for category in categories] == ["dedupcat_sauces"]
        assert categories[0]["source_category_keys"] == ["sauce"]

        run_response = client.post(
            "/api/dedup/runs",
            json={"project_name": "unit", "category_keys": ["dedupcat_sauces"], "wait": True},
        )
        assert run_response.status_code == 200
        run = run_response.json()["runs"][0]
        assert run["status"] == "success"
        assert run["threshold_strategy"] == "threshold_weighted_cost"
        assert run["threshold_same"] == pytest.approx(0.917444)
        assert run["manifest_json"]["retrieval_cache_status"] == "rebuilt"
        assert run["manifest_json"]["retrieval_cache_key"]
        assert run["manifest_json"]["embedding_shape"] == [2, 2]
        assert run["manifest_json"]["progress_percent"] == 100
        assert run["manifest_json"]["progress_stage"] == "success"

        groups_response = client.get(f"/api/dedup/runs/{run['run_id']}/export", params={"artifact": "groups"})
        assert groups_response.status_code == 200
        assert len(groups_response.json()["rows"]) == 2

        products_response = client.get(
            "/api/dedup/products",
            params={"project_name": "unit", "category_key": "dedupcat_sauces", "level": "expanded"},
        )
        assert products_response.status_code == 200
        product_rows = products_response.json()["rows"]
        assert [row["row_level"] for row in product_rows].count("canonical") == 1
        assert [row["row_level"] for row in product_rows].count("member") == 2

        products_export = client.get(
            "/api/dedup/products/export",
            params={"project_name": "unit", "category_key": "dedupcat_sauces", "level": "canonical"},
        )
        assert products_export.status_code == 200
        assert products_export.text.startswith("\ufeffproject_name;category_key;category_name;")

        with connect(settings.db_path) as con:
            tables = {
                row[0]
                for row in con.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_name LIKE 'dedup_%'
                       OR table_name = 'mpstats_products_dedup'
                    """
                ).fetchall()
            }
        assert {"dedup_runs", "dedup_sku_nodes", "dedup_sku_edges", "dedup_sku_groups", "mpstats_products_dedup"}.issubset(tables)
