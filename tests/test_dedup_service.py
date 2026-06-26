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


def seed_dedup_cube(
    repository: DuckDbAppRepository,
    settings: AppSettings,
    root: Path,
    *,
    rows_count: int = 3,
    category_key: str = "sauce",
    category_name: str = "Соус",
    marketplace_code: str = "oz",
    marketplace: str = "Ozon",
) -> None:
    rows = []
    for index in range(rows_count):
        rows.append(
            {
                "Маркетплейс": marketplace,
                "Категория": category_name,
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
    source_file = root / f"dedup-source-{category_key}-{rows_count}.csv"
    write_semicolon_csv(pd.DataFrame(rows), source_file)
    inserted = repository.import_products_file_idempotent(
        run_id=f"run-dedup-source-{category_key}-{rows_count}",
        csv_path=source_file,
        table_name=settings.products_table,
        project_name="unit",
        year=2026,
        month=5,
        marketplace_code=marketplace_code,
        category_key=category_key,
        category_name=category_name,
    )
    repository.upsert_cube_entry(
        {
            "project_name": "unit",
            "year": 2026,
            "month": 5,
            "marketplace": marketplace,
            "marketplace_code": marketplace_code,
            "category_key": category_key,
            "category_name": category_name,
            "rows_count": inserted,
            "source_processed_file_path": str(source_file),
            "file_hash": f"dedup-source-{category_key}-{rows_count}",
        }
    )


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        self.calls += 1
        return np.eye(len(texts), dtype="float32")


class FakeCrossEncoder:
    def __init__(self, score: float = 0.99) -> None:
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


def make_service(
    root: Path,
    *,
    cross_encoder_factory=None,
    embedding_model: FakeEmbeddingModel | None = None,
) -> tuple[DedupService, DuckDbAppRepository, FakeFaissModule, AppSettings]:
    settings = make_settings(root)
    repository = DuckDbAppRepository(settings)
    repository.ensure_ready()
    fake_faiss = FakeFaissModule()
    service = DedupService(
        settings=settings,
        repository=repository,
        embedding_model_factory=lambda _: embedding_model or FakeEmbeddingModel(),
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


def test_eligible_categories_collapse_source_keys_by_business_category(tmp_path: Path) -> None:
    _, repository, _, settings = make_service(tmp_path)
    seed_dedup_cube(
        repository,
        settings,
        tmp_path,
        rows_count=2,
        category_key="sauce_oz",
        category_name="Соус",
        marketplace_code="oz",
        marketplace="Ozon",
    )
    seed_dedup_cube(
        repository,
        settings,
        tmp_path,
        rows_count=3,
        category_key="sauce_wb",
        category_name="Соусы",
        marketplace_code="wb",
        marketplace="WB",
    )

    categories = repository.list_dedup_eligible_categories(project_name="unit")

    assert len(categories) == 1
    category = categories[0]
    assert category["category_key"] == "dedupcat_sauces"
    assert category["category_name"] == "Соусы"
    assert category["rows_count"] == 5
    assert category["slices_count"] == 2
    assert category["source_categories_count"] == 2
    assert set(category["source_category_keys"]) == {"sauce_oz", "sauce_wb"}
    assert set(category["marketplaces"]) == {"Ozon", "WB"}


def test_service_marks_stale_dedup_runs_failed_on_start(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    repository = DuckDbAppRepository(settings)
    repository.ensure_ready()
    repository.create_dedup_run(
        {
            "run_id": "stale-run",
            "project_name": "unit",
            "category_key": "dedupcat_sauces",
            "category_name": "Соусы",
            "status": "running",
            **DedupProfile().to_dict(),
            "manifest": {},
        }
    )

    DedupService(
        settings=settings,
        repository=repository,
        embedding_model_factory=lambda _: FakeEmbeddingModel(),
        cross_encoder_factory=lambda _: FakeCrossEncoder(),
        faiss_module=FakeFaissModule(),
    )

    run = repository.get_dedup_run("stale-run")
    assert run
    assert run["status"] == "failed"
    assert "backend restarted" in str(run["error"])
    assert run["finished_at"] is not None


def test_service_prunes_repeated_failed_dedup_runs_on_start(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    repository = DuckDbAppRepository(settings)
    repository.ensure_ready()
    for index, category_key in enumerate(("source_a", "source_b", "source_c"), start=1):
        repository.create_dedup_run(
            {
                "run_id": f"failed-run-{index}",
                "project_name": "unit",
                "category_key": category_key,
                "category_name": "Кокосовое масло",
                "status": "failed",
                **DedupProfile().to_dict(),
                "manifest": {},
            }
        )

    DedupService(
        settings=settings,
        repository=repository,
        embedding_model_factory=lambda _: FakeEmbeddingModel(),
        cross_encoder_factory=lambda _: FakeCrossEncoder(),
        faiss_module=FakeFaissModule(),
    )

    runs = repository.list_dedup_runs(project_name="unit", limit=10)
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert runs[0]["category_name"] == "Кокосовое масло"


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


def test_faiss_candidate_retrieval_crosses_source_category_keys(tmp_path: Path) -> None:
    service, _, _, _ = make_service(tmp_path)
    nodes = pd.DataFrame(
        [
            {
                "node_id": "sku_oz",
                "category_key": "source_oz",
                "subcategory": "Томатные соусы",
                "sku": "Соус томатный 500 г",
            },
            {
                "node_id": "sku_wb",
                "category_key": "source_wb",
                "subcategory": "Томатные соусы",
                "sku": "Томатный соус 0.5 кг",
            },
        ]
    )
    embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype="float32")

    candidates = service._generate_candidates(nodes, embeddings, DedupProfile())

    assert len(candidates) == 1
    assert {candidates.iloc[0]["node_id_a"], candidates.iloc[0]["node_id_b"]} == {"sku_oz", "sku_wb"}


def test_run_fails_if_fine_tuned_model_unavailable(tmp_path: Path) -> None:
    def missing_model(_: str) -> FakeCrossEncoder:
        raise DedupRuntimeError("fine-tuned model unavailable")

    service, repository, _, settings = make_service(tmp_path, cross_encoder_factory=missing_model)
    seed_dedup_cube(repository, settings, tmp_path, rows_count=2)

    result = service.start_runs(project_name="unit", category_keys=["sauce"], wait=True)

    run = result["runs"][0]
    assert run["status"] == "failed"
    assert "fine-tuned model unavailable" in str(run["error"])
    assert run["manifest_json"]["progress_percent"] == 100
    assert run["manifest_json"]["progress_stage"] == "failed"
    assert "DedupRuntimeError" in run["manifest_json"]["progress_message"]


def test_failed_dedup_runs_are_replaced_by_next_attempt(tmp_path: Path) -> None:
    def missing_model(_: str) -> FakeCrossEncoder:
        raise DedupRuntimeError("fine-tuned model unavailable")

    service, repository, _, settings = make_service(tmp_path, cross_encoder_factory=missing_model)
    seed_dedup_cube(repository, settings, tmp_path, rows_count=2)

    first = service.start_runs(project_name="unit", category_keys=["sauce"], wait=True)["runs"][0]
    second = service.start_runs(project_name="unit", category_keys=["sauce"], wait=True)["runs"][0]

    runs = repository.list_dedup_runs(project_name="unit", category_key="dedupcat_sauces", limit=10)
    assert first["status"] == "failed"
    assert second["status"] == "failed"
    assert first["run_id"] != second["run_id"]
    assert [run["run_id"] for run in runs] == [second["run_id"]]


def test_active_dedup_run_is_reused_instead_of_duplicated(tmp_path: Path) -> None:
    service, repository, _, settings = make_service(tmp_path)
    seed_dedup_cube(repository, settings, tmp_path, rows_count=2)
    active = repository.create_dedup_run(
        {
            "run_id": "active-run",
            "project_name": "unit",
            "category_key": "dedupcat_sauces",
            "category_name": "Соусы",
            "status": "running",
            **DedupProfile().to_dict(),
            "manifest": {"progress_percent": 30, "progress_message": "Уже работает"},
        }
    )

    result = service.start_runs(project_name="unit", category_keys=["sauce"], wait=True)

    assert result["runs"][0]["run_id"] == active["run_id"]
    runs = repository.list_dedup_runs(project_name="unit", category_key="dedupcat_sauces", limit=10)
    assert [run["run_id"] for run in runs] == ["active-run"]


def test_success_run_writes_identity_tables_and_export_join_preserves_rows(tmp_path: Path) -> None:
    service, repository, _, settings = make_service(tmp_path)
    seed_dedup_cube(repository, settings, tmp_path, rows_count=3)
    with connect(settings.db_path) as con:
        before_count = con.execute(f"SELECT COUNT(*) FROM {settings.products_table}").fetchone()[0]

    result = service.start_runs(project_name="unit", category_keys=["sauce"], wait=True)
    run = result["runs"][0]

    assert run["status"] == "success"
    assert run["category_key"] == "dedupcat_sauces"
    assert run["threshold_strategy"] == "threshold_weighted_cost"
    assert run["threshold_same"] == pytest.approx(0.917444)
    assert run["faiss_top_k"] == 30
    assert run["manifest_json"]["source_category_keys"] == ["sauce"]
    assert run["manifest_json"]["progress_percent"] == 100
    assert run["manifest_json"]["progress_stage"] == "success"
    assert "mpstats_products_dedup" in run["manifest_json"]["progress_message"]
    assert run["manifest_json"]["retrieval_cache_status"] == "rebuilt"
    assert run["manifest_json"]["cache_rebuild_reason"] == "miss"
    assert run["manifest_json"]["embedding_shape"] == [3, 3]

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

    browser = repository.fetch_dedup_products(project_name="unit", category_key="sauce", level="expanded", limit=20)
    assert browser["total"] == 4
    assert [row["row_level"] for row in browser["rows"]].count("canonical") == 1
    assert [row["row_level"] for row in browser["rows"]].count("member") == 3
    grouped_browser = repository.fetch_dedup_products(project_name="unit", category_key="dedupcat_sauces", level="expanded", limit=20)
    assert grouped_browser["total"] == 4
    canonical = repository.fetch_dedup_products(project_name="unit", category_key="sauce", level="canonical", limit=20)
    assert canonical["total"] == 1
    assert {row["normalized_sku"] for row in canonical["rows"]} == {row["canonical_sku"] for row in canonical["rows"]}

    export_path = tmp_path / "dedup-products.csv"
    export_result = repository.export_dedup_products_csv(
        project_name="unit",
        category_key="sauce",
        level="canonical",
        target=export_path,
    )
    assert export_result.output_path == export_path
    assert export_path.read_text(encoding="utf-8").startswith("\ufeffproject_name;category_key;category_name;")

    with connect(settings.db_path) as con:
        after_count = con.execute(f"SELECT COUNT(*) FROM {settings.products_table}").fetchone()[0]
    assert after_count == before_count


def test_retrieval_cache_hit_reuses_embeddings_without_model_encode(tmp_path: Path) -> None:
    embedding_model = FakeEmbeddingModel()
    service, repository, _, settings = make_service(tmp_path, embedding_model=embedding_model)
    seed_dedup_cube(repository, settings, tmp_path, rows_count=3)

    first = service.start_runs(project_name="unit", category_keys=["sauce"], wait=True)["runs"][0]
    second = service.start_runs(project_name="unit", category_keys=["sauce"], wait=True)["runs"][0]

    assert first["status"] == "success"
    assert first["manifest_json"]["retrieval_cache_status"] == "rebuilt"
    assert Path(str(first["manifest_json"]["retrieval_cache_path"])).joinpath("embeddings.npy").is_file()
    assert second["status"] == "success"
    assert second["manifest_json"]["retrieval_cache_status"] == "hit"
    assert embedding_model.calls == 1


def test_retrieval_cache_model_change_invalidates_embeddings(tmp_path: Path) -> None:
    embedding_model = FakeEmbeddingModel()
    service, repository, _, settings = make_service(tmp_path, embedding_model=embedding_model)
    seed_dedup_cube(repository, settings, tmp_path, rows_count=3)

    first = service.start_runs(project_name="unit", category_keys=["sauce"], wait=True)["runs"][0]
    service.save_settings({"embedding_model_name": "unit/other-embedding-model"})
    second = service.start_runs(project_name="unit", category_keys=["sauce"], wait=True)["runs"][0]

    assert first["manifest_json"]["retrieval_cache_status"] == "rebuilt"
    assert second["status"] == "success"
    assert second["embedding_model_name"] == "unit/other-embedding-model"
    assert second["manifest_json"]["retrieval_cache_status"] == "rebuilt"
    assert second["manifest_json"]["cache_rebuild_reason"] == "miss"
    assert embedding_model.calls == 2


def test_retrieval_cache_corrupt_node_ids_rebuilds_without_fatal_error(tmp_path: Path) -> None:
    embedding_model = FakeEmbeddingModel()
    service, repository, _, settings = make_service(tmp_path, embedding_model=embedding_model)
    seed_dedup_cube(repository, settings, tmp_path, rows_count=3)

    first = service.start_runs(project_name="unit", category_keys=["sauce"], wait=True)["runs"][0]
    cache_path = Path(str(first["manifest_json"]["retrieval_cache_path"]))
    cache_path.joinpath("node_ids.json").write_text("[\"broken-node\"]", encoding="utf-8")
    second = service.start_runs(project_name="unit", category_keys=["sauce"], wait=True)["runs"][0]

    assert second["status"] == "success"
    assert second["manifest_json"]["retrieval_cache_status"] == "rebuilt"
    assert str(second["manifest_json"]["cache_rebuild_reason"]).startswith("corrupt:")
    assert embedding_model.calls == 2
