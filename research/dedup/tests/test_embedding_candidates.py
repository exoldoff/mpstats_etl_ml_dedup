from __future__ import annotations

import numpy as np
import pandas as pd

from research.dedup import (
    CandidateGenerationConfig,
    FaissCandidateGenerationConfig,
    generate_faiss_candidate_pairs,
    prepare_product_records,
)


class _FakeIndexFlatIP:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._vectors: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        assert vectors.shape[1] == self.dimension
        self._vectors = vectors

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._vectors is not None
        scores = queries @ self._vectors.T
        order = np.argsort(-scores, axis=1)[:, :k]
        sorted_scores = np.take_along_axis(scores, order, axis=1)
        return sorted_scores.astype("float32"), order.astype("int64")


class _FakeFaiss:
    IndexFlatIP = _FakeIndexFlatIP

    @staticmethod
    def normalize_L2(vectors: np.ndarray) -> None:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors /= norms


def test_faiss_candidate_generation_uses_embedding_neighbors() -> None:
    products = pd.DataFrame(
        [
            {"Маркетплейс": "WB", "Артикул": "1", "SKU": "Соус томатный 200 г", "Бренд": "A"},
            {"Маркетплейс": "Ozon", "Артикул": "2", "SKU": "Томатный соус 0.2 кг", "Бренд": "A"},
            {"Маркетплейс": "WB", "Артикул": "3", "SKU": "Уксус яблочный 500 мл", "Бренд": "B"},
        ]
    )
    records = prepare_product_records(products)
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )

    pairs = generate_faiss_candidate_pairs(
        records,
        embeddings,
        FaissCandidateGenerationConfig(top_k=1, min_similarity=0.8, max_candidates=None),
        faiss_module=_FakeFaiss,
    )

    assert len(pairs) == 1
    pair = pairs.iloc[0]
    assert {pair["sku_a"], pair["sku_b"]} == {"1", "2"}
    assert pair["candidate_source"] == "faiss_embedding_topk"
    assert pair["embedding_similarity_score"] == pair["baseline_similarity_score"]
    assert bool(pair["is_cross_marketplace_pair"]) is True
    assert pair["blocking_scope"] == "global"


def test_faiss_subcategory_blocking_keeps_known_subcategories_separate() -> None:
    products = pd.DataFrame(
        [
            {"Маркетплейс": "WB", "Артикул": "1", "SKU": "Мыло жидкое 300 мл", "Бренд": "A", "Подкатегория": "Жидкое"},
            {"Маркетплейс": "Ozon", "Артикул": "2", "SKU": "Мыло жидкое 300 мл запас", "Бренд": "A", "Подкатегория": "Жидкое"},
            {"Маркетплейс": "WB", "Артикул": "3", "SKU": "Мыло твердое 90 г", "Бренд": "A", "Подкатегория": "Твердое"},
            {"Маркетплейс": "Ozon", "Артикул": "4", "SKU": "Мыло твердое 100 г", "Бренд": "A", "Подкатегория": "Твердое"},
        ]
    )
    records = prepare_product_records(
        products,
        CandidateGenerationConfig(
            collapse_empty_brand_exact_titles=False,
            collapse_exact_title_same_brand=False,
        ),
    )
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.99, 0.01],
            [0.0, 1.0],
        ],
        dtype="float32",
    )

    pairs = generate_faiss_candidate_pairs(
        records,
        embeddings,
        FaissCandidateGenerationConfig(
            top_k=1,
            max_candidates=None,
            global_safety_top_k=0,
            unknown_subcategory_top_k=0,
        ),
        faiss_module=_FakeFaiss,
    )

    pair_skus = {frozenset((row.sku_a, row.sku_b)) for row in pairs.itertuples(index=False)}
    assert frozenset(("1", "3")) not in pair_skus
    assert frozenset(("1", "2")) in pair_skus
    assert set(pairs["blocking_scope"]) == {"same_subcategory"}
    assert set(pairs["subcategory_relation"]) == {"same_subcategory"}


def test_faiss_unknown_subcategory_searches_global_scope() -> None:
    products = pd.DataFrame(
        [
            {"Маркетплейс": "WB", "Артикул": "1", "SKU": "Мыло жидкое 300 мл", "Бренд": "A", "Подкатегория": "Жидкое"},
            {"Маркетплейс": "Ozon", "Артикул": "2", "SKU": "Мыло жидкое 300 мл", "Бренд": "A", "Подкатегория": ""},
            {"Маркетплейс": "WB", "Артикул": "3", "SKU": "Мыло твердое 90 г", "Бренд": "A", "Подкатегория": "Твердое"},
        ]
    )
    records = prepare_product_records(
        products,
        CandidateGenerationConfig(
            collapse_empty_brand_exact_titles=False,
            collapse_exact_title_same_brand=False,
        ),
    )
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ],
        dtype="float32",
    )

    pairs = generate_faiss_candidate_pairs(
        records,
        embeddings,
        FaissCandidateGenerationConfig(
            top_k=1,
            max_candidates=None,
            global_safety_top_k=0,
            unknown_subcategory_top_k=1,
        ),
        faiss_module=_FakeFaiss,
    )

    assert len(pairs) == 1
    pair = pairs.iloc[0]
    assert {pair["sku_a"], pair["sku_b"]} == {"1", "2"}
    assert pair["blocking_scope"] == "unknown_subcategory"
    assert pair["subcategory_relation"] == "unknown_subcategory"


def test_faiss_falls_back_to_global_when_subcategory_is_absent() -> None:
    products = pd.DataFrame(
        [
            {"Маркетплейс": "WB", "Артикул": "1", "SKU": "Кокосовое масло 500 мл", "Бренд": "A"},
            {"Маркетплейс": "Ozon", "Артикул": "2", "SKU": "Масло кокосовое 0.5 л", "Бренд": "A"},
            {"Маркетплейс": "WB", "Артикул": "3", "SKU": "Кокосовая стружка 100 г", "Бренд": "B"},
        ]
    )
    records = prepare_product_records(products)
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
        ],
        dtype="float32",
    )

    pairs = generate_faiss_candidate_pairs(
        records,
        embeddings,
        FaissCandidateGenerationConfig(top_k=1, max_candidates=None),
        faiss_module=_FakeFaiss,
    )

    pair_skus = {frozenset((row.sku_a, row.sku_b)) for row in pairs.itertuples(index=False)}
    assert frozenset(("1", "2")) in pair_skus
    assert set(pairs["blocking_scope"]) == {"global"}
