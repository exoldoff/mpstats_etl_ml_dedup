from __future__ import annotations

import sys
import types

from research.dedup.model_registry import (
    CROSS_ENCODER_BACKEND,
    POLZA_EMBEDDING_BACKEND,
    SENTENCE_TRANSFORMER_BACKEND,
    TRANSFORMERS_AUTO_MODEL_BACKEND,
    ModelManager,
    PolzaEmbeddingModel,
    clear_model_pool,
    fetch_polza_models,
    model_pool_size,
    model_text_prefix,
    resolve_embedding_model_spec,
    resolve_model_spec,
)


def test_resolve_known_aliases_and_custom_model_ids() -> None:
    spec = resolve_model_spec("qwen3_4b")

    assert spec.alias == "reranker_qwen3_4b"
    assert spec.model_name == "Qwen/Qwen3-Reranker-4B"
    assert spec.backend == CROSS_ENCODER_BACKEND
    assert spec.device == "cpu"

    qwen_small = resolve_model_spec("qwen3_0_6b")
    assert qwen_small.alias == "reranker_qwen3_0_6b"
    assert qwen_small.model_name == "Qwen/Qwen3-Reranker-0.6B"
    assert qwen_small.backend == CROSS_ENCODER_BACKEND

    bge = resolve_model_spec("bge_m3")
    assert bge.alias == "reranker_bge_v2_m3"
    assert bge.model_name == "BAAI/bge-reranker-v2-m3"
    assert bge.backend == CROSS_ENCODER_BACKEND

    custom = resolve_model_spec("vendor/custom-e5-model", backend=SENTENCE_TRANSFORMER_BACKEND)
    assert custom.alias == "vendor/custom-e5-model"
    assert custom.model_name == "vendor/custom-e5-model"
    assert model_text_prefix("vendor/custom-e5-model") == "passage: "

    custom_embedding = resolve_embedding_model_spec("vendor/custom-e5-model")
    assert custom_embedding.backend == SENTENCE_TRANSFORMER_BACKEND

    polza = resolve_model_spec("polza_text_embedding_3_small")
    assert polza.alias == "polza_embedding_3_small"
    assert polza.model_name == "openai/text-embedding-3-small"
    assert polza.backend == POLZA_EMBEDDING_BACKEND
    assert model_text_prefix("polza_embedding_3_small") is None


def test_sentence_transformer_loader_uses_explicit_cache_and_pool(monkeypatch, tmp_path) -> None:
    clear_model_pool()
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            calls.append((model_name, kwargs))

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    manager = ModelManager(cache_dir=tmp_path, local_files_only=True)
    first = manager.load_sentence_transformer("embedding_e5_small")
    second = manager.load_sentence_transformer("embedding_e5_small")

    assert first is second
    assert model_pool_size() == 1
    assert calls == [
        (
            "intfloat/multilingual-e5-small",
            {
                "cache_folder": str(tmp_path),
                "local_files_only": True,
            },
        )
    ]


def test_cross_encoder_loader_falls_back_to_cache_dir_kwarg(monkeypatch, tmp_path) -> None:
    clear_model_pool()
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeCrossEncoder:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            calls.append((model_name, kwargs))
            if "cache_folder" in kwargs:
                raise TypeError("__init__() got an unexpected keyword argument 'cache_folder'")

    fake_module = types.SimpleNamespace(CrossEncoder=FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    manager = ModelManager(cache_dir=tmp_path, local_files_only=True)
    manager.load_cross_encoder("cross_encoder_mmarco")

    assert calls[0][1]["cache_folder"] == str(tmp_path)
    assert calls[1] == (
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        {
            "cache_dir": str(tmp_path),
            "local_files_only": True,
        },
    )


def test_cross_encoder_loader_uses_registry_device(monkeypatch, tmp_path) -> None:
    clear_model_pool()
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeCrossEncoder:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            calls.append((model_name, kwargs))

    fake_module = types.SimpleNamespace(CrossEncoder=FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    manager = ModelManager(cache_dir=tmp_path, local_files_only=True)
    manager.load_cross_encoder("qwen3_4b")

    assert calls == [
        (
            "Qwen/Qwen3-Reranker-4B",
            {
                "cache_folder": str(tmp_path),
                "local_files_only": True,
                "device": "cpu",
                "prompts": {
                    "sku_match": (
                        "Decide whether two ecommerce sauce products are the same SKU. "
                        "Pay attention to brand, flavor or purpose, unit weight, total weight, and pack count."
                    )
                },
                "default_prompt_name": "sku_match",
            },
        )
    ]


def test_transformers_auto_model_loader_uses_registry_defaults(monkeypatch, tmp_path) -> None:
    clear_model_pool()
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeModel:
        def __init__(self) -> None:
            self.eval_called = False

        def eval(self) -> None:
            self.eval_called = True

    fake_model = FakeModel()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: object) -> FakeModel:
            calls.append((model_name, kwargs))
            return fake_model

    fake_module = types.SimpleNamespace(AutoModel=FakeAutoModel)
    monkeypatch.setitem(sys.modules, "transformers", fake_module)

    manager = ModelManager(cache_dir=tmp_path, local_files_only=True)
    loaded = manager.load_transformers_auto_model("jina_v3", dtype="auto")

    assert loaded is fake_model
    assert fake_model.eval_called is True
    assert calls == [
        (
            "jinaai/jina-reranker-v3",
            {
                "cache_dir": str(tmp_path),
                "local_files_only": True,
                "trust_remote_code": True,
                "dtype": "auto",
            },
        )
    ]
    assert resolve_model_spec("jina_v3").backend == TRANSFORMERS_AUTO_MODEL_BACKEND


def test_polza_embedding_model_posts_openai_compatible_payload() -> None:
    calls: list[dict[str, object]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": [
                    {"index": 1, "embedding": [0.0, 3.0, 4.0]},
                    {"index": 0, "embedding": [3.0, 4.0, 0.0]},
                ]
            }

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse()

    model = PolzaEmbeddingModel(
        "openai/text-embedding-3-small",
        api_key="secret",
        base_url="https://polza.ai/api/v1/",
        request_post=fake_post,
    )
    embeddings = model.encode(["a", "b"], batch_size=16, normalize_embeddings=True, convert_to_numpy=True)

    assert embeddings.shape == (2, 3)
    assert round(float(embeddings[0][0]), 2) == 0.6
    assert calls == [
        {
            "url": "https://polza.ai/api/v1/embeddings",
            "headers": {
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
            "json": {
                "model": "openai/text-embedding-3-small",
                "input": ["a", "b"],
                "encoding_format": "float",
            },
            "timeout": 600,
        }
    ]


def test_model_manager_loads_polza_embedding_model_from_alias(monkeypatch) -> None:
    clear_model_pool()
    monkeypatch.setenv("POLZA_API_KEY", "secret")

    manager = ModelManager(polza_base_url="https://example.test/api/v1")
    model = manager.load_embedding_model("polza_embedding_3_small")

    assert isinstance(model, PolzaEmbeddingModel)
    assert model.model_name == "openai/text-embedding-3-small"
    assert model.base_url == "https://example.test/api/v1"
    assert model_pool_size() == 1


def test_fetch_polza_models_uses_embedding_catalog_parameters() -> None:
    calls: list[dict[str, object]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "openai/text-embedding-3-small"}]}

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse()

    models = fetch_polza_models(base_url="https://polza.ai/api/v1/", request_get=fake_get)

    assert models == [{"id": "openai/text-embedding-3-small"}]
    assert calls == [
        {
            "url": "https://polza.ai/api/v1/models",
            "params": {"type": "embedding", "include_providers": "true"},
            "timeout": 60,
        }
    ]
