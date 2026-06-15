from __future__ import annotations

import sys
import types

from research.dedup.model_registry import (
    CROSS_ENCODER_BACKEND,
    SENTENCE_TRANSFORMER_BACKEND,
    TRANSFORMERS_AUTO_MODEL_BACKEND,
    ModelManager,
    clear_model_pool,
    model_pool_size,
    model_text_prefix,
    resolve_model_spec,
)


def test_resolve_known_aliases_and_custom_model_ids() -> None:
    spec = resolve_model_spec("qwen3_4b")

    assert spec.alias == "reranker_qwen3_4b"
    assert spec.model_name == "Qwen/Qwen3-Reranker-4B"
    assert spec.backend == CROSS_ENCODER_BACKEND

    custom = resolve_model_spec("vendor/custom-e5-model", backend=SENTENCE_TRANSFORMER_BACKEND)
    assert custom.alias == "vendor/custom-e5-model"
    assert custom.model_name == "vendor/custom-e5-model"
    assert model_text_prefix("vendor/custom-e5-model") == "passage: "


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
