from __future__ import annotations

import sys
import types

from research.dedup.training.train_cross_encoder import (
    _maybe_apply_lora,
    _maybe_merge_lora_for_export,
    _parse_lora_target_modules,
)


def test_parse_lora_target_modules_trims_empty_items() -> None:
    assert _parse_lora_target_modules(" q_proj, ,v_proj,o_proj ") == ["q_proj", "v_proj", "o_proj"]


def test_parse_lora_target_modules_returns_none_for_empty_value() -> None:
    assert _parse_lora_target_modules(" , ") is None


def test_apply_lora_wraps_inner_transformer_model(monkeypatch) -> None:
    calls: list[object] = []
    base_model = object()

    class FakeTransformer:
        modality_config = {"text": {}}

        def __init__(self) -> None:
            self.model = base_model

    class FakeCrossEncoder:
        def __init__(self) -> None:
            self.transformer = FakeTransformer()

        def __iter__(self):
            return iter([self.transformer])

        def __setattr__(self, name: str, value: object) -> None:
            if name == "model":
                raise AssertionError("LoRA must not replace the top-level CrossEncoder child")
            super().__setattr__(name, value)

    class FakeLoraConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_task_type = types.SimpleNamespace(SEQ_CLS="SEQ_CLS")

    class FakePeftModel:
        def print_trainable_parameters(self) -> None:
            calls.append("printed")

    peft_model = FakePeftModel()

    def fake_get_peft_model(model: object, config: FakeLoraConfig) -> FakePeftModel:
        calls.extend([model, config.kwargs["target_modules"]])
        return peft_model

    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(
            LoraConfig=FakeLoraConfig,
            TaskType=fake_task_type,
            get_peft_model=fake_get_peft_model,
        ),
    )

    args = types.SimpleNamespace(
        use_peft_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules="q_proj,v_proj",
    )
    cross_encoder = FakeCrossEncoder()

    assert _maybe_apply_lora(cross_encoder, args) is cross_encoder
    assert cross_encoder.transformer.model is peft_model
    assert calls == [base_model, ["q_proj", "v_proj"], "printed"]


def test_merge_lora_updates_inner_transformer_model() -> None:
    merged_model = object()

    class FakePeftModel:
        def merge_and_unload(self) -> object:
            return merged_model

    class FakeTransformer:
        modality_config = {"text": {}}

        def __init__(self) -> None:
            self.model = FakePeftModel()

    class FakeCrossEncoder:
        def __init__(self) -> None:
            self.transformer = FakeTransformer()

        def __iter__(self):
            return iter([self.transformer])

    args = types.SimpleNamespace(use_peft_lora=True, merge_peft_lora=True)
    cross_encoder = FakeCrossEncoder()

    assert _maybe_merge_lora_for_export(cross_encoder, args) is cross_encoder
    assert cross_encoder.transformer.model is merged_model
