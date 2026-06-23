from __future__ import annotations

from research.dedup.training.train_cross_encoder import _parse_lora_target_modules


def test_parse_lora_target_modules_trims_empty_items() -> None:
    assert _parse_lora_target_modules(" q_proj, ,v_proj,o_proj ") == ["q_proj", "v_proj", "o_proj"]


def test_parse_lora_target_modules_returns_none_for_empty_value() -> None:
    assert _parse_lora_target_modules(" , ") is None
