from __future__ import annotations

from tools.dedup_labeling_bot.callbacks import make_label_callback, parse_label_callback


def test_label_callback_roundtrip() -> None:
    value = make_label_callback(42, "exact_duplicate")

    parsed = parse_label_callback(value)

    assert parsed.row_index == 42
    assert parsed.label == "exact_duplicate"
