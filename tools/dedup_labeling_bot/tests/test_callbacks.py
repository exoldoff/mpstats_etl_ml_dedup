from __future__ import annotations

from tools.dedup_labeling_bot.callbacks import (
    make_label_callback,
    make_nav_callback,
    make_noop_callback,
    parse_label_callback,
    parse_nav_callback,
)


def test_label_callback_roundtrip() -> None:
    value = make_label_callback(42, "exact_duplicate")

    parsed = parse_label_callback(value)

    assert parsed.row_index == 42
    assert parsed.label == "exact_duplicate"


def test_nav_callback_roundtrip() -> None:
    value = make_nav_callback(42, "next")

    parsed = parse_nav_callback(value)

    assert parsed.row_index == 42
    assert parsed.direction == "next"


def test_noop_callback_has_row_index() -> None:
    assert make_noop_callback(42) == "noop:42"
