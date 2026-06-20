from __future__ import annotations

from tools.dedup_labeling_bot.handlers import build_keyboard


def test_build_keyboard_contains_label_and_next_callbacks() -> None:
    markup = build_keyboard(7)
    buttons = [button for row in markup.inline_keyboard for button in row]
    callback_data = {button.callback_data for button in buttons}

    assert "label:7:exact_duplicate" in callback_data
    assert "label:7:different_product" in callback_data
    assert "label:7:uncertain" in callback_data
    assert "next" in callback_data
