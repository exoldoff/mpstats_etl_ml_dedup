from __future__ import annotations

from tools.dedup_labeling_bot.handlers import build_bot_commands, build_discussion_keyboard, build_keyboard


def test_build_keyboard_contains_label_and_next_callbacks() -> None:
    markup = build_keyboard(7)
    buttons = [button for row in markup.inline_keyboard for button in row]
    callback_data = {button.callback_data for button in buttons}

    assert "label:7:exact_duplicate" in callback_data
    assert "label:7:different_product" in callback_data
    assert "label:7:uncertain" in callback_data
    assert "next" in callback_data


def test_build_discussion_keyboard_has_vote_buttons_without_next() -> None:
    markup = build_discussion_keyboard(7)
    buttons = [button for row in markup.inline_keyboard for button in row]
    callback_data = {button.callback_data for button in buttons}

    assert "discussion_label:7:exact_duplicate" in callback_data
    assert "discussion_label:7:different_product" in callback_data
    assert "discussion_label:7:uncertain" in callback_data
    assert "next" not in callback_data


def test_build_bot_commands_contains_menu_commands() -> None:
    commands = {command.command: command.description for command in build_bot_commands()}

    assert commands["next"] == "получить пару"
    assert commands["menu"] == "показать команды"
    assert commands["stats"] == "общий прогресс"
