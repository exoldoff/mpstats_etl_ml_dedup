from __future__ import annotations

import asyncio

from telegram.error import TimedOut

from tools.dedup_labeling_bot.handlers import (
    TelegramLabelingHandlers,
    build_bot_commands,
    build_combo_revive_keyboard,
    build_discussion_keyboard,
    build_keyboard,
)


def test_build_keyboard_contains_label_and_next_callbacks() -> None:
    markup = build_keyboard(7)
    buttons = [button for row in markup.inline_keyboard for button in row]
    callback_data = {button.callback_data for button in buttons}

    assert "label:7:exact_duplicate" in callback_data
    assert "label:7:different_product" in callback_data
    assert "label:7:uncertain" in callback_data
    assert "nav:7:prev" in callback_data
    assert "nav:7:next" in callback_data


def test_build_keyboard_marks_selected_label() -> None:
    markup = build_keyboard(7, selected_label="exact_duplicate")
    buttons = [button for row in markup.inline_keyboard for button in row]
    texts = {button.text for button in buttons}
    callback_data = {button.callback_data for button in buttons}

    assert "✓ Дубль" in texts
    assert "noop:7" in callback_data
    assert "label:7:different_product" in callback_data
    assert "label:7:uncertain" in callback_data
    assert "nav:7:prev" in callback_data
    assert "nav:7:next" in callback_data


def test_build_discussion_keyboard_has_vote_buttons_without_next() -> None:
    markup = build_discussion_keyboard(7)
    buttons = [button for row in markup.inline_keyboard for button in row]
    callback_data = {button.callback_data for button in buttons}

    assert "discussion_label:7:exact_duplicate" in callback_data
    assert "discussion_label:7:different_product" in callback_data
    assert "discussion_label:7:uncertain" in callback_data
    assert "next" not in callback_data


def test_build_combo_revive_keyboard_shows_remaining_count() -> None:
    markup = build_combo_revive_keyboard(reset_event_id=7, revives_remaining=5)

    assert markup is not None
    button = markup.inline_keyboard[0][0]
    assert button.text == "Восстановить (5)"
    assert button.callback_data == "combo_revive:7"


def test_build_combo_revive_keyboard_returns_none_without_remaining_revives() -> None:
    assert build_combo_revive_keyboard(reset_event_id=7, revives_remaining=0) is None


def test_build_bot_commands_contains_menu_commands() -> None:
    commands = {command.command: command.description for command in build_bot_commands()}

    assert commands["next"] == "получить пару"
    assert commands["menu"] == "показать команды"
    assert commands["stats"] == "общий прогресс"
    assert commands["team"] == "вступить в команду"
    assert commands["teams"] == "топ команд"
    assert commands["players"] == "топ игроков"
    assert commands["achievements"] == "мои ачивки"
    assert commands["revive"] == "восстановить серию команды"


def test_safe_answer_ignores_telegram_timeout() -> None:
    class TimeoutQuery:
        async def answer(self, text=None):
            raise TimedOut("boom")

    async def run() -> None:
        handlers = TelegramLabelingHandlers(service=None)
        await handlers._safe_answer(TimeoutQuery(), "ok")

    asyncio.run(run())


def test_shutdown_cancels_combo_tasks() -> None:
    async def run() -> None:
        handlers = TelegramLabelingHandlers(service=None)
        task = asyncio.create_task(asyncio.sleep(60))
        handlers.combo_tasks[1] = task

        await handlers.shutdown(application=None)

        assert task.cancelled()
        assert handlers.combo_tasks == {}

    asyncio.run(run())
