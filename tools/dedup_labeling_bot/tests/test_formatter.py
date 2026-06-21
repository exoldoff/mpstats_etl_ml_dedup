from __future__ import annotations

import pandas as pd

from tools.dedup_labeling_bot.formatter import (
    format_achievement_message,
    format_combo_hot_message,
    format_combo_reset_message,
    format_discussion_message,
    format_help_text,
    format_milestone_message,
    format_pair_message,
    format_player_leaderboard_message,
    format_team_lead_message,
)


def test_format_pair_message_uses_html_and_code_for_sku() -> None:
    row = pd.Series(
        {
            "category_run": "soap",
            "title_a": "Мыло <fresh> & clean",
            "title_b": "Мыло fresh",
            "sku_a": "SKU<1>&A",
            "sku_b": "SKU-2",
            "marketplace_a": "Ozon",
            "marketplace_b": "WB",
            "brand_a": "Brand",
            "brand_b": "Brand",
            "unit_amount_a": "0.5",
            "unit_amount_b": "0.5",
            "total_amount_a": "1.0",
            "total_amount_b": "1.0",
            "multipack_count_a": "2",
            "multipack_count_b": "2",
            "candidate_source": "faiss",
            "embedding_similarity_score": "0.97",
            "is_cross_marketplace_pair": True,
        }
    )

    message = format_pair_message(row, 0, 10)

    assert "<b>Пара 1/10</b>" in message
    assert "<b>SKU A</b>\n🔢 <code>SKU&lt;1&gt;&amp;A</code>" in message
    assert "🛒 Ozon\n🏷️ Brand\n🧾 Мыло &lt;fresh&gt; &amp; clean\n📦 ед. 0.5 / всего 1.0 / x2" in message
    assert "<b>SKU B</b>\n🔢 <code>SKU-2</code>" in message
    assert "🛒 WB\n🏷️ Brand\n🧾 Мыло fresh\n📦 ед. 0.5 / всего 1.0 / x2" in message
    assert "<b>🧷 Справочно</b>" in message
    assert "<b>📊 Сигналы</b>" in message
    assert message.index("<b>🧷 Справочно</b>") < message.index("<b>📊 Сигналы</b>")
    assert message.index("<b>📊 Сигналы</b>") < message.index("<b>🛍️ Товары</b>")
    assert message.index("<b>SKU A</b>") < message.index("<b>SKU B</b>")


def test_format_pair_message_shows_selected_label() -> None:
    row = pd.Series({"title_a": "A", "title_b": "B", "sku_a": "1", "sku_b": "2"})

    message = format_pair_message(row, 0, 10, selected_label="exact_duplicate")

    assert "<b>Решение:</b> Дубль" in message


def test_format_help_text_escapes_start_placeholder() -> None:
    text = format_help_text()

    assert "<code>/start &lt;пароль&gt;</code>" in text
    assert "<code>/team &lt;название&gt;</code>" in text
    assert "<code>/players</code>" in text
    assert "<code>/achievements</code>" in text


def test_format_milestone_message_contains_progress_and_remaining() -> None:
    message = format_milestone_message(threshold=100, labeled_rows=101, total_rows=300, remaining_rows=199)

    assert "🔥 <b>100 заполнено!</b>" in message
    assert "📈 Прогресс: <b>101/300</b>" in message
    assert "⏳ Осталось: <b>199</b>" in message


def test_format_discussion_message_has_initiator() -> None:
    row = pd.Series({"title_a": "A", "title_b": "B", "sku_a": "1", "sku_b": "2"})

    message = format_discussion_message(row, 2, 5, "@user <id>")

    assert "<b>Нужна общая проверка</b>" in message
    assert "<b>Инициатор:</b> @user &lt;id&gt;" in message


def test_format_game_messages_escape_team_name() -> None:
    lead = format_team_lead_message(team_name="A < B", score=12, previous_leader_score=11)
    combo = format_combo_hot_message(team_name="A < B", combo_count=5)
    reset = format_combo_reset_message(team_name="A < B", combo_count=5)
    achievement = format_achievement_message(
        scope="team",
        subject_name="A < B",
        title="Титул <x>",
        description="Описание & детали",
    )

    assert "<b>A &lt; B</b>" in lead
    assert "<b>A &lt; B</b>" in combo
    assert "5 минут" in combo
    assert "<b>A &lt; B</b>" in reset
    assert "Пять минут" in reset
    assert "<b>A &lt; B</b>" in achievement
    assert "<b>Титул &lt;x&gt;</b>" in achievement
    assert "Описание &amp; детали" in achievement


def test_format_player_leaderboard_empty_state() -> None:
    message = format_player_leaderboard_message([])

    assert "<b>Топ игроков</b>" in message
    assert "Пока нет игроков" in message
