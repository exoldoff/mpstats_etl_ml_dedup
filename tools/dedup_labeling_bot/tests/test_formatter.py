from __future__ import annotations

import pandas as pd

from tools.dedup_labeling_bot.formatter import format_discussion_message, format_help_text, format_pair_message


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
    assert "<b>Название:</b> Мыло &lt;fresh&gt; &amp; clean" in message
    assert "<b>SKU:</b> <code>SKU&lt;1&gt;&amp;A</code>" in message
    assert "<b>Сигналы</b>" in message


def test_format_help_text_escapes_start_placeholder() -> None:
    text = format_help_text()

    assert "<code>/start &lt;пароль&gt;</code>" in text


def test_format_discussion_message_has_initiator() -> None:
    row = pd.Series({"title_a": "A", "title_b": "B", "sku_a": "1", "sku_b": "2"})

    message = format_discussion_message(row, 2, 5, "@user <id>")

    assert "<b>Нужна общая проверка</b>" in message
    assert "<b>Инициатор:</b> @user &lt;id&gt;" in message
