from __future__ import annotations

from html import escape

import pandas as pd

from research.dedup.annotation import (
    LABEL_DIFFERENT_PRODUCT,
    LABEL_EXACT_DUPLICATE,
    LABEL_UNCERTAIN,
    cell,
    first_cell,
    yes_no,
)


LABEL_TITLES = {
    LABEL_EXACT_DUPLICATE: "Дубль",
    LABEL_DIFFERENT_PRODUCT: "Разные товары",
    LABEL_UNCERTAIN: "Не уверен",
}


def h(value: object) -> str:
    return escape(str(value), quote=False)


def code(value: object) -> str:
    return f"<code>{h(value)}</code>"


def _field(label: str, value: object) -> str:
    return f"<b>{h(label)}:</b> {h(value) if str(value).strip() else '-'}"


def _pack_summary(row: pd.Series, side: str) -> str:
    unit = first_cell(row, f"unit_amount_{side}")
    total = first_cell(row, f"total_amount_{side}")
    pack = first_cell(row, f"multipack_count_{side}")
    return f"ед. {unit} / всего {total} / x{pack}"


def _item_block(row: pd.Series, side: str) -> str:
    marketplace = first_cell(row, f"marketplace_{side}", f"marketplaces_{side}")
    sku = first_cell(row, f"sku_{side}")
    brand = first_cell(row, f"brand_{side}")
    title = first_cell(row, f"title_{side}")
    pack = _pack_summary(row, side)
    return "\n".join(
        [
            f"<b>SKU {side.upper()}</b>",
            f"🔢 {code(sku)}",
            f"🛒 {h(marketplace)}",
            f"🏷️ {h(brand)}",
            f"🧾 {h(title)}",
            f"📦 {h(pack)}",
        ]
    )


def _reference_block(row: pd.Series) -> str:
    subcategory_a = first_cell(row, "subcategory_a")
    subcategory_b = first_cell(row, "subcategory_b")
    return "\n".join(
        [
            _field("Подкатегория A", subcategory_a),
            _field("Подкатегория B", subcategory_b),
        ]
    )


def _signal_block(row: pd.Series) -> str:
    return "\n".join(
        [
            _field("Источник", first_cell(row, "candidate_source")),
            _field("Scope", first_cell(row, "blocking_scope")),
            _field("Subcat", first_cell(row, "subcategory_relation")),
            _field("Rank", first_cell(row, "candidate_rank")),
            _field("Score", first_cell(row, "embedding_similarity_score", "baseline_similarity_score")),
            _field("Стратегия", first_cell(row, "labeling_stratum")),
            _field("Межмаркетплейс", yes_no(cell(row, "is_cross_marketplace_pair"))),
            _field("Сложный негатив", yes_no(cell(row, "is_hard_negative_candidate"))),
            _field("Вариант упаковки", yes_no(cell(row, "is_pack_variant_candidate"))),
        ]
    )


def _label_title(label: str) -> str:
    return LABEL_TITLES.get(label, label)


def format_pair_message(
    row: pd.Series,
    row_index: int,
    total_rows: int,
    *,
    selected_label: str | None = None,
) -> str:
    category_parts = [
        cell(row, "category_run"),
        cell(row, "category_name"),
        cell(row, "project_name"),
    ]
    category = " / ".join(part for part in category_parts if part)
    notes = cell(row, "notes")

    header_lines = [f"<b>Пара {row_index + 1}/{total_rows}</b>"]
    if category:
        header_lines.append(_field("Категория", category))
    if selected_label:
        header_lines.append(_field("Решение", _label_title(selected_label)))
    if notes:
        header_lines.append(_field("Заметка", notes))
    header = "\n".join(header_lines)

    return (
        f"{header}\n\n"
        f"<b>🧷 Справочно</b>\n"
        f"{_reference_block(row)}\n\n"
        f"<b>📊 Сигналы</b>\n"
        f"{_signal_block(row)}\n\n"
        f"<b>🛍️ Товары</b>\n"
        f"{_item_block(row, 'a')}\n\n"
        f"{_item_block(row, 'b')}"
    )


def format_milestone_message(*, threshold: int, labeled_rows: int, total_rows: int, remaining_rows: int) -> str:
    templates = {
        100: "Молодцы, банда, первые сто добили. Дальше уже пошла нормальная заруба.",
        300: "Вот это темп. Таблица начала трещать, а мы только разогреваемся.",
        500: "Пять сотен закрыто. Красавцы, ебашим как надо.",
        1000: "Тысяча взята. Это уже не разметка, это производственный угар.",
        1500: "Полторы тысячи в кармане. Не расслабляемся, добиваем эту махину.",
        2000: "Две тысячи закрыто. Жёсткая работа, очень по делу.",
        2500: "Две с половиной тысячи. Уже пахнет финишем и лёгким безумием.",
        3000: "Три тысячи. Всё, это мощно. Команда реально вывезла.",
    }
    line = templates.get(threshold, "Рубеж взят. Хорошая работа, продолжаем давить.")
    return (
        f"🔥 <b>{threshold} заполнено!</b>\n"
        f"{h(line)}\n"
        f"📈 Прогресс: <b>{labeled_rows}/{total_rows}</b>\n"
        f"⏳ Осталось: <b>{remaining_rows}</b>\n"
        "🚀 Погнали дальше."
    )


def format_discussion_message(row: pd.Series, row_index: int, total_rows: int, user_display: str) -> str:
    return (
        "<b>Нужна общая проверка</b>\n"
        f"<b>Инициатор:</b> {h(user_display)}\n\n"
        f"{format_pair_message(row, row_index, total_rows)}"
    )


def format_help_text() -> str:
    return (
        "<b>Команды</b>\n"
        f"{code('/start <пароль>')} — войти\n"
        f"{code('/next')} — получить пару\n"
        f"{code('/me')} — моя статистика\n"
        f"{code('/stats')} — общий прогресс\n"
        f"{code('/release')} — освободить мои неразмеченные пары\n"
        f"{code('/logout')} — выйти\n"
        f"{code('/menu')} — показать это меню"
    )
