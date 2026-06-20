from __future__ import annotations

from html import escape

import pandas as pd

from research.dedup.annotation import cell, first_cell, yes_no


def h(value: object) -> str:
    return escape(str(value), quote=False)


def code(value: object) -> str:
    return f"<code>{h(value)}</code>"


def _field(label: str, value: object) -> str:
    return f"<b>{h(label)}:</b> {h(value) if str(value).strip() else '-'}"


def _item_block(row: pd.Series, side: str) -> str:
    marketplace = first_cell(row, f"marketplace_{side}", f"marketplaces_{side}")
    sku = first_cell(row, f"sku_{side}")
    brand = first_cell(row, f"brand_{side}")
    subcategory = first_cell(row, f"subcategory_{side}")
    unit = first_cell(row, f"unit_amount_{side}")
    total = first_cell(row, f"total_amount_{side}")
    pack = first_cell(row, f"multipack_count_{side}")
    return "\n".join(
        [
            _field("Название", cell(row, f"title_{side}")),
            f"<b>SKU:</b> {code(sku)}",
            _field("Маркетплейс", marketplace),
            _field("Бренд", brand),
            _field("Подкатегория", subcategory),
            f"<b>Фасовка:</b> ед. {h(unit)} / всего {h(total)} / x{h(pack)}",
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


def format_pair_message(row: pd.Series, row_index: int, total_rows: int) -> str:
    category_parts = [
        cell(row, "category_run"),
        cell(row, "category_name"),
        cell(row, "project_name"),
    ]
    category = " / ".join(part for part in category_parts if part)
    category_line = f"\n{_field('Категория', category)}" if category else ""
    notes = cell(row, "notes")
    notes_line = f"\n\n{_field('Заметка', notes)}" if notes else ""

    return (
        f"<b>Пара {row_index + 1}/{total_rows}</b>{category_line}\n\n"
        f"<b>A</b>\n"
        f"{_item_block(row, 'a')}\n\n"
        f"<b>B</b>\n"
        f"{_item_block(row, 'b')}\n\n"
        f"<b>Сигналы</b>\n"
        f"{_signal_block(row)}"
        f"{notes_line}"
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
