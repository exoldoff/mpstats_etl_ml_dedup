from __future__ import annotations

import pandas as pd

from research.dedup.annotation import cell, item_summary, signal_summary


def format_pair_message(row: pd.Series, row_index: int, total_rows: int) -> str:
    category_parts = [
        cell(row, "category_run"),
        cell(row, "category_name"),
        cell(row, "project_name"),
    ]
    category = " / ".join(part for part in category_parts if part)
    category_line = f"\nКатегория: {category}" if category else ""
    notes = cell(row, "notes")
    notes_line = f"\nЗаметка: {notes}" if notes else ""

    return (
        f"Пара {row_index + 1}/{total_rows}{category_line}\n\n"
        f"A: {cell(row, 'title_a')}\n"
        f"   {item_summary(row, 'a')}\n\n"
        f"B: {cell(row, 'title_b')}\n"
        f"   {item_summary(row, 'b')}\n\n"
        f"Сигналы: {signal_summary(row)}"
        f"{notes_line}"
    )


def format_help_text() -> str:
    return (
        "Команды:\n"
        "/start <пароль> — войти\n"
        "/next — получить пару\n"
        "/me — моя статистика\n"
        "/stats — общий прогресс\n"
        "/release — освободить мои неразмеченные пары\n"
        "/logout — выйти"
    )
