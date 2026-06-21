from __future__ import annotations

from html import escape
from typing import Sequence

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


def _answer_word(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return "ответ"
    if count % 10 in {2, 3, 4} and count % 100 not in {12, 13, 14}:
        return "ответа"
    return "ответов"


def _user_display(user_id: int, username: str, first_name: str) -> str:
    if username:
        return f"@{username}"
    if first_name:
        return first_name
    return str(user_id)


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


def format_team_joined_message(*, team_name: str, leaderboard: str) -> str:
    return (
        f"🏁 Вы в команде <b>{h(team_name)}</b>.\n"
        "Теперь ваши финальные ответы идут в командный зачёт.\n\n"
        f"{leaderboard}"
    )


def format_team_leaderboard_message(teams: Sequence[object]) -> str:
    lines = [
        "🏆 <b>Топ команд разметки</b>",
        "Очки начисляются за финально решённые пары.",
    ]
    if not teams:
        lines.append("")
        lines.append("Пока нет команд. Создайте команду через /team Название")
        return "\n".join(lines)

    medals = ("🥇", "🥈", "🥉")
    for idx, team in enumerate(teams, start=1):
        medal = medals[idx - 1] if idx <= len(medals) else f"{idx}."
        score = int(getattr(team, "score"))
        combo_count = int(getattr(team, "combo_count"))
        combo = f" · ⚡ x{combo_count}" if combo_count > 0 else ""
        lines.append("")
        lines.append(f"{medal} <b>{h(getattr(team, 'name'))}</b> — {score} {_answer_word(score)}{combo}")
        members = list(getattr(team, "members", ()))
        if members:
            member_parts = [
                f"{h(_user_display(member.user_id, member.username, member.first_name))}: {member.score}"
                for member in members[:4]
            ]
            hidden_count = len(members) - len(member_parts)
            if hidden_count > 0:
                member_parts.append(f"+{hidden_count}")
            lines.append(f"   👥 {' · '.join(member_parts)}")
    return "\n".join(lines)


def format_player_leaderboard_message(players: Sequence[object]) -> str:
    lines = [
        "🏅 <b>Топ игроков</b>",
        "Считаются все финальные ответы, даже без команды.",
    ]
    if not players:
        lines.append("")
        lines.append("Пока нет игроков в зачёте.")
        return "\n".join(lines)

    medals = ("🥇", "🥈", "🥉")
    for idx, player in enumerate(players, start=1):
        medal = medals[idx - 1] if idx <= len(medals) else f"{idx}."
        score = int(getattr(player, "score"))
        achievements = int(getattr(player, "achievement_count"))
        team_name = getattr(player, "team_name", None)
        team = f" · 🏁 {h(team_name)}" if team_name else ""
        badges = f" · 🎖️ {achievements}" if achievements else ""
        name = _user_display(player.user_id, player.username, player.first_name)
        lines.append(f"{medal} <b>{h(name)}</b> — {score} {_answer_word(score)}{badges}{team}")
    return "\n".join(lines)


def format_team_lead_message(*, team_name: str, score: int, previous_leader_score: int) -> str:
    return (
        "🚨🚨🚨 <b>Смена лидера!</b> 🚨🚨🚨\n"
        f"🏎️💨 Команда <b>{h(team_name)}</b> вырывается вперёд!\n"
        f"🏆 Сейчас у них <b>{score}</b> {_answer_word(score)}.\n"
        f"📊 Предыдущая планка лидера: <b>{previous_leader_score}</b>.\n"
        "🔥⚡️💪 Держим темп, догоняем, обгоняем!"
    )


def format_achievement_message(*, scope: str, subject_name: str, title: str, description: str) -> str:
    subject = "Команда" if scope == "team" else "Игрок"
    return (
        "🎖️🎉 <b>Ачивка разблокирована!</b> 🎉🎖️\n"
        f"{subject} <b>{h(subject_name)}</b>\n"
        f"🏆 <b>{h(title)}</b>\n"
        f"✨ {h(description)}"
    )


def format_achievements_list(
    *,
    user_achievements: Sequence[object],
    team_name: str | None = None,
    team_achievements: Sequence[object] = (),
) -> str:
    lines = ["🎖️ <b>Ачивки</b>", "", "<b>Личные</b>"]
    if not user_achievements:
        lines.append("Пока пусто. Первый финальный ответ откроет стартовую ачивку.")
    else:
        for achievement in user_achievements:
            lines.append(f"• <b>{h(achievement.title)}</b> — {h(achievement.description)}")

    lines.append("")
    if team_name:
        lines.append(f"<b>Команда {h(team_name)}</b>")
        if not team_achievements:
            lines.append("Пока нет командных ачивок.")
        else:
            for achievement in team_achievements:
                lines.append(f"• <b>{h(achievement.title)}</b> — {h(achievement.description)}")
    else:
        lines.append("<b>Команда</b>")
        lines.append("Вы пока без команды. Вступите через /team <название>.")
    return "\n".join(lines)


def format_combo_hot_message(*, team_name: str, combo_count: int) -> str:
    return (
        f"⚡️⚡️⚡️ <b>Комбо x{combo_count}!</b> ⚡️⚡️⚡️\n"
        f"Команда <b>{h(team_name)}</b> закрывает ответы серией без паузы.\n"
        "⏱️ Есть 5 минут, чтобы продлить серию.\n"
        "🔥🔥🔥 Вот это темп!"
    )


def format_combo_reset_message(*, team_name: str, combo_count: int) -> str:
    return (
        "💥 <b>Комбо сброшено</b>\n"
        f"Команда <b>{h(team_name)}</b> остановилась на x{combo_count}.\n"
        "⏱️ Пять минут прошли без нового финального ответа.\n"
        "🔁 Собираемся и запускаем новую серию!"
    )


def format_help_text() -> str:
    return (
        "<b>Команды</b>\n"
        f"{code('/start <пароль>')} — войти\n"
        f"{code('/next')} — получить пару\n"
        f"{code('/me')} — моя статистика\n"
        f"{code('/team <название>')} — вступить в команду или создать её\n"
        f"{code('/teams')} — топ команд\n"
        f"{code('/players')} — топ игроков\n"
        f"{code('/achievements')} — мои ачивки\n"
        f"{code('/stats')} — общий прогресс\n"
        f"{code('/release')} — освободить мои неразмеченные пары\n"
        f"{code('/logout')} — выйти\n"
        f"{code('/menu')} — показать это меню"
    )
