from __future__ import annotations

from dataclasses import dataclass
import random
from datetime import datetime

import pandas as pd

from .config import LabelingBotConfig
from .csv_repository import CsvStats, LabelingCsvRepository
from .formatter import (
    format_achievement_message,
    format_achievements_list,
    format_combo_hot_message,
    format_combo_reset_message,
    format_discussion_message,
    format_milestone_message,
    format_pair_message,
    format_player_leaderboard_message,
    format_team_joined_message,
    format_team_leaderboard_message,
    format_team_lead_message,
)
from .state_store import LabelingStateStore


MILESTONE_THRESHOLDS = (100, 300, 500, 1000, 1500, 2000, 2500, 3000)
COMBO_ANNOUNCEMENT_THRESHOLDS = (3, 5, 10, 20, 30, 50, 100)
USER_SCORE_ACHIEVEMENTS = (
    (1, "Первый удар", "Первый финальный ответ в зачёте. Лёд тронулся."),
    (10, "Десятник", "10 финальных ответов. Уже виден рабочий темп."),
    (25, "Снайпер разметки", "25 финальных ответов. Рука набита, глаз пристрелян."),
    (50, "Полтинник", "50 финальных ответов. Это уже личная смена в разметке."),
    (100, "Сотня без паники", "100 финальных ответов. Машина прогресса завелась."),
)
TEAM_SCORE_ACHIEVEMENTS = (
    (10, "Командный старт", "10 финальных ответов на командном счёте."),
    (25, "Слаженная смена", "25 финальных ответов команды."),
    (50, "Командный разгон", "50 финальных ответов команды."),
    (100, "Отряд зачистки", "100 финальных ответов команды."),
    (250, "Цех разметки", "250 финальных ответов команды."),
)
TEAM_COMBO_ACHIEVEMENTS = (
    (3, "Искра серии", "Команда удержала комбо x3."),
    (5, "Горячая рука", "Команда удержала комбо x5."),
    (10, "Без тормозов", "Команда удержала комбо x10."),
    (20, "Комбо-машина", "Команда удержала комбо x20."),
)


@dataclass(frozen=True)
class AssignedPair:
    row_index: int
    total_rows: int
    message: str
    selected_label: str | None = None


@dataclass(frozen=True)
class MilestoneAnnouncement:
    threshold: int
    labeled_rows: int
    total_rows: int
    remaining_rows: int
    message: str


@dataclass(frozen=True)
class ComboTimer:
    team_id: int
    deadline_at: datetime


@dataclass(frozen=True)
class GameAnnouncement:
    message: str
    group_only: bool = False


@dataclass(frozen=True)
class LabelOutcome:
    status: str
    message: str
    final_label: str | None = None
    milestones: tuple[MilestoneAnnouncement, ...] = ()
    game_announcements: tuple[GameAnnouncement, ...] = ()
    combo_timers: tuple[ComboTimer, ...] = ()


class LabelingBotService:
    def __init__(
        self,
        config: LabelingBotConfig,
        *,
        repository: LabelingCsvRepository | None = None,
        store: LabelingStateStore | None = None,
        assignment_rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self.repository = repository or LabelingCsvRepository(config.csv_path)
        self.store = store or LabelingStateStore(config.state_path)
        self.assignment_rng = assignment_rng or random.SystemRandom()

    def authorize(self, user_id: int, password: str, *, username: str = "", first_name: str = "") -> bool:
        if password != self.config.access_password:
            return False
        self.store.authorize_user(user_id, username=username, first_name=first_name)
        return True

    def logout(self, user_id: int) -> int:
        released = self.store.release_user_assignments(user_id)
        self.store.logout_user(user_id)
        return released

    def is_authorized(self, user_id: int) -> bool:
        return self.store.is_authorized(user_id)

    def touch_user(self, user_id: int, *, username: str = "", first_name: str = "") -> None:
        self.store.touch_user(user_id, username=username, first_name=first_name)

    def next_pair(self, user_id: int) -> AssignedPair | None:
        self.store.expire_stale_assignments()
        available_rows = self.repository.available_row_indices()
        if not available_rows:
            return None
        available_set = set(available_rows)

        assigned = self.store.user_assigned_rows(user_id, available_rows=available_set)
        if not assigned:
            new_rows = self.store.select_available_rows(
                available_rows=self._assignment_order(available_rows),
                user_id=user_id,
                mode=self.config.assignment_mode,
                limit=self.config.batch_size,
                overlap_votes=self.config.overlap_votes,
            )
            self.store.assign_rows(user_id, new_rows, ttl=self.config.assignment_ttl)
            assigned = new_rows

        if not assigned:
            return None
        return self.pair_for_row(user_id, assigned[0])

    def navigate_pair(self, user_id: int, current_row_index: int, direction: str) -> AssignedPair | None:
        if direction not in {"prev", "next"}:
            raise ValueError(f"Unsupported navigation direction: {direction!r}")

        self.store.expire_stale_assignments()
        available_rows = self.repository.available_row_indices()
        available_set = set(available_rows)
        navigation_rows = self.store.user_navigation_rows(user_id, available_rows=available_set)

        if current_row_index in navigation_rows:
            current_position = navigation_rows.index(current_row_index)
            if direction == "prev":
                if current_position == 0:
                    return None
                return self.pair_for_row(user_id, navigation_rows[current_position - 1])
            for row_index in navigation_rows[current_position + 1 :]:
                if row_index in available_set:
                    return self.pair_for_row(user_id, row_index)
        elif direction == "prev" and navigation_rows:
            previous_rows = [row_index for row_index in navigation_rows if row_index < current_row_index]
            if previous_rows:
                return self.pair_for_row(user_id, previous_rows[-1])

        if direction == "prev":
            return None

        new_rows = self.store.select_available_rows(
            available_rows=self._assignment_order(available_rows),
            user_id=user_id,
            mode=self.config.assignment_mode,
            limit=self.config.batch_size,
            overlap_votes=self.config.overlap_votes,
        )
        self.store.assign_rows(user_id, new_rows, ttl=self.config.assignment_ttl)
        if not new_rows:
            return None
        return self.pair_for_row(user_id, new_rows[0])

    def pair_for_row(self, user_id: int, row_index: int) -> AssignedPair:
        frame = self.repository.load()
        row = frame.iloc[row_index]
        selected_label = self.store.user_row_label(user_id, row_index)
        return AssignedPair(
            row_index=row_index,
            total_rows=len(frame),
            selected_label=selected_label,
            message=format_pair_message(row, row_index, len(frame), selected_label=selected_label),
        )

    def _assignment_order(self, available_rows: list[int]) -> list[int]:
        rows = list(available_rows)
        if len(rows) <= 1:
            return rows

        batch_size = max(1, self.config.batch_size)
        bucket_count = min(batch_size, len(rows))
        buckets = [
            rows[start:end]
            for start, end in (
                (idx * len(rows) // bucket_count, (idx + 1) * len(rows) // bucket_count)
                for idx in range(bucket_count)
            )
        ]

        picked: list[int] = []
        leftovers: list[int] = []
        for bucket in buckets:
            shuffled_bucket = list(bucket)
            self.assignment_rng.shuffle(shuffled_bucket)
            picked.append(shuffled_bucket[0])
            leftovers.extend(shuffled_bucket[1:])

        self.assignment_rng.shuffle(picked)
        self.assignment_rng.shuffle(leftovers)
        return picked + leftovers

    def label_row(self, user_id: int, row_index: int, label: str) -> LabelOutcome:
        is_relabel = self.store.user_row_label(user_id, row_index) is not None
        if not is_relabel and not self.repository.is_row_available(row_index):
            raise ValueError("Пара уже решена, беру следующую актуальную")
        result = self.store.record_vote(
            row_index=row_index,
            user_id=user_id,
            label=label,
            mode=self.config.assignment_mode,
            overlap_votes=self.config.overlap_votes,
        )
        if result.finalized:
            previous_labeled_rows = self.repository.stats().labeled_rows
            self.repository.set_label(row_index, result.final_label or label)
            milestones = self._claim_milestones(previous_labeled_rows=previous_labeled_rows)
            game_announcements, combo_timers = self._record_game_updates(
                row_index=row_index,
                final_label=result.final_label or label,
                finalized_by_user_id=user_id,
                previous_labeled_rows=previous_labeled_rows,
            )
            return LabelOutcome(
                status=result.status,
                final_label=result.final_label,
                message=f"Сохранено в CSV: {result.final_label}",
                milestones=milestones,
                game_announcements=game_announcements,
                combo_timers=combo_timers,
            )
        if result.conflict:
            return LabelOutcome(
                status=result.status,
                message="Голоса разошлись. В CSV ничего не записано, строка помечена как conflict.",
            )
        return LabelOutcome(
            status=result.status,
            message="Голос сохранён. Ждём консенсус по этой строке.",
        )

    def move_row_to_discussion(self, user_id: int, row_index: int, label: str) -> LabelOutcome:
        if not self.repository.is_row_available(row_index):
            raise ValueError("Пара уже решена, беру следующую актуальную")
        result = self.store.move_assigned_row_to_discussion(
            row_index=row_index,
            user_id=user_id,
            label=label,
        )
        return LabelOutcome(
            status=result.status,
            message="Пара отправлена в общий чат. CSV пока не обновляю.",
        )

    def vote_discussion_row(self, user_id: int, row_index: int, label: str) -> LabelOutcome:
        result = self.store.record_discussion_vote(
            row_index=row_index,
            user_id=user_id,
            label=label,
            overlap_votes=self.config.overlap_votes,
        )
        if result.finalized:
            previous_labeled_rows = self.repository.stats().labeled_rows
            self.repository.set_label(row_index, result.final_label or label)
            milestones = self._claim_milestones(previous_labeled_rows=previous_labeled_rows)
            game_announcements, combo_timers = self._record_game_updates(
                row_index=row_index,
                final_label=result.final_label or label,
                finalized_by_user_id=user_id,
                previous_labeled_rows=previous_labeled_rows,
            )
            return LabelOutcome(
                status=result.status,
                final_label=result.final_label,
                message=f"Есть консенсус. Сохранено в CSV: {result.final_label}",
                milestones=milestones,
                game_announcements=game_announcements,
                combo_timers=combo_timers,
            )
        return LabelOutcome(
            status=result.status,
            message="Голос в общем обсуждении сохранён. Ждём консенсус.",
        )

    def discussion_message(self, row_index: int, user_display: str) -> str:
        frame = self.repository.load()
        row = frame.iloc[row_index]
        return format_discussion_message(row, row_index, len(frame), user_display)

    def should_send_discussion(self, row_index: int) -> bool:
        return self.config.discussion_chat_id is not None and not self.store.discussion_was_posted(row_index)

    def record_discussion_post(self, row_index: int, user_id: int, *, chat_id: int | str, message_id: int) -> None:
        self.store.record_discussion_post(
            row_index=row_index,
            user_id=user_id,
            chat_id=str(chat_id),
            message_id=message_id,
        )

    def release_user_assignments(self, user_id: int) -> int:
        return self.store.release_user_assignments(user_id)

    def milestone_recipients(self) -> list[int]:
        return self.store.authorized_user_ids()

    def announcement_recipients(self) -> list[int | str]:
        recipients: list[int | str] = list(self.store.authorized_user_ids())
        if self.config.discussion_chat_id is not None:
            recipients.append(self.config.discussion_chat_id)
        if self.config.leaderboard_chat_id is not None:
            recipients.append(self.config.leaderboard_chat_id)
        return list(dict.fromkeys(recipients))

    def group_announcement_recipients(self) -> list[int | str]:
        recipients: list[int | str] = []
        if self.config.discussion_chat_id is not None:
            recipients.append(self.config.discussion_chat_id)
        if self.config.leaderboard_chat_id is not None:
            recipients.append(self.config.leaderboard_chat_id)
        return list(dict.fromkeys(recipients))

    def join_team(self, user_id: int, team_name: str) -> str:
        membership = self.store.set_user_team(user_id, team_name)
        return format_team_joined_message(
            team_name=membership.name,
            leaderboard=self.team_leaderboard(),
        )

    def team_leaderboard(self) -> str:
        return format_team_leaderboard_message(self.store.team_leaderboard(limit=10))

    def player_leaderboard(self) -> str:
        return format_player_leaderboard_message(self.store.player_leaderboard(limit=10))

    def user_achievements(self, user_id: int) -> str:
        team = self.store.user_team(user_id)
        team_achievements = self.store.team_achievements(team.team_id) if team is not None else []
        return format_achievements_list(
            user_achievements=self.store.user_achievements(user_id),
            team_name=team.name if team is not None else None,
            team_achievements=team_achievements,
        )

    def leaderboard_pin(self) -> tuple[str, int] | None:
        pin = self.store.leaderboard_pin()
        if pin is None:
            return None
        return pin.chat_id, pin.message_id

    def record_leaderboard_pin(self, *, chat_id: int | str, message_id: int) -> None:
        self.store.record_leaderboard_pin(chat_id=chat_id, message_id=message_id)

    def expire_team_combo(self, *, team_id: int, deadline_at: datetime) -> GameAnnouncement | None:
        reset = self.store.expire_team_combo(team_id=team_id, expected_deadline_at=deadline_at)
        if reset is None:
            return None
        return GameAnnouncement(
            message=format_combo_reset_message(
                team_name=reset.team_name,
                combo_count=reset.combo_count,
            ),
            group_only=True,
        )

    def _record_game_updates(
        self,
        *,
        row_index: int,
        final_label: str,
        finalized_by_user_id: int,
        previous_labeled_rows: int,
    ) -> tuple[tuple[GameAnnouncement, ...], tuple[ComboTimer, ...]]:
        csv_stats = self.repository.stats()
        if csv_stats.labeled_rows <= previous_labeled_rows:
            return (), ()

        updates = self.store.record_game_answers_for_finalized_row(
            row_index=row_index,
            final_label=final_label,
            finalized_by_user_id=finalized_by_user_id,
            combo_timeout=self.config.combo_timeout,
        )
        announcements: list[GameAnnouncement] = []
        timers: list[ComboTimer] = []

        for user_score in updates.user_score_changes:
            announcements.extend(self._claim_user_score_achievements(user_score))

        for team_score in updates.team_score_changes:
            announcements.extend(self._claim_team_score_achievements(team_score))

        for reset in updates.combo_resets:
            announcements.append(
                GameAnnouncement(
                    message=format_combo_reset_message(
                        team_name=reset.team_name,
                        combo_count=reset.combo_count,
                    ),
                    group_only=True,
                )
            )

        for lead in updates.lead_changes:
            announcements.append(
                GameAnnouncement(
                    message=format_team_lead_message(
                        team_name=lead.team_name,
                        score=lead.score,
                        previous_leader_score=lead.previous_leader_score,
                    )
                )
            )

        for combo in updates.combo_updates:
            crossed_thresholds = [
                threshold
                for threshold in COMBO_ANNOUNCEMENT_THRESHOLDS
                if combo.previous_count < threshold <= combo.combo_count
            ]
            if crossed_thresholds:
                announcements.append(
                    GameAnnouncement(
                        message=format_combo_hot_message(
                            team_name=combo.team_name,
                            combo_count=combo.combo_count,
                        ),
                        group_only=True,
                    )
                )
                announcements.extend(self._claim_team_combo_achievements(combo.team_id, combo.team_name, crossed_thresholds))
            timers.append(ComboTimer(team_id=combo.team_id, deadline_at=combo.deadline_at))

        return tuple(announcements), tuple(timers)

    def _claim_user_score_achievements(self, user_score: object) -> list[GameAnnouncement]:
        display_name = self._format_user_name(user_score.user_id, user_score.username, user_score.first_name)
        result: list[GameAnnouncement] = []
        for threshold, title, description in USER_SCORE_ACHIEVEMENTS:
            if user_score.score < threshold:
                continue
            achievement = self.store.claim_achievement(
                scope="user",
                owner_id=user_score.user_id,
                code=f"solo_{threshold}",
                title=title,
                description=description,
            )
            if achievement is not None:
                result.append(
                    GameAnnouncement(
                        message=format_achievement_message(
                            scope="user",
                            subject_name=display_name,
                            title=achievement.title,
                            description=achievement.description,
                        )
                    )
                )
        return result

    def _claim_team_score_achievements(self, team_score: object) -> list[GameAnnouncement]:
        result: list[GameAnnouncement] = []
        for threshold, title, description in TEAM_SCORE_ACHIEVEMENTS:
            if team_score.score < threshold:
                continue
            achievement = self.store.claim_achievement(
                scope="team",
                owner_id=team_score.team_id,
                code=f"team_score_{threshold}",
                title=title,
                description=description,
            )
            if achievement is not None:
                result.append(
                    GameAnnouncement(
                        message=format_achievement_message(
                            scope="team",
                            subject_name=team_score.team_name,
                            title=achievement.title,
                            description=achievement.description,
                        )
                    )
                )
        return result

    def _claim_team_combo_achievements(
        self,
        team_id: int,
        team_name: str,
        crossed_thresholds: list[int],
    ) -> list[GameAnnouncement]:
        result: list[GameAnnouncement] = []
        for threshold, title, description in TEAM_COMBO_ACHIEVEMENTS:
            if threshold not in crossed_thresholds:
                continue
            achievement = self.store.claim_achievement(
                scope="team",
                owner_id=team_id,
                code=f"team_combo_{threshold}",
                title=title,
                description=description,
            )
            if achievement is not None:
                result.append(
                    GameAnnouncement(
                        message=format_achievement_message(
                            scope="team",
                            subject_name=team_name,
                            title=achievement.title,
                            description=achievement.description,
                        ),
                        group_only=True,
                    )
                )
        return result

    def _claim_milestones(self, *, previous_labeled_rows: int) -> tuple[MilestoneAnnouncement, ...]:
        csv_stats = self.repository.stats()
        crossed_thresholds = tuple(
            threshold
            for threshold in MILESTONE_THRESHOLDS
            if previous_labeled_rows < threshold <= csv_stats.labeled_rows
        )
        if not crossed_thresholds:
            return ()
        thresholds = self.store.claim_due_milestones(
            thresholds=crossed_thresholds,
            labeled_rows=csv_stats.labeled_rows,
            remaining_rows=csv_stats.unlabeled_rows,
        )
        return tuple(
            MilestoneAnnouncement(
                threshold=threshold,
                labeled_rows=csv_stats.labeled_rows,
                total_rows=csv_stats.total_rows,
                remaining_rows=csv_stats.unlabeled_rows,
                message=format_milestone_message(
                    threshold=threshold,
                    labeled_rows=csv_stats.labeled_rows,
                    total_rows=csv_stats.total_rows,
                    remaining_rows=csv_stats.unlabeled_rows,
                ),
            )
            for threshold in thresholds
        )

    def stats(self) -> str:
        csv_stats = self.repository.stats()
        users = self.store.all_user_stats()
        teams = self.store.team_leaderboard(limit=10, include_members=False)
        lines = [
            "Общий прогресс:",
            f"CSV: {csv_stats.labeled_rows}/{csv_stats.total_rows} размечено, {csv_stats.unlabeled_rows} осталось",
            f"Голосов в боте: {self.store.vote_count()}",
            f"Конфликтов: {self.store.conflict_count()}",
            "",
            "Команды:",
        ]
        if not teams:
            lines.append("пока нет команд")
        for idx, team in enumerate(teams, start=1):
            combo = f", комбо x{team.combo_count}" if team.combo_count > 0 else ""
            lines.append(f"{idx}. {team.name}: {team.score}{combo}")
        lines.extend(
            [
                "",
                "Пользователи:",
            ]
        )
        if not users:
            lines.append("пока нет авторизованных пользователей")
        for user in users:
            name = self._format_user_name(user.user_id, user.username, user.first_name)
            team = self.store.user_team(user.user_id)
            game_count = self.store.user_game_answer_count(user.user_id)
            team_suffix = f", команда {team.name}" if team is not None else ", без команды"
            lines.append(
                f"{name}: игровых ответов {game_count}, меток {user.labeled_count}, "
                f"активных пар {user.active_count}{team_suffix}"
            )
        return "\n".join(lines)

    def user_stats(self, user_id: int) -> str:
        stats = self.store.user_stats(user_id)
        if stats is None:
            return "Вы ещё не авторизованы."
        csv_stats = self.repository.stats()
        name = self._format_user_name(stats.user_id, stats.username, stats.first_name)
        team = self.store.user_team(user_id)
        game_count = self.store.user_game_answer_count(user_id)
        if team is None:
            team_line = "Команда: нет, вступите через /team <название>"
        else:
            team_line = f"Команда: {team.name}, командный счёт: {self.store.team_score(team.team_id)}"
        return (
            f"{name}\n"
            f"{team_line}\n"
            f"Игровые ответы: {game_count}\n"
            f"Ваши метки: {stats.labeled_count}\n"
            f"Активные пары: {stats.active_count}\n"
            f"Общий CSV-прогресс: {csv_stats.labeled_rows}/{csv_stats.total_rows}"
        )

    @staticmethod
    def _format_user_name(user_id: int, username: str, first_name: str) -> str:
        if username:
            return f"@{username} ({user_id})"
        if first_name:
            return f"{first_name} ({user_id})"
        return str(user_id)
