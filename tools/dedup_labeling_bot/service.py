from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import LabelingBotConfig
from .csv_repository import CsvStats, LabelingCsvRepository
from .formatter import format_discussion_message, format_milestone_message, format_pair_message
from .state_store import LabelingStateStore


MILESTONE_THRESHOLDS = (100, 300, 500, 1000, 1500, 2000, 2500, 3000)


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
class LabelOutcome:
    status: str
    message: str
    final_label: str | None = None
    milestones: tuple[MilestoneAnnouncement, ...] = ()


class LabelingBotService:
    def __init__(
        self,
        config: LabelingBotConfig,
        *,
        repository: LabelingCsvRepository | None = None,
        store: LabelingStateStore | None = None,
    ) -> None:
        self.config = config
        self.repository = repository or LabelingCsvRepository(config.csv_path)
        self.store = store or LabelingStateStore(config.state_path)

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
                available_rows=available_rows,
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
            if current_position + 1 < len(navigation_rows):
                return self.pair_for_row(user_id, navigation_rows[current_position + 1])
        elif direction == "prev" and navigation_rows:
            previous_rows = [row_index for row_index in navigation_rows if row_index < current_row_index]
            if previous_rows:
                return self.pair_for_row(user_id, previous_rows[-1])

        if direction == "prev":
            return None

        new_rows = self.store.select_available_rows(
            available_rows=available_rows,
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

    def label_row(self, user_id: int, row_index: int, label: str) -> LabelOutcome:
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
            return LabelOutcome(
                status=result.status,
                final_label=result.final_label,
                message=f"Сохранено в CSV: {result.final_label}",
                milestones=milestones,
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
            return LabelOutcome(
                status=result.status,
                final_label=result.final_label,
                message=f"Есть консенсус. Сохранено в CSV: {result.final_label}",
                milestones=milestones,
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
        lines = [
            "Общий прогресс:",
            f"CSV: {csv_stats.labeled_rows}/{csv_stats.total_rows} размечено, {csv_stats.unlabeled_rows} осталось",
            f"Голосов в боте: {self.store.vote_count()}",
            f"Конфликтов: {self.store.conflict_count()}",
            "",
            "Пользователи:",
        ]
        if not users:
            lines.append("пока нет авторизованных пользователей")
        for user in users:
            name = self._format_user_name(user.user_id, user.username, user.first_name)
            lines.append(f"{name}: меток {user.labeled_count}, активных пар {user.active_count}")
        return "\n".join(lines)

    def user_stats(self, user_id: int) -> str:
        stats = self.store.user_stats(user_id)
        if stats is None:
            return "Вы ещё не авторизованы."
        csv_stats = self.repository.stats()
        name = self._format_user_name(stats.user_id, stats.username, stats.first_name)
        return (
            f"{name}\n"
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
