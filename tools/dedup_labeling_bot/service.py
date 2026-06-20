from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import LabelingBotConfig
from .csv_repository import CsvStats, LabelingCsvRepository
from .formatter import format_pair_message
from .state_store import LabelingStateStore


@dataclass(frozen=True)
class AssignedPair:
    row_index: int
    total_rows: int
    message: str


@dataclass(frozen=True)
class LabelOutcome:
    status: str
    message: str
    final_label: str | None = None


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
        row_index = assigned[0]
        frame = self.repository.load()
        row = frame.iloc[row_index]
        return AssignedPair(
            row_index=row_index,
            total_rows=len(frame),
            message=format_pair_message(row, row_index, len(frame)),
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
            self.repository.set_label(row_index, result.final_label or label)
            return LabelOutcome(
                status=result.status,
                final_label=result.final_label,
                message=f"Сохранено в CSV: {result.final_label}",
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

    def release_user_assignments(self, user_id: int) -> int:
        return self.store.release_user_assignments(user_id)

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
