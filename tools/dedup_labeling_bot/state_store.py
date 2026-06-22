from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

from research.dedup.annotation import VALID_LABELS


ASSIGNED = "assigned"
LABELED = "labeled"
RELEASED = "released"
EXPIRED = "expired"

ROW_FINALIZED = "finalized"
ROW_CONFLICT = "conflict"
ROW_DISCUSSION = "discussion"
COMBO_REVIVE_LIMIT = 5


@dataclass(frozen=True)
class VoteResult:
    status: str
    final_label: str | None = None

    @property
    def finalized(self) -> bool:
        return self.status == ROW_FINALIZED and self.final_label is not None

    @property
    def conflict(self) -> bool:
        return self.status == ROW_CONFLICT


@dataclass(frozen=True)
class UserLabelStats:
    user_id: int
    username: str
    first_name: str
    labeled_count: int
    active_count: int


@dataclass(frozen=True)
class TeamMembership:
    team_id: int
    name: str


@dataclass(frozen=True)
class TeamMemberStats:
    user_id: int
    username: str
    first_name: str
    score: int


@dataclass(frozen=True)
class PlayerStats:
    user_id: int
    username: str
    first_name: str
    score: int
    team_name: str | None = None
    achievement_count: int = 0


@dataclass(frozen=True)
class TeamStats:
    team_id: int
    name: str
    score: int
    combo_count: int
    combo_deadline_at: datetime | None
    members: tuple[TeamMemberStats, ...] = ()


@dataclass(frozen=True)
class LeaderboardPin:
    chat_id: str
    message_id: int


@dataclass(frozen=True)
class TeamLeadChange:
    team_id: int
    team_name: str
    score: int
    previous_leader_score: int


@dataclass(frozen=True)
class TeamComboUpdate:
    team_id: int
    team_name: str
    previous_count: int
    combo_count: int
    deadline_at: datetime


@dataclass(frozen=True)
class TeamComboReset:
    team_id: int
    team_name: str
    combo_count: int
    reset_event_id: int | None = None
    revives_remaining: int = 0


@dataclass(frozen=True)
class TeamComboRevive:
    team_id: int
    team_name: str
    combo_count: int
    deadline_at: datetime
    revives_remaining: int


@dataclass(frozen=True)
class DiscussionPost:
    row_index: int
    requested_by_user_id: int
    requested_at: datetime
    chat_id: str
    message_id: int


@dataclass(frozen=True)
class DiscussionVote:
    user_id: int
    username: str
    first_name: str
    label: str
    voted_at: datetime


@dataclass(frozen=True)
class UserScoreChange:
    user_id: int
    username: str
    first_name: str
    team_id: int | None
    team_name: str | None
    score: int


@dataclass(frozen=True)
class TeamScoreChange:
    team_id: int
    team_name: str
    score: int


@dataclass(frozen=True)
class AchievementRecord:
    scope: str
    owner_id: int
    code: str
    title: str
    description: str
    unlocked_at: datetime


@dataclass(frozen=True)
class GameUpdateResult:
    lead_changes: tuple[TeamLeadChange, ...] = ()
    combo_updates: tuple[TeamComboUpdate, ...] = ()
    combo_resets: tuple[TeamComboReset, ...] = ()
    user_score_changes: tuple[UserScoreChange, ...] = ()
    team_score_changes: tuple[TeamScoreChange, ...] = ()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _from_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _optional_from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return _from_iso(value)


def _normalize_team_name(value: str) -> tuple[str, str]:
    display_name = " ".join(value.strip().split())
    if not display_name:
        raise ValueError("Team name is empty")
    if len(display_name) > 48:
        raise ValueError("Team name is too long")
    return display_name, display_name.casefold()


class LabelingStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL DEFAULT '',
                    first_name TEXT NOT NULL DEFAULT '',
                    authorized_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assignments (
                    row_index INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    label TEXT,
                    assigned_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    labeled_at TEXT,
                    released_at TEXT,
                    PRIMARY KEY (row_index, user_id)
                );
                CREATE TABLE IF NOT EXISTS row_states (
                    row_index INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    final_label TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS discussion_posts (
                    row_index INTEGER PRIMARY KEY,
                    requested_by_user_id INTEGER NOT NULL,
                    requested_at TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    message_id INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS milestone_notifications (
                    threshold INTEGER PRIMARY KEY,
                    notified_at TEXT NOT NULL,
                    labeled_rows INTEGER NOT NULL,
                    remaining_rows INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS teams (
                    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS team_members (
                    user_id INTEGER PRIMARY KEY,
                    team_id INTEGER NOT NULL,
                    joined_at TEXT NOT NULL,
                    FOREIGN KEY (team_id) REFERENCES teams(team_id)
                );
                CREATE TABLE IF NOT EXISTS game_answers (
                    row_index INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    team_id INTEGER NOT NULL,
                    answered_at TEXT NOT NULL,
                    PRIMARY KEY (row_index, user_id),
                    FOREIGN KEY (team_id) REFERENCES teams(team_id)
                );
                CREATE TABLE IF NOT EXISTS player_answers (
                    row_index INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    answered_at TEXT NOT NULL,
                    PRIMARY KEY (row_index, user_id)
                );
                CREATE TABLE IF NOT EXISTS team_combos (
                    team_id INTEGER PRIMARY KEY,
                    combo_count INTEGER NOT NULL,
                    last_answer_at TEXT,
                    deadline_at TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (team_id) REFERENCES teams(team_id)
                );
                CREATE TABLE IF NOT EXISTS team_combo_reset_events (
                    reset_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id INTEGER NOT NULL,
                    combo_count INTEGER NOT NULL,
                    reset_at TEXT NOT NULL,
                    revived_at TEXT,
                    revived_by_user_id INTEGER,
                    FOREIGN KEY (team_id) REFERENCES teams(team_id)
                );
                CREATE TABLE IF NOT EXISTS leaderboard_pin (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    chat_id TEXT NOT NULL,
                    message_id INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS achievements (
                    scope TEXT NOT NULL,
                    owner_id INTEGER NOT NULL,
                    code TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    unlocked_at TEXT NOT NULL,
                    PRIMARY KEY (scope, owner_id, code)
                );
                CREATE INDEX IF NOT EXISTS idx_assignments_status
                    ON assignments(status, expires_at);
                CREATE INDEX IF NOT EXISTS idx_assignments_user_status
                    ON assignments(user_id, status);
                CREATE INDEX IF NOT EXISTS idx_player_answers_user
                    ON player_answers(user_id);
                CREATE INDEX IF NOT EXISTS idx_game_answers_team
                    ON game_answers(team_id);
                CREATE INDEX IF NOT EXISTS idx_game_answers_user
                    ON game_answers(user_id);
                CREATE INDEX IF NOT EXISTS idx_achievements_owner
                    ON achievements(scope, owner_id);
                CREATE INDEX IF NOT EXISTS idx_team_combo_reset_events_team
                    ON team_combo_reset_events(team_id, reset_at);
                INSERT OR IGNORE INTO player_answers (row_index, user_id, answered_at)
                SELECT row_index, user_id, answered_at
                FROM game_answers;
                """
            )

    def authorize_user(
        self,
        user_id: int,
        *,
        username: str = "",
        first_name: str = "",
        now: datetime | None = None,
    ) -> None:
        current = now or utcnow()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users (user_id, username, first_name, authorized_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    username = excluded.username,
                    first_name = excluded.first_name,
                    last_seen_at = excluded.last_seen_at
                """,
                (user_id, username or "", first_name or "", _to_iso(current), _to_iso(current)),
            )

    def logout_user(self, user_id: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

    def is_authorized(self, user_id: int) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return row is not None

    def touch_user(
        self,
        user_id: int,
        *,
        username: str = "",
        first_name: str = "",
        now: datetime | None = None,
    ) -> None:
        current = now or utcnow()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE users
                SET username = ?, first_name = ?, last_seen_at = ?
                WHERE user_id = ?
                """,
                (username or "", first_name or "", _to_iso(current), user_id),
            )

    def authorized_user_ids(self) -> list[int]:
        with self._connect() as connection:
            rows = connection.execute("SELECT user_id FROM users ORDER BY user_id").fetchall()
        return [int(row["user_id"]) for row in rows]

    def expire_stale_assignments(self, now: datetime | None = None) -> int:
        current = now or utcnow()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE assignments
                SET status = ?
                WHERE status = ? AND expires_at <= ?
                """,
                (EXPIRED, ASSIGNED, _to_iso(current)),
            )
            return int(cursor.rowcount)

    def release_user_assignments(self, user_id: int, now: datetime | None = None) -> int:
        current = now or utcnow()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE assignments
                SET status = ?, released_at = ?
                WHERE user_id = ? AND status = ?
                """,
                (RELEASED, _to_iso(current), user_id, ASSIGNED),
            )
            return int(cursor.rowcount)

    def assign_rows(
        self,
        user_id: int,
        row_indices: list[int],
        *,
        ttl: timedelta,
        now: datetime | None = None,
    ) -> None:
        if not row_indices:
            return
        current = now or utcnow()
        expires_at = current + ttl
        rows = [
            (row_index, user_id, ASSIGNED, _to_iso(current), _to_iso(expires_at))
            for row_index in row_indices
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO assignments (row_index, user_id, status, assigned_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(row_index, user_id) DO UPDATE SET
                    status = excluded.status,
                    assigned_at = excluded.assigned_at,
                    expires_at = excluded.expires_at,
                    label = NULL,
                    labeled_at = NULL,
                    released_at = NULL
                """,
                rows,
            )

    def user_assigned_rows(
        self,
        user_id: int,
        *,
        available_rows: set[int],
        now: datetime | None = None,
    ) -> list[int]:
        current = now or utcnow()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT row_index
                FROM assignments
                WHERE user_id = ? AND status = ? AND expires_at > ?
                ORDER BY assigned_at, row_index
                """,
                (user_id, ASSIGNED, _to_iso(current)),
            ).fetchall()
        return [int(row["row_index"]) for row in rows if int(row["row_index"]) in available_rows]

    def user_navigation_rows(
        self,
        user_id: int,
        *,
        available_rows: set[int],
        now: datetime | None = None,
    ) -> list[int]:
        current = now or utcnow()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT row_index, status, expires_at
                FROM assignments
                WHERE user_id = ? AND status IN (?, ?)
                ORDER BY assigned_at, row_index
                """,
                (user_id, ASSIGNED, LABELED),
            ).fetchall()

        result: list[int] = []
        for row in rows:
            row_index = int(row["row_index"])
            if row["status"] == LABELED:
                result.append(row_index)
            elif row_index in available_rows and _from_iso(row["expires_at"]) > current:
                result.append(row_index)
        return result

    def user_row_label(self, user_id: int, row_index: int) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT label
                FROM assignments
                WHERE user_id = ? AND row_index = ? AND status = ? AND label IS NOT NULL
                """,
                (user_id, row_index, LABELED),
            ).fetchone()
        if row is None:
            return None
        return str(row["label"])

    def claim_due_milestones(
        self,
        *,
        thresholds: tuple[int, ...],
        labeled_rows: int,
        remaining_rows: int,
        now: datetime | None = None,
    ) -> list[int]:
        current = now or utcnow()
        due_thresholds = [threshold for threshold in thresholds if threshold <= labeled_rows]
        if not due_thresholds:
            return []

        claimed: list[int] = []
        with self._connect() as connection:
            for threshold in due_thresholds:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO milestone_notifications (
                        threshold,
                        notified_at,
                        labeled_rows,
                        remaining_rows
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (threshold, _to_iso(current), labeled_rows, remaining_rows),
                )
                if cursor.rowcount:
                    claimed.append(threshold)
        return claimed

    def select_available_rows(
        self,
        *,
        available_rows: list[int],
        user_id: int,
        mode: str,
        limit: int,
        overlap_votes: int,
        now: datetime | None = None,
    ) -> list[int]:
        current = now or utcnow()
        available_set = set(available_rows)
        with self._connect() as connection:
            assignment_rows = connection.execute("SELECT * FROM assignments").fetchall()
            state_rows = connection.execute("SELECT * FROM row_states").fetchall()

        row_states = {int(row["row_index"]): row["status"] for row in state_rows}
        by_row: dict[int, list[sqlite3.Row]] = {}
        for row in assignment_rows:
            row_index = int(row["row_index"])
            if row_index in available_set:
                by_row.setdefault(row_index, []).append(row)

        pending_overlap_rows: list[int] = []
        fresh_rows: list[int] = []
        for row_index in available_rows:
            if row_states.get(row_index) in {ROW_FINALIZED, ROW_CONFLICT, ROW_DISCUSSION}:
                continue
            assignments = by_row.get(row_index, [])
            if any(int(row["user_id"]) == user_id and row["status"] in {ASSIGNED, LABELED} for row in assignments):
                continue
            if mode == "unique":
                has_active = any(
                    row["status"] == ASSIGNED and _from_iso(row["expires_at"]) > current
                    for row in assignments
                )
                has_vote = any(row["status"] == LABELED for row in assignments)
                if has_active or has_vote:
                    continue
                fresh_rows.append(row_index)
            else:
                participants = 0
                for row in assignments:
                    if row["status"] == LABELED:
                        participants += 1
                    elif row["status"] == ASSIGNED and _from_iso(row["expires_at"]) > current:
                        participants += 1
                if participants >= overlap_votes:
                    continue
                if participants:
                    pending_overlap_rows.append(row_index)
                else:
                    fresh_rows.append(row_index)
        selected = pending_overlap_rows + fresh_rows
        return selected[:limit]

    def record_vote(
        self,
        *,
        row_index: int,
        user_id: int,
        label: str,
        mode: str,
        overlap_votes: int,
        now: datetime | None = None,
    ) -> VoteResult:
        if label not in VALID_LABELS:
            raise ValueError(f"Unsupported label: {label!r}")
        current = now or utcnow()
        with self._connect() as connection:
            assignment = connection.execute(
                """
                SELECT *
                FROM assignments
                WHERE row_index = ? AND user_id = ?
                """,
                (row_index, user_id),
            ).fetchone()
            if assignment is None or assignment["status"] not in {ASSIGNED, LABELED}:
                raise ValueError("Row is not assigned to this user")
            is_relabel = assignment["status"] == LABELED
            if is_relabel and mode != "unique":
                raise ValueError("Row is already labeled by this user")
            if assignment["status"] == ASSIGNED and _from_iso(assignment["expires_at"]) <= current:
                connection.execute(
                    """
                    UPDATE assignments
                    SET status = ?
                    WHERE row_index = ? AND user_id = ? AND status = ?
                    """,
                    (EXPIRED, row_index, user_id, ASSIGNED),
                )
                raise ValueError("Assignment is expired")

            connection.execute(
                """
                UPDATE assignments
                SET status = ?, label = ?, labeled_at = ?
                WHERE row_index = ? AND user_id = ?
                """,
                (LABELED, label, _to_iso(current), row_index, user_id),
            )

            if mode == "unique":
                connection.execute(
                    """
                    INSERT INTO row_states (row_index, status, final_label, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(row_index) DO UPDATE SET
                        status = excluded.status,
                        final_label = excluded.final_label,
                        updated_at = excluded.updated_at
                    """,
                    (row_index, ROW_FINALIZED, label, _to_iso(current)),
                )
                return VoteResult(status=ROW_FINALIZED, final_label=label)

            vote_rows = connection.execute(
                """
                SELECT label
                FROM assignments
                WHERE row_index = ? AND status = ? AND label IS NOT NULL
                """,
                (row_index, LABELED),
            ).fetchall()
            labels = [str(row["label"]) for row in vote_rows]
            counts = Counter(labels)
            for candidate_label, count in counts.items():
                if count >= overlap_votes:
                    connection.execute(
                        """
                        INSERT INTO row_states (row_index, status, final_label, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(row_index) DO UPDATE SET
                            status = excluded.status,
                            final_label = excluded.final_label,
                            updated_at = excluded.updated_at
                        """,
                        (row_index, ROW_FINALIZED, candidate_label, _to_iso(current)),
                    )
                    return VoteResult(status=ROW_FINALIZED, final_label=candidate_label)

            if len(labels) >= overlap_votes:
                connection.execute(
                    """
                    INSERT INTO row_states (row_index, status, final_label, updated_at)
                    VALUES (?, ?, NULL, ?)
                    ON CONFLICT(row_index) DO UPDATE SET
                        status = excluded.status,
                        final_label = NULL,
                        updated_at = excluded.updated_at
                    """,
                    (row_index, ROW_CONFLICT, _to_iso(current)),
                )
                return VoteResult(status=ROW_CONFLICT)

        return VoteResult(status="pending")

    def move_assigned_row_to_discussion(
        self,
        *,
        row_index: int,
        user_id: int,
        label: str,
        now: datetime | None = None,
    ) -> VoteResult:
        if label not in VALID_LABELS:
            raise ValueError(f"Unsupported label: {label!r}")
        current = now or utcnow()
        with self._connect() as connection:
            assignment = connection.execute(
                """
                SELECT *
                FROM assignments
                WHERE row_index = ? AND user_id = ?
                """,
                (row_index, user_id),
            ).fetchone()
            if assignment is None or assignment["status"] != ASSIGNED:
                raise ValueError("Row is not assigned to this user")
            if _from_iso(assignment["expires_at"]) <= current:
                connection.execute(
                    """
                    UPDATE assignments
                    SET status = ?
                    WHERE row_index = ? AND user_id = ? AND status = ?
                    """,
                    (EXPIRED, row_index, user_id, ASSIGNED),
                )
                raise ValueError("Assignment is expired")
            connection.execute(
                """
                UPDATE assignments
                SET status = ?, label = ?, labeled_at = ?
                WHERE row_index = ? AND user_id = ?
                """,
                (LABELED, label, _to_iso(current), row_index, user_id),
            )
            connection.execute(
                """
                INSERT INTO row_states (row_index, status, final_label, updated_at)
                VALUES (?, ?, NULL, ?)
                ON CONFLICT(row_index) DO UPDATE SET
                    status = excluded.status,
                    final_label = NULL,
                    updated_at = excluded.updated_at
                """,
                (row_index, ROW_DISCUSSION, _to_iso(current)),
            )
        return VoteResult(status=ROW_DISCUSSION)

    def record_discussion_vote(
        self,
        *,
        row_index: int,
        user_id: int,
        label: str,
        overlap_votes: int,
        now: datetime | None = None,
    ) -> VoteResult:
        if label not in VALID_LABELS:
            raise ValueError(f"Unsupported label: {label!r}")
        current = now or utcnow()
        with self._connect() as connection:
            row_state = connection.execute(
                "SELECT * FROM row_states WHERE row_index = ?",
                (row_index,),
            ).fetchone()
            if row_state is not None and row_state["status"] == ROW_FINALIZED:
                return VoteResult(status=ROW_FINALIZED, final_label=str(row_state["final_label"]))
            connection.execute(
                """
                INSERT INTO assignments (
                    row_index,
                    user_id,
                    status,
                    label,
                    assigned_at,
                    expires_at,
                    labeled_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(row_index, user_id) DO UPDATE SET
                    status = excluded.status,
                    label = excluded.label,
                    labeled_at = excluded.labeled_at
                """,
                (row_index, user_id, LABELED, label, _to_iso(current), _to_iso(current), _to_iso(current)),
            )
            connection.execute(
                """
                INSERT INTO row_states (row_index, status, final_label, updated_at)
                VALUES (?, ?, NULL, ?)
                ON CONFLICT(row_index) DO UPDATE SET
                    status = CASE
                        WHEN row_states.status = ? THEN row_states.status
                        ELSE excluded.status
                    END,
                    updated_at = excluded.updated_at
                """,
                (row_index, ROW_DISCUSSION, _to_iso(current), ROW_FINALIZED),
            )
            vote_rows = connection.execute(
                """
                SELECT label
                FROM assignments
                WHERE row_index = ? AND status = ? AND label IS NOT NULL
                """,
                (row_index, LABELED),
            ).fetchall()
            labels = [str(row["label"]) for row in vote_rows]
            counts = Counter(labels)
            for candidate_label, count in counts.items():
                if count >= overlap_votes:
                    connection.execute(
                        """
                        INSERT INTO row_states (row_index, status, final_label, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(row_index) DO UPDATE SET
                            status = excluded.status,
                            final_label = excluded.final_label,
                            updated_at = excluded.updated_at
                        """,
                        (row_index, ROW_FINALIZED, candidate_label, _to_iso(current)),
                    )
                    return VoteResult(status=ROW_FINALIZED, final_label=candidate_label)
        return VoteResult(status=ROW_DISCUSSION)

    def expire_discussion_row(
        self,
        *,
        row_index: int,
        expected_requested_at: datetime,
        final_label: str,
        now: datetime | None = None,
    ) -> bool:
        if final_label not in VALID_LABELS:
            raise ValueError(f"Unsupported label: {final_label!r}")
        current = now or utcnow()
        expected_requested = _to_iso(expected_requested_at)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT rs.status, dp.requested_at
                FROM row_states rs
                JOIN discussion_posts dp ON dp.row_index = rs.row_index
                WHERE rs.row_index = ?
                """,
                (row_index,),
            ).fetchone()
            if row is None or row["status"] != ROW_DISCUSSION or str(row["requested_at"]) != expected_requested:
                return False
            connection.execute(
                """
                INSERT INTO row_states (row_index, status, final_label, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(row_index) DO UPDATE SET
                    status = excluded.status,
                    final_label = excluded.final_label,
                    updated_at = excluded.updated_at
                """,
                (row_index, ROW_FINALIZED, final_label, _to_iso(current)),
            )
        return True

    def discussion_was_posted(self, row_index: int) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM discussion_posts WHERE row_index = ?",
                (row_index,),
            ).fetchone()
        return row is not None

    def record_discussion_post(
        self,
        *,
        row_index: int,
        user_id: int,
        chat_id: str,
        message_id: int,
        now: datetime | None = None,
    ) -> None:
        current = now or utcnow()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO discussion_posts (
                    row_index,
                    requested_by_user_id,
                    requested_at,
                    chat_id,
                    message_id
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(row_index) DO NOTHING
                """,
                (row_index, user_id, _to_iso(current), chat_id, message_id),
            )

    def discussion_post(self, row_index: int) -> DiscussionPost | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT row_index, requested_by_user_id, requested_at, chat_id, message_id
                FROM discussion_posts
                WHERE row_index = ?
                """,
                (row_index,),
            ).fetchone()
        if row is None:
            return None
        return DiscussionPost(
            row_index=int(row["row_index"]),
            requested_by_user_id=int(row["requested_by_user_id"]),
            requested_at=_from_iso(str(row["requested_at"])),
            chat_id=str(row["chat_id"]),
            message_id=int(row["message_id"]),
        )

    def pending_discussion_posts(self) -> list[DiscussionPost]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT dp.row_index, dp.requested_by_user_id, dp.requested_at, dp.chat_id, dp.message_id
                FROM discussion_posts dp
                JOIN row_states rs ON rs.row_index = dp.row_index
                WHERE rs.status = ?
                ORDER BY dp.requested_at, dp.row_index
                """,
                (ROW_DISCUSSION,),
            ).fetchall()
        return [
            DiscussionPost(
                row_index=int(row["row_index"]),
                requested_by_user_id=int(row["requested_by_user_id"]),
                requested_at=_from_iso(str(row["requested_at"])),
                chat_id=str(row["chat_id"]),
                message_id=int(row["message_id"]),
            )
            for row in rows
        ]

    def discussion_votes(self, row_index: int) -> list[DiscussionVote]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    a.user_id,
                    u.username,
                    u.first_name,
                    a.label,
                    a.labeled_at
                FROM assignments a
                LEFT JOIN users u ON u.user_id = a.user_id
                WHERE a.row_index = ? AND a.status = ? AND a.label IS NOT NULL
                ORDER BY a.labeled_at, a.user_id
                """,
                (row_index, LABELED),
            ).fetchall()
        return [
            DiscussionVote(
                user_id=int(row["user_id"]),
                username=str(row["username"] or ""),
                first_name=str(row["first_name"] or ""),
                label=str(row["label"]),
                voted_at=_from_iso(str(row["labeled_at"])),
            )
            for row in rows
        ]

    def set_user_team(
        self,
        user_id: int,
        team_name: str,
        *,
        now: datetime | None = None,
    ) -> TeamMembership:
        display_name, normalized_name = _normalize_team_name(team_name)
        current = now or utcnow()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO teams (name, normalized_name, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(normalized_name) DO NOTHING
                """,
                (display_name, normalized_name, _to_iso(current)),
            )
            team = connection.execute(
                "SELECT team_id, name FROM teams WHERE normalized_name = ?",
                (normalized_name,),
            ).fetchone()
            if team is None:
                raise ValueError("Team was not created")
            connection.execute(
                """
                INSERT INTO team_members (user_id, team_id, joined_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    team_id = excluded.team_id,
                    joined_at = excluded.joined_at
                """,
                (user_id, int(team["team_id"]), _to_iso(current)),
            )
        return TeamMembership(team_id=int(team["team_id"]), name=str(team["name"]))

    def user_team(self, user_id: int) -> TeamMembership | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT t.team_id, t.name
                FROM team_members tm
                JOIN teams t ON t.team_id = tm.team_id
                WHERE tm.user_id = ?
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return TeamMembership(team_id=int(row["team_id"]), name=str(row["name"]))

    def team_leaderboard(self, *, limit: int = 10, include_members: bool = True) -> list[TeamStats]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    t.team_id,
                    t.name,
                    COUNT(ga.row_index) AS score,
                    COALESCE(tc.combo_count, 0) AS combo_count,
                    tc.deadline_at AS combo_deadline_at
                FROM teams t
                LEFT JOIN game_answers ga ON ga.team_id = t.team_id
                LEFT JOIN team_combos tc ON tc.team_id = t.team_id
                GROUP BY t.team_id, t.name, tc.combo_count, tc.deadline_at
                ORDER BY score DESC, combo_count DESC, lower(t.name)
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            result: list[TeamStats] = []
            for row in rows:
                team_id = int(row["team_id"])
                members = self._team_member_stats(connection, team_id) if include_members else ()
                result.append(
                    TeamStats(
                        team_id=team_id,
                        name=str(row["name"]),
                        score=int(row["score"]),
                        combo_count=int(row["combo_count"]),
                        combo_deadline_at=_optional_from_iso(row["combo_deadline_at"]),
                        members=members,
                    )
                )
        return result

    def player_leaderboard(self, *, limit: int = 10) -> list[PlayerStats]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    u.user_id,
                    u.username,
                    u.first_name,
                    t.name AS team_name,
                    COUNT(DISTINCT pa.row_index) AS score,
                    COUNT(DISTINCT a.code) AS achievement_count
                FROM users u
                LEFT JOIN player_answers pa ON pa.user_id = u.user_id
                LEFT JOIN team_members tm ON tm.user_id = u.user_id
                LEFT JOIN teams t ON t.team_id = tm.team_id
                LEFT JOIN achievements a ON a.scope = 'user' AND a.owner_id = u.user_id
                GROUP BY u.user_id, u.username, u.first_name, t.name
                ORDER BY score DESC, achievement_count DESC, u.user_id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            PlayerStats(
                user_id=int(row["user_id"]),
                username=str(row["username"] or ""),
                first_name=str(row["first_name"] or ""),
                score=int(row["score"]),
                team_name=str(row["team_name"]) if row["team_name"] else None,
                achievement_count=int(row["achievement_count"]),
            )
            for row in rows
        ]

    def user_game_answer_count(self, user_id: int) -> int:
        with self._connect() as connection:
            count = connection.execute(
                "SELECT COUNT(*) AS count FROM player_answers WHERE user_id = ?",
                (user_id,),
            ).fetchone()["count"]
        return int(count)

    def team_score(self, team_id: int) -> int:
        with self._connect() as connection:
            count = connection.execute(
                "SELECT COUNT(*) AS count FROM game_answers WHERE team_id = ?",
                (team_id,),
            ).fetchone()["count"]
        return int(count)

    def record_leaderboard_pin(
        self,
        *,
        chat_id: int | str,
        message_id: int,
        now: datetime | None = None,
    ) -> None:
        current = now or utcnow()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO leaderboard_pin (id, chat_id, message_id, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    chat_id = excluded.chat_id,
                    message_id = excluded.message_id,
                    updated_at = excluded.updated_at
                """,
                (str(chat_id), message_id, _to_iso(current)),
            )

    def leaderboard_pin(self) -> LeaderboardPin | None:
        with self._connect() as connection:
            row = connection.execute("SELECT chat_id, message_id FROM leaderboard_pin WHERE id = 1").fetchone()
        if row is None:
            return None
        return LeaderboardPin(chat_id=str(row["chat_id"]), message_id=int(row["message_id"]))

    def record_game_answers_for_finalized_row(
        self,
        *,
        row_index: int,
        final_label: str,
        finalized_by_user_id: int,
        combo_timeout: timedelta,
        now: datetime | None = None,
    ) -> GameUpdateResult:
        current = now or utcnow()
        with self._connect() as connection:
            before_scores = self._team_scores_by_id(connection)
            participants = connection.execute(
                """
                SELECT
                    a.user_id,
                    u.username,
                    u.first_name,
                    tm.team_id,
                    t.name AS team_name
                FROM assignments a
                LEFT JOIN users u ON u.user_id = a.user_id
                LEFT JOIN team_members tm ON tm.user_id = a.user_id
                LEFT JOIN teams t ON t.team_id = tm.team_id
                WHERE a.row_index = ? AND a.status = ? AND a.label = ?
                ORDER BY a.labeled_at, a.user_id
                """,
                (row_index, LABELED, final_label),
            ).fetchall()

            if not participants:
                participants = connection.execute(
                    """
                    SELECT
                        u.user_id,
                        u.username,
                        u.first_name,
                        tm.team_id,
                        t.name AS team_name
                    FROM users u
                    LEFT JOIN team_members tm ON tm.user_id = u.user_id
                    LEFT JOIN teams t ON t.team_id = tm.team_id
                    WHERE u.user_id = ?
                    """,
                    (finalized_by_user_id,),
                ).fetchall()

            inserted_by_team: Counter[int] = Counter()
            inserted_user_ids: set[int] = set()
            team_names: dict[int, str] = {}
            user_rows: dict[int, sqlite3.Row] = {}
            for participant in participants:
                user_id = int(participant["user_id"])
                user_rows[user_id] = participant
                player_cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO player_answers (row_index, user_id, answered_at)
                    VALUES (?, ?, ?)
                    """,
                    (row_index, user_id, _to_iso(current)),
                )
                if player_cursor.rowcount:
                    inserted_user_ids.add(user_id)

                if participant["team_id"] is None:
                    continue
                team_id = int(participant["team_id"])
                team_names[team_id] = str(participant["team_name"] or "")
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO game_answers (row_index, user_id, team_id, answered_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (row_index, user_id, team_id, _to_iso(current)),
                )
                if cursor.rowcount:
                    inserted_by_team[team_id] += 1

            if not inserted_user_ids and not inserted_by_team:
                return GameUpdateResult()

            after_scores = self._team_scores_by_id(connection)
            lead_changes: list[TeamLeadChange] = []
            combo_updates: list[TeamComboUpdate] = []
            combo_resets: list[TeamComboReset] = []
            user_score_changes: list[UserScoreChange] = []
            team_score_changes: list[TeamScoreChange] = []

            for user_id in sorted(inserted_user_ids):
                user_row = user_rows[user_id]
                user_score_changes.append(
                    UserScoreChange(
                        user_id=user_id,
                        username=str(user_row["username"] or ""),
                        first_name=str(user_row["first_name"] or ""),
                        team_id=int(user_row["team_id"]) if user_row["team_id"] is not None else None,
                        team_name=str(user_row["team_name"]) if user_row["team_name"] else None,
                        score=self._player_score(connection, user_id),
                    )
                )

            for team_id, increment in inserted_by_team.items():
                previous_score = before_scores.get(team_id, 0)
                previous_other_top = max(
                    (score for other_team_id, score in before_scores.items() if other_team_id != team_id),
                    default=0,
                )
                current_score = after_scores.get(team_id, 0)
                if previous_other_top > 0 and previous_score <= previous_other_top < current_score:
                    lead_changes.append(
                        TeamLeadChange(
                            team_id=team_id,
                            team_name=team_names[team_id],
                            score=current_score,
                            previous_leader_score=previous_other_top,
                        )
                    )
                team_score_changes.append(
                    TeamScoreChange(
                        team_id=team_id,
                        team_name=team_names[team_id],
                        score=current_score,
                    )
                )

                combo_update, combo_reset = self._record_team_combo_answer(
                    connection,
                    team_id=team_id,
                    team_name=team_names[team_id],
                    increment=increment,
                    combo_timeout=combo_timeout,
                    now=current,
                )
                combo_updates.append(combo_update)
                if combo_reset is not None:
                    combo_resets.append(combo_reset)

        return GameUpdateResult(
            lead_changes=tuple(lead_changes),
            combo_updates=tuple(combo_updates),
            combo_resets=tuple(combo_resets),
            user_score_changes=tuple(user_score_changes),
            team_score_changes=tuple(team_score_changes),
        )

    def expire_team_combo(
        self,
        *,
        team_id: int,
        expected_deadline_at: datetime,
        now: datetime | None = None,
    ) -> TeamComboReset | None:
        current = now or utcnow()
        expected_deadline = _to_iso(expected_deadline_at)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT t.name, tc.combo_count, tc.deadline_at
                FROM team_combos tc
                JOIN teams t ON t.team_id = tc.team_id
                WHERE tc.team_id = ?
                """,
                (team_id,),
            ).fetchone()
            if row is None or not row["deadline_at"]:
                return None
            deadline = _from_iso(str(row["deadline_at"]))
            combo_count = int(row["combo_count"])
            if combo_count <= 0 or deadline > current or _to_iso(deadline) != expected_deadline:
                return None
            connection.execute(
                """
                UPDATE team_combos
                SET combo_count = 0,
                    last_answer_at = NULL,
                    deadline_at = NULL,
                    updated_at = ?
                WHERE team_id = ?
                """,
                (_to_iso(current), team_id),
            )
            reset_event_id, revives_remaining = self._record_team_combo_reset_event(
                connection,
                team_id=team_id,
                combo_count=combo_count,
                now=current,
            )
        return TeamComboReset(
            team_id=team_id,
            team_name=str(row["name"]),
            combo_count=combo_count,
            reset_event_id=reset_event_id,
            revives_remaining=revives_remaining,
        )

    def revive_team_combo(
        self,
        *,
        reset_event_id: int,
        user_id: int,
        combo_timeout: timedelta,
        now: datetime | None = None,
    ) -> TeamComboRevive:
        current = now or utcnow()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT e.reset_event_id, e.team_id, e.combo_count, e.revived_at, t.name
                FROM team_combo_reset_events e
                JOIN teams t ON t.team_id = e.team_id
                WHERE e.reset_event_id = ?
                """,
                (reset_event_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Сброс серии уже не найден.")
            team_id = int(row["team_id"])
            membership = connection.execute(
                "SELECT 1 FROM team_members WHERE user_id = ? AND team_id = ?",
                (user_id, team_id),
            ).fetchone()
            if membership is None:
                raise ValueError("Восстановить серию может только участник этой команды.")
            if row["revived_at"]:
                raise ValueError("Эту серию уже восстановили.")

            combo = connection.execute(
                "SELECT combo_count, deadline_at FROM team_combos WHERE team_id = ?",
                (team_id,),
            ).fetchone()
            if combo is not None:
                deadline = _optional_from_iso(combo["deadline_at"])
                if int(combo["combo_count"]) > 0 and deadline is not None and deadline > current:
                    raise ValueError("У команды уже есть живая серия.")

            used_count = self._team_combo_revives_used(connection, team_id)
            if used_count >= COMBO_REVIVE_LIMIT:
                raise ValueError("Лимит возрождений команды уже потрачен.")

            combo_count = int(row["combo_count"])
            deadline_at = current + combo_timeout
            connection.execute(
                """
                UPDATE team_combo_reset_events
                SET revived_at = ?, revived_by_user_id = ?
                WHERE reset_event_id = ? AND revived_at IS NULL
                """,
                (_to_iso(current), user_id, reset_event_id),
            )
            connection.execute(
                """
                INSERT INTO team_combos (team_id, combo_count, last_answer_at, deadline_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(team_id) DO UPDATE SET
                    combo_count = excluded.combo_count,
                    last_answer_at = excluded.last_answer_at,
                    deadline_at = excluded.deadline_at,
                    updated_at = excluded.updated_at
                """,
                (team_id, combo_count, _to_iso(current), _to_iso(deadline_at), _to_iso(current)),
            )
            remaining = max(0, COMBO_REVIVE_LIMIT - used_count - 1)
        return TeamComboRevive(
            team_id=team_id,
            team_name=str(row["name"]),
            combo_count=combo_count,
            deadline_at=deadline_at,
            revives_remaining=remaining,
        )

    def claim_achievement(
        self,
        *,
        scope: str,
        owner_id: int,
        code: str,
        title: str,
        description: str,
        now: datetime | None = None,
    ) -> AchievementRecord | None:
        current = now or utcnow()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO achievements (
                    scope,
                    owner_id,
                    code,
                    title,
                    description,
                    unlocked_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (scope, owner_id, code, title, description, _to_iso(current)),
            )
        if not cursor.rowcount:
            return None
        return AchievementRecord(
            scope=scope,
            owner_id=owner_id,
            code=code,
            title=title,
            description=description,
            unlocked_at=current,
        )

    def user_achievements(self, user_id: int) -> list[AchievementRecord]:
        return self._achievements(scope="user", owner_id=user_id)

    def team_achievements(self, team_id: int) -> list[AchievementRecord]:
        return self._achievements(scope="team", owner_id=team_id)

    def user_stats(self, user_id: int, now: datetime | None = None) -> UserLabelStats | None:
        current = now or utcnow()
        with self._connect() as connection:
            user = connection.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if user is None:
                return None
            labeled = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM assignments
                WHERE user_id = ? AND status = ?
                """,
                (user_id, LABELED),
            ).fetchone()["count"]
            active = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM assignments
                WHERE user_id = ? AND status = ? AND expires_at > ?
                """,
                (user_id, ASSIGNED, _to_iso(current)),
            ).fetchone()["count"]
        return UserLabelStats(
            user_id=user_id,
            username=str(user["username"] or ""),
            first_name=str(user["first_name"] or ""),
            labeled_count=int(labeled),
            active_count=int(active),
        )

    def all_user_stats(self, now: datetime | None = None) -> list[UserLabelStats]:
        current = now or utcnow()
        with self._connect() as connection:
            users = connection.execute("SELECT * FROM users ORDER BY user_id").fetchall()
            result: list[UserLabelStats] = []
            for user in users:
                user_id = int(user["user_id"])
                labeled = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM assignments
                    WHERE user_id = ? AND status = ?
                    """,
                    (user_id, LABELED),
                ).fetchone()["count"]
                active = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM assignments
                    WHERE user_id = ? AND status = ? AND expires_at > ?
                    """,
                    (user_id, ASSIGNED, _to_iso(current)),
                ).fetchone()["count"]
                result.append(
                    UserLabelStats(
                        user_id=user_id,
                        username=str(user["username"] or ""),
                        first_name=str(user["first_name"] or ""),
                        labeled_count=int(labeled),
                        active_count=int(active),
                    )
                )
        return result

    def conflict_count(self) -> int:
        with self._connect() as connection:
            count = connection.execute(
                "SELECT COUNT(*) AS count FROM row_states WHERE status = ?",
                (ROW_CONFLICT,),
            ).fetchone()["count"]
        return int(count)

    def vote_count(self) -> int:
        with self._connect() as connection:
            count = connection.execute(
                "SELECT COUNT(*) AS count FROM assignments WHERE status = ?",
                (LABELED,),
            ).fetchone()["count"]
        return int(count)

    def _team_scores_by_id(self, connection: sqlite3.Connection) -> dict[int, int]:
        rows = connection.execute(
            """
            SELECT team_id, COUNT(*) AS score
            FROM game_answers
            GROUP BY team_id
            """
        ).fetchall()
        return {int(row["team_id"]): int(row["score"]) for row in rows}

    def _player_score(self, connection: sqlite3.Connection, user_id: int) -> int:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM player_answers WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return int(row["count"])

    def _achievements(self, *, scope: str, owner_id: int) -> list[AchievementRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT scope, owner_id, code, title, description, unlocked_at
                FROM achievements
                WHERE scope = ? AND owner_id = ?
                ORDER BY unlocked_at, code
                """,
                (scope, owner_id),
            ).fetchall()
        return [
            AchievementRecord(
                scope=str(row["scope"]),
                owner_id=int(row["owner_id"]),
                code=str(row["code"]),
                title=str(row["title"]),
                description=str(row["description"]),
                unlocked_at=_from_iso(str(row["unlocked_at"])),
            )
            for row in rows
        ]

    def _team_member_stats(
        self,
        connection: sqlite3.Connection,
        team_id: int,
    ) -> tuple[TeamMemberStats, ...]:
        rows = connection.execute(
            """
            SELECT
                u.user_id,
                u.username,
                u.first_name,
                COUNT(ga.row_index) AS score
            FROM team_members tm
            JOIN users u ON u.user_id = tm.user_id
            LEFT JOIN game_answers ga
                ON ga.user_id = tm.user_id AND ga.team_id = tm.team_id
            WHERE tm.team_id = ?
            GROUP BY u.user_id, u.username, u.first_name
            ORDER BY score DESC, u.user_id
            """,
            (team_id,),
        ).fetchall()
        return tuple(
            TeamMemberStats(
                user_id=int(row["user_id"]),
                username=str(row["username"] or ""),
                first_name=str(row["first_name"] or ""),
                score=int(row["score"]),
            )
            for row in rows
        )

    def _record_team_combo_answer(
        self,
        connection: sqlite3.Connection,
        *,
        team_id: int,
        team_name: str,
        increment: int,
        combo_timeout: timedelta,
        now: datetime,
    ) -> tuple[TeamComboUpdate, TeamComboReset | None]:
        row = connection.execute(
            """
            SELECT combo_count, deadline_at
            FROM team_combos
            WHERE team_id = ?
            """,
            (team_id,),
        ).fetchone()

        previous_count = 0
        reset: TeamComboReset | None = None
        if row is not None:
            stored_count = int(row["combo_count"])
            deadline = _optional_from_iso(row["deadline_at"])
            if deadline is not None and deadline <= now and stored_count > 0:
                reset = TeamComboReset(team_id=team_id, team_name=team_name, combo_count=stored_count)
            else:
                previous_count = stored_count

        combo_count = previous_count + increment
        deadline_at = now + combo_timeout
        connection.execute(
            """
            INSERT INTO team_combos (team_id, combo_count, last_answer_at, deadline_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(team_id) DO UPDATE SET
                combo_count = excluded.combo_count,
                last_answer_at = excluded.last_answer_at,
                deadline_at = excluded.deadline_at,
                updated_at = excluded.updated_at
            """,
            (team_id, combo_count, _to_iso(now), _to_iso(deadline_at), _to_iso(now)),
        )
        return (
            TeamComboUpdate(
                team_id=team_id,
                team_name=team_name,
                previous_count=previous_count,
                combo_count=combo_count,
                deadline_at=deadline_at,
            ),
            reset,
        )

    def _record_team_combo_reset_event(
        self,
        connection: sqlite3.Connection,
        *,
        team_id: int,
        combo_count: int,
        now: datetime,
    ) -> tuple[int, int]:
        cursor = connection.execute(
            """
            INSERT INTO team_combo_reset_events (team_id, combo_count, reset_at)
            VALUES (?, ?, ?)
            """,
            (team_id, combo_count, _to_iso(now)),
        )
        used_count = self._team_combo_revives_used(connection, team_id)
        return int(cursor.lastrowid), max(0, COMBO_REVIVE_LIMIT - used_count)

    def _team_combo_revives_used(self, connection: sqlite3.Connection, team_id: int) -> int:
        row = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM team_combo_reset_events
            WHERE team_id = ? AND revived_at IS NOT NULL
            """,
            (team_id,),
        ).fetchone()
        return int(row["count"])
