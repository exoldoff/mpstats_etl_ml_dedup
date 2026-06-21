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


@dataclass(frozen=True)
class GameUpdateResult:
    lead_changes: tuple[TeamLeadChange, ...] = ()
    combo_updates: tuple[TeamComboUpdate, ...] = ()
    combo_resets: tuple[TeamComboReset, ...] = ()


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
                CREATE TABLE IF NOT EXISTS team_combos (
                    team_id INTEGER PRIMARY KEY,
                    combo_count INTEGER NOT NULL,
                    last_answer_at TEXT,
                    deadline_at TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (team_id) REFERENCES teams(team_id)
                );
                CREATE TABLE IF NOT EXISTS leaderboard_pin (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    chat_id TEXT NOT NULL,
                    message_id INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_assignments_status
                    ON assignments(status, expires_at);
                CREATE INDEX IF NOT EXISTS idx_assignments_user_status
                    ON assignments(user_id, status);
                CREATE INDEX IF NOT EXISTS idx_game_answers_team
                    ON game_answers(team_id);
                CREATE INDEX IF NOT EXISTS idx_game_answers_user
                    ON game_answers(user_id);
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

    def user_game_answer_count(self, user_id: int) -> int:
        with self._connect() as connection:
            count = connection.execute(
                "SELECT COUNT(*) AS count FROM game_answers WHERE user_id = ?",
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
                SELECT a.user_id, tm.team_id, t.name AS team_name
                FROM assignments a
                JOIN team_members tm ON tm.user_id = a.user_id
                JOIN teams t ON t.team_id = tm.team_id
                WHERE a.row_index = ? AND a.status = ? AND a.label = ?
                ORDER BY a.labeled_at, a.user_id
                """,
                (row_index, LABELED, final_label),
            ).fetchall()

            if not participants:
                participants = connection.execute(
                    """
                    SELECT tm.user_id, tm.team_id, t.name AS team_name
                    FROM team_members tm
                    JOIN teams t ON t.team_id = tm.team_id
                    WHERE tm.user_id = ?
                    """,
                    (finalized_by_user_id,),
                ).fetchall()

            inserted_by_team: Counter[int] = Counter()
            team_names: dict[int, str] = {}
            for participant in participants:
                team_id = int(participant["team_id"])
                team_names[team_id] = str(participant["team_name"])
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO game_answers (row_index, user_id, team_id, answered_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (row_index, int(participant["user_id"]), team_id, _to_iso(current)),
                )
                if cursor.rowcount:
                    inserted_by_team[team_id] += 1

            if not inserted_by_team:
                return GameUpdateResult()

            after_scores = self._team_scores_by_id(connection)
            lead_changes: list[TeamLeadChange] = []
            combo_updates: list[TeamComboUpdate] = []
            combo_resets: list[TeamComboReset] = []

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
        return TeamComboReset(team_id=team_id, team_name=str(row["name"]), combo_count=combo_count)

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
