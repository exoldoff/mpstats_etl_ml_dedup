from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os
from pathlib import Path
from typing import Mapping


TOKEN_ENV = "DEDUP_TELEGRAM_BOT_TOKEN"
PASSWORD_ENV = "DEDUP_TELEGRAM_ACCESS_PASSWORD"
ADMIN_IDS_ENV = "DEDUP_TELEGRAM_ADMIN_USER_IDS"
DISCUSSION_CHAT_ID_ENV = "DEDUP_TELEGRAM_DISCUSSION_CHAT_ID"
LEADERBOARD_CHAT_ID_ENV = "DEDUP_TELEGRAM_LEADERBOARD_CHAT_ID"
BATCH_SIZE_ENV = "DEDUP_TELEGRAM_BATCH_SIZE"
ASSIGNMENT_MODE_ENV = "DEDUP_TELEGRAM_ASSIGNMENT_MODE"
OVERLAP_VOTES_ENV = "DEDUP_TELEGRAM_OVERLAP_VOTES"
ASSIGNMENT_TTL_HOURS_ENV = "DEDUP_TELEGRAM_ASSIGNMENT_TTL_HOURS"
COMBO_TIMEOUT_SECONDS_ENV = "DEDUP_TELEGRAM_COMBO_TIMEOUT_SECONDS"
STATE_PATH_ENV = "DEDUP_TELEGRAM_STATE_PATH"

UNIQUE_MODE = "unique"
OVERLAP_MODE = "overlap"
ASSIGNMENT_MODES = {UNIQUE_MODE, OVERLAP_MODE}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LabelingBotConfig:
    token: str
    access_password: str
    csv_path: Path
    state_path: Path
    admin_user_ids: frozenset[int]
    discussion_chat_id: int | str | None = None
    leaderboard_chat_id: int | str | None = None
    batch_size: int = 10
    assignment_mode: str = UNIQUE_MODE
    overlap_votes: int = 2
    assignment_ttl: timedelta = timedelta(hours=24)
    combo_timeout: timedelta = timedelta(minutes=2)


def parse_user_ids(value: str | None) -> frozenset[int]:
    if not value:
        return frozenset()
    user_ids: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            user_ids.add(int(part))
        except ValueError as exc:
            raise ConfigError(f"Invalid Telegram user id in {ADMIN_IDS_ENV}: {part!r}") from exc
    return frozenset(user_ids)


def parse_chat_id(value: str | None) -> int | str | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip()
    if normalized.lstrip("-").isdigit():
        return int(normalized)
    return normalized


def _int_env(env: Mapping[str, str], key: str, default: int, *, minimum: int) -> int:
    raw_value = env.get(key)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer") from exc
    if value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}")
    return value


def _float_env(env: Mapping[str, str], key: str, default: float, *, minimum: float) -> float:
    raw_value = env.get(key)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number") from exc
    if value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}")
    return value


def default_state_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(csv_path.suffix + ".telegram_state.sqlite")


def load_config(csv_path: Path, env: Mapping[str, str] | None = None) -> LabelingBotConfig:
    values = env or os.environ
    token = values.get(TOKEN_ENV, "").strip()
    if not token:
        raise ConfigError(f"{TOKEN_ENV} is required")
    password = values.get(PASSWORD_ENV, "")
    if not password:
        raise ConfigError(f"{PASSWORD_ENV} is required")

    resolved_csv_path = csv_path.expanduser().resolve()
    state_path_value = values.get(STATE_PATH_ENV, "").strip()
    state_path = Path(state_path_value).expanduser().resolve() if state_path_value else default_state_path(resolved_csv_path)

    batch_size = _int_env(values, BATCH_SIZE_ENV, 10, minimum=1)
    assignment_mode = values.get(ASSIGNMENT_MODE_ENV, UNIQUE_MODE).strip().casefold() or UNIQUE_MODE
    if assignment_mode not in ASSIGNMENT_MODES:
        raise ConfigError(f"{ASSIGNMENT_MODE_ENV} must be one of: {', '.join(sorted(ASSIGNMENT_MODES))}")
    overlap_votes = _int_env(values, OVERLAP_VOTES_ENV, 2, minimum=2)
    ttl_hours = _float_env(values, ASSIGNMENT_TTL_HOURS_ENV, 24.0, minimum=0.01)
    combo_timeout_seconds = _float_env(values, COMBO_TIMEOUT_SECONDS_ENV, 120.0, minimum=1.0)
    discussion_chat_id = parse_chat_id(values.get(DISCUSSION_CHAT_ID_ENV))
    leaderboard_chat_id = parse_chat_id(values.get(LEADERBOARD_CHAT_ID_ENV)) or discussion_chat_id

    return LabelingBotConfig(
        token=token,
        access_password=password,
        csv_path=resolved_csv_path,
        state_path=state_path,
        admin_user_ids=parse_user_ids(values.get(ADMIN_IDS_ENV)),
        discussion_chat_id=discussion_chat_id,
        leaderboard_chat_id=leaderboard_chat_id,
        batch_size=batch_size,
        assignment_mode=assignment_mode,
        overlap_votes=overlap_votes,
        assignment_ttl=timedelta(hours=ttl_hours),
        combo_timeout=timedelta(seconds=combo_timeout_seconds),
    )
