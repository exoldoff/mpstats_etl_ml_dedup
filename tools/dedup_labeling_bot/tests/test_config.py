from __future__ import annotations

from pathlib import Path
from datetime import timedelta

import pytest

from tools.dedup_labeling_bot.config import ConfigError, load_config


def test_load_config_reads_password_mode_and_state_path(tmp_path) -> None:
    csv_path = tmp_path / "labeling.csv"
    env = {
        "DEDUP_TELEGRAM_BOT_TOKEN": "token",
        "DEDUP_TELEGRAM_ACCESS_PASSWORD": "secret",
        "DEDUP_TELEGRAM_ADMIN_USER_IDS": "1, 2",
        "DEDUP_TELEGRAM_DISCUSSION_CHAT_ID": "-100123",
        "DEDUP_TELEGRAM_LEADERBOARD_CHAT_ID": "-100456",
        "DEDUP_TELEGRAM_BATCH_SIZE": "5",
        "DEDUP_TELEGRAM_ASSIGNMENT_MODE": "overlap",
        "DEDUP_TELEGRAM_OVERLAP_VOTES": "3",
        "DEDUP_TELEGRAM_ASSIGNMENT_TTL_HOURS": "2",
        "DEDUP_TELEGRAM_COMBO_TIMEOUT_SECONDS": "90",
    }

    config = load_config(csv_path, env)

    assert config.token == "token"
    assert config.access_password == "secret"
    assert config.admin_user_ids == frozenset({1, 2})
    assert config.discussion_chat_id == -100123
    assert config.leaderboard_chat_id == -100456
    assert config.batch_size == 5
    assert config.assignment_mode == "overlap"
    assert config.overlap_votes == 3
    assert config.assignment_ttl == timedelta(hours=2)
    assert config.combo_timeout == timedelta(seconds=90)
    assert config.state_path == Path(str(csv_path.resolve()) + ".telegram_state.sqlite")


def test_load_config_requires_password(tmp_path) -> None:
    with pytest.raises(ConfigError, match="DEDUP_TELEGRAM_ACCESS_PASSWORD"):
        load_config(
            tmp_path / "labeling.csv",
            {"DEDUP_TELEGRAM_BOT_TOKEN": "token"},
        )


def test_load_config_defaults_combo_timeout_to_five_minutes(tmp_path) -> None:
    config = load_config(
        tmp_path / "labeling.csv",
        {
            "DEDUP_TELEGRAM_BOT_TOKEN": "token",
            "DEDUP_TELEGRAM_ACCESS_PASSWORD": "secret",
        },
    )

    assert config.combo_timeout == timedelta(minutes=5)
