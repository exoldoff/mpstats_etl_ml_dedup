from __future__ import annotations

from datetime import timedelta
import random

import pandas as pd
import pytest

from tools.dedup_labeling_bot.config import LabelingBotConfig
from tools.dedup_labeling_bot.csv_repository import LabelingCsvRepository
from tools.dedup_labeling_bot.service import LabelingBotService
from tools.dedup_labeling_bot.state_store import LabelingStateStore


def make_csv(tmp_path, rows: int = 4, labeled_count: int = 0):
    csv_path = tmp_path / "labeling.csv"
    pd.DataFrame(
        {
            "label": ["exact_duplicate" if idx < labeled_count else "" for idx in range(rows)],
            "notes": ["" for _ in range(rows)],
            "title_a": [f"a{idx}" for idx in range(rows)],
            "title_b": [f"b{idx}" for idx in range(rows)],
            "marketplace_a": ["Ozon" for _ in range(rows)],
            "marketplace_b": ["WB" for _ in range(rows)],
            "sku_a": [f"sku-a-{idx}" for idx in range(rows)],
            "sku_b": [f"sku-b-{idx}" for idx in range(rows)],
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def make_service(
    tmp_path,
    *,
    mode: str = "unique",
    overlap_votes: int = 2,
    batch_size: int = 2,
    discussion_chat_id=None,
    leaderboard_chat_id=None,
    rows: int = 4,
    labeled_count: int = 0,
    combo_timeout: timedelta = timedelta(minutes=2),
    assignment_rng: random.Random | None = None,
):
    csv_path = make_csv(tmp_path, rows=rows, labeled_count=labeled_count)
    config = LabelingBotConfig(
        token="token",
        access_password="secret",
        csv_path=csv_path,
        state_path=tmp_path / "state.sqlite",
        admin_user_ids=frozenset(),
        discussion_chat_id=discussion_chat_id,
        leaderboard_chat_id=leaderboard_chat_id,
        batch_size=batch_size,
        assignment_mode=mode,
        overlap_votes=overlap_votes,
        assignment_ttl=timedelta(hours=24),
        combo_timeout=combo_timeout,
    )
    service = LabelingBotService(
        config,
        repository=LabelingCsvRepository(csv_path),
        store=LabelingStateStore(config.state_path),
        assignment_rng=assignment_rng or random.Random(12345),
    )
    return service, csv_path


def test_authorize_checks_password(tmp_path) -> None:
    service, _ = make_service(tmp_path)

    assert not service.authorize(1, "bad")
    assert service.authorize(1, "secret", username="user")
    assert service.is_authorized(1)


def test_team_scores_sum_final_answers_and_show_members(tmp_path) -> None:
    service, _ = make_service(tmp_path, mode="unique", batch_size=1, rows=3)
    service.authorize(1, "secret", username="alice")
    service.authorize(2, "secret", username="bob")

    service.join_team(1, "Шустрые")
    service.join_team(2, "Шустрые")
    first = service.next_pair(1)
    assert first is not None
    service.label_row(1, first.row_index, "exact_duplicate")
    second = service.next_pair(2)
    assert second is not None
    service.label_row(2, second.row_index, "different_product")

    leaderboard = service.team_leaderboard()
    user_stats = service.user_stats(1)

    assert "<b>Шустрые</b> — 2 ответа" in leaderboard
    assert "@alice: 1" in leaderboard
    assert "@bob: 1" in leaderboard
    assert "Команда: Шустрые, командный счёт: 2" in user_stats
    assert "Игровые ответы: 1" in user_stats


def test_solo_score_and_first_achievement_do_not_require_team(tmp_path) -> None:
    service, _ = make_service(tmp_path, mode="unique", batch_size=1, rows=2)
    service.authorize(1, "secret", username="solo")

    first = service.next_pair(1)
    assert first is not None
    outcome = service.label_row(1, first.row_index, "exact_duplicate")

    messages = "\n".join(announcement.message for announcement in outcome.game_announcements)
    assert "Первый удар" in messages
    assert "<b>@solo (1)</b>" in messages
    assert "<b>@solo</b> — 1 ответ" in service.player_leaderboard()
    assert "Первый удар" in service.user_achievements(1)
    assert service.store.user_game_answer_count(1) == 1


def test_team_overtake_creates_lead_announcement(tmp_path) -> None:
    service, _ = make_service(tmp_path, mode="unique", batch_size=1, rows=4)
    service.authorize(1, "secret")
    service.authorize(2, "secret")
    service.join_team(1, "Красные")
    service.join_team(2, "Синие")

    red = service.next_pair(1)
    assert red is not None
    service.label_row(1, red.row_index, "exact_duplicate")
    blue_first = service.next_pair(2)
    assert blue_first is not None
    service.label_row(2, blue_first.row_index, "different_product")
    blue_second = service.next_pair(2)
    assert blue_second is not None
    outcome = service.label_row(2, blue_second.row_index, "different_product")

    messages = "\n".join(announcement.message for announcement in outcome.game_announcements)
    assert "Смена лидера" in messages
    assert "Синие" in messages
    assert "<b>2</b> ответа" in messages


def test_combo_threshold_and_timer_are_returned(tmp_path) -> None:
    service, _ = make_service(tmp_path, mode="unique", batch_size=1, rows=4)
    service.authorize(1, "secret")
    service.join_team(1, "Молнии")

    outcome = None
    for _ in range(3):
        pair = service.next_pair(1)
        assert pair is not None
        outcome = service.label_row(1, pair.row_index, "exact_duplicate")

    assert outcome is not None
    messages = "\n".join(announcement.message for announcement in outcome.game_announcements)
    assert "Комбо x3" in messages
    assert "Искра серии" in messages
    assert len(outcome.combo_timers) == 1
    assert outcome.combo_timers[0].team_id == service.store.user_team(1).team_id


def test_combo_expiry_resets_team_combo(tmp_path) -> None:
    service, _ = make_service(
        tmp_path,
        mode="unique",
        batch_size=1,
        rows=2,
        combo_timeout=timedelta(seconds=1),
    )
    service.authorize(1, "secret")
    service.join_team(1, "Таймеры")

    pair = service.next_pair(1)
    assert pair is not None
    outcome = service.label_row(1, pair.row_index, "exact_duplicate")
    timer = outcome.combo_timers[0]

    reset = service.store.expire_team_combo(
        team_id=timer.team_id,
        expected_deadline_at=timer.deadline_at,
        now=timer.deadline_at + timedelta(seconds=1),
    )

    assert reset is not None
    assert reset.team_name == "Таймеры"
    assert reset.combo_count == 1
    assert "⚡ x" not in service.team_leaderboard()


def test_relabel_does_not_add_second_game_answer(tmp_path) -> None:
    service, _ = make_service(tmp_path, mode="unique", batch_size=1, rows=2)
    service.authorize(1, "secret")
    service.join_team(1, "Аккуратные")

    pair = service.next_pair(1)
    assert pair is not None
    first = service.label_row(1, pair.row_index, "exact_duplicate")
    second = service.label_row(1, pair.row_index, "different_product")

    assert len(first.combo_timers) == 1
    assert second.combo_timers == ()
    assert service.store.user_game_answer_count(1) == 1


def test_forward_navigation_skips_already_labeled_history_rows(tmp_path) -> None:
    service, _ = make_service(tmp_path, mode="unique", batch_size=3, rows=4)
    service.authorize(1, "secret")

    first = service.next_pair(1)
    assert first is not None
    assigned_rows = service.store.user_assigned_rows(1, available_rows={0, 1, 2, 3})
    assert len(assigned_rows) == 3
    service.label_row(1, assigned_rows[0], "exact_duplicate")
    service.label_row(1, assigned_rows[1], "different_product")

    next_pair = service.navigate_pair(1, assigned_rows[0], "next")

    assert next_pair is not None
    assert next_pair.row_index == assigned_rows[2]
    assert next_pair.selected_label is None


def test_stale_assigned_row_already_labeled_in_csv_is_not_overwritten(tmp_path) -> None:
    service, csv_path = make_service(tmp_path, mode="unique", batch_size=1, rows=2)
    service.authorize(1, "secret")

    first = service.next_pair(1)
    assert first is not None
    service.repository.set_label(first.row_index, "different_product")

    with pytest.raises(ValueError, match="уже решена"):
        service.label_row(1, first.row_index, "exact_duplicate")

    assert pd.read_csv(csv_path).at[first.row_index, "label"] == "different_product"


def test_unique_batches_do_not_overlap_and_write_csv(tmp_path) -> None:
    service, csv_path = make_service(tmp_path, mode="unique", batch_size=2)
    service.authorize(1, "secret")
    service.authorize(2, "secret")

    first = service.next_pair(1)
    second = service.next_pair(2)

    assert first is not None
    assert second is not None
    first_rows = set(service.store.user_assigned_rows(1, available_rows={0, 1, 2, 3}))
    second_rows = set(service.store.user_assigned_rows(2, available_rows={0, 1, 2, 3}))
    assert len(first_rows) == 2
    assert len(second_rows) == 2
    assert first_rows.isdisjoint(second_rows)

    outcome = service.label_row(1, first.row_index, "exact_duplicate")

    assert outcome.final_label == "exact_duplicate"
    assert pd.read_csv(csv_path).at[first.row_index, "label"] == "exact_duplicate"


def test_unique_batch_samples_from_different_csv_segments(tmp_path) -> None:
    service, _ = make_service(
        tmp_path,
        mode="unique",
        batch_size=10,
        rows=100,
        assignment_rng=random.Random(7),
    )
    service.authorize(1, "secret")

    first = service.next_pair(1)

    assert first is not None
    assigned_rows = service.store.user_assigned_rows(1, available_rows=set(range(100)))
    assert len(assigned_rows) == 10
    assert assigned_rows != list(range(10))
    assert {row_index // 10 for row_index in assigned_rows} == set(range(10))


def test_milestone_announcements_are_claimed_once(tmp_path) -> None:
    service, _ = make_service(tmp_path, mode="unique", batch_size=1, rows=101, labeled_count=99)
    service.authorize(1, "secret")
    service.authorize(2, "secret")

    first = service.next_pair(1)
    assert first is not None
    outcome = service.label_row(1, first.row_index, "exact_duplicate")

    assert len(outcome.milestones) == 1
    milestone = outcome.milestones[0]
    assert milestone.threshold == 100
    assert milestone.labeled_rows == 100
    assert milestone.total_rows == 101
    assert milestone.remaining_rows == 1
    assert "100 заполнено" in milestone.message
    assert service.milestone_recipients() == [1, 2]

    second = service.next_pair(2)
    assert second is not None
    second_outcome = service.label_row(2, second.row_index, "different_product")

    assert second_outcome.milestones == ()


def test_existing_progress_does_not_backfill_old_milestones(tmp_path) -> None:
    service, _ = make_service(tmp_path, mode="unique", batch_size=1, rows=102, labeled_count=100)
    service.authorize(1, "secret")

    first = service.next_pair(1)
    assert first is not None
    outcome = service.label_row(1, first.row_index, "exact_duplicate")

    assert outcome.milestones == ()


def test_navigation_moves_within_batch_and_back_to_labeled_pair(tmp_path) -> None:
    service, _ = make_service(tmp_path, mode="unique", batch_size=2)
    service.authorize(1, "secret")

    first = service.next_pair(1)
    assert first is not None
    assigned_rows = service.store.user_assigned_rows(1, available_rows={0, 1, 2, 3})
    assert len(assigned_rows) == 2

    second = service.navigate_pair(1, first.row_index, "next")
    assert second is not None
    assert second.row_index == assigned_rows[1]

    service.label_row(1, first.row_index, "exact_duplicate")
    back = service.navigate_pair(1, second.row_index, "prev")

    assert back is not None
    assert back.row_index == first.row_index
    assert back.selected_label == "exact_duplicate"
    assert "<b>Решение:</b> Дубль" in back.message


def test_unique_labeled_pair_can_be_changed_after_going_back(tmp_path) -> None:
    service, csv_path = make_service(tmp_path, mode="unique", batch_size=2)
    service.authorize(1, "secret")

    first = service.next_pair(1)
    assert first is not None
    second = service.navigate_pair(1, first.row_index, "next")
    assert second is not None
    service.label_row(1, first.row_index, "exact_duplicate")

    back = service.navigate_pair(1, second.row_index, "prev")
    assert back is not None
    assert back.row_index == first.row_index
    changed = service.label_row(1, back.row_index, "different_product")
    refreshed = service.pair_for_row(1, back.row_index)

    assert changed.final_label == "different_product"
    assert pd.read_csv(csv_path).at[back.row_index, "label"] == "different_product"
    assert refreshed.selected_label == "different_product"
    assert "<b>Решение:</b> Разные" in refreshed.message


def test_navigation_assigns_next_batch_after_label(tmp_path) -> None:
    service, _ = make_service(tmp_path, mode="unique", batch_size=1)
    service.authorize(1, "secret")

    first = service.next_pair(1)
    assert first is not None
    service.label_row(1, first.row_index, "different_product")

    next_pair = service.navigate_pair(1, first.row_index, "next")

    assert next_pair is not None
    assert next_pair.row_index != first.row_index


def test_release_frees_unlabeled_unique_assignments(tmp_path) -> None:
    service, _ = make_service(tmp_path, mode="unique", batch_size=2, rows=2)
    service.authorize(1, "secret")
    service.authorize(2, "secret")

    assert service.next_pair(1) is not None
    first_rows = set(service.store.user_assigned_rows(1, available_rows={0, 1}))
    assert service.release_user_assignments(1) == 2

    second = service.next_pair(2)

    assert second is not None
    assert second.row_index in first_rows


def test_overlap_writes_csv_only_after_consensus(tmp_path) -> None:
    service, csv_path = make_service(tmp_path, mode="overlap", batch_size=1, overlap_votes=2)
    service.authorize(1, "secret")
    service.authorize(2, "secret")

    first = service.next_pair(1)
    assert first is not None
    pending = service.label_row(1, first.row_index, "exact_duplicate")

    assert pending.status == "pending"
    assert pd.isna(pd.read_csv(csv_path).at[first.row_index, "label"]) or pd.read_csv(csv_path).at[first.row_index, "label"] == ""

    second = service.next_pair(2)
    assert second is not None
    assert second.row_index == first.row_index
    finalized = service.label_row(2, second.row_index, "exact_duplicate")

    assert finalized.final_label == "exact_duplicate"
    assert pd.read_csv(csv_path).at[first.row_index, "label"] == "exact_duplicate"


def test_overlap_conflict_does_not_write_csv(tmp_path) -> None:
    service, csv_path = make_service(tmp_path, mode="overlap", batch_size=1, overlap_votes=2)
    service.authorize(1, "secret")
    service.authorize(2, "secret")

    first = service.next_pair(1)
    assert first is not None
    service.label_row(1, first.row_index, "exact_duplicate")
    second = service.next_pair(2)
    assert second is not None

    conflict = service.label_row(2, second.row_index, "different_product")

    assert conflict.status == "conflict"
    value = pd.read_csv(csv_path).at[first.row_index, "label"]
    assert pd.isna(value) or value == ""
    assert "Конфликтов: 1" in service.stats()


def test_logout_releases_assignments_and_removes_authorization(tmp_path) -> None:
    service, _ = make_service(tmp_path, mode="unique", batch_size=1)
    service.authorize(1, "secret")
    assert service.next_pair(1) is not None

    released = service.logout(1)

    assert released == 1
    assert not service.is_authorized(1)


def test_uncertain_with_discussion_chat_does_not_write_csv_until_consensus(tmp_path) -> None:
    service, csv_path = make_service(tmp_path, discussion_chat_id=-100123, overlap_votes=2)
    service.authorize(1, "secret")
    service.authorize(2, "secret")

    first = service.next_pair(1)
    assert first is not None
    outcome = service.move_row_to_discussion(1, first.row_index, "uncertain")

    assert outcome.status == "discussion"
    value = pd.read_csv(csv_path).at[first.row_index, "label"]
    assert pd.isna(value) or value == ""
    assert service.should_send_discussion(first.row_index)

    vote = service.vote_discussion_row(2, first.row_index, "uncertain")

    assert vote.final_label == "uncertain"
    assert pd.read_csv(csv_path).at[first.row_index, "label"] == "uncertain"


def test_discussion_vote_waits_for_consensus_after_disagreement(tmp_path) -> None:
    service, csv_path = make_service(tmp_path, discussion_chat_id=-100123, overlap_votes=2)
    service.authorize(1, "secret")
    service.authorize(2, "secret")
    service.authorize(3, "secret")

    first = service.next_pair(1)
    assert first is not None
    service.move_row_to_discussion(1, first.row_index, "uncertain")
    pending = service.vote_discussion_row(2, first.row_index, "exact_duplicate")

    assert pending.status == "discussion"
    value = pd.read_csv(csv_path).at[first.row_index, "label"]
    assert pd.isna(value) or value == ""

    finalized = service.vote_discussion_row(3, first.row_index, "exact_duplicate")

    assert finalized.final_label == "exact_duplicate"
    assert pd.read_csv(csv_path).at[first.row_index, "label"] == "exact_duplicate"
