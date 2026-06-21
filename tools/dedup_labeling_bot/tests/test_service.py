from __future__ import annotations

from datetime import timedelta

import pandas as pd

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
    rows: int = 4,
    labeled_count: int = 0,
):
    csv_path = make_csv(tmp_path, rows=rows, labeled_count=labeled_count)
    config = LabelingBotConfig(
        token="token",
        access_password="secret",
        csv_path=csv_path,
        state_path=tmp_path / "state.sqlite",
        admin_user_ids=frozenset(),
        discussion_chat_id=discussion_chat_id,
        batch_size=batch_size,
        assignment_mode=mode,
        overlap_votes=overlap_votes,
        assignment_ttl=timedelta(hours=24),
    )
    service = LabelingBotService(
        config,
        repository=LabelingCsvRepository(csv_path),
        store=LabelingStateStore(config.state_path),
    )
    return service, csv_path


def test_authorize_checks_password(tmp_path) -> None:
    service, _ = make_service(tmp_path)

    assert not service.authorize(1, "bad")
    assert service.authorize(1, "secret", username="user")
    assert service.is_authorized(1)


def test_unique_batches_do_not_overlap_and_write_csv(tmp_path) -> None:
    service, csv_path = make_service(tmp_path, mode="unique", batch_size=2)
    service.authorize(1, "secret")
    service.authorize(2, "secret")

    first = service.next_pair(1)
    second = service.next_pair(2)

    assert first is not None
    assert second is not None
    assert first.row_index == 0
    assert second.row_index == 2

    outcome = service.label_row(1, first.row_index, "exact_duplicate")

    assert outcome.final_label == "exact_duplicate"
    assert pd.read_csv(csv_path).at[first.row_index, "label"] == "exact_duplicate"


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

    second = service.navigate_pair(1, first.row_index, "next")
    assert second is not None
    assert second.row_index == 1

    service.label_row(1, first.row_index, "exact_duplicate")
    back = service.navigate_pair(1, second.row_index, "prev")

    assert back is not None
    assert back.row_index == first.row_index
    assert back.selected_label == "exact_duplicate"
    assert "<b>Решение:</b> Дубль" in back.message


def test_navigation_assigns_next_batch_after_label(tmp_path) -> None:
    service, _ = make_service(tmp_path, mode="unique", batch_size=1)
    service.authorize(1, "secret")

    first = service.next_pair(1)
    assert first is not None
    service.label_row(1, first.row_index, "different_product")

    next_pair = service.navigate_pair(1, first.row_index, "next")

    assert next_pair is not None
    assert next_pair.row_index == 1


def test_release_frees_unlabeled_unique_assignments(tmp_path) -> None:
    service, _ = make_service(tmp_path, mode="unique", batch_size=2)
    service.authorize(1, "secret")
    service.authorize(2, "secret")

    assert service.next_pair(1) is not None
    assert service.release_user_assignments(1) == 2

    second = service.next_pair(2)

    assert second is not None
    assert second.row_index == 0


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
