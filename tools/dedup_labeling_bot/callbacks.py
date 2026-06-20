from __future__ import annotations

from dataclasses import dataclass


NEXT_CALLBACK = "next"
LABEL_CALLBACK_PREFIX = "label"
DISCUSSION_LABEL_CALLBACK_PREFIX = "discussion_label"


@dataclass(frozen=True)
class LabelCallback:
    row_index: int
    label: str


def make_label_callback(row_index: int, label: str) -> str:
    return f"{LABEL_CALLBACK_PREFIX}:{row_index}:{label}"


def make_discussion_label_callback(row_index: int, label: str) -> str:
    return f"{DISCUSSION_LABEL_CALLBACK_PREFIX}:{row_index}:{label}"


def parse_label_callback(value: str) -> LabelCallback:
    prefix, raw_row_index, label = value.split(":", 2)
    if prefix not in {LABEL_CALLBACK_PREFIX, DISCUSSION_LABEL_CALLBACK_PREFIX}:
        raise ValueError(f"Unsupported callback: {value!r}")
    return LabelCallback(row_index=int(raw_row_index), label=label)
