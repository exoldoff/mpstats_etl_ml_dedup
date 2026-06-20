from __future__ import annotations

from dataclasses import dataclass


NEXT_CALLBACK = "next"
LABEL_CALLBACK_PREFIX = "label"
DISCUSSION_LABEL_CALLBACK_PREFIX = "discussion_label"
NAV_CALLBACK_PREFIX = "nav"
NOOP_CALLBACK_PREFIX = "noop"


@dataclass(frozen=True)
class LabelCallback:
    row_index: int
    label: str


@dataclass(frozen=True)
class NavCallback:
    row_index: int
    direction: str


def make_label_callback(row_index: int, label: str) -> str:
    return f"{LABEL_CALLBACK_PREFIX}:{row_index}:{label}"


def make_discussion_label_callback(row_index: int, label: str) -> str:
    return f"{DISCUSSION_LABEL_CALLBACK_PREFIX}:{row_index}:{label}"


def make_nav_callback(row_index: int, direction: str) -> str:
    if direction not in {"prev", "next"}:
        raise ValueError(f"Unsupported navigation direction: {direction!r}")
    return f"{NAV_CALLBACK_PREFIX}:{row_index}:{direction}"


def make_noop_callback(row_index: int) -> str:
    return f"{NOOP_CALLBACK_PREFIX}:{row_index}"


def parse_label_callback(value: str) -> LabelCallback:
    prefix, raw_row_index, label = value.split(":", 2)
    if prefix not in {LABEL_CALLBACK_PREFIX, DISCUSSION_LABEL_CALLBACK_PREFIX}:
        raise ValueError(f"Unsupported callback: {value!r}")
    return LabelCallback(row_index=int(raw_row_index), label=label)


def parse_nav_callback(value: str) -> NavCallback:
    prefix, raw_row_index, direction = value.split(":", 2)
    if prefix != NAV_CALLBACK_PREFIX:
        raise ValueError(f"Unsupported navigation callback: {value!r}")
    if direction not in {"prev", "next"}:
        raise ValueError(f"Unsupported navigation direction: {direction!r}")
    return NavCallback(row_index=int(raw_row_index), direction=direction)
