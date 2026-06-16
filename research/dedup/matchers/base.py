from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


PAIR_LABELS = ("exact_duplicate", "different_product")


@dataclass(frozen=True)
class MatcherStatus:
    available: bool
    message: str = "ready"


class PairMatcher(ABC):
    """Common interface for research-only pair matching baselines."""

    name = "base"

    def status(self) -> MatcherStatus:
        return MatcherStatus(available=True)

    @abstractmethod
    def score(self, pair: Any) -> float:
        """Return a 0..1 similarity score for a candidate pair."""

    @abstractmethod
    def predict_label(self, pair: Any) -> str:
        """Return one of the binary pairwise labels used by matching baselines."""
