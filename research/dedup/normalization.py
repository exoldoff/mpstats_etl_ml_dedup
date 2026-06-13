from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Iterable


TOKEN_RE = re.compile(r"[0-9a-zа-я]+")

DEFAULT_STOP_TOKENS = frozenset(
    {
        "соус",
        "соусы",
        "соуса",
        "для",
        "без",
        "на",
        "и",
        "в",
        "с",
        "из",
        "по",
        "от",
        "до",
        "под",
        "г",
        "гр",
        "кг",
        "мл",
        "л",
        "шт",
        "штук",
        "уп",
        "упак",
        "упаковка",
        "бутылка",
        "стекло",
        "пэт",
        "x",
        "х",
    }
)

DEFAULT_FLAVOR_TOKENS = frozenset(
    {
        "соевый",
        "соевого",
        "паста",
        "чили",
        "острый",
        "острая",
        "остро",
        "уксус",
        "уксусный",
        "томатный",
        "томатная",
        "натуральный",
        "натуральная",
        "сладкий",
        "сладкая",
        "бальзамический",
        "бальзамическая",
        "мяса",
    }
)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except TypeError:
        return True


def normalize_title(value: object) -> str:
    """Lowercase, remove punctuation noise, and collapse whitespace."""
    if _is_missing(value):
        return ""
    text = str(value).casefold().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_title(value: object) -> list[str]:
    """Return normalized title tokens without punctuation."""
    return TOKEN_RE.findall(normalize_title(value))


def meaningful_title_tokens(
    value: object,
    *,
    stop_tokens: Iterable[str] = DEFAULT_STOP_TOKENS,
    keep_numeric: bool = False,
) -> list[str]:
    """Tokens used for title blocking and scoring.

    Generic category words and unit words are removed so that a common token
    like ``соус`` does not create an all-pairs block.
    """
    stops = set(stop_tokens)
    tokens: list[str] = []
    for token in tokenize_title(value):
        if token in stops:
            continue
        if len(token) < 2:
            continue
        if not keep_numeric and token.isdigit():
            continue
        tokens.append(token)
    return tokens


def normalize_brand(value: object) -> str:
    """Cheap surface normalization for source brand values."""
    if _is_missing(value):
        return ""
    text = str(value).casefold().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def flavor_token_set(value: object, vocabulary: Iterable[str] = DEFAULT_FLAVOR_TOKENS) -> set[str]:
    """Extract the current heuristic flavor/type tokens from a title."""
    vocab = set(vocabulary)
    return set(meaningful_title_tokens(value)) & vocab


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 0.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def containment_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    denominator = min(len(left_set), len(right_set))
    if denominator == 0:
        return 0.0
    return len(left_set & right_set) / denominator


def title_similarity(
    title_a: object,
    title_b: object,
    *,
    tokens_a: Iterable[str] | None = None,
    tokens_b: Iterable[str] | None = None,
) -> float:
    """Baseline fuzzy-ish similarity in the 0..1 range.

    Uses token Jaccard/containment plus stdlib SequenceMatcher.  This keeps the
    research package dependency-free while still giving a useful baseline.
    """
    left_tokens = list(tokens_a) if tokens_a is not None else meaningful_title_tokens(title_a)
    right_tokens = list(tokens_b) if tokens_b is not None else meaningful_title_tokens(title_b)
    token_jaccard = jaccard_similarity(left_tokens, right_tokens)
    token_containment = containment_similarity(left_tokens, right_tokens)
    char_score = SequenceMatcher(None, normalize_title(title_a), normalize_title(title_b)).ratio()
    return round(max(token_jaccard, 0.65 * token_containment + 0.35 * char_score), 6)
