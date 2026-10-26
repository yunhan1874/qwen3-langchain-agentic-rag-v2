from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence


_CJK = re.compile(r"[\u4e00-\u9fff]")
_ALNUM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


def stable_id(*parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese as unigrams+bigrams and preserve Latin medical terms."""
    lowered = text.lower()
    latin = _ALNUM.findall(lowered)
    chinese = _CJK.findall(lowered)
    bigrams = ["".join(chinese[index : index + 2]) for index in range(len(chinese) - 1)]
    return latin + chinese + bigrams


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def unique_in_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
