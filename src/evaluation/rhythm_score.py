"""Rhythm-oriented evaluation scores for symbolic music sequences."""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def rhythm_diversity_score(durations: Sequence[float]) -> float:
    """Compute unique-duration ratio from a note-duration sequence."""
    if not durations:
        return 0.0

    valid_durations = [round(float(duration), 4) for duration in durations if float(duration) > 0.0]
    if not valid_durations:
        return 0.0

    return float(len(set(valid_durations)) / len(valid_durations))


def rhythm_pattern_repetition_ratio(durations: Sequence[float], pattern_length: int = 4) -> float:
    """Compute repeated-pattern ratio for rhythm duration n-grams."""
    if pattern_length <= 0:
        raise ValueError("pattern_length must be positive.")

    rounded_durations = [round(float(duration), 4) for duration in durations if float(duration) > 0.0]
    if len(rounded_durations) < pattern_length:
        return 0.0

    patterns = [
        tuple(rounded_durations[index : index + pattern_length])
        for index in range(len(rounded_durations) - pattern_length + 1)
    ]
    if not patterns:
        return 0.0

    counts = Counter(patterns)
    repeated_patterns = sum(count - 1 for count in counts.values() if count > 1)
    return float(repeated_patterns / len(patterns))
