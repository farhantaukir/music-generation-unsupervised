"""Standalone evaluation metrics for generated symbolic music sequences."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence, Tuple

import numpy as np


def pitch_histogram_similarity(reference_pitches: Sequence[int], generated_pitches: Sequence[int]) -> float:
    """Compute L1 distance between reference and generated pitch-class histograms."""
    if not reference_pitches or not generated_pitches:
        return float("inf")

    reference_hist = np.zeros(12, dtype=np.float64)
    generated_hist = np.zeros(12, dtype=np.float64)

    for pitch in reference_pitches:
        reference_hist[pitch % 12] += 1.0
    for pitch in generated_pitches:
        generated_hist[pitch % 12] += 1.0

    reference_hist /= reference_hist.sum()
    generated_hist /= generated_hist.sum()
    return float(np.abs(reference_hist - generated_hist).sum())


def rhythm_diversity(durations: Sequence[float]) -> float:
    """Compute unique-duration ratio over all note durations."""
    if not durations:
        return 0.0

    rounded = [round(float(duration), 4) for duration in durations if duration > 0]
    if not rounded:
        return 0.0

    return float(len(set(rounded)) / len(rounded))


def repetition_ratio(tokens: Sequence[int], pattern_length: int = 4) -> float:
    """Compute repeated-pattern ratio in a tokenized musical sequence."""
    if pattern_length <= 0:
        raise ValueError("pattern_length must be positive.")
    if len(tokens) < pattern_length:
        return 0.0

    patterns = [tuple(tokens[index : index + pattern_length]) for index in range(len(tokens) - pattern_length + 1)]
    if not patterns:
        return 0.0

    counts = Counter(patterns)
    repeated_patterns = sum(count - 1 for count in counts.values() if count > 1)
    return float(repeated_patterns / len(patterns))


def extract_pitches_and_durations(note_events: Iterable[Tuple[int, float, float]]) -> Tuple[list[int], list[float]]:
    """Extract pitch and duration lists from iterable note tuples."""
    pitches: list[int] = []
    durations: list[float] = []

    for pitch, start, end in note_events:
        if end <= start:
            continue
        pitches.append(int(pitch))
        durations.append(float(end - start))

    return pitches, durations
