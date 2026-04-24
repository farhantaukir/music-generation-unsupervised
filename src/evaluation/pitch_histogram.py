"""Pitch-class histogram metrics for symbolic music evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def pitch_class_histogram(pitches: Sequence[int]) -> np.ndarray:
    """Convert a pitch sequence into a normalized 12-bin pitch-class histogram."""
    histogram = np.zeros(12, dtype=np.float64)
    if not pitches:
        return histogram

    for pitch in pitches:
        histogram[int(pitch) % 12] += 1.0

    total = float(histogram.sum())
    if total <= 0.0:
        return histogram
    return histogram / total


def histogram_l1_distance(reference_histogram: np.ndarray, generated_histogram: np.ndarray) -> float:
    """Compute L1 distance between two pitch-class histograms."""
    reference = np.asarray(reference_histogram, dtype=np.float64)
    generated = np.asarray(generated_histogram, dtype=np.float64)

    if reference.shape != (12,) or generated.shape != (12,):
        raise ValueError("Both histograms must have shape (12,).")

    return float(np.abs(reference - generated).sum())


def pitch_histogram_similarity(reference_pitches: Sequence[int], generated_pitches: Sequence[int]) -> float:
    """Compute pitch-histogram similarity as L1 distance over pitch classes."""
    if not reference_pitches or not generated_pitches:
        return float("inf")

    reference_histogram = pitch_class_histogram(reference_pitches)
    generated_histogram = pitch_class_histogram(generated_pitches)
    return histogram_l1_distance(reference_histogram, generated_histogram)
