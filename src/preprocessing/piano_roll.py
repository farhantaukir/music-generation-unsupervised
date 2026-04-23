"""Convert MIDI note events into binary piano-roll windows for model training."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
from tqdm import tqdm

from src.config import (
    MIDI_MAX_PITCH,
    MIDI_MIN_PITCH,
    MIN_NOTES_PER_WINDOW,
    SEQUENCE_LENGTH,
    STEPS_PER_BEAT,
    WINDOW_STEP,
)
from src.preprocessing.midi_parser import (
    NoteEvent,
    discover_midi_files,
    estimate_tempo_bpm,
    extract_note_events,
    load_pretty_midi,
)


def get_seconds_per_step(tempo_bpm: float, steps_per_beat: int = STEPS_PER_BEAT) -> float:
    """Convert BPM into seconds per quantized piano-roll step."""
    beats_per_second = tempo_bpm / 60.0
    return 1.0 / (beats_per_second * steps_per_beat)


def quantize_time_to_step(time_seconds: float, seconds_per_step: float) -> int:
    """Map a continuous timestamp in seconds to its nearest quantized step index."""
    if seconds_per_step <= 0:
        raise ValueError("seconds_per_step must be positive.")
    return max(0, int(round(time_seconds / seconds_per_step)))


def note_events_to_binary_roll(
    note_events: Sequence[NoteEvent],
    tempo_bpm: float,
    min_pitch: int = MIDI_MIN_PITCH,
    max_pitch: int = MIDI_MAX_PITCH,
) -> np.ndarray:
    """Convert a list of note events into a binary piano roll matrix."""
    if not note_events:
        return np.zeros((0, max_pitch - min_pitch + 1), dtype=np.float32)

    seconds_per_step = get_seconds_per_step(tempo_bpm)
    pitch_dim = max_pitch - min_pitch + 1

    last_end = max(event.end for event in note_events)
    total_steps = quantize_time_to_step(last_end, seconds_per_step) + 1
    piano_roll = np.zeros((total_steps, pitch_dim), dtype=np.float32)

    for event in note_events:
        start_idx = quantize_time_to_step(event.start, seconds_per_step)
        end_idx = max(start_idx + 1, quantize_time_to_step(event.end, seconds_per_step))
        pitch_idx = event.pitch - min_pitch
        if 0 <= pitch_idx < pitch_dim:
            piano_roll[start_idx:end_idx, pitch_idx] = 1.0

    return piano_roll


def window_piano_roll(
    piano_roll: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
    step_size: int = WINDOW_STEP,
    min_notes_per_window: int = MIN_NOTES_PER_WINDOW,
) -> List[np.ndarray]:
    """Slice a piano roll into fixed-length overlapping windows."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")

    total_steps = piano_roll.shape[0]
    windows: List[np.ndarray] = []

    if total_steps == 0:
        return windows

    if total_steps < sequence_length:
        padded = np.zeros((sequence_length, piano_roll.shape[1]), dtype=np.float32)
        padded[:total_steps] = piano_roll
        if float(padded.sum()) >= float(min_notes_per_window):
            windows.append(padded)
        return windows

    starts = list(range(0, total_steps - sequence_length + 1, step_size))
    final_start = total_steps - sequence_length
    if starts[-1] != final_start:
        starts.append(final_start)

    for start_idx in starts:
        end_idx = start_idx + sequence_length
        window = piano_roll[start_idx:end_idx]
        if float(window.sum()) >= float(min_notes_per_window):
            windows.append(window.astype(np.float32, copy=False))

    return windows


def load_windowed_piano_rolls(
    midi_files: Sequence[Path],
    sequence_length: int = SEQUENCE_LENGTH,
    step_size: int = WINDOW_STEP,
) -> np.ndarray:
    """Load MIDI files and return stacked piano-roll windows for training."""
    all_windows: List[np.ndarray] = []

    for midi_path in tqdm(midi_files, desc="Converting MIDI to windows"):
        try:
            midi = load_pretty_midi(midi_path)
            tempo_bpm = estimate_tempo_bpm(midi)
            note_events = extract_note_events(midi)
            piano_roll = note_events_to_binary_roll(note_events, tempo_bpm=tempo_bpm)
            windows = window_piano_roll(
                piano_roll,
                sequence_length=sequence_length,
                step_size=step_size,
            )
            all_windows.extend(windows)
        except Exception:
            continue

    if not all_windows:
        raise ValueError("No valid piano-roll windows were created from the provided MIDI files.")

    return np.stack(all_windows, axis=0).astype(np.float32, copy=False)


def load_windowed_piano_rolls_from_root(
    midi_root: Path | str,
    sequence_length: int = SEQUENCE_LENGTH,
    step_size: int = WINDOW_STEP,
    limit_files: int | None = None,
) -> np.ndarray:
    """Discover MIDI files under a root and return stacked piano-roll windows."""
    midi_files = discover_midi_files(midi_root)
    if limit_files is not None:
        midi_files = midi_files[:limit_files]
    return load_windowed_piano_rolls(
        midi_files,
        sequence_length=sequence_length,
        step_size=step_size,
    )
