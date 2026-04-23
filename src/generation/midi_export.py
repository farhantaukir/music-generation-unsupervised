"""Convert binary piano-roll sequences into MIDI files for listening and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

from src.config import DEFAULT_TEMPO_BPM, MIDI_MIN_PITCH, STEPS_PER_BEAT


def step_to_seconds(step_index: int, tempo_bpm: float, steps_per_beat: int = STEPS_PER_BEAT) -> float:
    """Convert a quantized step index into absolute seconds."""
    seconds_per_beat = 60.0 / tempo_bpm
    seconds_per_step = seconds_per_beat / float(steps_per_beat)
    return float(step_index) * seconds_per_step


def binary_piano_roll_to_midi(
    piano_roll: np.ndarray,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
    min_pitch: int = MIDI_MIN_PITCH,
) -> pretty_midi.PrettyMIDI:
    """Transform a binary piano-roll matrix into a PrettyMIDI object."""
    if piano_roll.ndim != 2:
        raise ValueError("piano_roll must be a 2D array with shape (time, pitch).")

    binary_roll = (piano_roll > 0.5).astype(np.int32, copy=False)
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(tempo_bpm))
    instrument = pretty_midi.Instrument(program=0, is_drum=False, name="Task1 Piano")

    active_starts: dict[int, int] = {}
    padded_roll = np.vstack([binary_roll, np.zeros((1, binary_roll.shape[1]), dtype=np.int32)])

    for time_idx in range(padded_roll.shape[0]):
        frame = padded_roll[time_idx]
        for pitch_idx, value in enumerate(frame):
            is_active = pitch_idx in active_starts

            if value == 1 and not is_active:
                active_starts[pitch_idx] = time_idx
            elif value == 0 and is_active:
                start_step = active_starts.pop(pitch_idx)
                if time_idx <= start_step:
                    continue
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=min_pitch + pitch_idx,
                    start=step_to_seconds(start_step, tempo_bpm=tempo_bpm),
                    end=step_to_seconds(time_idx, tempo_bpm=tempo_bpm),
                )
                instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi


def save_piano_roll_as_midi(
    piano_roll: np.ndarray,
    file_path: Path | str,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
) -> Path:
    """Serialize one piano-roll sample as a MIDI file on disk."""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    midi = binary_piano_roll_to_midi(piano_roll=piano_roll, tempo_bpm=tempo_bpm)
    midi.write(str(output_path))
    return output_path
