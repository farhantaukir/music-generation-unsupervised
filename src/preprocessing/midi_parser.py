"""MIDI parsing utilities for loading files and extracting note metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pretty_midi

from src.config import DEFAULT_TEMPO_BPM, MIDI_MAX_PITCH, MIDI_MIN_PITCH


@dataclass(frozen=True)
class NoteEvent:
    """Store one symbolic note event with timing and velocity."""

    pitch: int
    start: float
    end: float
    velocity: int


def discover_midi_files(root_path: Path | str, extensions: Sequence[str] = (".mid", ".midi")) -> List[Path]:
    """Return all MIDI file paths under a root directory recursively."""
    root = Path(root_path)
    if not root.exists():
        return []

    discovered: List[Path] = []
    for extension in extensions:
        discovered.extend(root.rglob(f"*{extension}"))

    return sorted(set(discovered))


def load_pretty_midi(midi_path: Path | str) -> pretty_midi.PrettyMIDI:
    """Load a MIDI file from disk into a PrettyMIDI object."""
    return pretty_midi.PrettyMIDI(str(midi_path))


def estimate_tempo_bpm(midi: pretty_midi.PrettyMIDI, default_tempo: float = DEFAULT_TEMPO_BPM) -> float:
    """Estimate the first available tempo in BPM from a PrettyMIDI object."""
    _, tempo_values = midi.get_tempo_changes()
    if len(tempo_values) == 0:
        return default_tempo
    return float(tempo_values[0])


def extract_note_events(
    midi: pretty_midi.PrettyMIDI,
    min_pitch: int = MIDI_MIN_PITCH,
    max_pitch: int = MIDI_MAX_PITCH,
) -> List[NoteEvent]:
    """Extract pitch-filtered notes from all non-drum instruments in a MIDI object."""
    events: List[NoteEvent] = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if min_pitch <= note.pitch <= max_pitch and note.end > note.start:
                events.append(
                    NoteEvent(
                        pitch=int(note.pitch),
                        start=float(note.start),
                        end=float(note.end),
                        velocity=int(note.velocity),
                    )
                )

    events.sort(key=lambda item: (item.start, item.pitch))
    return events


def load_midi_note_events(midi_path: Path | str) -> List[NoteEvent]:
    """Load one MIDI file and return sorted note events."""
    midi = load_pretty_midi(midi_path)
    return extract_note_events(midi)
