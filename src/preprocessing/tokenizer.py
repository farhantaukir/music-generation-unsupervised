"""Tokenization utilities for Task 3 Transformer-based symbolic music modeling."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from src.config import (
    MIDI_MAX_PITCH,
    MIDI_MIN_PITCH,
    TOKEN_BOS_ID,
    TOKEN_EOS_ID,
    TOKEN_NOTE_OFFSET,
    TOKEN_PAD_ID,
    TOKEN_SEQUENCE_LENGTH,
)
from src.preprocessing.midi_parser import discover_midi_files, extract_note_events, load_pretty_midi


def pitch_to_token(pitch: int) -> int:
    """Map a MIDI pitch in configured range to a note token ID."""
    if pitch < MIDI_MIN_PITCH or pitch > MIDI_MAX_PITCH:
        raise ValueError(f"Pitch {pitch} is outside configured range [{MIDI_MIN_PITCH}, {MIDI_MAX_PITCH}].")
    return TOKEN_NOTE_OFFSET + (pitch - MIDI_MIN_PITCH)


def token_to_pitch(token_id: int) -> int:
    """Map a note token ID back to its MIDI pitch value."""
    if token_id < TOKEN_NOTE_OFFSET:
        raise ValueError(f"Token ID {token_id} is not a note token.")
    return MIDI_MIN_PITCH + (token_id - TOKEN_NOTE_OFFSET)


def midi_path_to_note_tokens(midi_path: Path | str) -> List[int]:
    """Convert one MIDI file to an ordered list of note tokens."""
    midi = load_pretty_midi(midi_path)
    events = extract_note_events(midi)
    return [pitch_to_token(event.pitch) for event in events]


def add_special_tokens(tokens: Sequence[int]) -> List[int]:
    """Wrap token sequence with BOS and EOS special tokens."""
    return [TOKEN_BOS_ID, *list(tokens), TOKEN_EOS_ID]


def chunk_tokens(tokens: Sequence[int], chunk_length: int = TOKEN_SEQUENCE_LENGTH) -> List[List[int]]:
    """Split a token stream into fixed-length chunks with right padding."""
    if chunk_length <= 1:
        raise ValueError("chunk_length must be greater than 1.")

    chunks: List[List[int]] = []
    step = chunk_length - 2
    for start_index in range(0, len(tokens), step):
        token_slice = list(tokens[start_index : start_index + step])
        with_specials = add_special_tokens(token_slice)
        if len(with_specials) < chunk_length:
            with_specials.extend([TOKEN_PAD_ID] * (chunk_length - len(with_specials)))
        else:
            with_specials = with_specials[:chunk_length]
            with_specials[-1] = TOKEN_EOS_ID
        chunks.append(with_specials)

    return chunks


def build_token_chunks_from_files(
    midi_files: Sequence[Path],
    chunk_length: int = TOKEN_SEQUENCE_LENGTH,
) -> List[List[int]]:
    """Tokenize multiple MIDI files and return fixed-length training chunks."""
    all_chunks: List[List[int]] = []

    for midi_path in tqdm(midi_files, desc="Tokenizing MIDI files"):
        try:
            note_tokens = midi_path_to_note_tokens(midi_path)
            if not note_tokens:
                continue
            chunks = chunk_tokens(note_tokens, chunk_length=chunk_length)
            all_chunks.extend(chunks)
        except Exception:
            continue

    if not all_chunks:
        raise ValueError("No token chunks were produced from the provided MIDI files.")

    return all_chunks


def build_token_chunks_from_root(
    midi_root: Path | str,
    chunk_length: int = TOKEN_SEQUENCE_LENGTH,
    limit_files: int | None = None,
) -> List[List[int]]:
    """Discover MIDI files under a root and return token chunks."""
    midi_files = discover_midi_files(midi_root)
    if limit_files is not None:
        midi_files = midi_files[:limit_files]
    return build_token_chunks_from_files(midi_files, chunk_length=chunk_length)


def token_chunks_to_array(token_chunks: Sequence[Sequence[int]]) -> np.ndarray:
    """Convert a nested list of token chunks into an int64 NumPy array."""
    array = np.asarray(token_chunks, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("token_chunks must produce a 2D array.")
    return array


def token_sequence_to_midi_pitches(token_sequence: Sequence[int]) -> List[int]:
    """Extract MIDI pitches from a token sequence by dropping special and pad tokens."""
    pitches: List[int] = []
    for token in token_sequence:
        if token >= TOKEN_NOTE_OFFSET:
            pitches.append(token_to_pitch(int(token)))
    return pitches


def split_input_target(token_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create autoregressive inputs and shifted targets from token chunks."""
    if token_batch.ndim != 2 or token_batch.shape[1] < 2:
        raise ValueError("token_batch must be a 2D array with sequence length >= 2.")
    return token_batch[:, :-1], token_batch[:, 1:]
