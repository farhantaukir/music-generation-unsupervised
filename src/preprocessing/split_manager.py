"""Utilities to persist and reuse deterministic train/validation MIDI file splits."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from src.config import SEED, TRAIN_TEST_SPLIT_DIR, VALIDATION_SPLIT


def sanitize_split_name(split_name: str) -> str:
    """Normalize a split name for safe manifest file naming."""
    return "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in split_name)


def get_split_manifest_paths(split_name: str, split_dir: Path = TRAIN_TEST_SPLIT_DIR) -> Tuple[Path, Path]:
    """Return train and validation manifest paths for a named split."""
    safe_name = sanitize_split_name(split_name)
    return split_dir / f"{safe_name}_train.txt", split_dir / f"{safe_name}_val.txt"


def save_train_val_split(
    train_files: Sequence[Path],
    validation_files: Sequence[Path],
    split_name: str,
    split_dir: Path = TRAIN_TEST_SPLIT_DIR,
) -> None:
    """Write train and validation file lists to split manifest text files."""
    split_dir.mkdir(parents=True, exist_ok=True)
    train_manifest, validation_manifest = get_split_manifest_paths(split_name=split_name, split_dir=split_dir)

    train_manifest.write_text("\n".join(str(path) for path in train_files), encoding="utf-8")
    validation_manifest.write_text("\n".join(str(path) for path in validation_files), encoding="utf-8")


def load_train_val_split(
    split_name: str,
    available_files: Sequence[Path],
    split_dir: Path = TRAIN_TEST_SPLIT_DIR,
) -> Tuple[list[Path], list[Path]] | None:
    """Load a saved split when manifests exist and all listed files are still available."""
    train_manifest, validation_manifest = get_split_manifest_paths(split_name=split_name, split_dir=split_dir)
    if not train_manifest.exists() or not validation_manifest.exists():
        return None

    available_set = {str(path) for path in available_files}

    train_files = [
        Path(line.strip())
        for line in train_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.strip() in available_set
    ]
    validation_files = [
        Path(line.strip())
        for line in validation_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.strip() in available_set
    ]

    if not train_files or not validation_files:
        return None

    return train_files, validation_files


def deterministic_train_val_split(
    midi_files: Sequence[Path],
    validation_split: float = VALIDATION_SPLIT,
    seed: int = SEED,
) -> Tuple[list[Path], list[Path]]:
    """Create a deterministic train/validation split from MIDI file paths."""
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be in the range (0, 1).")

    files = list(midi_files)
    if len(files) < 2:
        raise ValueError("At least two MIDI files are required for a train/validation split.")

    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    split_index = max(1, int(len(files) * (1.0 - validation_split)))
    split_index = min(split_index, len(files) - 1)
    return files[:split_index], files[split_index:]


def get_or_create_train_val_split(
    midi_files: Sequence[Path],
    split_name: str,
    validation_split: float = VALIDATION_SPLIT,
    seed: int = SEED,
    split_dir: Path = TRAIN_TEST_SPLIT_DIR,
    force_rebuild: bool = False,
) -> Tuple[list[Path], list[Path]]:
    """Load an existing named split or create and persist a new deterministic split."""
    files = list(midi_files)
    if not force_rebuild:
        loaded = load_train_val_split(split_name=split_name, available_files=files, split_dir=split_dir)
        if loaded is not None:
            return loaded

    train_files, validation_files = deterministic_train_val_split(
        midi_files=files,
        validation_split=validation_split,
        seed=seed,
    )
    save_train_val_split(
        train_files=train_files,
        validation_files=validation_files,
        split_name=split_name,
        split_dir=split_dir,
    )
    return train_files, validation_files
