"""Training pipeline for Task 1 LSTM autoencoder on windowed piano-roll inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    DEVICE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    SEED,
    SEQUENCE_LENGTH,
    VALIDATION_SPLIT,
    WEIGHT_DECAY,
    WINDOW_STEP,
    ensure_output_dirs,
    get_data_root,
    set_global_seed,
)
from src.models.autoencoder import LSTMAutoencoder
from src.preprocessing.midi_parser import discover_midi_files
from src.preprocessing.piano_roll import load_windowed_piano_rolls
from src.preprocessing.split_manager import get_or_create_train_val_split


class PianoRollWindowDataset(Dataset[Tensor]):
    """Wrap fixed-length piano-roll windows into a torch dataset."""

    def __init__(self, windows: np.ndarray) -> None:
        """Store piano-roll windows as float32 tensors."""
        self.windows = torch.from_numpy(windows.astype(np.float32, copy=False))

    def __len__(self) -> int:
        """Return the number of available windows."""
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> Tensor:
        """Return one piano-roll window by index."""
        return self.windows[index]


def split_midi_files(
    midi_files: Sequence[Path],
    validation_split: float = VALIDATION_SPLIT,
    seed: int = SEED,
    split_name: str | None = None,
    force_rebuild: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """Split MIDI files deterministically and optionally persist/reuse split manifests."""
    files = list(midi_files)
    if split_name:
        train_files, validation_files = get_or_create_train_val_split(
            midi_files=files,
            split_name=split_name,
            validation_split=validation_split,
            seed=seed,
            force_rebuild=force_rebuild,
        )
        return train_files, validation_files

    train_files, validation_files = get_or_create_train_val_split(
        midi_files=files,
        split_name="temporary_in_memory_split",
        validation_split=validation_split,
        seed=seed,
        force_rebuild=True,
    )
    return train_files, validation_files


def build_dataloaders(
    data_root: Path | str | None = None,
    limit_files: int | None = None,
    split_name: str = "task1",
    force_rebuild_split: bool = False,
) -> Tuple[DataLoader[Tensor], DataLoader[Tensor]]:
    """Create train and validation DataLoaders from discovered MIDI files."""
    midi_root = Path(data_root) if data_root is not None else get_data_root()
    midi_files = discover_midi_files(midi_root)

    if limit_files is not None:
        midi_files = midi_files[:limit_files]

    if not midi_files:
        raise ValueError(f"No MIDI files found under: {midi_root}")

    train_files, validation_files = split_midi_files(
        midi_files,
        split_name=split_name,
        force_rebuild=force_rebuild_split,
    )

    train_windows = load_windowed_piano_rolls(
        train_files,
        sequence_length=SEQUENCE_LENGTH,
        step_size=WINDOW_STEP,
    )
    validation_windows = load_windowed_piano_rolls(
        validation_files,
        sequence_length=SEQUENCE_LENGTH,
        step_size=WINDOW_STEP,
    )

    train_dataset = PianoRollWindowDataset(train_windows)
    validation_dataset = PianoRollWindowDataset(validation_windows)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, validation_loader


def train_one_epoch(
    model: LSTMAutoencoder,
    train_loader: DataLoader[Tensor],
    criterion: nn.Module,
    optimizer: Adam,
) -> float:
    """Run one training epoch and return mean reconstruction loss."""
    model.train()
    total_loss = 0.0
    sample_count = 0

    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        reconstruction, _ = model(batch)
        loss = criterion(reconstruction, batch)
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        batch_size = int(batch.size(0))
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size

    return total_loss / max(sample_count, 1)


def evaluate(
    model: LSTMAutoencoder,
    validation_loader: DataLoader[Tensor],
    criterion: nn.Module,
) -> float:
    """Run validation and return mean reconstruction loss without gradients."""
    model.eval()
    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in validation_loader:
            batch = batch.to(DEVICE)
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)

            batch_size = int(batch.size(0))
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size

    return total_loss / max(sample_count, 1)


def train_autoencoder(
    train_loader: DataLoader[Tensor],
    validation_loader: DataLoader[Tensor],
    model: LSTMAutoencoder | None = None,
) -> Dict[str, List[float] | str]:
    """Train the Task 1 autoencoder and save the best validation checkpoint."""
    output_dirs = ensure_output_dirs()
    checkpoint_path = output_dirs["checkpoints"] / "task1_best_autoencoder.pt"

    model = model or LSTMAutoencoder()
    model = model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    train_losses: List[float] = []
    validation_losses: List[float] = []
    best_validation_loss = float("inf")

    epoch_iterator = tqdm(range(1, NUM_EPOCHS + 1), desc="Task 1 AE Training")
    for epoch in epoch_iterator:
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        validation_loss = evaluate(model, validation_loader, criterion)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                },
                checkpoint_path,
            )

        epoch_iterator.set_postfix(
            train_loss=f"{train_loss:.4f}",
            validation_loss=f"{validation_loss:.4f}",
            best_validation=f"{best_validation_loss:.4f}",
        )

    return {
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "checkpoint_path": str(checkpoint_path),
    }


def run_training(data_root: Path | str | None = None, limit_files: int | None = None) -> Dict[str, List[float] | str]:
    """Build data loaders and run end-to-end Task 1 autoencoder training."""
    set_global_seed(SEED)
    train_loader, validation_loader = build_dataloaders(
        data_root=data_root,
        limit_files=limit_files,
        split_name="task1",
    )
    return train_autoencoder(train_loader, validation_loader)


if __name__ == "__main__":
    results = run_training()
    print("Training finished.")
    print(f"Checkpoint: {results['checkpoint_path']}")
    print(f"Final train loss: {results['train_losses'][-1]:.4f}")
    print(f"Final validation loss: {results['validation_losses'][-1]:.4f}")
