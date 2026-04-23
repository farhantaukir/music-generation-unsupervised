"""Training pipeline for Task 3 Transformer autoregressive music generation."""

from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
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
    TOKEN_PAD_ID,
    TOKEN_SEQUENCE_LENGTH,
    TOKEN_VOCAB_SIZE,
    TR_LABEL_SMOOTHING,
    VALIDATION_SPLIT,
    WEIGHT_DECAY,
    ensure_output_dirs,
    get_data_root,
    set_global_seed,
)
from src.models.transformer import MusicTransformer
from src.preprocessing.midi_parser import discover_midi_files
from src.preprocessing.split_manager import get_or_create_train_val_split
from src.preprocessing.tokenizer import build_token_chunks_from_files, token_chunks_to_array


class TokenSequenceDataset(Dataset[Tuple[Tensor, Tensor]]):
    """Wrap token sequences into autoregressive input and target pairs."""

    def __init__(self, token_chunks: np.ndarray) -> None:
        """Store token chunks as int64 tensors for LM training."""
        if token_chunks.ndim != 2 or token_chunks.shape[1] < 2:
            raise ValueError("token_chunks must be a 2D array with sequence length >= 2.")

        tokens = torch.from_numpy(token_chunks.astype(np.int64, copy=False))
        self.inputs = tokens[:, :-1]
        self.targets = tokens[:, 1:]

    def __len__(self) -> int:
        """Return the number of training sequences."""
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Return one autoregressive input-target token pair."""
        return self.inputs[index], self.targets[index]


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
        return get_or_create_train_val_split(
            midi_files=files,
            split_name=split_name,
            validation_split=validation_split,
            seed=seed,
            force_rebuild=force_rebuild,
        )

    return get_or_create_train_val_split(
        midi_files=files,
        split_name="temporary_in_memory_split",
        validation_split=validation_split,
        seed=seed,
        force_rebuild=True,
    )


def build_dataloaders(
    data_root: Path | str | None = None,
    limit_files: int | None = None,
    chunk_length: int = TOKEN_SEQUENCE_LENGTH,
    split_name: str = "task3",
    force_rebuild_split: bool = False,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader[Tuple[Tensor, Tensor]], DataLoader[Tuple[Tensor, Tensor]]]:
    """Create train and validation token dataloaders from MIDI files."""
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

    train_chunks = token_chunks_to_array(build_token_chunks_from_files(train_files, chunk_length=chunk_length))
    validation_chunks = token_chunks_to_array(build_token_chunks_from_files(validation_files, chunk_length=chunk_length))

    train_dataset = TokenSequenceDataset(train_chunks)
    validation_dataset = TokenSequenceDataset(validation_chunks)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, validation_loader


def compute_perplexity(loss_value: float) -> float:
    """Convert average negative log-likelihood into perplexity."""
    return float(math.exp(min(loss_value, 20.0)))


def train_one_epoch(
    model: MusicTransformer,
    train_loader: DataLoader[Tuple[Tensor, Tensor]],
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
    max_batches: int | None = None,
    show_batch_progress: bool = False,
) -> float:
    """Run one Transformer training epoch and return average token loss."""
    model.train()

    total_loss = 0.0
    total_tokens = 0

    batch_progress = tqdm(train_loader, desc="Train batches", leave=False) if show_batch_progress else None
    batch_iterator = batch_progress if batch_progress is not None else train_loader
    for batch_index, (input_tokens, target_tokens) in enumerate(batch_iterator, start=1):
        if max_batches is not None and batch_index > max_batches:
            break

        input_tokens = input_tokens.to(DEVICE, non_blocking=True)
        target_tokens = target_tokens.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
        with autocast_context:
            logits = model(input_tokens)
            loss = criterion(logits.reshape(-1, TOKEN_VOCAB_SIZE), target_tokens.reshape(-1))

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

        token_count = int(target_tokens.ne(TOKEN_PAD_ID).sum().item())
        total_loss += float(loss.item()) * max(token_count, 1)
        total_tokens += max(token_count, 1)

        if batch_progress is not None:
            batch_progress.set_postfix(loss=f"{loss.item():.4f}")

    if batch_progress is not None:
        batch_progress.close()

    return total_loss / max(total_tokens, 1)


def evaluate(
    model: MusicTransformer,
    validation_loader: DataLoader[Tuple[Tensor, Tensor]],
    criterion: nn.Module,
    use_amp: bool = False,
    max_batches: int | None = None,
    show_batch_progress: bool = False,
) -> float:
    """Run no-grad validation and return average token loss."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        batch_progress = tqdm(validation_loader, desc="Validation batches", leave=False) if show_batch_progress else None
        batch_iterator = batch_progress if batch_progress is not None else validation_loader
        for batch_index, (input_tokens, target_tokens) in enumerate(batch_iterator, start=1):
            if max_batches is not None and batch_index > max_batches:
                break

            input_tokens = input_tokens.to(DEVICE, non_blocking=True)
            target_tokens = target_tokens.to(DEVICE, non_blocking=True)

            autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with autocast_context:
                logits = model(input_tokens)
                loss = criterion(logits.reshape(-1, TOKEN_VOCAB_SIZE), target_tokens.reshape(-1))

            token_count = int(target_tokens.ne(TOKEN_PAD_ID).sum().item())
            total_loss += float(loss.item()) * max(token_count, 1)
            total_tokens += max(token_count, 1)

            if batch_progress is not None:
                batch_progress.set_postfix(loss=f"{loss.item():.4f}")

        if batch_progress is not None:
            batch_progress.close()

    return total_loss / max(total_tokens, 1)


def train_transformer(
    train_loader: DataLoader[Tuple[Tensor, Tensor]],
    validation_loader: DataLoader[Tuple[Tensor, Tensor]],
    model: MusicTransformer | None = None,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = TR_LABEL_SMOOTHING,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    use_amp: bool | None = None,
    show_batch_progress: bool = False,
    show_epoch_progress: bool = True,
) -> Dict[str, List[float] | str | int]:
    """Train Task 3 Transformer and save best checkpoint by validation loss."""
    output_dirs = ensure_output_dirs()
    checkpoint_path = output_dirs["checkpoints"] / "task3_best_transformer.pt"

    model = model or MusicTransformer()
    model = model.to(DEVICE)

    num_layers = len(model.decoder.layers)
    if num_layers > 0:
        first_layer = model.decoder.layers[0]
        nhead = int(first_layer.self_attn.num_heads)
        dim_feedforward = int(first_layer.linear1.out_features)
        dropout = float(first_layer.dropout.p)
    else:
        nhead = 1
        dim_feedforward = int(model.d_model)
        dropout = 0.0

    model_config = {
        "vocab_size": int(model.vocab_size),
        "d_model": int(model.d_model),
        "nhead": nhead,
        "num_layers": int(num_layers),
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "pad_id": int(model.pad_id),
    }

    if use_amp is None:
        use_amp = DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD_ID, label_smoothing=label_smoothing)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses: List[float] = []
    validation_losses: List[float] = []
    validation_perplexities: List[float] = []

    best_validation_loss = float("inf")

    epoch_progress = tqdm(range(1, num_epochs + 1), desc="Task 3 Transformer Training") if show_epoch_progress else None
    epoch_iterator = epoch_progress if epoch_progress is not None else range(1, num_epochs + 1)
    for epoch in epoch_iterator:
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            max_batches=max_train_batches,
            show_batch_progress=show_batch_progress,
        )
        validation_loss = evaluate(
            model=model,
            validation_loader=validation_loader,
            criterion=criterion,
            use_amp=use_amp,
            max_batches=max_validation_batches,
            show_batch_progress=show_batch_progress,
        )
        validation_perplexity = compute_perplexity(validation_loss)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        validation_perplexities.append(validation_perplexity)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "validation_perplexity": validation_perplexity,
                    "num_epochs": num_epochs,
                    "max_train_batches": max_train_batches,
                    "max_validation_batches": max_validation_batches,
                    "show_batch_progress": show_batch_progress,
                    "show_epoch_progress": show_epoch_progress,
                    "model_config": model_config,
                },
                checkpoint_path,
            )

        if epoch_progress is not None:
            epoch_progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                validation_loss=f"{validation_loss:.4f}",
                perplexity=f"{validation_perplexity:.2f}",
            )
        else:
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"validation_loss={validation_loss:.4f} | "
                f"perplexity={validation_perplexity:.2f}"
            )

    if epoch_progress is not None:
        epoch_progress.close()

    epochs_completed = len(train_losses)
    print(f"Training finished: {epochs_completed}/{num_epochs} epochs completed.")

    return {
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "validation_perplexities": validation_perplexities,
        "checkpoint_path": str(checkpoint_path),
        "epochs_completed": epochs_completed,
        "requested_num_epochs": num_epochs,
    }


def run_training(
    data_root: Path | str | None = None,
    limit_files: int | None = None,
    chunk_length: int = TOKEN_SEQUENCE_LENGTH,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    num_epochs: int = NUM_EPOCHS,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    use_amp: bool | None = None,
    show_batch_progress: bool = False,
    show_epoch_progress: bool = True,
) -> Dict[str, List[float] | str | int]:
    """Build token dataloaders and run end-to-end Task 3 training."""
    set_global_seed(SEED)
    train_loader, validation_loader = build_dataloaders(
        data_root=data_root,
        limit_files=limit_files,
        chunk_length=chunk_length,
        split_name="task3",
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_transformer(
        train_loader=train_loader,
        validation_loader=validation_loader,
        num_epochs=num_epochs,
        max_train_batches=max_train_batches,
        max_validation_batches=max_validation_batches,
        use_amp=use_amp,
        show_batch_progress=show_batch_progress,
        show_epoch_progress=show_epoch_progress,
    )


if __name__ == "__main__":
    results = run_training()
    print("Training finished.")
    print(f"Checkpoint: {results['checkpoint_path']}")
    print(f"Final validation loss: {results['validation_losses'][-1]:.4f}")
    print(f"Final validation perplexity: {results['validation_perplexities'][-1]:.2f}")
