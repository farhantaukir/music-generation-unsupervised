"""Training pipeline for Task 2 variational autoencoder on piano-roll windows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DEVICE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    NUM_EPOCHS,
    SEED,
    VAE_BETA,
    WEIGHT_DECAY,
    ensure_output_dirs,
    set_global_seed,
)
from src.models.vae import MusicVAE, kl_divergence
from src.training.train_ae import build_dataloaders


def vae_loss(
    reconstruction: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = VAE_BETA,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute total, reconstruction, and KL losses for VAE optimization."""
    reconstruction_loss = nn.functional.mse_loss(reconstruction, target, reduction="mean")
    divergence_loss = kl_divergence(mu, logvar)
    total_loss = reconstruction_loss + beta * divergence_loss
    return total_loss, reconstruction_loss, divergence_loss


def train_one_epoch(
    model: MusicVAE,
    train_loader: DataLoader[Tensor],
    optimizer: Adam,
    beta: float = VAE_BETA,
) -> tuple[float, float, float]:
    """Run one VAE training epoch and return averaged total/recon/KL losses."""
    model.train()

    total_running = 0.0
    recon_running = 0.0
    kl_running = 0.0
    sample_count = 0

    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        reconstruction, mu, logvar, _ = model(batch)
        total_loss, recon_loss, divergence_loss = vae_loss(
            reconstruction=reconstruction,
            target=batch,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        total_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        batch_size = int(batch.size(0))
        total_running += float(total_loss.item()) * batch_size
        recon_running += float(recon_loss.item()) * batch_size
        kl_running += float(divergence_loss.item()) * batch_size
        sample_count += batch_size

    normalizer = max(sample_count, 1)
    return total_running / normalizer, recon_running / normalizer, kl_running / normalizer


def evaluate(
    model: MusicVAE,
    validation_loader: DataLoader[Tensor],
    beta: float = VAE_BETA,
) -> tuple[float, float, float]:
    """Run VAE validation and return averaged total/recon/KL losses."""
    model.eval()

    total_running = 0.0
    recon_running = 0.0
    kl_running = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in validation_loader:
            batch = batch.to(DEVICE)
            reconstruction, mu, logvar, _ = model(batch)
            total_loss, recon_loss, divergence_loss = vae_loss(
                reconstruction=reconstruction,
                target=batch,
                mu=mu,
                logvar=logvar,
                beta=beta,
            )

            batch_size = int(batch.size(0))
            total_running += float(total_loss.item()) * batch_size
            recon_running += float(recon_loss.item()) * batch_size
            kl_running += float(divergence_loss.item()) * batch_size
            sample_count += batch_size

    normalizer = max(sample_count, 1)
    return total_running / normalizer, recon_running / normalizer, kl_running / normalizer


def train_vae(
    train_loader: DataLoader[Tensor],
    validation_loader: DataLoader[Tensor],
    model: MusicVAE | None = None,
    beta: float = VAE_BETA,
) -> Dict[str, List[float] | str]:
    """Train Task 2 VAE and save the best validation-loss checkpoint."""
    output_dirs = ensure_output_dirs()
    checkpoint_path = output_dirs["checkpoints"] / "task2_best_vae.pt"

    model = model or MusicVAE()
    model = model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_total_losses: List[float] = []
    train_recon_losses: List[float] = []
    train_kl_losses: List[float] = []

    validation_total_losses: List[float] = []
    validation_recon_losses: List[float] = []
    validation_kl_losses: List[float] = []

    best_validation_total_loss = float("inf")

    epoch_iterator = tqdm(range(1, NUM_EPOCHS + 1), desc="Task 2 VAE Training")
    for epoch in epoch_iterator:
        train_total, train_recon, train_kl = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            beta=beta,
        )
        validation_total, validation_recon, validation_kl = evaluate(
            model=model,
            validation_loader=validation_loader,
            beta=beta,
        )

        train_total_losses.append(train_total)
        train_recon_losses.append(train_recon)
        train_kl_losses.append(train_kl)

        validation_total_losses.append(validation_total)
        validation_recon_losses.append(validation_recon)
        validation_kl_losses.append(validation_kl)

        if validation_total < best_validation_total_loss:
            best_validation_total_loss = validation_total
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_total_loss": train_total,
                    "validation_total_loss": validation_total,
                    "train_reconstruction_loss": train_recon,
                    "validation_reconstruction_loss": validation_recon,
                    "train_kl_loss": train_kl,
                    "validation_kl_loss": validation_kl,
                    "beta": beta,
                },
                checkpoint_path,
            )

        epoch_iterator.set_postfix(
            train_total=f"{train_total:.4f}",
            validation_total=f"{validation_total:.4f}",
            validation_recon=f"{validation_recon:.4f}",
            validation_kl=f"{validation_kl:.4f}",
        )

    return {
        "train_total_losses": train_total_losses,
        "train_reconstruction_losses": train_recon_losses,
        "train_kl_losses": train_kl_losses,
        "validation_total_losses": validation_total_losses,
        "validation_reconstruction_losses": validation_recon_losses,
        "validation_kl_losses": validation_kl_losses,
        "checkpoint_path": str(checkpoint_path),
    }


def run_training(
    data_root: Path | str | None = None,
    limit_files: int | None = None,
    beta: float = VAE_BETA,
) -> Dict[str, List[float] | str]:
    """Build data loaders and run end-to-end Task 2 VAE training."""
    set_global_seed(SEED)
    train_loader, validation_loader = build_dataloaders(
        data_root=data_root,
        limit_files=limit_files,
        split_name="task2",
    )
    return train_vae(train_loader=train_loader, validation_loader=validation_loader, beta=beta)


if __name__ == "__main__":
    results = run_training()
    print("Training finished.")
    print(f"Checkpoint: {results['checkpoint_path']}")
    print(f"Final train total loss: {results['train_total_losses'][-1]:.4f}")
    print(f"Final validation total loss: {results['validation_total_losses'][-1]:.4f}")
