"""Latent sampling helpers for autoencoder and VAE generation workflows."""

from __future__ import annotations

import torch
from torch import Tensor

from src.config import AE_LATENT_DIM, DEVICE, SEED, VAE_LATENT_DIM


def make_generator(seed: int = SEED, device: torch.device = DEVICE) -> torch.Generator:
    """Create a seeded torch generator for reproducible latent sampling."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def sample_standard_normal_latents(
    num_samples: int,
    latent_dim: int = AE_LATENT_DIM,
    device: torch.device = DEVICE,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample Gaussian latent vectors as tensor shape (num_samples, latent_dim)."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")

    return torch.randn(
        int(num_samples),
        int(latent_dim),
        device=device,
        generator=generator,
    )


def sample_uniform_latents(
    num_samples: int,
    latent_dim: int = AE_LATENT_DIM,
    low: float = -1.0,
    high: float = 1.0,
    device: torch.device = DEVICE,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample uniform latent vectors in [low, high] with shape (num_samples, latent_dim)."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")
    if high <= low:
        raise ValueError("high must be greater than low.")

    latents = torch.rand(
        int(num_samples),
        int(latent_dim),
        device=device,
        generator=generator,
    )
    return low + (high - low) * latents


def sample_vae_latents(
    num_samples: int,
    latent_dim: int = VAE_LATENT_DIM,
    device: torch.device = DEVICE,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample VAE latent vectors as Gaussian tensor shape (num_samples, latent_dim)."""
    return sample_standard_normal_latents(
        num_samples=num_samples,
        latent_dim=latent_dim,
        device=device,
        generator=generator,
    )


def interpolate_latents(
    start_latent: Tensor,
    end_latent: Tensor,
    num_steps: int,
) -> Tensor:
    """Interpolate two latent vectors into a tensor of shape (num_steps, latent_dim)."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2.")
    if start_latent.ndim != 1 or end_latent.ndim != 1:
        raise ValueError("start_latent and end_latent must be 1D tensors.")
    if start_latent.shape != end_latent.shape:
        raise ValueError("start_latent and end_latent must have the same shape.")

    weights = torch.linspace(0.0, 1.0, steps=int(num_steps), device=start_latent.device).unsqueeze(1)
    start = start_latent.unsqueeze(0)
    end = end_latent.unsqueeze(0)
    return start * (1.0 - weights) + end * weights
