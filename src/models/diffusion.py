"""Denoising diffusion model definitions for symbolic piano-roll music generation."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.config import (
    DEVICE,
    DIFF_BETA_END,
    DIFF_BETA_START,
    DIFF_DROPOUT,
    DIFF_HIDDEN_DIM,
    DIFF_INPUT_DIM,
    DIFF_MODEL_DIM,
    DIFF_NUM_TIMESTEPS,
    DIFF_SEQUENCE_LENGTH,
    DIFF_TIME_EMBED_DIM,
)


def _gather_schedule_values(values: Tensor, timesteps: Tensor, target_shape: torch.Size) -> Tensor:
    """Gather per-timestep schedule values and reshape to broadcast over inputs."""
    gathered = values.gather(0, timesteps.long())
    view_shape = (target_shape[0],) + (1,) * (len(target_shape) - 1)
    return gathered.view(view_shape)


class SinusoidalTimeEmbedding(nn.Module):
    """Embed integer diffusion timesteps into continuous sinusoidal vectors."""

    def __init__(self, embedding_dim: int = DIFF_TIME_EMBED_DIM) -> None:
        """Initialize sinusoidal timestep embedding size."""
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: Tensor) -> Tensor:
        """Map a timestep batch into sinusoidal embedding features."""
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            raise ValueError("embedding_dim must be at least 2 for sinusoidal embedding.")

        frequency_factors = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half_dim - 1, 1))
        )
        phases = timesteps.float().unsqueeze(1) * frequency_factors.unsqueeze(0)
        embedding = torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)

        if self.embedding_dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros((embedding.size(0), 1), device=timesteps.device, dtype=embedding.dtype)],
                dim=1,
            )

        return embedding


class DiffusionDenoiser(nn.Module):
    """Predict additive Gaussian noise for noised piano-roll windows."""

    def __init__(
        self,
        input_dim: int = DIFF_INPUT_DIM,
        sequence_length: int = DIFF_SEQUENCE_LENGTH,
        model_dim: int = DIFF_MODEL_DIM,
        time_embed_dim: int = DIFF_TIME_EMBED_DIM,
        hidden_dim: int = DIFF_HIDDEN_DIM,
        dropout: float = DIFF_DROPOUT,
    ) -> None:
        """Construct a timestep-conditioned MLP denoiser for flattened windows."""
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.flat_dim = input_dim * sequence_length

        self.time_embedding = SinusoidalTimeEmbedding(embedding_dim=time_embed_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(time_embed_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

        self.input_projection = nn.Linear(self.flat_dim, model_dim)
        self.network = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.flat_dim),
        )

    def forward(self, noisy_inputs: Tensor, timesteps: Tensor) -> Tensor:
        """Predict noise from noised windows and their diffusion timesteps."""
        if noisy_inputs.ndim != 3:
            raise ValueError("noisy_inputs must have shape (batch, sequence_length, input_dim).")

        batch_size = noisy_inputs.size(0)
        flattened = noisy_inputs.reshape(batch_size, -1)
        input_features = self.input_projection(flattened)
        time_features = self.time_projection(self.time_embedding(timesteps))
        predicted_noise = self.network(input_features + time_features)
        return predicted_noise.reshape(batch_size, self.sequence_length, self.input_dim)


class MusicDiffusion(nn.Module):
    """Wrap diffusion schedules, denoiser network, and sampling utilities."""

    def __init__(
        self,
        input_dim: int = DIFF_INPUT_DIM,
        sequence_length: int = DIFF_SEQUENCE_LENGTH,
        model_dim: int = DIFF_MODEL_DIM,
        time_embed_dim: int = DIFF_TIME_EMBED_DIM,
        hidden_dim: int = DIFF_HIDDEN_DIM,
        num_timesteps: int = DIFF_NUM_TIMESTEPS,
        beta_start: float = DIFF_BETA_START,
        beta_end: float = DIFF_BETA_END,
        dropout: float = DIFF_DROPOUT,
    ) -> None:
        """Initialize diffusion schedules and timestep-conditioned denoiser."""
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.num_timesteps = num_timesteps

        self.denoiser = DiffusionDenoiser(
            input_dim=input_dim,
            sequence_length=sequence_length,
            model_dim=model_dim,
            time_embed_dim=time_embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_cumprod", alpha_cumprod, persistent=False)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev, persistent=False)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod",
            torch.sqrt(1.0 - alpha_cumprod),
            persistent=False,
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas), persistent=False)
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod),
            persistent=False,
        )

    def q_sample(self, clean_inputs: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Add Gaussian noise to clean inputs at selected timesteps."""
        if noise is None:
            noise = torch.randn_like(clean_inputs)

        sqrt_alpha_cumprod_t = _gather_schedule_values(self.sqrt_alpha_cumprod, timesteps, clean_inputs.shape)
        sqrt_one_minus_alpha_cumprod_t = _gather_schedule_values(
            self.sqrt_one_minus_alpha_cumprod,
            timesteps,
            clean_inputs.shape,
        )
        noised = sqrt_alpha_cumprod_t * clean_inputs + sqrt_one_minus_alpha_cumprod_t * noise
        return noised, noise

    def training_loss(self, clean_inputs: Tensor, timesteps: Tensor | None = None) -> Tensor:
        """Compute MSE loss between true and predicted diffusion noise."""
        if timesteps is None:
            timesteps = torch.randint(0, self.num_timesteps, (clean_inputs.size(0),), device=clean_inputs.device)

        noised, target_noise = self.q_sample(clean_inputs=clean_inputs, timesteps=timesteps)
        predicted_noise = self.denoiser(noised, timesteps)
        return nn.functional.mse_loss(predicted_noise, target_noise)

    def p_sample(self, noisy_inputs: Tensor, timesteps: Tensor) -> Tensor:
        """Run one reverse-diffusion step to denoise current samples."""
        betas_t = _gather_schedule_values(self.betas, timesteps, noisy_inputs.shape)
        sqrt_one_minus_alpha_cumprod_t = _gather_schedule_values(
            self.sqrt_one_minus_alpha_cumprod,
            timesteps,
            noisy_inputs.shape,
        )
        sqrt_recip_alphas_t = _gather_schedule_values(self.sqrt_recip_alphas, timesteps, noisy_inputs.shape)

        predicted_noise = self.denoiser(noisy_inputs, timesteps)
        model_mean = sqrt_recip_alphas_t * (noisy_inputs - betas_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)

        posterior_variance_t = _gather_schedule_values(self.posterior_variance, timesteps, noisy_inputs.shape)
        nonzero_mask = (timesteps != 0).float().view(noisy_inputs.size(0), *([1] * (noisy_inputs.ndim - 1)))
        posterior_noise = torch.randn_like(noisy_inputs)
        return model_mean + nonzero_mask * torch.sqrt(torch.clamp(posterior_variance_t, min=1e-20)) * posterior_noise

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | None = None) -> Tensor:
        """Generate new piano-roll windows via iterative reverse diffusion."""
        model_device = device if device is not None else next(self.parameters()).device
        current = torch.randn(num_samples, self.sequence_length, self.input_dim, device=model_device)

        for timestep in reversed(range(self.num_timesteps)):
            batch_timesteps = torch.full((num_samples,), timestep, device=model_device, dtype=torch.long)
            current = self.p_sample(current, batch_timesteps)

        return torch.sigmoid(current)


def build_diffusion_model(device: torch.device = DEVICE) -> MusicDiffusion:
    """Build a diffusion model instance and move it to the requested device."""
    return MusicDiffusion().to(device)
