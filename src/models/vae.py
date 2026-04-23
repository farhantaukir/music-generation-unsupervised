"""Variational autoencoder model definitions for Task 2 music generation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.config import (
    SEQUENCE_LENGTH,
    VAE_DROPOUT,
    VAE_HIDDEN_SIZE,
    VAE_INPUT_DIM,
    VAE_LATENT_DIM,
    VAE_NUM_LAYERS,
)


class VAEEncoder(nn.Module):
    """Encode input sequences into latent distribution parameters."""

    def __init__(
        self,
        input_dim: int = VAE_INPUT_DIM,
        hidden_size: int = VAE_HIDDEN_SIZE,
        num_layers: int = VAE_NUM_LAYERS,
        latent_dim: int = VAE_LATENT_DIM,
        dropout: float = VAE_DROPOUT,
    ) -> None:
        """Initialize LSTM encoder and mean/log-variance projection heads."""
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.to_mu = nn.Linear(hidden_size, latent_dim)
        self.to_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return latent mean and log-variance for a batch of sequences."""
        _, (hidden_state, _) = self.lstm(inputs)
        final_hidden = hidden_state[-1]
        mu = self.to_mu(final_hidden)
        logvar = self.to_logvar(final_hidden)
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decode latent samples into reconstructed piano-roll sequences."""

    def __init__(
        self,
        output_dim: int = VAE_INPUT_DIM,
        hidden_size: int = VAE_HIDDEN_SIZE,
        num_layers: int = VAE_NUM_LAYERS,
        latent_dim: int = VAE_LATENT_DIM,
        dropout: float = VAE_DROPOUT,
    ) -> None:
        """Initialize latent projection, recurrent decoder, and output head."""
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.latent_to_decoder_input = nn.Linear(latent_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.to_output = nn.Linear(hidden_size, output_dim)

    def forward(self, latent: Tensor, sequence_length: int = SEQUENCE_LENGTH) -> Tensor:
        """Generate reconstruction probabilities from latent vectors."""
        decoder_seed = self.latent_to_decoder_input(latent)
        repeated_seed = decoder_seed.unsqueeze(1).expand(-1, sequence_length, -1)
        decoded, _ = self.lstm(repeated_seed)
        logits = self.to_output(decoded)
        return torch.sigmoid(logits)


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """Compute mean KL divergence between q(z|x) and standard normal prior."""
    kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl_per_sample.mean()


class MusicVAE(nn.Module):
    """Wrap encoder and decoder into a variational autoencoder architecture."""

    def __init__(
        self,
        input_dim: int = VAE_INPUT_DIM,
        hidden_size: int = VAE_HIDDEN_SIZE,
        num_layers: int = VAE_NUM_LAYERS,
        latent_dim: int = VAE_LATENT_DIM,
        dropout: float = VAE_DROPOUT,
    ) -> None:
        """Construct the full VAE model with shared configuration values."""
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = VAEDecoder(
            output_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            latent_dim=latent_dim,
            dropout=dropout,
        )

    def encode(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a batch of sequences into mean and log-variance tensors."""
        return self.encoder(inputs)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample latent vectors with the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, latent: Tensor, sequence_length: int = SEQUENCE_LENGTH) -> Tensor:
        """Decode latent vectors into sequence reconstructions."""
        return self.decoder(latent, sequence_length=sequence_length)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return reconstruction, latent mean, latent log-variance, and sample."""
        mu, logvar = self.encode(inputs)
        latent = self.reparameterize(mu, logvar)
        reconstruction = self.decode(latent, sequence_length=inputs.size(1))
        return reconstruction, mu, logvar, latent

    def sample(self, num_samples: int, sequence_length: int = SEQUENCE_LENGTH) -> Tensor:
        """Sample random latent vectors and decode them into new sequences."""
        latent = torch.randn(num_samples, self.latent_dim, device=next(self.parameters()).device)
        return self.decode(latent, sequence_length=sequence_length)
