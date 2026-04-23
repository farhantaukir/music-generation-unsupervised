"""LSTM autoencoder model definitions for Task 1 music reconstruction and generation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.config import (
    AE_DROPOUT,
    AE_HIDDEN_SIZE,
    AE_INPUT_DIM,
    AE_LATENT_DIM,
    AE_NUM_LAYERS,
    SEQUENCE_LENGTH,
)


class Encoder(nn.Module):
    """Encode a piano-roll sequence into a fixed-dimensional latent vector."""

    def __init__(
        self,
        input_dim: int = AE_INPUT_DIM,
        hidden_size: int = AE_HIDDEN_SIZE,
        num_layers: int = AE_NUM_LAYERS,
        latent_dim: int = AE_LATENT_DIM,
        dropout: float = AE_DROPOUT,
    ) -> None:
        """Initialize the LSTM encoder and latent projection layers."""
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.to_latent = nn.Linear(hidden_size, latent_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """Map a batch of sequences to latent vectors."""
        _, (hidden_state, _) = self.lstm(inputs)
        final_hidden = hidden_state[-1]
        return self.to_latent(final_hidden)


class Decoder(nn.Module):
    """Decode latent vectors into reconstructed piano-roll sequences."""

    def __init__(
        self,
        output_dim: int = AE_INPUT_DIM,
        hidden_size: int = AE_HIDDEN_SIZE,
        num_layers: int = AE_NUM_LAYERS,
        latent_dim: int = AE_LATENT_DIM,
        dropout: float = AE_DROPOUT,
    ) -> None:
        """Initialize the latent expansion, LSTM decoder, and output projection."""
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
        """Generate reconstructed piano-roll probabilities from latent vectors."""
        decoder_seed = self.latent_to_decoder_input(latent)
        repeated_seed = decoder_seed.unsqueeze(1).expand(-1, sequence_length, -1)
        decoded, _ = self.lstm(repeated_seed)
        logits = self.to_output(decoded)
        return torch.sigmoid(logits)


class LSTMAutoencoder(nn.Module):
    """Wrap an encoder and decoder into an end-to-end reconstruction model."""

    def __init__(
        self,
        input_dim: int = AE_INPUT_DIM,
        hidden_size: int = AE_HIDDEN_SIZE,
        num_layers: int = AE_NUM_LAYERS,
        latent_dim: int = AE_LATENT_DIM,
        dropout: float = AE_DROPOUT,
    ) -> None:
        """Construct encoder and decoder modules with shared configuration."""
        super().__init__()
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = Decoder(
            output_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            latent_dim=latent_dim,
            dropout=dropout,
        )

    def encode(self, inputs: Tensor) -> Tensor:
        """Encode input sequences into latent vectors."""
        return self.encoder(inputs)

    def decode(self, latent: Tensor, sequence_length: int = SEQUENCE_LENGTH) -> Tensor:
        """Decode latent vectors into sequence reconstructions."""
        return self.decoder(latent, sequence_length=sequence_length)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return reconstruction and latent vectors for a batch of inputs."""
        latent = self.encode(inputs)
        reconstruction = self.decode(latent, sequence_length=inputs.size(1))
        return reconstruction, latent
