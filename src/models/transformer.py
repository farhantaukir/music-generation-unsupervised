"""Decoder-only Transformer model for Task 3 autoregressive music token generation."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.config import (
    TOKEN_PAD_ID,
    TOKEN_VOCAB_SIZE,
    TR_DROPOUT,
    TR_FF_DIM,
    TR_MODEL_DIM,
    TR_NUM_HEADS,
    TR_NUM_LAYERS,
)


class PositionalEncoding(nn.Module):
    """Provide sinusoidal positional encodings for token embeddings."""

    def __init__(self, d_model: int, dropout: float = TR_DROPOUT, max_len: int = 8192) -> None:
        """Precompute sinusoidal position vectors and initialize dropout."""
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Add positional encodings to token embeddings."""
        sequence_length = embeddings.size(1)
        embeddings = embeddings + self.pe[:, :sequence_length]
        return self.dropout(embeddings)


class MusicTransformer(nn.Module):
    """Implement a decoder-only Transformer for autoregressive token prediction."""

    def __init__(
        self,
        vocab_size: int = TOKEN_VOCAB_SIZE,
        d_model: int = TR_MODEL_DIM,
        nhead: int = TR_NUM_HEADS,
        num_layers: int = TR_NUM_LAYERS,
        dim_feedforward: int = TR_FF_DIM,
        dropout: float = TR_DROPOUT,
        pad_id: int = TOKEN_PAD_ID,
    ) -> None:
        """Construct embedding, positional encoding, decoder stack, and output projection."""
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, sequence_length: int, device: torch.device) -> Tensor:
        """Create a causal attention mask to block future token access."""
        mask = torch.triu(
            torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=device),
            diagonal=1,
        )
        return mask

    def forward(self, token_ids: Tensor) -> Tensor:
        """Predict next-token logits for each sequence position."""
        padding_mask = token_ids.eq(self.pad_id)
        causal_mask = self.generate_causal_mask(sequence_length=token_ids.size(1), device=token_ids.device)

        hidden = self.token_embedding(token_ids) * math.sqrt(self.d_model)
        hidden = self.position_encoding(hidden)
        hidden = self.decoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.output_projection(hidden)

    def generate(
        self,
        start_tokens: Tensor,
        max_new_tokens: int,
        eos_id: int,
        temperature: float = 1.0,
        top_k: int | None = 16,
    ) -> Tensor:
        """Autoregressively sample tokens from model logits."""
        generated = start_tokens

        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                top_values, top_indices = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))
                filtered_logits = torch.full_like(next_token_logits, float("-inf"))
                filtered_logits.scatter_(1, top_indices, top_values)
                next_token_logits = filtered_logits

            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat([generated, next_tokens], dim=1)

            if torch.all(next_tokens.squeeze(1).eq(eos_id)):
                break

        return generated
