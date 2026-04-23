"""Generate MIDI samples from trained models across project tasks."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping

import numpy as np
import torch
from torch import Tensor

from src.config import (
    AE_LATENT_DIM,
    DEFAULT_TEMPO_BPM,
    DEVICE,
    RLHF_MAX_NEW_TOKENS,
    RLHF_NUM_SAMPLES,
    SEQUENCE_LENGTH,
    TOKEN_BOS_ID,
    TOKEN_EOS_ID,
    TOKEN_NOTE_OFFSET,
    TOKEN_PAD_ID,
    TOKEN_VOCAB_SIZE,
    TR_DROPOUT,
    TR_GENERATION_MAX_TOKENS,
    TR_NUM_HEADS,
    TR_NUM_SAMPLES,
    VAE_INTERPOLATION_STEPS,
    VAE_LATENT_DIM,
    VAE_NUM_SAMPLES,
    ensure_output_dirs,
)
from src.models.autoencoder import LSTMAutoencoder
from src.models.transformer import MusicTransformer
from src.models.vae import MusicVAE
from src.preprocessing.tokenizer import token_to_pitch


def _save_piano_roll_as_midi(
    piano_roll: np.ndarray,
    file_path: Path | str,
    tempo_bpm: float,
) -> Path:
    """Lazy-import MIDI export to avoid hard dependency during non-export flows."""
    from src.generation.midi_export import save_piano_roll_as_midi

    return save_piano_roll_as_midi(
        piano_roll=piano_roll,
        file_path=file_path,
        tempo_bpm=tempo_bpm,
    )


def load_trained_autoencoder(
    checkpoint_path: Path | str,
    device: torch.device = DEVICE,
) -> LSTMAutoencoder:
    """Load Task 1 model weights from a saved checkpoint path."""
    model = LSTMAutoencoder().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def sample_latent_vectors(
    num_samples: int,
    latent_dim: int = AE_LATENT_DIM,
    device: torch.device = DEVICE,
) -> Tensor:
    """Draw random Gaussian latent vectors for generation."""
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")
    return torch.randn(num_samples, latent_dim, device=device)


def decode_latent_vectors(
    model: LSTMAutoencoder,
    latent_vectors: Tensor,
    sequence_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Decode latent vectors into binary piano-roll arrays without gradients."""
    model.eval()
    with torch.no_grad():
        decoded = model.decode(latent_vectors, sequence_length=sequence_length)
        binary = (decoded >= 0.5).float()
    return binary.detach().cpu().numpy()


def generate_task1_samples(
    checkpoint_path: Path | str,
    num_samples: int = 5,
    output_dir: Path | str | None = None,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
) -> List[Path]:
    """Generate and save Task 1 MIDI samples from a trained checkpoint."""
    directories = ensure_output_dirs()
    base_dir = Path(output_dir) if output_dir is not None else directories["generated_midis"]
    base_dir.mkdir(parents=True, exist_ok=True)

    model = load_trained_autoencoder(checkpoint_path=checkpoint_path)
    latent_vectors = sample_latent_vectors(num_samples=num_samples)
    piano_rolls = decode_latent_vectors(model=model, latent_vectors=latent_vectors)

    generated_paths: List[Path] = []
    for sample_index in range(num_samples):
        file_path = base_dir / f"task1_sample_{sample_index + 1}.mid"
        _save_piano_roll_as_midi(
            piano_roll=piano_rolls[sample_index],
            file_path=file_path,
            tempo_bpm=tempo_bpm,
        )
        generated_paths.append(file_path)

    return generated_paths


def load_trained_vae(
    checkpoint_path: Path | str,
    device: torch.device = DEVICE,
) -> MusicVAE:
    """Load Task 2 VAE weights from a saved checkpoint path."""
    model = MusicVAE().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def decode_vae_latent_vectors(
    model: MusicVAE,
    latent_vectors: Tensor,
    sequence_length: int = SEQUENCE_LENGTH,
    binarize: bool = True,
    threshold: float = 0.5,
) -> np.ndarray:
    """Decode VAE latent vectors into piano-roll arrays without gradients."""
    model.eval()
    with torch.no_grad():
        decoded = model.decode(latent_vectors, sequence_length=sequence_length)
        if binarize:
            decoded = (decoded >= threshold).float()
    return decoded.detach().cpu().numpy()


def generate_task2_samples(
    checkpoint_path: Path | str,
    num_samples: int = VAE_NUM_SAMPLES,
    output_dir: Path | str | None = None,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
    latent_dim: int = VAE_LATENT_DIM,
) -> List[Path]:
    """Generate and save Task 2 MIDI samples from a trained VAE checkpoint."""
    directories = ensure_output_dirs()
    base_dir = Path(output_dir) if output_dir is not None else directories["generated_midis"]
    base_dir.mkdir(parents=True, exist_ok=True)

    model = load_trained_vae(checkpoint_path=checkpoint_path)
    latent_vectors = sample_latent_vectors(num_samples=num_samples, latent_dim=latent_dim)
    piano_rolls = decode_vae_latent_vectors(
        model=model,
        latent_vectors=latent_vectors,
        binarize=True,
    )

    generated_paths: List[Path] = []
    for sample_index in range(num_samples):
        file_path = base_dir / f"task2_sample_{sample_index + 1}.mid"
        _save_piano_roll_as_midi(
            piano_roll=piano_rolls[sample_index],
            file_path=file_path,
            tempo_bpm=tempo_bpm,
        )
        generated_paths.append(file_path)

    return generated_paths


def build_task2_latent_interpolation(
    checkpoint_path: Path | str,
    num_steps: int = VAE_INTERPOLATION_STEPS,
    sequence_length: int = SEQUENCE_LENGTH,
    latent_dim: int = VAE_LATENT_DIM,
) -> np.ndarray:
    """Decode linear latent interpolation between two random points into piano rolls."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2.")

    model = load_trained_vae(checkpoint_path=checkpoint_path)
    start = torch.randn(1, latent_dim, device=DEVICE)
    end = torch.randn(1, latent_dim, device=DEVICE)
    weights = torch.linspace(0.0, 1.0, steps=num_steps, device=DEVICE).unsqueeze(1)
    interpolation = start * (1.0 - weights) + end * weights
    return decode_vae_latent_vectors(
        model=model,
        latent_vectors=interpolation,
        sequence_length=sequence_length,
        binarize=False,
    )


def _extract_transformer_config_from_checkpoint(checkpoint: object) -> dict[str, int | float]:
    """Extract Transformer architecture settings saved inside checkpoint metadata."""
    if not isinstance(checkpoint, dict):
        return {}

    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, dict):
        return {}

    allowed_keys = (
        "vocab_size",
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "dropout",
        "pad_id",
    )
    extracted: dict[str, int | float] = {}
    for key in allowed_keys:
        value = raw_config.get(key)
        if value is not None:
            extracted[key] = value
    return extracted


def _guess_attention_heads(d_model: int) -> int:
    """Guess a sensible number of attention heads when metadata is unavailable."""
    if d_model % 32 == 0:
        inferred = d_model // 32
        if 1 <= inferred <= 16:
            return inferred

    if TR_NUM_HEADS > 0 and d_model % TR_NUM_HEADS == 0:
        return int(TR_NUM_HEADS)

    for candidate in (8, 6, 4, 3, 2, 1):
        if d_model % candidate == 0:
            return candidate
    return 1


def _infer_transformer_config_from_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, int | float]:
    """Infer Transformer architecture from checkpoint state_dict tensor shapes."""
    token_embedding = state_dict.get("token_embedding.weight")
    if token_embedding is None or token_embedding.ndim != 2:
        return {}

    vocab_size = int(token_embedding.shape[0])
    d_model = int(token_embedding.shape[1])

    layer_indices: set[int] = set()
    prefix = "decoder.layers."
    for key in state_dict:
        if key.startswith(prefix):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_indices.add(int(parts[2]))
    num_layers = int(max(layer_indices) + 1) if layer_indices else 0

    linear1_weight = state_dict.get("decoder.layers.0.linear1.weight")
    if linear1_weight is not None and linear1_weight.ndim == 2:
        dim_feedforward = int(linear1_weight.shape[0])
    else:
        dim_feedforward = int(d_model * 4)

    return {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "nhead": _guess_attention_heads(d_model),
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": float(TR_DROPOUT),
        "pad_id": int(TOKEN_PAD_ID),
    }


def load_trained_transformer(
    checkpoint_path: Path | str,
    device: torch.device = DEVICE,
    model_config: Mapping[str, int | float] | None = None,
) -> MusicTransformer:
    """Load Task 3 Transformer weights from a saved checkpoint path."""
    # Always deserialize checkpoints on CPU first to avoid VRAM spikes from
    # loading non-model tensors (for example optimizer state) onto CUDA.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    resolved_config = _extract_transformer_config_from_checkpoint(checkpoint)
    if not resolved_config and isinstance(state_dict, Mapping):
        resolved_config = _infer_transformer_config_from_state_dict(state_dict)

    if model_config is not None:
        for key, value in model_config.items():
            if value is not None:
                resolved_config[key] = value

    model = MusicTransformer(**resolved_config)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise RuntimeError(
            "Error loading Task 3 checkpoint into MusicTransformer. "
            f"Resolved model config: {resolved_config}. "
            "Pass exact training architecture via model_config in generate_task3_samples."
        ) from error

    model = model.to(device)
    model.eval()
    return model


def transformer_tokens_to_piano_roll(
    token_sequence: Tensor,
    sequence_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Convert generated Transformer note tokens into a binary piano-roll matrix."""
    from src.config import MIDI_MAX_PITCH, MIDI_MIN_PITCH

    pitch_dim = MIDI_MAX_PITCH - MIDI_MIN_PITCH + 1
    piano_roll = np.zeros((sequence_length, pitch_dim), dtype=np.float32)

    token_values = token_sequence.detach().cpu().tolist()
    current_time = 0
    for token in token_values:
        if token in (TOKEN_PAD_ID, TOKEN_BOS_ID):
            continue
        if token == TOKEN_EOS_ID:
            break
        if token >= TOKEN_NOTE_OFFSET:
            try:
                pitch = token_to_pitch(int(token))
                pitch_index = pitch - MIDI_MIN_PITCH
                if 0 <= current_time < sequence_length and 0 <= pitch_index < pitch_dim:
                    piano_roll[current_time, pitch_index] = 1.0
                    current_time += 1
            except ValueError:
                continue

    return piano_roll


def generate_task3_samples(
    checkpoint_path: Path | str,
    num_samples: int = TR_NUM_SAMPLES,
    max_new_tokens: int = TR_GENERATION_MAX_TOKENS,
    output_dir: Path | str | None = None,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
    model_config: Mapping[str, int | float] | None = None,
) -> List[Path]:
    """Generate and save Task 3 MIDI samples from a trained Transformer checkpoint."""
    directories = ensure_output_dirs()
    base_dir = Path(output_dir) if output_dir is not None else directories["generated_midis"]
    base_dir.mkdir(parents=True, exist_ok=True)

    model = load_trained_transformer(checkpoint_path=checkpoint_path, model_config=model_config)

    generated_paths: List[Path] = []
    for sample_index in range(num_samples):
        start_tokens = torch.tensor([[TOKEN_BOS_ID]], dtype=torch.long, device=DEVICE)
        generated_tokens = model.generate(
            start_tokens=start_tokens,
            max_new_tokens=max_new_tokens,
            eos_id=TOKEN_EOS_ID,
            temperature=1.0,
            top_k=16,
        )

        piano_roll = transformer_tokens_to_piano_roll(
            token_sequence=generated_tokens.squeeze(0),
            sequence_length=SEQUENCE_LENGTH,
        )
        file_path = base_dir / f"task3_sample_{sample_index + 1}.mid"
        _save_piano_roll_as_midi(
            piano_roll=piano_roll,
            file_path=file_path,
            tempo_bpm=tempo_bpm,
        )
        generated_paths.append(file_path)

    return generated_paths


def generate_task4_samples(
    checkpoint_path: Path | str,
    num_samples: int = RLHF_NUM_SAMPLES,
    max_new_tokens: int = RLHF_MAX_NEW_TOKENS,
    output_dir: Path | str | None = None,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
    model_config: Mapping[str, int | float] | None = None,
) -> List[Path]:
    """Generate and save Task 4 MIDI samples from an RLHF-tuned Transformer checkpoint."""
    directories = ensure_output_dirs()
    base_dir = Path(output_dir) if output_dir is not None else directories["generated_midis"]
    base_dir.mkdir(parents=True, exist_ok=True)

    model = load_trained_transformer(checkpoint_path=checkpoint_path, model_config=model_config)

    generated_paths: List[Path] = []
    for sample_index in range(num_samples):
        start_tokens = torch.tensor([[TOKEN_BOS_ID]], dtype=torch.long, device=DEVICE)
        generated_tokens = model.generate(
            start_tokens=start_tokens,
            max_new_tokens=max_new_tokens,
            eos_id=TOKEN_EOS_ID,
            temperature=1.0,
            top_k=16,
        )

        piano_roll = transformer_tokens_to_piano_roll(
            token_sequence=generated_tokens.squeeze(0),
            sequence_length=SEQUENCE_LENGTH,
        )
        file_path = base_dir / f"task4_sample_{sample_index + 1}.mid"
        _save_piano_roll_as_midi(
            piano_roll=piano_roll,
            file_path=file_path,
            tempo_bpm=tempo_bpm,
        )
        generated_paths.append(file_path)

    return generated_paths
