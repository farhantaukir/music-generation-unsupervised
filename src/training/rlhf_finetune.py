"""Policy-gradient RLHF fine-tuning for Task 4 music generation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm

from src.config import (
    DEVICE,
    GRAD_CLIP_NORM,
    HUMAN_SCORE_MAX,
    HUMAN_SCORE_MIN,
    RLHF_EVALUATION_SAMPLES,
    RLHF_LEARNING_RATE,
    RLHF_MAX_NEW_TOKENS,
    RLHF_NUM_EPOCHS,
    RLHF_SAMPLES_PER_EPOCH,
    RLHF_TEMPERATURE,
    RLHF_TOP_K,
    RLHF_WEIGHT_DECAY,
    SEED,
    SURVEY_SCORES_PATH,
    TOKEN_BOS_ID,
    TOKEN_EOS_ID,
    TOKEN_NOTE_OFFSET,
    ensure_output_dirs,
    set_global_seed,
)
from src.evaluation.metrics import repetition_ratio, rhythm_diversity
from src.generation.generate_music import load_trained_transformer
from src.models.transformer import MusicTransformer
from src.preprocessing.tokenizer import token_to_pitch


def _apply_top_k_filter(logits: Tensor, top_k: int | None) -> Tensor:
    """Apply top-k filtering to logits before categorical sampling."""
    if top_k is None or top_k <= 0:
        return logits

    k = min(int(top_k), int(logits.size(-1)))
    top_values, top_indices = torch.topk(logits, k=k, dim=-1)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(1, top_indices, top_values)
    return filtered


def _extract_model_config(model: MusicTransformer) -> Dict[str, int | float]:
    """Extract serializable Transformer architecture settings from a model instance."""
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

    return {
        "vocab_size": int(model.vocab_size),
        "d_model": int(model.d_model),
        "nhead": nhead,
        "num_layers": int(num_layers),
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "pad_id": int(model.pad_id),
    }


def _token_sequence_to_pitch_runs(token_sequence: Sequence[int]) -> Tuple[List[int], List[float], List[int]]:
    """Convert note tokens into pitch list, run-length durations, and note-token list."""
    pitches: List[int] = []
    durations: List[float] = []
    note_tokens: List[int] = []

    current_pitch: int | None = None
    current_run = 0

    for token in token_sequence:
        token_id = int(token)
        if token_id == TOKEN_EOS_ID:
            break
        if token_id < TOKEN_NOTE_OFFSET:
            continue

        try:
            pitch = int(token_to_pitch(token_id))
        except ValueError:
            continue

        note_tokens.append(token_id)

        if current_pitch is None:
            current_pitch = pitch
            current_run = 1
            continue

        if pitch == current_pitch:
            current_run += 1
        else:
            pitches.append(current_pitch)
            durations.append(float(current_run))
            current_pitch = pitch
            current_run = 1

    if current_pitch is not None:
        pitches.append(current_pitch)
        durations.append(float(current_run))

    return pitches, durations, note_tokens


def _score_sequence_metrics(token_sequence: Sequence[int]) -> Dict[str, float]:
    """Compute lightweight token-based musical quality metrics for reward shaping."""
    pitches, durations, note_tokens = _token_sequence_to_pitch_runs(token_sequence)

    if not pitches:
        return {
            "pitch_diversity": 0.0,
            "rhythm_diversity": 0.0,
            "repetition_ratio": 1.0,
            "length_quality": 0.0,
            "quality": 0.0,
        }

    pitch_classes = {pitch % 12 for pitch in pitches}
    pitch_diversity = float(len(pitch_classes) / 12.0)

    rhythm_score = float(rhythm_diversity(durations))
    repetition = float(repetition_ratio(note_tokens, pattern_length=4))
    target_note_count = 64.0
    length_quality = float(min(len(note_tokens) / target_note_count, 1.0))

    repetition_quality = 1.0 - min(max(repetition, 0.0), 1.0)
    quality = (
        0.35 * pitch_diversity
        + 0.25 * rhythm_score
        + 0.25 * repetition_quality
        + 0.15 * length_quality
    )

    return {
        "pitch_diversity": pitch_diversity,
        "rhythm_diversity": rhythm_score,
        "repetition_ratio": repetition,
        "length_quality": length_quality,
        "quality": float(min(max(quality, 0.0), 1.0)),
    }


def _load_human_scores(survey_csv_path: Path | str | None) -> Dict[str, float | int | list[float]]:
    """Load numeric human listening scores from a survey CSV file if available."""
    if survey_csv_path is None:
        return {
            "scores": [],
            "rows": 0,
            "valid_scores": 0,
            "participants": 0,
        }

    csv_path = Path(survey_csv_path)
    if not csv_path.exists():
        return {
            "scores": [],
            "rows": 0,
            "valid_scores": 0,
            "participants": 0,
        }

    scores: list[float] = []
    participants: set[str] = set()
    row_count = 0

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {
                "scores": [],
                "rows": 0,
                "valid_scores": 0,
                "participants": 0,
            }

        normalized = {name.lower().strip(): name for name in reader.fieldnames}
        score_field = None
        for candidate in ("score", "rating", "human_score"):
            if candidate in normalized:
                score_field = normalized[candidate]
                break

        participant_field = None
        for candidate in ("participant_id", "participant", "rater", "user"):
            if candidate in normalized:
                participant_field = normalized[candidate]
                break

        if score_field is None:
            return {
                "scores": [],
                "rows": 0,
                "valid_scores": 0,
                "participants": 0,
            }

        for row in reader:
            row_count += 1
            if participant_field:
                participant_value = str(row.get(participant_field, "")).strip()
                if participant_value:
                    participants.add(participant_value)

            raw_score = str(row.get(score_field, "")).strip()
            if not raw_score:
                continue

            try:
                score = float(raw_score)
            except ValueError:
                continue

            if HUMAN_SCORE_MIN <= score <= HUMAN_SCORE_MAX:
                scores.append(score)

    return {
        "scores": scores,
        "rows": row_count,
        "valid_scores": len(scores),
        "participants": len(participants),
    }


def _build_reward_scale(human_scores: Sequence[float]) -> float:
    """Convert survey score distribution into a scalar reward calibration factor."""
    if not human_scores:
        return 1.0

    midpoint = (HUMAN_SCORE_MIN + HUMAN_SCORE_MAX) / 2.0
    mean_score = float(np.mean(human_scores))
    scale = mean_score / max(midpoint, 1e-6)
    return float(np.clip(scale, 0.5, 1.5))


def _score_token_sequence(token_sequence: Sequence[int], reward_scale: float = 1.0) -> Tuple[float, Dict[str, float]]:
    """Compute a bounded reward score and metrics from one generated token sequence."""
    metrics = _score_sequence_metrics(token_sequence)

    base_reward = HUMAN_SCORE_MIN + (HUMAN_SCORE_MAX - HUMAN_SCORE_MIN) * metrics["quality"]
    calibrated_reward = float(np.clip(base_reward * reward_scale, HUMAN_SCORE_MIN, HUMAN_SCORE_MAX))
    return calibrated_reward, metrics


def _compute_sequence_log_prob(
    model: MusicTransformer,
    token_sequence: Tensor,
    temperature: float = RLHF_TEMPERATURE,
    top_k: int | None = RLHF_TOP_K,
) -> Tensor:
    """Compute cumulative log-probability of a sampled sequence under current policy."""
    if token_sequence.ndim != 1:
        token_sequence = token_sequence.view(-1)

    if token_sequence.numel() < 2:
        return torch.zeros((), dtype=torch.float32, device=DEVICE)

    input_tokens = token_sequence[:-1].unsqueeze(0).to(device=DEVICE, dtype=torch.long)
    target_tokens = token_sequence[1:].unsqueeze(0).to(device=DEVICE, dtype=torch.long)

    logits = model(input_tokens) / max(float(temperature), 1e-5)
    if top_k is not None and top_k > 0:
        batch, seq_len, vocab_size = logits.shape
        reshaped = logits.reshape(batch * seq_len, vocab_size)
        reshaped = _apply_top_k_filter(reshaped, top_k)
        logits = reshaped.reshape(batch, seq_len, vocab_size)

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum()


def _sample_sequence_for_evaluation(
    model: MusicTransformer,
    max_new_tokens: int,
    temperature: float = RLHF_TEMPERATURE,
    top_k: int | None = RLHF_TOP_K,
) -> Tensor:
    """Sample one sequence without gradients for policy evaluation."""
    start_tokens = torch.tensor([[TOKEN_BOS_ID]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        generated = model.generate(
            start_tokens=start_tokens,
            max_new_tokens=max_new_tokens,
            eos_id=TOKEN_EOS_ID,
            temperature=temperature,
            top_k=top_k,
        )
    return generated.squeeze(0)


def _evaluate_policy(
    model: MusicTransformer,
    num_samples: int,
    max_new_tokens: int,
    reward_scale: float,
    temperature: float = RLHF_TEMPERATURE,
    top_k: int | None = RLHF_TOP_K,
) -> Dict[str, float]:
    """Estimate mean reward and quality metrics from sampled policy rollouts."""
    rewards: list[float] = []
    pitch_diversities: list[float] = []
    rhythm_scores: list[float] = []
    repetition_scores: list[float] = []

    model.eval()
    for _ in range(num_samples):
        sequence = _sample_sequence_for_evaluation(
            model=model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        reward, metrics = _score_token_sequence(sequence.tolist(), reward_scale=reward_scale)

        rewards.append(reward)
        pitch_diversities.append(metrics["pitch_diversity"])
        rhythm_scores.append(metrics["rhythm_diversity"])
        repetition_scores.append(metrics["repetition_ratio"])

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_pitch_diversity": float(np.mean(pitch_diversities)) if pitch_diversities else 0.0,
        "mean_rhythm_diversity": float(np.mean(rhythm_scores)) if rhythm_scores else 0.0,
        "mean_repetition_ratio": float(np.mean(repetition_scores)) if repetition_scores else 0.0,
    }


def _save_before_after_plot(before_stats: Dict[str, float], after_stats: Dict[str, float], output_path: Path | str) -> Path:
    """Save a Task 4 before-vs-after metric comparison figure to disk."""
    plot_path = Path(output_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Reward", "PitchDiv", "RhythmDiv", "Repetition"]
    before_values = [
        before_stats["mean_reward"],
        before_stats["mean_pitch_diversity"],
        before_stats["mean_rhythm_diversity"],
        before_stats["mean_repetition_ratio"],
    ]
    after_values = [
        after_stats["mean_reward"],
        after_stats["mean_pitch_diversity"],
        after_stats["mean_rhythm_diversity"],
        after_stats["mean_repetition_ratio"],
    ]

    positions = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(10, 5))
    plt.bar(positions - width / 2, before_values, width=width, label="Before RLHF")
    plt.bar(positions + width / 2, after_values, width=width, label="After RLHF")
    plt.xticks(positions, labels)
    plt.ylabel("Metric Value")
    plt.title("Task 4 - Before vs After RLHF")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return plot_path


def rlhf_finetune_transformer(
    base_checkpoint_path: Path | str,
    survey_csv_path: Path | str | None = SURVEY_SCORES_PATH,
    num_epochs: int = RLHF_NUM_EPOCHS,
    samples_per_epoch: int = RLHF_SAMPLES_PER_EPOCH,
    learning_rate: float = RLHF_LEARNING_RATE,
    weight_decay: float = RLHF_WEIGHT_DECAY,
    max_new_tokens: int = RLHF_MAX_NEW_TOKENS,
    temperature: float = RLHF_TEMPERATURE,
    top_k: int | None = RLHF_TOP_K,
    evaluation_samples: int = RLHF_EVALUATION_SAMPLES,
) -> Dict[str, object]:
    """Fine-tune Task 3 Transformer with policy gradients and survey-calibrated rewards."""
    set_global_seed(SEED)
    output_dirs = ensure_output_dirs()

    base_path = Path(base_checkpoint_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base Task 3 checkpoint not found: {base_path}")

    model = load_trained_transformer(checkpoint_path=base_path, device=DEVICE)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    survey_info = _load_human_scores(survey_csv_path)
    survey_scores = survey_info["scores"] if isinstance(survey_info["scores"], list) else []
    reward_scale = _build_reward_scale(survey_scores)

    before_stats = _evaluate_policy(
        model=model,
        num_samples=evaluation_samples,
        max_new_tokens=max_new_tokens,
        reward_scale=reward_scale,
        temperature=temperature,
        top_k=top_k,
    )

    history_rewards: list[float] = []
    history_losses: list[float] = []
    best_reward = -float("inf")
    checkpoint_path = output_dirs["checkpoints"] / "task4_best_rlhf_transformer.pt"

    epoch_iterator = tqdm(range(1, num_epochs + 1), desc="Task 4 RLHF Training")
    for epoch in epoch_iterator:
        model.eval()

        sampled_sequences: list[Tensor] = []
        rewards: list[float] = []

        for _ in range(samples_per_epoch):
            generated_tokens = _sample_sequence_for_evaluation(
                model=model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            reward_value, _ = _score_token_sequence(generated_tokens.tolist(), reward_scale=reward_scale)
            sampled_sequences.append(generated_tokens.detach())
            rewards.append(reward_value)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        reward_std = rewards_tensor.std(unbiased=False)
        advantages = (rewards_tensor - rewards_tensor.mean()) / (reward_std + 1e-6)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        per_sample_losses: list[float] = []
        for sequence, advantage in zip(sampled_sequences, advantages):
            sequence_log_prob = _compute_sequence_log_prob(
                model=model,
                token_sequence=sequence,
                temperature=temperature,
                top_k=top_k,
            )
            sample_loss = -(advantage.detach() * sequence_log_prob)
            per_sample_losses.append(float(sample_loss.detach().item()))
            (sample_loss / max(len(sampled_sequences), 1)).backward()

        clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        policy_loss_value = float(np.mean(per_sample_losses)) if per_sample_losses else 0.0

        epoch_reward = float(np.mean(rewards)) if rewards else 0.0
        history_rewards.append(epoch_reward)
        history_losses.append(policy_loss_value)

        if epoch_reward > best_reward:
            best_reward = epoch_reward
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mean_reward": epoch_reward,
                    "policy_loss": policy_loss_value,
                    "num_epochs": num_epochs,
                    "samples_per_epoch": samples_per_epoch,
                    "max_new_tokens": max_new_tokens,
                    "temperature": float(temperature),
                    "top_k": int(top_k) if top_k is not None else None,
                    "survey_csv_path": str(survey_csv_path) if survey_csv_path is not None else "",
                    "reward_scale": float(reward_scale),
                    "model_config": _extract_model_config(model),
                },
                checkpoint_path,
            )

        epoch_iterator.set_postfix(
            reward=f"{epoch_reward:.3f}",
            policy_loss=f"{policy_loss_value:.4f}",
            best_reward=f"{best_reward:.3f}",
        )

    best_model = load_trained_transformer(checkpoint_path=checkpoint_path, device=DEVICE)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after_stats = _evaluate_policy(
        model=best_model,
        num_samples=evaluation_samples,
        max_new_tokens=max_new_tokens,
        reward_scale=reward_scale,
        temperature=temperature,
        top_k=top_k,
    )

    comparison_plot_path = _save_before_after_plot(
        before_stats=before_stats,
        after_stats=after_stats,
        output_path=output_dirs["plots"] / "task4_before_after_comparison.png",
    )

    return {
        "checkpoint_path": str(checkpoint_path),
        "comparison_plot_path": str(comparison_plot_path),
        "history_rewards": history_rewards,
        "history_losses": history_losses,
        "before_stats": before_stats,
        "after_stats": after_stats,
        "survey_rows": int(survey_info["rows"]),
        "valid_scores": int(survey_info["valid_scores"]),
        "participants": int(survey_info["participants"]),
        "reward_scale": float(reward_scale),
    }


def run_finetuning(
    base_checkpoint_path: Path | str | None = None,
    survey_csv_path: Path | str | None = SURVEY_SCORES_PATH,
    num_epochs: int = RLHF_NUM_EPOCHS,
    samples_per_epoch: int = RLHF_SAMPLES_PER_EPOCH,
    learning_rate: float = RLHF_LEARNING_RATE,
    weight_decay: float = RLHF_WEIGHT_DECAY,
    max_new_tokens: int = RLHF_MAX_NEW_TOKENS,
    temperature: float = RLHF_TEMPERATURE,
    top_k: int | None = RLHF_TOP_K,
    evaluation_samples: int = RLHF_EVALUATION_SAMPLES,
) -> Dict[str, object]:
    """Run end-to-end Task 4 RLHF fine-tuning from Task 3 checkpoint defaults."""
    output_dirs = ensure_output_dirs()
    default_checkpoint = output_dirs["checkpoints"] / "task3_best_transformer.pt"
    checkpoint = Path(base_checkpoint_path) if base_checkpoint_path is not None else default_checkpoint

    return rlhf_finetune_transformer(
        base_checkpoint_path=checkpoint,
        survey_csv_path=survey_csv_path,
        num_epochs=num_epochs,
        samples_per_epoch=samples_per_epoch,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        evaluation_samples=evaluation_samples,
    )


if __name__ == "__main__":
    results = run_finetuning()
    print("Task 4 RLHF finished.")
    print(f"Checkpoint: {results['checkpoint_path']}")
    print(f"Comparison plot: {results['comparison_plot_path']}")
    print(f"Before mean reward: {results['before_stats']['mean_reward']:.3f}")
    print(f"After mean reward: {results['after_stats']['mean_reward']:.3f}")
