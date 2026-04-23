"""Central configuration for data paths, hyperparameters, and runtime behavior."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = 42

KAGGLE_DATASET_PATH = Path("/kaggle/input/datasets/jackvial/themaestrodatasetv2")
KAGGLE_OUTPUT_PATH = Path("/kaggle/working/outputs")

LOCAL_DATA_ROOT = PROJECT_ROOT / "data" / "raw_midi"
LOCAL_OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_TEST_SPLIT_DIR = PROJECT_ROOT / "data" / "train_test_split"

CHECKPOINT_DIR = LOCAL_OUTPUT_ROOT / "checkpoints"
PLOTS_DIR = LOCAL_OUTPUT_ROOT / "plots"
GENERATED_MIDI_DIR = LOCAL_OUTPUT_ROOT / "generated_midis"
SURVEY_RESULTS_DIR = LOCAL_OUTPUT_ROOT / "survey_results"

MIDI_MIN_PITCH = 21
MIDI_MAX_PITCH = 108
PITCH_DIM = MIDI_MAX_PITCH - MIDI_MIN_PITCH + 1

STEPS_PER_BAR = 16
BEATS_PER_BAR = 4
STEPS_PER_BEAT = STEPS_PER_BAR // BEATS_PER_BAR
DEFAULT_TEMPO_BPM = 120.0

SEQUENCE_LENGTH = 256
WINDOW_STEP = 128
MIN_NOTES_PER_WINDOW = 8

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 30
VALIDATION_SPLIT = 0.2
NUM_WORKERS = 0
GRAD_CLIP_NORM = 1.0

AE_INPUT_DIM = PITCH_DIM
AE_HIDDEN_SIZE = 256
AE_NUM_LAYERS = 2
AE_LATENT_DIM = 64
AE_DROPOUT = 0.2

VAE_INPUT_DIM = PITCH_DIM
VAE_HIDDEN_SIZE = 256
VAE_NUM_LAYERS = 2
VAE_LATENT_DIM = 64
VAE_DROPOUT = 0.2
VAE_BETA = 1e-3
VAE_NUM_SAMPLES = 8
VAE_INTERPOLATION_STEPS = 10

TOKEN_PAD_ID = 0
TOKEN_BOS_ID = 1
TOKEN_EOS_ID = 2
TOKEN_NOTE_OFFSET = 3
TOKEN_VOCAB_SIZE = TOKEN_NOTE_OFFSET + PITCH_DIM
TOKEN_SEQUENCE_LENGTH = 512

TR_MODEL_DIM = 256
TR_NUM_HEADS = 8
TR_NUM_LAYERS = 6
TR_FF_DIM = 1024
TR_DROPOUT = 0.1
TR_LABEL_SMOOTHING = 0.05
TR_NUM_SAMPLES = 10
TR_GENERATION_MAX_TOKENS = 512

DIFF_INPUT_DIM = PITCH_DIM
DIFF_SEQUENCE_LENGTH = SEQUENCE_LENGTH
DIFF_MODEL_DIM = 256
DIFF_TIME_EMBED_DIM = 128
DIFF_HIDDEN_DIM = 512
DIFF_NUM_TIMESTEPS = 1000
DIFF_BETA_START = 1e-4
DIFF_BETA_END = 2e-2
DIFF_DROPOUT = 0.1

RLHF_NUM_EPOCHS = 10
RLHF_SAMPLES_PER_EPOCH = 16
RLHF_LEARNING_RATE = 1e-5
RLHF_WEIGHT_DECAY = 1e-5
RLHF_TEMPERATURE = 1.0
RLHF_TOP_K = 16
RLHF_EVALUATION_SAMPLES = 10
RLHF_NUM_SAMPLES = 10
RLHF_MAX_NEW_TOKENS = TR_GENERATION_MAX_TOKENS
HUMAN_SCORE_MIN = 1.0
HUMAN_SCORE_MAX = 5.0
SURVEY_SCORES_PATH = SURVEY_RESULTS_DIR / "scores.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int = SEED) -> None:
    """Seed random, numpy, and torch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_root() -> Path:
    """Return Kaggle dataset path when available, otherwise local raw MIDI path."""
    if KAGGLE_DATASET_PATH.exists():
        return KAGGLE_DATASET_PATH
    return LOCAL_DATA_ROOT


def get_output_root() -> Path:
    """Return Kaggle output path inside notebooks, otherwise local outputs path."""
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return KAGGLE_OUTPUT_PATH
    return LOCAL_OUTPUT_ROOT


def ensure_output_dirs() -> Dict[str, Path]:
    """Create output directories and return their resolved paths."""
    output_root = get_output_root()
    checkpoints = output_root / "checkpoints"
    plots = output_root / "plots"
    generated_midis = output_root / "generated_midis"
    survey_results = output_root / "survey_results"

    for directory in (checkpoints, plots, generated_midis, survey_results):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "output_root": output_root,
        "checkpoints": checkpoints,
        "plots": plots,
        "generated_midis": generated_midis,
        "survey_results": survey_results,
    }
