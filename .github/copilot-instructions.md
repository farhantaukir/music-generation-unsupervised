# Copilot Instructions — Music Generation Project

## Context

This is a university Neural Networks project. The workflow is split across two environments:

- **VS Code** — write, organize, and maintain all source code
- **Kaggle Notebooks** — run all training, preprocessing, and generation (free T4 GPU)

Code is written locally in VS Code and copied into Kaggle notebook cells to execute. All model outputs (trained weights, MIDI files, plots) are downloaded from Kaggle and saved into the local `outputs/` folder.

---

## Environment Facts

- **Runtime:** Kaggle (Python 3.14, PyTorch, CUDA T4 GPU)
- **Dataset path on Kaggle:** `/kaggle/input/datasets/jackvial/themaestrodatasetv2`
- **Output path on Kaggle:** `/kaggle/working/outputs`
- **Local machine:** Windows 10, no GPU, Python venv in project root
- **Do not** generate code that requires local GPU or assumes local training

---

## Code Style Rules

- All hyperparameters must live in `src/config.py` — never hardcode values inline
- Every source file must have a module-level docstring explaining its purpose
- Use `tqdm` for all loops over files or training epochs
- Use `torch.no_grad()` for all inference and validation passes
- Always seed: `random`, `numpy`, and `torch` with `SEED` from config
- Save the best model checkpoint based on validation loss, not final epoch
- Functions must have a one-line docstring describing input → output

---

## File Responsibilities

| File | Responsibility |
|---|---|
| `src/config.py` | All hyperparameters, paths, device setup |
| `src/preprocessing/piano_roll.py` | MIDI → piano roll, windowing |
| `src/preprocessing/midi_parser.py` | MIDI file loading, metadata extraction |
| `src/preprocessing/tokenizer.py` | Event-based tokenization for Transformer |
| `src/models/autoencoder.py` | Encoder, Decoder, LSTMAutoencoder classes |
| `src/models/vae.py` | VAE with reparameterization trick |
| `src/models/transformer.py` | Transformer decoder for Task 3 |
| `src/training/train_ae.py` | Training loop for Task 1 |
| `src/training/train_vae.py` | Training loop for Task 2 |
| `src/training/train_transformer.py` | Training loop for Task 3 |
| `src/evaluation/metrics.py` | Pitch histogram, rhythm diversity, repetition ratio |
| `src/generation/midi_export.py` | Piano roll → MIDI file conversion |
| `src/generation/generate_music.py` | Latent sampling and music generation |
| `notebooks/` | Full runnable Kaggle notebooks, one per task |

---

## Task Build Order

Complete tasks strictly in this order. Each task builds on the previous one.

```
Task 1 (LSTM Autoencoder)
    → Task 2 (VAE, extends Task 1 encoder/decoder pattern)
        → Task 3 (Transformer, new architecture)
            → Task 4 (RLHF, fine-tunes Task 3 output)
```

Do not start a task until the previous task's deliverables are confirmed working.

---

## Per-Task Deliverables Checklist

### Task 1
- [ ] `src/models/autoencoder.py` — Encoder, Decoder, LSTMAutoencoder
- [ ] `src/training/train_ae.py` — training + validation loop
- [ ] `outputs/plots/task1_loss_curve.png`
- [ ] `outputs/generated_midis/task1_sample_{1-5}.mid`

### Task 2
- [ ] `src/models/vae.py` — VAE with KL divergence
- [ ] `src/training/train_vae.py`
- [ ] `outputs/plots/task2_loss_curve.png`
- [ ] `outputs/plots/task2_latent_interpolation.png`
- [ ] `outputs/generated_midis/task2_sample_{1-8}.mid`

### Task 3
- [ ] `src/preprocessing/tokenizer.py`
- [ ] `src/models/transformer.py`
- [ ] `src/training/train_transformer.py`
- [ ] `outputs/plots/task3_perplexity_curve.png`
- [ ] `outputs/generated_midis/task3_sample_{1-10}.mid`

### Task 4
- [ ] `src/training/rlhf_finetune.py`
- [ ] `outputs/survey_results/scores.csv`
- [ ] `outputs/generated_midis/task4_sample_{1-10}.mid`
- [ ] `outputs/plots/task4_before_after_comparison.png`

---

## Kaggle Notebook Structure

Each task gets its own Kaggle notebook. Structure every notebook as numbered cells in this order:

1. GPU check + package installs
2. All imports + config values (copied from `src/config.py`)
3. Data loading and preprocessing
4. Model definition (copied from the relevant `src/models/` file)
5. Training loop (copied from the relevant `src/training/` file)
6. Loss / metric plots — save to `/kaggle/working/outputs/plots/`
7. Music generation — save MIDIs to `/kaggle/working/outputs/midi/`
8. Summary cell printing all deliverables

After a notebook runs successfully on Kaggle, download it as `.ipynb` and save it to the local `notebooks/` folder.

---

## Evaluation Metrics

Implement all three in `src/evaluation/metrics.py`:

```python
# Pitch histogram similarity
H(p, q) = sum(|pi - qi|) for i in 12 pitch classes

# Rhythm diversity
D = unique_durations / total_notes

# Repetition ratio
R = repeated_patterns / total_patterns
```

The metrics module must be importable standalone — no dependency on model files.

---

## Baseline Models

Implement in `notebooks/baseline_markov.ipynb`:
- **Random generator** — uniform random note selection
- **Markov chain** — first-order transition matrix trained on MAESTRO

Both baselines must output MIDI files and be scored with the same metrics as Tasks 1–4 so the comparison table can be filled in directly.

---

## What to Avoid

- Do not use `display()` or `IPython` calls inside `src/` files — those are notebook-only
- Do not hardcode `/kaggle/...` paths inside `src/` files — paths live only in `config.py`
- Do not train anything locally — all training happens on Kaggle
- Do not install heavy packages in the local venv — keep it lightweight for editing only
- Do not mix Task 1 and Task 2 model classes in the same file
