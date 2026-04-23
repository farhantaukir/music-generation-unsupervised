# Project Specifications — Unsupervised Neural Network for Multi-Genre Music Generation

**Course:** Neural Networks

---

## Goal

Build a deep unsupervised model that learns musical representations without explicit genre labels and generates novel music pieces across multiple genres (Classical, Jazz, Rock, Pop, Electronic).

---

## Music Representation

A music sequence is represented as:

```
X = {x1, x2, ..., xT}
```

where each `xt` is a symbolic event: note-on, note-off, velocity, or duration.

The model learns a generative distribution `pθ(X)` and samples `X̂ ~ pθ(X)`.

---

## Datasets

Use at least one of the following MIDI datasets:

| Dataset | Genre | Link |
|---|---|---|
| MAESTRO v2 | Classical Piano | https://magenta.tensorflow.org/datasets/maestro |
| Lakh MIDI | Multi-Genre | https://colinraffel.com/projects/lakh |
| Groove MIDI | Jazz / Drums | https://magenta.tensorflow.org/datasets/groove |

**Preprocessing pipeline:**
1. Convert MIDI to piano-roll or token-based representation
2. Normalize timing resolution (e.g. 16 steps per bar)
3. Segment sequences into fixed-length windows

---

## Tasks

### Task 1 — LSTM Autoencoder (Easy)

**Goal:** Reconstruct and generate short music sequences from a single genre.

**Model:**
- Encoder: `z = fϕ(X)`
- Decoder: `X̂ = gθ(z)`
- Loss: `L_AE = Σ ||xt - x̂t||²`

**Deliverables:**
- Autoencoder implementation code
- Reconstruction loss curve
- 5 generated MIDI samples

---

### Task 2 — Variational Autoencoder / VAE (Medium)

**Goal:** Generate diverse music across multiple genres.

**Model:**
- Latent distribution: `qϕ(z|X) = N(µ(X), σ(X))`
- Reparameterization: `z = µ + σ ⊙ ε, ε ~ N(0, I)`
- Loss: `L_VAE = L_recon + β * DKL(qϕ(z|X) || p(z))`

**Deliverables:**
- VAE code with KL-divergence loss
- Multi-genre generation outputs (8 samples)
- Latent interpolation experiment
- Metric comparison vs Task 1

---

### Task 3 — Transformer-Based Generator (Hard)

**Goal:** Generate long coherent music sequences autoregressively.

**Model:**
- Autoregressive: `p(X) = Π p(xt | x<t)`
- Training loss: `L_TR = -Σ log pθ(xt | x<t)`
- Perplexity: `exp(1/T * L_TR)`
- Genre embedding: `ht = Emb(xt) + Emb(genre)`

**Deliverables:**
- Transformer architecture implementation
- Perplexity evaluation report
- 10 long-sequence generated compositions
- Baseline comparison results

---

### Task 4 — RLHF Human Preference Tuning (Advanced)

**Goal:** Fine-tune the generator using human feedback scores.

**Model:**
- Generate: `X_gen ~ pθ(X)`
- Reward: `r = HumanScore(X_gen)`
- Objective: `max_θ E[r(X_gen)]`
- Policy gradient: `∇θ J(θ) = E[r ∇θ log pθ(X)]`

**Deliverables:**
- Human listening survey dataset (minimum 10 participants)
- Reward model or scoring function
- RL fine-tuned generator outputs (10 samples)
- Before vs after comparison

---

## Evaluation Metrics

| Metric | Formula |
|---|---|
| Pitch Histogram Similarity | `H(p,q) = Σ |pi - qi|` (over 12 pitch classes) |
| Rhythm Diversity Score | `D = #unique durations / #total notes` |
| Repetition Ratio | `R = #repeated patterns / #total patterns` |
| Human Listening Score | `Score ∈ [1, 5]` via survey |

---

## Baseline Models

Must compare against at least two:
- Random Note Generator (naive)
- Markov Chain Music Model
- MuseGAN (optional)

**Expected performance benchmarks:**

| Model | Loss | Perplexity | Rhythm Diversity | Human Score |
|---|---|---|---|---|
| Random Generator | — | — | Low | 1.1 |
| Markov Chain | — | — | Medium | 2.3 |
| Task 1: Autoencoder | 0.82 | — | Medium | 3.1 |
| Task 2: VAE | 0.65 | — | High | 3.8 |
| Task 3: Transformer | — | 12.5 | Very High | 4.4 |
| Task 4: RLHF-Tuned | — | 11.2 | Very High | 4.8 |

---

## Grading Rubric

| Category | Weight |
|---|---|
| Dataset preparation & preprocessing | 15% |
| Model implementation & stable training | 25% |
| Music generation quality | 20% |
| Baseline comparison | 15% |
| Evaluation metrics (quantitative + qualitative) | 15% |
| Final report & presentation | 10% |

---

## Submission Requirements

1. GitHub repository or ZIP source code
2. Generated MIDI samples (5–15 files)
3. Evaluation results and plots
4. Final report PDF (6–10 pages) using one of the provided LaTeX templates

**Report templates (choose one):**
- IEEE: https://www.overleaf.com/latex/templates/ieee-conference-template
- NeurIPS 2024: https://www.overleaf.com/latex/templates/neurips2024
- ICLR 2025: https://www.overleaf.com/latex/templates/template-for-iclr-2025-conference-submission
- ICML 2025: https://www.overleaf.com/latex/templates/icml2025-template
- AAAI: https://www.overleaf.com/latex/templates/aaai-press-latex-template

---

## Required Project Structure

```
music-generation-unsupervised/
├── README.md
├── requirements.txt
├── data/
│   ├── raw_midi/
│   ├── processed/
│   └── train_test_split/
├── notebooks/
│   ├── preprocessing.ipynb
│   └── baseline_markov.ipynb
├── src/
│   ├── config.py
│   ├── preprocessing/
│   │   ├── midi_parser.py
│   │   ├── tokenizer.py
│   │   └── piano_roll.py
│   ├── models/
│   │   ├── autoencoder.py
│   │   ├── vae.py
│   │   ├── transformer.py
│   │   └── diffusion.py
│   ├── training/
│   │   ├── train_ae.py
│   │   ├── train_vae.py
│   │   └── train_transformer.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── pitch_histogram.py
│   │   └── rhythm_score.py
│   └── generation/
│       ├── sample_latent.py
│       ├── generate_music.py
│       └── midi_export.py
├── outputs/
│   ├── generated_midis/
│   ├── plots/
│   └── survey_results/
└── report/
    ├── final_report.tex
    ├── architecture_diagrams/
    └── references.bib
```
