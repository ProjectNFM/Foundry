# Tokenizer Follow-Up Experiments

This document describes a structured research plan to address the open questions
from the AJILE12 tokenizer ablation study (`notebooks/tokenizer_comparison.ipynb`).
The goal is to move from observational rankings to causal, falsifiable experiments
that isolate why certain tokenizer designs outperform others.

Progress is tracked inline with checkboxes.

---

## Table of contents

1. [Baseline results and open questions](#1-baseline-results-and-open-questions)
2. [Hardware and infrastructure](#2-hardware-and-infrastructure)
3. [Data regime](#3-data-regime)
4. [Question 1 — Temporal inductive bias (CWT vs CNN)](#4-question-1--temporal-inductive-bias-cwt-vs-cnn)
5. [Question 2 — Spatial routing mechanism (perchannel vs spatialsession)](#5-question-2--spatial-routing-mechanism-per_channel-vs-spatial_session)
6. [Question 3 — Cross-session generalization (LOSO)](#6-question-3--cross-session-generalization-loso)
7. [PR plan and execution order](#7-pr-plan-and-execution-order)
8. [Appendix — Architecture reference](#appendix--architecture-reference)

---

## 1. Baseline results and open questions

### 1.1 Results from the initial ablation

The initial tokenizer ablation evaluated 11 tokenizer configurations across two
tasks on the AJILE12 ECoG dataset, using **intrasession** splits, a single set
of shared hyperparameters (LR=6e-4, WD=0.09, effective batch 1024), and 2 folds.

**Behavior classification (val AUROC, mean ± std over 2 folds):**


| Tokenizer                          | Val AUROC       |
| ---------------------------------- | --------------- |
| perchannelcwtdim512                | 0.9095 ± 0.0112 |
| perchannelcwt                      | 0.9005 ± 0.0115 |
| perchannelresamplecnn              | 0.8909 ± 0.0063 |
| perchannelpertimepointlineardim512 | 0.8847 ± 0.0014 |
| spatialsessionpertimepointlinear   | 0.8709 ± 0.0049 |
| spatialsessioncwtcommon            | 0.8687 ± 0.0114 |
| spatialsessionpertimepointidentity | 0.8684 ± 0.0037 |
| spatialsessioncwt                  | 0.8591 ± 0.0140 |
| spatialsessioncwtdim512            | 0.8493 ± 0.0138 |
| spatialsessionresamplecnn          | 0.8390 ± 0.0068 |
| perchannelpertimepointlinear       | 0.8360 ± 0.0850 |


**Pose estimation (val R², fold 0 only):**


| Tokenizer                        | Val R² |
| -------------------------------- | ------ |
| perchannelresamplecnn            | 0.3237 |
| perchannelcwt                    | 0.3119 |
| spatialsessioncwtcommon          | 0.3065 |
| spatialsessionpertimepointlinear | 0.2258 |
| spatialsessionresamplecnn        | 0.2198 |
| perchannelpertimepointlinear     | 0.1989 |


### 1.2 Key observations

1. `**per_channel` consistently outperforms `spatial_session**` on behavior (3–6
  AUROC points). Pose margins are smaller.
2. **CWT-based tokenizers rank highest** on both tasks.
3. **Large train–val gaps** for `spatial_session` variants suggest overfitting.
4. **Shared hyperparameters** across architecturally different tokenizers may bias
  the comparison — tokenizers with vastly different parameter counts / token
   counts received the same LR/WD.
5. **Intrasession-only evaluation** proves within-session decoding but says nothing
  about cross-session or cross-subject generalization.

### 1.3 Validated limitations


| #   | Limitation                                        | Impact                                                                    |
| --- | ------------------------------------------------- | ------------------------------------------------------------------------- |
| L1  | Aliasing from `grid_sample` resampling            | Valid for `ResampleCNNEmbedding`; CWT resamples band-limited coefficients |
| L2  | CNN receptive field ≪ CWT                         | ~65 ms vs full 1 s window; backbone attention partially compensates       |
| L3  | Static spatial projection removes dynamic routing | `spatial_session` collapses channels before backbone cross-attention      |
| L5  | Shared hyperparameters                            | Direction of bias unknown without per-architecture sweeps                 |
| L6  | Intrasession-only evaluation                      | Cross-session / cross-subject conclusions cannot be drawn                 |


### 1.4 Research questions


| Q#  | Question                                                                                      |
| --- | --------------------------------------------------------------------------------------------- |
| Q1  | Is the CWT inherently superior, or did the CNN baseline fail due to aliasing + narrow RF?     |
| Q2  | Does `spatial_session` fail from information loss or from removing dynamic spatial attention? |
| Q3  | How robust are these strategies under cross-session (LOSO) evaluation?                        |


---

## 2. Hardware and infrastructure

### 2.1 Compute

All experiments run on a **single machine with 8× NVIDIA A100 GPUs** (80 GB each).

- The existing `local_gpu` launcher (`configs/hydra/launcher/local_gpu.yaml`,
implemented in `hydra_plugins/foundry_launcher/local_gpu_launcher.py`) spawns
one subprocess per sweep job, pinning each to a distinct GPU via
`CUDA_VISIBLE_DEVICES`.
- Up to **8 single-GPU jobs** can run concurrently per sweep.
- The `EffectiveBatchSizeCallback` (`foundry/training/callbacks.py`) auto-scales
the micro-batch size via a power-of-2 memory search, then sets
`accumulate_grad_batches` to hit the target effective batch size.
`per_channel` tokenizers produce ~12,800 tokens per sample (64 channels × 200
time tokens) vs 200 for `spatial_session`, so micro-batch sizes will differ
dramatically between architectures.

### 2.2 Training pipeline recap


| Component           | Location                                                  | Notes                                                      |
| ------------------- | --------------------------------------------------------- | ---------------------------------------------------------- |
| Entry point         | `main.py`                                                 | Hydra-driven; calls `_build_model_and_data`, `trainer.fit` |
| Model               | `foundry/models/poyo_eeg.py` (`POYOEEGModel`)             | Perceiver IO backbone + EEGTokenizer                       |
| Tokenizer           | `foundry/models/tokenizer.py` (`EEGTokenizer`)            | Composes `ChannelStrategy` + `TemporalEmbedding`           |
| Channel strategies  | `foundry/models/embeddings/channel/processors.py`         | `PerChannelStrategy`, `SpatialProjectionStrategy`          |
| Spatial projectors  | `foundry/models/embeddings/channel/spatial_projectors.py` | `SessionSpatialProjector`, `PerceiverSpatialProjector`     |
| Temporal embeddings | `foundry/models/embeddings/temporal/`                     | `CWTEmbedding`, `ResampleCNNEmbedding`, `PerTimepoint*`    |
| Training modules    | `foundry/training/task_modules.py`                        | `RegressionModule`, `ClassificationModule` (Lightning)     |
| Optimizer           | AdamW + CosineAnnealingLR                                 | Configured in `BaseMultitaskModule.configure_optimizers`   |
| Data                | `foundry/data/datamodules/ajile.py`                       | Wraps `brainsets.PetersonBruntonPoseTrajectory2022`        |
| Experiment configs  | `configs/experiment/tokenizer_explore/`                   |                                                            |


### 2.3 Current hyperparameter tuning infrastructure

- **Hydra multirun / sweeper** with `choice(...)` or comma-separated grids — the
only tuning mechanism currently wired up.
- **No Optuna integration** exists in the codebase.
- **No Lightning LR Finder** is hooked up in `main.py`.
- The `EffectiveBatchSizeCallback` handles batch-size search only.

---

## 3. Data regime

### 3.1 AJILE12 dataset structure

The AJILE12 dataset (Peterson & Brunton 2022, via the `brainsets` package)
contains:

- **12 subjects** (P01–P12)
- **55 total sessions** (~3–6 sessions per subject)
- **Variable ECoG channel counts** per subject (grid/strip/depth coverage varies)

The current `configs/data/ajile/multisess.yaml` lists **12 recording IDs — one
cherry-picked recording per subject**. `allsess.yaml` uses all 55 sessions
(`recording_ids: null`).

**Split types** (controlled by `split_type` in the data config):


| Split type     | Behavior                                                                    |
| -------------- | --------------------------------------------------------------------------- |
| `intrasession` | Stratified train/val/test within each recording (time-segment-level split)  |
| `intersession` | Whole sessions assigned to train/val/test per `fold_number`                 |
| `intersubject` | All sessions of a subject share the same train/val/test assignment per fold |


`**fold_number**`: The brainsets dataset defines 3 predefined folds (0, 1, 2) for
each split type.

### 3.2 Tier definitions

#### Tier 1 — Fast sweep subset (all 12 subjects, 1 session each)

**Purpose**: Hyperparameter sweeps, architecture debugging, sanity checks.

**Design**: Use the existing `configs/data/ajile/multisess.yaml` as-is. This
already provides **one recording per subject (12 total)**, which preserves
diversity across all 12 electrode grids while keeping the dataset small.

This is preferable to dropping subjects entirely because:

- It exposes the model to all 12 grid geometries (critical for spatial tokenizer
evaluation).
- `SessionSpatialProjector` needs `session_configs` entries for each session it
will encounter — using all 12 subjects means the projector vocabulary stays the
same between Tier 1 and Tier 2.
- Training on a subset of subjects would bias Tier 1 HP decisions toward those
particular electrode configurations.

**Config**: No new file needed — `configs/data/ajile/multisess.yaml` already
defines this subset with `split_type` set by the experiment config.

**Estimated training time per run** (single GPU, early stopping patience=20):
~30–90 min per tokenizer depending on token count. A Tier 1 sweep of N trials
across 8 GPUs completes in approximately `ceil(N / 8) × 90 min` wall-clock.

#### Tier 2 — Full multi-session evaluation (all 12 subjects, all sessions)

**Purpose**: Final head-to-head comparisons; LOSO generalization experiments.

**Design**: Use `configs/data/ajile/allsess.yaml` which loads all 55 sessions
(`recording_ids: null`, `split_type: intersession`).

**Engineering required**:

- Create a new experiment config for Tier 2 head-to-heads
(`configs/experiment/tokenizer_explore/tier2_headtohead_pose.yaml` and
`..._behavior.yaml`).
- Verify that `allsess.yaml` correctly discovers all 55 H5 files in
`./data/processed/peterson_brunton_pose_trajectory_2022/`. Run a quick
data-loading smoke test: `uv run python -c "from foundry.data.datamodules.ajile import AjileDataModule; ..."`.
- Verify that `_populate_data_driven_hyperparams` in `main.py` correctly
builds `session_configs` for all 55 sessions (each with its own channel
count).
- Estimate GPU-hours per Tier 2 run and update time budgets below.

**Estimated training time per run**: ~2–6 hours per tokenizer (single GPU) with
55 sessions. With 8 GPUs, a Tier 2 head-to-head of 2 tokenizers × 3 folds = 6
runs completes in ~6 hours wall-clock.

### 3.3 Data validation checklist

Before running any experiment:

- Confirm all 55 `.h5` files are present in `./data/processed/`.
- Run a Tier 1 training job to completion for both `per_channel_cwt` and
`spatial_session_cwt_common` to establish baseline wall-clock times on A100.
- Record per-subject channel counts from the `session_configs` output in the
training log (needed for receptive-field and parameter-count calculations).

---

## 4. Question 1 — Temporal inductive bias (CWT vs CNN)

### 4.1 Hypothesis

> The performance gap between CWT and ResampleCNN is an artifact of the baseline
> CNN design. Once the CNN is protected from aliasing and granted a receptive
> field comparable to the CWT (≥500 ms), it will achieve parity with or
> outperform the CWT at equivalent parameter count.

### 4.2 Current CNN architecture analysis

The `ResampleCNNEmbedding` (`foundry/models/embeddings/temporal/resample_cnn.py`)
has two stages:

1. **Resampling**: `F.grid_sample` with `mode="bilinear"` maps `(B, C, Max_T)` →
  `(B, C, target_time_tokens)` where `target_time_tokens=200`. Bilinear
   interpolation on 1-D is a tent filter: **not** an ideal low-pass. Its
   frequency response has non-zero sidelobes and only -6 dB attenuation at the
   new Nyquist. For a 500 Hz signal resampled to 200 tokens in 1 s, frequencies
   above ~100 Hz may alias.
2. **Conv1d stack**: 2 layers, `kernel_size=7`, same-padded, with GELU
  activation. The effective receptive field is `1 + 2 × (7-1) = 13 tokens`. At
   200 tokens/s this is **65 ms**.

The CWT (`ContinuousCWTLayer` in `foundry/models/embeddings/temporal/cwt.py`)
uses learnable Morlet wavelets (9 frequencies, 2–100 Hz, `n_cycles≈7`) with
**FFT-based convolution** over the entire 1 s window. At the lowest frequency
(~2 Hz), the Gaussian envelope spans `σ = n_cycles / (2π·f) ≈ 0.56 s` — the CWT
sees the full trial. Resampling applies to the **already-band-limited** TF
coefficients, so aliasing is inherently less severe.

### 4.3 Required code changes

#### 4.3.1 Anti-aliased CNN embedding

**File**: `foundry/models/embeddings/temporal/resample_cnn.py`

**Option A — Low-pass filter before `grid_sample**`:

Insert a 1-D FIR (windowed-sinc) low-pass filter with cutoff at the target
Nyquist (`target_time_tokens / (2 × sequence_length)` Hz) before the existing
`grid_sample` call.

Steps:

- Implement a `_lowpass_filter` method on `ResampleCNNEmbedding` that:
  1. Computes a windowed-sinc kernel (Kaiser window, order ≈ 63–127 taps) for each
    batch item's sampling rate.
  2. Applies it via `F.conv1d` with appropriate padding (or FFT-based for long
    kernels).
- Call `_lowpass_filter` inside `forward()` before `self._resample(...)`.
- Add a `use_antialias: bool = True` constructor parameter so the old behavior
can be toggled for comparison.

**Option B — Replace `grid_sample` with learned strided Conv1d**:

Remove `_resample` entirely. Instead, use a strided `Conv1d` (or a sequence of
strided convolutions) to downsample from `Max_T` to `target_time_tokens`. The
learned kernel acts as an adaptive low-pass + downsample.

Steps:

- Add a `ResampleMode` enum or string parameter (`"grid_sample"` /
`"strided_conv"`) to `ResampleCNNEmbedding.__init_`_.
- Implement a `_strided_downsample` path that computes the required stride as
`ceil(Max_T / target_time_tokens)` and uses `Conv1d(in_channels, in_channels, kernel_size, stride)`.
- Handle variable-length inputs by masking post-convolution (same approach as
the existing `seq_lens` handling).

**Recommendation**: Implement **Option A** first — it is a smaller, more
controlled change that directly addresses limitation L1. Option B is a more
aggressive refactor and could be a follow-up if Option A does not close the gap.

Engineering challenge: the FIR filter cutoff depends on the per-item sampling  
rate (`input_sampling_rate`) and the target token count. Since different batch  
items may have different sampling rates, the filter must either operate  
per-item (loop) or assume a homogeneous batch (which the current code already  
does for patching).

#### 4.3.2 Receptive-field expansion

**File**: `foundry/models/embeddings/temporal/resample_cnn.py`

The target is ≥500 ms of receptive field. At 200 tokens/s, this is ≥100 tokens.


| Config                                       | RF (tokens) | RF (ms) |
| -------------------------------------------- | ----------- | ------- |
| Current: 2 layers, kernel=7                  | 13          | 65      |
| 4 layers, kernel=7                           | 25          | 125     |
| 6 layers, kernel=7                           | 37          | 185     |
| 4 layers, kernel=15                          | 57          | 285     |
| 6 layers, kernel=15                          | 85          | 425     |
| 8 layers, kernel=15                          | 113         | 565 ✓   |
| 4 layers, kernel=7, dilation=[1,2,4,8]       | 61          | 305     |
| 6 layers, kernel=7, dilation=[1,2,4,8,16,32] | 253         | 1265 ✓  |


**Dilated convolutions** are the most parameter-efficient path to a wide RF.

Steps:

- Add `dilations: list[int] | None = None` parameter to
`ResampleCNNEmbedding.__init__`. When provided, each `Conv1d` layer uses the
corresponding dilation factor. Padding becomes `dilation * (kernel_size // 2)`
for same-padding.
- Create new tokenizer configs:
  - `configs/model/tokenizer/per_channel_resample_cnn_aa.yaml` (anti-aliased,
  same layer/kernel as original)
  - `configs/model/tokenizer/per_channel_resample_cnn_widerf.yaml` (anti-aliased
    - dilated convolutions for ≥500 ms RF)
  - Mirror for `spatial_session_`* variants.
- Update tests in `tests/test_models/test_tokenizer.py` for the new
parameters (shape checks, gradient flow).

### 4.4 Per-architecture hyperparameter tuning

The initial ablation used a single set of hyperparameters for all tokenizers.
Different architectures (e.g. a 2-layer CNN with 64 filters vs a CWT with 9
learned frequencies) have very different parameter counts and optimization
landscapes. This step gives each architecture its own tuned HP set.

#### 4.4.1 What to sweep

For each tokenizer architecture, sweep over:


| Hyperparameter  | Search space                           | Notes                |
| --------------- | -------------------------------------- | -------------------- |
| `learning_rate` | `[1e-5, 3e-5, 1e-4, 3e-4, 6e-4, 1e-3]` | 6 values; log-spaced |
| `weight_decay`  | `[0.001, 0.01, 0.05, 0.09, 0.2]`       | 5 values             |


That is **30 combinations per architecture**. On Tier 1 with 8 GPUs and
~60 min/run, a full grid sweep for one architecture takes approximately
`ceil(30 / 8) × 60 min ≈ 4 hours` wall-clock.

With 4 tokenizer variants to sweep (CWT, ResampleCNN-original, ResampleCNN-AA,
ResampleCNN-wideRF) that is **~16 hours** wall-clock to sweep all Q1
architectures — or ~8 hours if we run two sweeps in parallel on 4 GPUs each.

To reduce this, consider either:

- Reducing the LR grid to 4 values and WD grid to 3 values (12 combos → ~2 hrs
per architecture).
- Using Optuna with a TPE sampler instead of grid search (see 4.4.2).

#### 4.4.2 Optuna integration (optional)

If grid search is too expensive, integrate Optuna via the
`hydra-optuna-sweeper` plugin.

Steps:

- Add `hydra-optuna-sweeper` to `pyproject.toml` dependencies.
- Create `configs/hydra/sweeper/optuna.yaml` with:
  ```yaml
  _target_: hydra_plugins.hydra_optuna_sweeper.OptunaSweeper
  direction: maximize
  n_trials: 15
  sampler:
    _target_: optuna.samplers.TPESampler
    seed: 42
  ```
- Create per-architecture sweep experiment configs that reference
`hydra/sweeper=optuna` and define the search space via
`hydra.sweeper.params`.

Engineering challenge: the `local_gpu` launcher and Optuna sweeper must work
together. The `local_gpu` launcher pins GPUs per job, so Optuna trials should
be submitted sequentially to the launcher's job queue. Verify this works by
running a 2-trial dry run on Tier 1.

### 4.5 Head-to-head evaluation (Tier 2)

After HP tuning on Tier 1, run final comparisons on Tier 2 (all 55 sessions).

**Runs**:

- `per_channel_cwt` (with its Tier-1-tuned HPs)
- `per_channel_resample_cnn` (original, with its Tier-1-tuned HPs)
- `per_channel_resample_cnn_aa` (anti-aliased, with its Tier-1-tuned HPs)
- `per_channel_resample_cnn_widerf` (anti-aliased + wide RF, with its Tier-1-tuned HPs)

Each tokenizer × 3 folds × 2 tasks = **24 runs** total.

With 8 GPUs and ~4 hrs/run, this takes approximately
`ceil(24 / 8) × 4 hrs = 12 hours` wall-clock.

Steps:

- Create `configs/experiment/tokenizer_explore/q1_tier2_pose.yaml` and
`q1_tier2_behavior.yaml` based on
`poyo_ajile_alltokenizers_matrix_pose.yaml`, but:
  - Override `data` to `ajile/allsess`.
  - Set `split_type: intersession`.
  - Sweep `fold_number: "0,1,2"`.
  - Per-tokenizer HP overrides via `hydra.sweeper.params`.
- Run the sweep and collect results into a new analysis notebook.

### 4.6 Interpretation criteria


| Outcome                                 | Conclusion                                                             |
| --------------------------------------- | ---------------------------------------------------------------------- |
| Refactored CNN ≈ CWT                    | CWT advantage was an artifact of aliasing + narrow RF                  |
| Refactored CNN < CWT (but gap narrowed) | Aliasing/RF contributed, but CWT's frequency decomposition adds value  |
| Refactored CNN ≈ original CNN           | Aliasing and RF were not the primary bottleneck; investigate elsewhere |


---

## 5. Question 2 — Spatial routing mechanism (perchannel vs spatialsession)

### 5.1 Hypothesis

> The failure of `spatial_session` is primarily driven by the loss of dynamic
> spatial routing. Forcing the Perceiver to operate on a static, globally-mixed
> spatial signal cripples its expressivity compared to `per_channel`, where the
> backbone can attend across all channel × time tokens.

### 5.2 Experiment 2A — Spatial MLP (testing nonlinearity)

**Question**: Does replacing the single `Linear` layer in
`SessionSpatialProjector` with a deeper MLP recover performance?

**Implementation**: `SessionSpatialProjector` already supports a `hidden_dim`
parameter. When `hidden_dim` is set, it constructs a per-session 2-layer MLP:
`Linear(num_channels, hidden_dim) → GELU → Linear(hidden_dim, num_sources)`.

Steps:

- Verify that the `hidden_dim` path in `SessionSpatialProjector.forward`
works end-to-end (create a quick test in
`tests/test_models/test_tokenizer.py`).
- Create `configs/model/tokenizer/spatial_session_mlp_cwt.yaml`:
  ```yaml
  # Same as spatial_session_cwt.yaml but with hidden_dim
  channel_strategy:
    projector:
      _target_: foundry.models.embeddings.SessionSpatialProjector
      session_configs: ${hyperparameters.session_configs}
      num_sources: ${model.tokenizer.channel_strategy.num_sources}
      hidden_dim: 128  # or sweep [64, 128, 256]
  ```
- Add regularization controls to `SessionSpatialProjector`:
  - Add `dropout: float = 0.0` parameter; apply `nn.Dropout(dropout)`
  after GELU in the MLP path.
  - Create test for dropout behavior.
- **Tier 1 HP sweep**: Sweep `hidden_dim` ∈ {64, 128, 256}, `dropout` ∈
{0.0, 0.1, 0.3}, LR and WD as in Q1.
  - With `hidden_dim` (3) × `dropout` (3) × LR (4) × WD (3) = 108 combos, this
  is expensive. Consider fixing LR/WD to the best values from the original
  ablation and sweeping only `hidden_dim` × `dropout` (9 combos → ~2 hrs with
  8 GPUs). Then do a targeted LR/WD sweep for the winning `hidden_dim`/`dropout`.
- **Tier 2 evaluation**: Train the best spatial MLP config on Tier 2 and
compare against baseline `spatial_session_cwt` and `per_channel_cwt`.

### 5.3 Experiment 2B — Perceiver Blindfold (testing dynamic routing)

**Question**: Is the Perceiver's spatial cross-attention the true driver of
`per_channel`'s advantage?

**Design**: Take the winning `per_channel_cwt` architecture and force a `mean()`
pool across the channel dimension immediately **before** tokens enter the
Perceiver encoder's cross-attention. This destroys per-channel identity while
keeping everything else identical.

Steps:

- Add a `collapse_channels: bool = False` parameter to `EEGTokenizer`:
  ```python
  # In EEGTokenizer.forward, after _reassemble_per_channel:
  if self.collapse_channels:
      # (B, C*N, D) → (B, N, D) via mean over channel groups
      tokens = tokens.view(B, C_pad, N, D).mean(dim=1)
      # Adjust input_mask and input_timestamps accordingly
  ```
  This is the most surgical intervention: the temporal embedding still processes
  each channel independently (preserving per-channel feature extraction), but the
  backbone only sees spatially-averaged tokens.
- Create `configs/model/tokenizer/per_channel_cwt_blindfold.yaml`:
same as `per_channel_cwt.yaml` with `collapse_channels: true`.
- Add unit tests verifying output shape changes correctly when
`collapse_channels=True`.
- **No HP sweep needed**: Use the exact same HPs as the winning
`per_channel_cwt` run to isolate the single variable (dynamic routing vs
not).
- **Tier 2 evaluation**: Train on Tier 2 with `intersession` splits, 3 folds.

### 5.4 Interpretation criteria


| Outcome (2A)                  | Outcome (2B)                  | Conclusion                                                                         |
| ----------------------------- | ----------------------------- | ---------------------------------------------------------------------------------- |
| MLP ≈ Linear (no improvement) | Blindfold crashes to baseline | Dynamic spatial attention is the true driver                                       |
| MLP > Linear (closes gap)     | Blindfold is fine             | Information loss was the bottleneck, not routing                                   |
| MLP > Linear                  | Blindfold crashes             | Both factors contribute; MLP adds capacity while backbone provides dynamic routing |


---

## 6. Question 3 — Cross-session generalization (LOSO)

### 6.1 Hypothesis

> Pure zero-shot spatial transfer will fail across heterogeneous ECoG grids.
> However, a structured "few-shot calibration" phase — updating only the spatial
> projector/embeddings — will recover strong performance, proving the backbone has
> learned generalizable neural representations.

### 6.2 Phase 1 — N-1 pretraining

**Design**: Select the best tokenizer combination from Q1 and Q2. Create a
Leave-One-Session-Out split using the brainsets `intersession` split mechanism.

Steps:

- Understand how `intersession` folds assign sessions. The brainsets dataset
stores `splits.intersession_fold_{fold_number}_assignment` per recording
(values: `train`, `valid`, `test`). With 3 folds and 55 sessions, this is
a 3-way partition, **not** a true LOSO (55-fold) scheme.
- **Option A (use existing folds)**: For practical compute budget, use the
existing 3-fold `intersession` split. This holds out ~18 sessions per fold
as test, with the remainder split into train/val. This is not LOSO but is
a reasonable first step for cross-session generalization.
- **Option B (true LOSO)**: Implement a custom LOSO splitter that iterates
over each of the 55 sessions as the held-out test set, training on the
remaining 54. This requires:
  - A new `split_type` in the `AjileDataModule` or a wrapper script that loops
  over `recording_ids` and programmatically excludes one per iteration.
  - 55 full training runs per tokenizer — likely infeasible for full training,
  but feasible for the calibration-only Phase 3 (see below).
- **Recommendation**: Start with Option A (3-fold intersession). If results
are promising, pursue Option B for the calibration experiments only.
- Create experiment config:
`configs/experiment/tokenizer_explore/q3_phase1_pretrain.yaml`
  - `data: ajile/allsess`
  - `split_type: intersession`
  - Sweep `fold_number: "0,1,2"`
  - Use Tier-1-tuned HPs from the winning tokenizer.

### 6.3 Phase 2 — Freeze backbone, initialize new spatial adapter

**Design**: After Phase 1 pretraining, freeze the backbone, temporal embeddings,
and readout heads. Initialize a fresh spatial adapter for the held-out session(s).

Steps:

- Implement a `FreezeBackboneCallback` (new file or extend
`foundry/training/callbacks.py`):
  ```python
  class FreezeBackboneCallback(L.Callback):
      """Freeze all parameters except those matching `trainable_patterns`."""
      def __init__(self, trainable_patterns: list[str]):
          self.trainable_patterns = trainable_patterns

      def on_fit_start(self, trainer, pl_module):
          for name, param in pl_module.named_parameters():
              if not any(fnmatch(name, p) for p in self.trainable_patterns):
                  param.requires_grad = False
  ```
- For `per_channel` tokenizers, the "spatial adapter" is the
`channel_emb` (`InfiniteVocabEmbedding`). The `trainable_patterns` would be
`["*channel_emb*"]`.
- For `spatial_session` tokenizers, the spatial adapter is the
`SessionSpatialProjector`'s per-session layers. The `trainable_patterns`
would be `["*session_layers*"]`. A new session layer must be added for the
held-out session's `session_id` — this requires calling
`initialize_vocabs` with the new session before calibration.
- Add checkpoint loading support: the experiment config needs a
`ckpt_path` field pointing to the Phase 1 best checkpoint. The existing
`_get_resume_checkpoint_path` in `main.py` handles this for SLURM restarts,
but we need a clean way to load a pretrained checkpoint for fine-tuning.
  - Add a `pretrained_checkpoint` field to the experiment config. In
  `main.py`, load the checkpoint weights into the model before
  `trainer.fit` (but do **not** resume the optimizer/scheduler state).
- Add tests for the freeze callback (verify parameter grad states).

### 6.4 Phase 3 — Few-shot calibration evaluation

**Design**: Train only the new spatial parameters on a small "calibration split"
(first 5% of trials from the held-out session). Evaluate on the remaining 95%.

Steps:

- Implement a calibration split mechanism. The simplest approach: add a
`calibration_fraction: float = 0.0` parameter to `AjileDataModule` that,
when > 0, uses the first N% of the held-out session's intervals as the
training set and the rest as validation.
  - This requires modifying the brainsets dataset's interval logic or
  post-filtering intervals in the datamodule's `setup()`.
- Use a **higher learning rate** for calibration (the spatial adapter is
training from scratch, not fine-tuning a deep network). Sweep
LR ∈ {1e-3, 3e-3, 1e-2} on Tier 1 with a few held-out sessions.
- Training is fast (only spatial params, small data) — estimate ~5–15 min
per calibration run. Even true LOSO (55 runs) is feasible:
`55 × 15 min / 8 GPUs ≈ 1.7 hours`.
- Create experiment config:
`configs/experiment/tokenizer_explore/q3_phase3_calibrate.yaml`
  - References Phase 1 checkpoint.
  - Adds `FreezeBackboneCallback` with appropriate patterns.
  - Overrides LR to calibration-appropriate values.

### 6.5 Interpretation criteria


| Phase 1 (zero-shot)           | Phase 3 (calibrated)              | Conclusion                                                                 |
| ----------------------------- | --------------------------------- | -------------------------------------------------------------------------- |
| Poor cross-session perf       | Strong recovery after calibration | Backbone learned generalizable features; spatial adapter is the bottleneck |
| Poor cross-session perf       | Still poor after calibration      | Backbone did not learn transferable representations                        |
| Reasonable cross-session perf | N/A                               | Tokenizer already handles cross-session variation                          |


---

## 7. PR plan and execution order

### Phase 0 — Infrastructure and data (prerequisite for all questions)


| #   | PR                            | Description                                                                                                | Depends on |
| --- | ----------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------- |
| 0.1 | **Data validation**           | Verify all 55 H5 files present; smoke-test `allsess.yaml` loading; record per-subject channel counts       | —          |
| 0.2 | **Tier 1 baseline timing**    | Run `per_channel_cwt` and `spatial_session_cwt_common` on Tier 1 to establish wall-clock baselines on A100 | 0.1        |
| 0.3 | **Tier 2 experiment configs** | Create `q1_tier2_pose.yaml`, `q1_tier2_behavior.yaml`, `q2_tier2_*.yaml`, `q3_phase1_pretrain.yaml`        | 0.1        |


### Phase 1 — Question 1 (Temporal)


| #   | PR                                         | Description                                                                                      | Depends on |
| --- | ------------------------------------------ | ------------------------------------------------------------------------------------------------ | ---------- |
| 1.1 | **Anti-aliased ResampleCNN**               | Add `use_antialias` param + FIR low-pass filter to `ResampleCNNEmbedding`; unit tests            | —          |
| 1.2 | **Dilated-conv receptive field expansion** | Add `dilations` param to `ResampleCNNEmbedding`; create `*_widerf` tokenizer configs; unit tests | 1.1        |
| 1.3 | **New tokenizer configs for Q1**           | `per_channel_resample_cnn_aa.yaml`, `per_channel_resample_cnn_widerf.yaml`, spatial mirrors      | 1.2        |
| 1.4 | **(Optional) Optuna sweeper integration**  | Add `hydra-optuna-sweeper` dep; create sweeper config; verify with `local_gpu` launcher          | —          |
| 1.5 | **Tier 1 HP sweep for Q1 architectures**   | Run LR × WD sweeps for CWT, original CNN, AA CNN, wideRF CNN on Tier 1                           | 1.3, 0.2   |
| 1.6 | **Tier 2 Q1 head-to-head**                 | Final comparison on all 55 sessions with per-architecture HPs; analysis notebook                 | 1.5, 0.3   |


### Phase 2 — Question 2 (Spatial)


| #   | PR                                       | Description                                                           | Depends on    |
| --- | ---------------------------------------- | --------------------------------------------------------------------- | ------------- |
| 2.1 | **Dropout for SessionSpatialProjector**  | Add `dropout` param to `SessionSpatialProjector` MLP path; unit tests | —             |
| 2.2 | **Spatial MLP tokenizer config**         | `spatial_session_mlp_cwt.yaml` with `hidden_dim` and `dropout`        | 2.1           |
| 2.3 | `**collapse_channels` for EEGTokenizer** | Implement channel-mean pooling before backbone; unit tests            | —             |
| 2.4 | **Blindfold tokenizer config**           | `per_channel_cwt_blindfold.yaml`                                      | 2.3           |
| 2.5 | **Tier 1 sweep for Exp 2A**              | Sweep `hidden_dim` × `dropout` (× optional LR/WD) on Tier 1           | 2.2, 0.2      |
| 2.6 | **Tier 2 Q2 evaluation**                 | Train Exp 2A best + Exp 2B on Tier 2; analysis notebook               | 2.4, 2.5, 0.3 |


### Phase 3 — Question 3 (LOSO)


| #   | PR                                | Description                                                      | Depends on                         |
| --- | --------------------------------- | ---------------------------------------------------------------- | ---------------------------------- |
| 3.1 | **FreezeBackboneCallback**        | Implement in `foundry/training/callbacks.py`; unit tests         | —                                  |
| 3.2 | **Pretrained checkpoint loading** | Add `pretrained_checkpoint` support to `main.py`                 | —                                  |
| 3.3 | **Calibration split mechanism**   | Add `calibration_fraction` to `AjileDataModule`                  | —                                  |
| 3.4 | **Q3 Phase 1 pretraining**        | Run N-1 pretraining with best tokenizer on Tier 2 intersession   | 3.1, 3.2, 0.3, Phase 1 & 2 results |
| 3.5 | **Q3 Phase 3 calibration**        | Run few-shot calibration with frozen backbone; analysis notebook | 3.3, 3.4                           |


### Execution order summary

```
Phase 0 ──────────────────────────────────────────────
  0.1 Data validation
  0.2 Tier 1 baseline timing
  0.3 Tier 2 experiment configs

Phase 1 (Q1 — Temporal) ─────────────────────────────
  1.1 Anti-aliased CNN
  1.2 Dilated-conv RF expansion
  1.3 New tokenizer configs
  1.4 (Optional) Optuna integration
  1.5 Tier 1 HP sweep
  1.6 Tier 2 head-to-head

Phase 2 (Q2 — Spatial) ──────── can start in parallel
  2.1 Dropout for spatial projector
  2.2 Spatial MLP config
  2.3 collapse_channels
  2.4 Blindfold config
  2.5 Tier 1 sweep for 2A
  2.6 Tier 2 evaluation

Phase 3 (Q3 — LOSO) ────────── depends on Q1+Q2 winner
  3.1 FreezeBackboneCallback
  3.2 Pretrained checkpoint loading
  3.3 Calibration split
  3.4 N-1 pretraining
  3.5 Few-shot calibration
```

---

## Appendix — Architecture reference

### Token counts per 1 s window


| Configuration                         | Tokens            |
| ------------------------------------- | ----------------- |
| `per_channel` + CWT/CNN (64 ch)       | 64 × 200 = 12,800 |
| `per_channel` + PerTimepoint (500 Hz) | 64 × 500 = 32,000 |
| `spatial_session` + CWT/CNN           | 200               |
| `spatial_session` + PerTimepoint      | 500               |


### ResampleCNN parameter counts


| Variant                          | numfilters | kernel | layers | Params (approx) |
| -------------------------------- | ---------- | ------ | ------ | --------------- |
| `spatial_session` (original)     | 288        | 7      | 2      | ~1.17 M         |
| `per_channel` (original)         | 64         | 7      | 2      | ~38 K           |
| `per_channel` (wide RF, dilated) | 64         | 7      | 6      | ~77 K           |


### CWT parameter counts


| Variant               | numfreqs | numsources | Params (approx)     |
| --------------------- | -------- | ---------- | ------------------- |
| `spatial_session_cwt` | 9        | 256        | ~1.18 M             |
| `per_channel_cwt`     | 9        | 1          | ~4.6 K + 256 linear |


### Backbone (fixed across all experiments)


| Parameter              | Value |
| ---------------------- | ----- |
| `embed_dim`            | 256   |
| `depth`                | 4     |
| `dim_head`             | 128   |
| `cross_heads`          | 8     |
| `self_heads`           | 8     |
| `ffn_dropout`          | 0.2   |
| `lin_dropout`          | 0.4   |
| `atn_dropout`          | 0.2   |
| `latent_step`          | 0.1   |
| `num_latents_per_step` | 16    |


