# Foundry

Foundry is a modular brain data experimentation framework designed for flexible neuroscience research. It provides composable building blocks (tokenizers, embeddings, backbones, readouts) and keeps the core minimal so you can focus on your experiments instead of glue code.

**Use Foundry when you want to:**
- Train neural network models on neural data (e.g., EEG, iEEG)
- Experiment with different architectures and tokenization strategies
- Run sweeps across hyperparameters on local or cluster (SLURM) environments
- Log experiments and track results with Weights & Biases or CSV

---

## Quick start

Get up and running in 3 steps:

### 1. Install dependencies

```bash
# Requires Python ≥ 3.11
# Install uv if you don't have it: pip install uv

uv sync
```

This installs Foundry and all required dependencies. Several are downloaded from git (`brainsets`, `torch-brain`, `temporaldata`), so you need internet access.

### 2. Prepare your data

Place processed neural data in `./data/processed/` (or override the path in configs).

### 3. Set up environment variables (optional but recommended for logging)

Create a `.env` file in the project root with your credentials. See **Environment setup** section below for details.

### 4. Run a training experiment

```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep
```

That's it! Your model will train and log results to Weights & Biases by default (if credentials are set up). Outputs and checkpoints go to `./outputs` (or to the `SCRATCH` directory if set).

---

## Repository structure

```
foundry/
├── data/                    # Data loading: datasets and datamodules
│   ├── datasets/            # Raw dataset implementations
│   ├── datamodules/         # Lightning datamodules for train/val/test splits
│   └── transforms/          # Preprocessing and patching utilities
├── models/
│   ├── embeddings/          # Token/patch embedding layers
│   ├── backbones/           # Core model architectures (Perceiver, etc.)
│   └── poyo_eeg.py          # Reference neural/brain signal model implementation
├── training/                # Lightning modules and training logic
│   └── module.py            # Main training module for neural networks
├── tools/                   # Utility scripts
│   └── stage_data.py        # Copy/compress data for SLURM cluster
├── callbacks.py             # PyTorch Lightning callbacks
├── config_resolvers.py      # Custom Hydra config helpers
└── core.py                  # Shared utilities

configs/
├── config.yaml              # Root config (composes all groups below)
├── experiment/              # Pre-configured experiment combinations
│   ├── tokenizer_explore/            # Tokenizer-focused experiments
│   ├── neurosoft/                    # Neurosoft project experiments
│   └── ...
├── data/                    # Data configuration by dataset group
│   ├── ajile/
│   ├── neurosoft_minipigs/
│   ├── physionet/
│   └── ...
├── model/                   # Model configuration
├── module/                  # Training loop configuration
├── trainer/                 # PyTorch Lightning trainer settings
├── logger/                  # Logging backends (wandb, csv)
├── profiling/               # Performance profiling settings
└── hydra/launcher/          # Cluster job submission (SLURM)

main.py                       # Training entrypoint (Hydra + PyTorch Lightning)
profile_training.py           # Profiling entrypoint (same config system)
pyproject.toml               # Dependencies and project metadata
uv.lock                      # Locked dependency versions (for reproducibility)

tests/                        # Unit tests (pytest)
```

---

## Prerequisites

- **Python 3.11+** (required by PyTorch and core dependencies)
- **uv** (package manager; much faster than pip)
  - Install: `pip install uv`
  - Learn more: [astral.sh/uv](https://docs.astral.sh/uv/)
- **Internet access** (to download git-sourced dependencies during install)

---

## Environment setup

### Standard install (all you need to train models)

```bash
cd /path/to/Foundry
uv sync
```

This reads `pyproject.toml` and `uv.lock` to install Foundry and all dependencies in an isolated virtual environment. It's fast and reproducible.

### Development install (includes testing and linting tools)

```bash
uv sync --group dev
```

This adds `pytest`, `ruff`, and `pre-commit` for development workflows.

### Verify installation

```bash
uv run python -c "import foundry; print(foundry.__version__)"
```

If this succeeds, you're ready to go.

### Setting up Weights & Biases (WandB) for logging

By default, Foundry logs experiments to [Weights & Biases](https://wandb.ai/), a platform for tracking and visualizing machine learning experiments.

**Step 1: Create a WandB account**

1. Go to [https://wandb.ai/](https://wandb.ai/) and sign up for a free account
2. Once logged in, navigate to [https://wandb.ai/authorize](https://wandb.ai/authorize) to generate an API key

**Step 2: Create a `.env` file with your credentials**

Create a file named `.env` in the project root directory (same level as `main.py`) with the following content:

```bash
# .env file - NEVER share this publicly as it contains your API key!
WANDB_API_KEY=<your_api_key_from_wandb>
WANDB_ENTITY=<your_username_or_team_name>
```

Replace:
- `<your_api_key_from_wandb>` with your actual WandB API key
- `<your_username_or_team_name>` with your WandB username or team name

**⚠️ Important security note:** The `.env` file contains sensitive credentials and should **NEVER** be committed to git or publicly shared. It's already listed in `.gitignore`, but double-check if you use other version control systems.

**Step 3: Verify setup**

The first time you run Foundry, it will automatically read the `WANDB_API_KEY` from your `.env` file and authenticate. You should see your experiment appear on the [WandB dashboard](https://wandb.ai/).

**If you prefer not to use WandB**, you can disable it and use CSV logging instead:

```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep logger=csv
```

---

## How configuration works

Foundry uses **Hydra**, a configuration framework that lets you compose settings from YAML files without editing code. The config system works by merging defaults in layers:

1. **Root config** ([`configs/config.yaml`](configs/config.yaml)) defines the base structure with required groups:
   - `experiment` (mandatory) — the experiment name you must specify
   - `data`, `model`, `module`, `trainer`, `logger`, `profiling`, `hydra/launcher`

2. **Experiments** ([`configs/experiment/`](configs/experiment/)) combine multiple groups into a ready-to-run setup.
   - Example: `experiment=tokenizer_explore/poyo_ajile_sweep` picks `model=poyo_eeg`, `data=ajile/singlesess`, and SLURM launcher settings.

3. **Config groups** ([`configs/data/`](configs/data/), [`configs/model/`](configs/model/), etc.) define options within each category.

4. **Command-line overrides** let you tweak settings without editing files:
   ```bash
   uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep data.root=./my_data model.hidden_dim=256
   ```

### Key config groups

| Group | Location | Purpose |
|-------|----------|---------|
| `experiment` | `configs/experiment/` | Pre-composed experiment combinations (which data, model, logger, etc.) |
| `data` | `configs/data/` | Dataset and dataloader settings |
| `model` | `configs/model/` | Model architecture and hyperparameters |
| `module` | `configs/module/` | Training loop (optimizer, loss, learning rate) |
| `trainer` | `configs/trainer/` | PyTorch Lightning trainer settings |
| `logger` | `configs/logger/` | Logging backend (WandB, CSV) |
| `hydra/launcher` | `configs/hydra/launcher/` | Job submission (local or SLURM cluster) |

---

## Running experiments

### Local training (single run)

```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep
```

**What happens:**
1. Hydra composes the config from `experiment=tokenizer_explore/poyo_ajile_sweep` and all linked config groups
2. Foundry loads data from `./data/processed/` (or your configured `data.root`)
3. The model trains on GPU (if available) and logs metrics to Weights & Biases by default
4. Checkpoints and logs save to `./outputs/` (or `$SCRATCH/runs/` if the `SCRATCH` environment variable is set)

**Output structure:**
```
./outputs/runs/
├── POYO_AJILE_SWEEP/                    # experiment group
│   └── ajile_poyo_sweep_bs32_lr0.001/   # run name (from config)
│       ├── checkpoints/
│       │   ├── last.ckpt                # last checkpoint (auto-resume from here if preempted)
│       │   └── best-*-*.ckpt            # best checkpoint by validation metric
│       └── .hydra/
│           └── config.yaml              # saved config snapshot
```

### Multi-run sweeps (hyperparameter grid)

To run multiple configurations in parallel (e.g., trying different batch sizes and learning rates):

```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep -m
```

The `-m` flag enables **multirun mode**. Hydra will:
1. Parse the sweep parameters from the experiment config (e.g., `batch_size: choice(32, 64, 128)`)
2. Generate all combinations (3 batch sizes × 3 learning rates = 9 runs)
3. Run each one sequentially (or in parallel on SLURM)

Each run gets its own output folder under the experiment group.

**View results on the WandB dashboard:**

Once your experiment starts running and you have set up your `.env` file with WandB credentials, you can view:
- Real-time training curves and metrics
- Compare runs across different experiments
- Download final artifacts and model weights
- Access logs and system information

---

## Common errors and fixes

### Error: `experiment: ???` is mandatory

**Problem:** You ran `uv run python main.py` without specifying an experiment.

**Fix:** Always pass an experiment:
```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep
```

### Error: No data found at `./data/processed/`

**Problem:** Foundry can't find your processed dataset.

**Fix:** You have two options:

1. **Override the data path** in your command:
   ```bash
   uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep data.root=/path/to/my/data
   ```

2. **Create a symbolic link** to your processed data location:
   ```bash
   ln -s /path/to/your/processed/data ./data/processed
   ```
   This is convenient if your data lives elsewhere—just point the symlink to the actual location, and Foundry will find it at `./data/processed/`.

### Error: W&B offline / API key not found

**Problem:** You're using the default `logger=wandb` but your WandB credentials aren't set up properly.

**Fix:**
- Set up your `.env` file with `WANDB_API_KEY` as described in the **Environment setup** section above, or
- Switch to CSV logging:
  ```bash
  uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep logger=csv
  ```

### Error: GPU out of memory

**Problem:** Model + batch size exceeds GPU memory.

**Fix:**
- Reduce batch size:
  ```bash
  uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep hyperparameters.batch_size=32
  ```
- Or train on CPU (slow but works):
  ```bash
  uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep trainer.accelerator=cpu
  ```

### Error: Checkpoint not found during SLURM restart

**Problem:** Job was preempted and `last.ckpt` is missing.

**Fix:** Foundry logs a warning and starts from scratch. This is expected behavior. If you want to manually resume from an old checkpoint:
```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep run.resume_if_checkpoint_exists=true
```

---

## Contributing to Foundry

If you want to contribute code changes to Foundry, please follow these quality checks before submitting a pull request:

### Run unit tests

```bash
uv run pytest
```

This runs all tests in the `tests/` folder to verify your changes don't break existing functionality.

### Check code style

```bash
uv run ruff check .
```

This checks for code style violations. The linter is enforced in CI, so please fix any issues it flags.

### Or automatically format code

```bash
uv run ruff format .
```

This automatically formats your code to match the project's style standards. Run this before committing to ensure consistent formatting.

### Install pre-commit hooks (recommended)

```bash
uv sync --group dev
pre-commit install
```

This automatically runs linting and formatting checks before each commit, catching issues early.

### Development workflow

1. Create a feature branch from `main`
2. **Add unit tests** for any new code in the `tests/` folder
   - All new features and bug fixes must include corresponding tests
   - Tests should cover the main functionality and edge cases
   - Run `uv run pytest` to verify your tests pass
3. Make your changes and run the checks above
4. Ensure all tests pass and style checks are clean
5. Submit a pull request with a clear description of your changes

Thank you for contributing!
