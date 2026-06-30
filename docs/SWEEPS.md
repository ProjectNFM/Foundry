# Hyperparameter Sweeps in Foundry

This document describes how to run hyperparameter sweeps in Foundry using two complementary approaches:

1. **Optuna sweeps** — For local experimentation and intelligent search (Bayesian optimization)
2. **WandB sweeps** — For production runs, team collaboration, and advanced monitoring

## Quick Start

### Running an Optuna sweep locally

```bash
uv run python -m foundry.optuna_launcher \
  --experiment=auditory_decoding/poyo_neurosoft_8band_hp_sweep \
  --n-trials=50 \
  --sampler=bayes
```

### Running a WandB sweep

```bash
# Create sweep config
wandb sweep configs/sweep/auditory_decoding_hp.yaml

# Launch agent(s)
wandb agent <sweep-id>
```

## Architecture

### Optuna Sweeps (Recommended for Development)

**Best for:**

- Local iteration and prototyping
- Intelligent search (Bayesian optimization)
- Early stopping and pruning
- SLURM cluster integration (each trial as a separate job)
- Resumable studies (checkpoint and restart)

**Components:**

- `foundry/optuna_integration.py` — Core `HydraOptunaOptimizer` class
- `foundry/optuna_launcher.py` — CLI entry point
- `configs/optuna/*.yaml` — Sampler and pruner configurations
- Experiment configs with `sweep.search_space` definition

**Workflow:**

```
CLI args → Optuna Launcher
    ↓
Load Hydra config + sweep.search_space
    ↓
Create HydraOptunaOptimizer (TPE/Random/Grid sampler)
    ↓
For each trial:
  - Generate hyperparameter suggestions
  - Inject into Hydra config
  - Run training
  - Extract validation metric
  - Report to Optuna
  - Prune if converging badly
    ↓
Return best hyperparameters + trial history
```

### WandB Sweeps (Recommended for Production)

**Best for:**

- Team collaboration and result sharing
- WandB UI monitoring and analysis
- Advanced search methods (Bayesian via Optuna backend)
- Production runs with full reproducibility logging

**Components:**

- `foundry/sweep.py` — WandB sweep utilities
- `configs/sweep/*.yaml` — WandB sweep definitions
- `main.py` enhancement — Sweep mode detection and hyperparameter injection

**Workflow:**

```
wandb sweep config.yaml
    ↓
WandB backend initializes sweep
    ↓
wandb agent <sweep-id>
    ↓
For each trial (agent loop):
  - Pull hyperparameters from wandb.config
  - Run training (main.py with WANDB_SWEEP_ID set)
  - main.py detects sweep mode
  - Injects hyperparams from wandb.config
  - Training logs metrics back to WandB
    ↓
WandB aggregates results in UI
```

## Usage Guide

### 1. Optuna Sweeps (Local Development)

#### Setup

Your experiment config must include a `sweep` section:

```yaml
sweep:
  metric_name: val/accuracy  # Metric to optimize
  metric_direction: maximize  # or "minimize"
  search_space:
    hyperparameters.learning_rate:
      type: float
      low: 1e-5
      high: 1e-2
      log: true  # Log scale
    hyperparameters.batch_size:
      type: categorical
      choices: [32, 64, 128]
    hyperparameters.weight_decay:
      type: float
      low: 0.0
      high: 0.1
```

#### Launch a sweep

```bash
# Bayesian optimization (default)
uv run python -m foundry.optuna_launcher \
  --experiment=auditory_decoding/poyo_neurosoft_8band_hp_sweep \
  --n-trials=50 \
  --sampler=bayes

# Random search (for baseline / exploration)
uv run python -m foundry.optuna_launcher \
  --experiment=auditory_decoding/poyo_neurosoft_8band_hp_sweep \
  --n-trials=30 \
  --sampler=random

# Grid search (only for tiny spaces)
uv run python -m foundry.optuna_launcher \
  --experiment=tokenizer_explore/poyo_ajile_hp_sweep \
  --sampler=grid
```

#### Resume an incomplete sweep

Optuna stores trial history in a local SQLite database. To resume:

```bash
# Increase n_trials beyond what was already run
uv run python -m foundry.optuna_launcher \
  --experiment=auditory_decoding/poyo_neurosoft_8band_hp_sweep \
  --n-trials=100  # Will run 50 more trials (if 50 were completed)
```

#### Run on SLURM cluster

Each trial becomes a separate SLURM job:

```bash
uv run python -m foundry.optuna_launcher \
  --experiment=auditory_decoding/poyo_neurosoft_8band_hp_sweep \
  --n-trials=50 \
  --n-jobs=4  # Run 4 trials in parallel on cluster
```

#### Monitor progress

Watch the terminal output for trial results and early stopping decisions:

```
[2026-06-26 12:34:56] Trial 0: {'learning_rate': 0.0003, 'batch_size': 64}
[2026-06-26 12:35:12] Trial 0 finished with value 0.92
[2026-06-26 12:35:28] Trial 1: {'learning_rate': 0.0001, 'batch_size': 128}
...
[2026-06-26 12:50:00] Optimization complete: best_value=0.96 best_trial=12
Best hyperparameters: {'learning_rate': 0.00045, 'batch_size': 64}
```

### 2. WandB Sweeps (Production / Collaboration)

#### Important: Task-Sweep Alignment

WandB sweeps are **experiment-agnostic** — they don't directly reference which task or experiment they should run with. Instead, they specify a **metric name to watch for**. 

**You are responsible for ensuring the experiment matches the sweep's metric expectations:**

```yaml
# configs/sweep/auditory_decoding_hp.yaml
metric:
  name: val/neurosoft_acoustic_stim_8band_f1  # Expects this metric
  
# Must launch with an experiment that produces this metric:
# (see configs/experiment/auditory_decoding/poyo_neurosoft_8band_hp_sweep.yaml)
task_configs:
  - neurosoft_acoustic_stim_8band  # This task produces val/neurosoft_acoustic_stim_8band_f1
```

If you run the sweep with the wrong experiment, WandB will wait forever for a metric that never appears.

**Best practice:** Each sweep config has a **recommended experiment** documented in its comments. Always cross-check before launching.

#### Create a sweep config

WandB uses YAML configs in `configs/sweep/`:

```yaml
method: bayes  # Search method: bayes, random, grid
metric:
  name: val/accuracy
  goal: maximize
early_stopping:
  patience: 15
  min_iter: 3
parameters:
  hyperparameters.learning_rate:
    distribution: log_uniform
    min: 1e-5
    max: 1e-2
  hyperparameters.batch_size:
    values: [32, 64, 128]
```

#### Initialize the sweep

```bash
wandb sweep configs/sweep/auditory_decoding_hp.yaml
# Output: "Created sweep with ID: user/project/abcd1234"
```

#### Launch sweep agent(s)

```bash
# Single agent (with experiment specification to link to task)
wandb agent user/project/abcd1234 -- main.py experiment=auditory_decoding/poyo_neurosoft_8band_hp_sweep

# Multiple agents (parallelism)
for i in {1..4}; do
  wandb agent user/project/abcd1234 -- main.py experiment=auditory_decoding/poyo_neurosoft_8band_hp_sweep &
done
```

**Important:** The `-- main.py experiment=...` part is **critical**. It tells the agent which experiment to use, which determines which task is loaded and which metric gets logged. The sweep config watches for `val/neurosoft_acoustic_stim_8band_f1`, which only appears if the `neurosoft_acoustic_stim_8band` task is loaded.

#### Monitor in WandB UI

- Open [https://wandb.ai/user/project/sweeps/abcd1234](https://wandb.ai/user/project/sweeps/abcd1234)
- See real-time trial results, parallel coordinates plots, importance weights
- Compare hyperparameter values against metrics
- Download results as CSV

### Using WandB Sweeps with SLURM

If you want to use WandB sweeps with SLURM cluster parallelism, launch multiple agents as a **SLURM job array**. Each array task runs one independent agent that pulls trials from the sweep.

#### Job Array Approach (Recommended)

Create a script:

```bash
#!/bin/bash
# submit_wandb_sweep_array.sh
#SBATCH --job-name=wandb_sweep
#SBATCH --array=1-4                # Create 4 agents (one per array task)
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/wandb_agent_%a.log

# Each array task runs one agent independently
wandb agent user/project/abcd1234 -- main.py experiment=auditory_decoding/poyo_neurosoft_8band_hp_sweep
```

Submit with:

```bash
mkdir -p logs
sbatch submit_wandb_sweep_array.sh
```

Monitor progress:

```bash
# Check job array status
squeue -j <job_id>

# Check individual agent logs
cat logs/wandb_agent_1.log
cat logs/wandb_agent_2.log

# View sweep in WandB UI
# https://wandb.ai/user/project/sweeps/abcd1234
```

**How it works:**
- Each array task (`SLURM_ARRAY_TASK_ID=1,2,3,4`) runs one agent independently
- Each agent connects to the WandB sweep, pulls the next available trial, runs it, and repeats
- WandB backend tracks which trials are in progress and which are pending
- Multiple agents run in **true parallelism** on different SLURM nodes/GPUs

#### Advanced: Pin Agents to Specific GPUs

If you want each agent to use a specific GPU:

```bash
#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00

# Array task IDs map to GPU IDs (0, 1, 2, 3)
GPU_ID=$SLURM_ARRAY_TASK_ID
CUDA_VISIBLE_DEVICES=$GPU_ID wandb agent user/project/abcd1234 \
  -- main.py experiment=auditory_decoding/poyo_neurosoft_8band_hp_sweep
```

#### Comparison: SLURM Parallelism Approaches

| Method | Setup | Monitoring | Best for |
|--------|-------|-----------|----------|
| **Job Array** | Simple (1 script) | Native SLURM | ✅ Production |
| **srun loop** | Complex | Manual | Testing |
| **Single agent** | Trivial | Basic | Local testing |

**Advantages of job array:**
- ✅ Native SLURM feature (designed for this exact use case)
- ✅ Each agent gets its own job (easy to cancel individual agents)
- ✅ Clean logs per agent (`wandb_agent_1.log`, `wandb_agent_2.log`, etc.)
- ✅ Better resource accounting and billing
- ✅ Automatic dependency and resource management

### 3. Comparing Approaches


| Aspect                 | Optuna                          | WandB                                |
| ---------------------- | ------------------------------- | ------------------------------------ |
| **Best for**           | Local development               | Production / teams                   |
| **Search methods**     | TPE, Random, Grid               | Bayes (Optuna backend), Random, Grid |
| **Parallelism**        | SLURM native (each trial = job) | Manual (multiple agents)             |
| **Trial Monitoring**   | Terminal output                 | WandB UI + W&B API                   |
| **Resumability**       | SQLite checkpoint               | Native (sweep ID persists)           |
| **Team collaboration** | Manual sharing                  | Full integration                     |
| **Cost**               | Free (local)                    | Free tier available                  |
| **SLURM integration**  | Automatic                       | User-managed                         |


**For SLURM clusters, Optuna is recommended** because it automatically submits each trial as a separate job. WandB sweeps require manually launching multiple agent processes.

**Recommendation:**

- Use **Optuna** while iterating on SLURM cluster
- Use **WandB** for final production runs and sharing results with team

## Configuration Reference

### WandB Sweep ↔ Experiment ↔ Task Mapping

This table shows which sweep configs link to which experiments and tasks:


| WandB Sweep Config                | Recommended Experiment                                            | Task                            | Metric                                       |
| --------------------------------- | ----------------------------------------------------------------- | ------------------------------- | -------------------------------------------- |
| `sweep/auditory_decoding_hp.yaml` | `experiment/auditory_decoding/poyo_neurosoft_8band_hp_sweep.yaml` | `neurosoft_acoustic_stim_8band` | `val/neurosoft_acoustic_stim_8band_f1`       |
| `sweep/sleep_staging_hp.yaml`     | `experiment/sleep_staging/eegnet_neurosoft_hp_sweep.yaml`         | `neurosoft_sleep_stage`         | `val_loss` or `val/neurosoft_sleep_stage_f1` |
| `sweep/tokenizer_explore_hp.yaml` | `experiment/tokenizer_explore/poyo_ajile_hp_sweep.yaml`           | `ajile_inactive_active`         | `val_loss`                                   |


**Always verify:** Sweep metric name = `val/{task_name}_{metric_key}` from task's `metric_summary_modes`

### Optuna Samplers

Location: `configs/optuna/*.yaml`

#### TPE (Tree-structured Parzen Estimator)

Default, Bayesian optimization. Best all-around choice.

```yaml
sampler:
  type: tpe
  seed: 42
  n_startup_trials: 10  # Trials before learning kicks in
  consider_prior: true
  consider_magic_clip: true
```

#### Random

Baseline search. Good for exploration and comparison.

```yaml
sampler:
  type: random
  seed: 42
```

#### Grid

Exhaustive enumeration. Only for tiny spaces.

```yaml
sampler:
  type: grid
```

### Optuna Pruners

#### Successive Halving (Recommended)

Aggressive early stopping: eliminate bad trials after few epochs.

```yaml
pruner:
  type: successive_halving
  min_resource: 3     # Minimum epochs before pruning
  reduction_factor: 3 # Drop worst 2/3 of trials
```

#### None

No early stopping, all trials complete full training.

```yaml
pruner:
  type: none
```

### Search Space Definition

In experiment config under `sweep.search_space`:

#### Float parameters

```yaml
hyperparameters.learning_rate:
  type: float
  low: 1e-5
  high: 1e-2
  log: true  # Use log scale (default: false)
```

#### Integer parameters

```yaml
hyperparameters.num_layers:
  type: int
  low: 1
  high: 10
  log: false
```

#### Categorical parameters

```yaml
hyperparameters.batch_size:
  type: categorical
  choices: [32, 64, 128, 256]
```

## Experiment Templates

Pre-configured sweep experiments:

### Auditory Decoding

- `auditory_decoding/poyo_neurosoft_8band_hp_sweep.yaml`
- Tunes: learning_rate, weight_decay, batch_size, cwt_lr_multiplier
- Metric: F1 score on 8-band acoustic stimulus task

### Sleep Staging

- `sleep_staging/eegnet_neurosoft_hp_sweep.yaml`
- Tunes: learning_rate, weight_decay, batch_size
- Metric: F1 score on sleep stage classification

### Tokenizer Exploration

- `tokenizer_explore/poyo_ajile_hp_sweep.yaml`
- Tunes: batch_size, learning_rate, weight_decay
- Metric: Validation loss on active/inactive prediction

### Minimal Test (for CI)

- `test/minimal_hp_sweep.yaml`
- Quick 3-trial sweep for validation
- Use to test sweep infrastructure before full runs

## Troubleshooting

### Sweep not creating runs in WandB

**Symptom:** `wandb sweep` succeeds but no runs appear in project.

**Solution:**

- Ensure your experiment config has `run.project` set
- Check that `logger: wandb` is in your Hydra defaults
- Verify WANDB_API_KEY and WANDB_ENTITY environment variables

```bash
export WANDB_API_KEY=<your-key>
export WANDB_ENTITY=<team-or-username>
```

### WandB sweep waiting for metric that never appears

**Symptom:** Sweep runs but shows "No reported metrics" and waits indefinitely.

**Likely cause:** The sweep config specifies a metric (e.g., `val/neurosoft_acoustic_stim_8band_f1`) but the experiment you launched produces a different metric or uses a different task.

**Solution:**

1. Check the sweep config's documentation comments for the recommended experiment
2. Verify the experiment uses the correct task:
  ```bash
   grep "task_configs:" configs/experiment/your_experiment.yaml
  ```
3. Verify the task produces the expected metric:
  ```bash
   grep "metric_summary_modes:" configs/tasks/your_task.yaml
  ```
4. If metrics don't match, either:
  - Use the recommended experiment, OR
  - Update the sweep config's metric to match your experiment's task

Example:

```yaml
# configs/sweep/auditory_decoding_hp.yaml expects:
metric:
  name: val/neurosoft_acoustic_stim_8band_f1

# If you want to use a different task (neurosoft_acoustic_stim_2band),
# update the sweep to match:
metric:
  name: val/neurosoft_acoustic_stim_2band_f1
```

### Optuna optimizer hangs

**Symptom:** Optuna launcher stuck after creating study.

**Solution:**

- Check that your training function returns a numeric metric value
- Verify the experiment config is valid: `uv run python main.py experiment=... --help`
- Ensure `sweep.metric_name` matches an actual metric logged during training

### Early stopping too aggressive

**Symptom:** Many trials pruned very early, few complete.

**Solution:**

- Increase `pruner.min_resource` (minimum training steps before pruning allowed)
- Use `pruner: {type: none}` to disable pruning temporarily for debugging

```yaml
pruner:
  type: successive_halving
  min_resource: 5  # Was 3, now more lenient
```

### "WANDB_SWEEP_ID not in environment"

**Symptom:** Sweep hyperparameters not being injected into config.

**Solution:**

- This happens only when running as `wandb agent`; normal training ignores it
- For development, use Optuna instead
- Or manually verify sweep env vars:

```bash
echo $WANDB_SWEEP_ID  # Should be set when under wandb agent
```

## Advanced Topics

### Custom Training Functions for Optuna

If you want to run Optuna sweeps without modifying `main.py`, create a custom train function:

```python
from foundry.optuna_integration import HydraOptunaOptimizer
from omegaconf import DictConfig

def my_train_fn(cfg: DictConfig) -> float:
    """Train one trial and return metric value."""
    # Your training logic here
    # Must return a single float (metric to optimize)
    return val_loss

optimizer = HydraOptunaOptimizer(
    experiment_name="custom_sweep",
    search_space=search_space,
    metric_name="val_loss",
    metric_direction="minimize",
)

best_params = optimizer.optimize(cfg, my_train_fn, n_jobs=1)
```

### Programmatic Sweep Launch

```python
from foundry.optuna_launcher import launch_optuna_sweep

best_params = launch_optuna_sweep(
    experiment="auditory_decoding/poyo_neurosoft_8band_hp_sweep",
    n_trials=100,
    sampler="bayes",
    pruner="successive_halving",
)

print(f"Best hyperparameters: {best_params}")
```

### Analyzing Results

After a sweep completes, inspect results:

```bash
# List all trials in study
python -c "
import optuna
storage = optuna.storages.RDBStorage('sqlite:///./auditory_decoding_poyo_neurosoft_8band_hp_sweep_optuna_study.db')
study = optuna.load_study('auditory_decoding_poyo_neurosoft_8band_hp_sweep', storage)
for trial in study.trials:
    print(f'Trial {trial.number}: params={trial.params} value={trial.value}')
"

# Export to CSV for analysis
python -c "
import optuna
import pandas as pd
storage = optuna.storages.RDBStorage('sqlite:///./auditory_decoding_poyo_neurosoft_8band_hp_sweep_optuna_study.db')
study = optuna.load_study('auditory_decoding_poyo_neurosoft_8band_hp_sweep', storage)
df = study.trials_dataframe()
df.to_csv('sweep_results.csv', index=False)
print(df.head())
"
```

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [WandB Sweeps Guide](https://docs.wandb.ai/guides/sweeps)
- [Hydra Multirun Documentation](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)

