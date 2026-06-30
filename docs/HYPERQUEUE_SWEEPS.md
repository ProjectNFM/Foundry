# Running WandB Agents with HyperQueue

This guide explains how to use HyperQueue to efficiently run multiple WandB sweep agents on CSCS clusters.

## Why HyperQueue?

HyperQueue is a meta-scheduler that bundles multiple small tasks into a single, larger Slurm job. This approach:

- **Reduces scheduling overhead** — One Slurm job instead of N array jobs
- **Improves GPU efficiency** — HyperQueue manages task-to-GPU allocation more flexibly
- **Enables resumability** — Automatically resume incomplete tasks after interruptions
- **Faster startup** — No repeated Slurm scheduling delays between tasks

See [CSCS HyperQueue documentation](https://docs.cscs.ch/running/hyperqueue/#example-workflow) for more details.

## Quick Start

### 1. Basic Usage (4 agents)

```bash
sbatch --gpus-per-node=4 sweep_hyperqueue.sh 4
```

This submits a job that:

- Allocates 1 node with **4 GPUs** (one per agent)
- Starts 4 HyperQueue workers (one per GPU)
- Submits 4 wandb agent tasks that run **in parallel**
- Each agent pulls trials from the sweep until it completes or time runs out

**Important:** Each wandb agent is long-running. You must allocate **one GPU per agent** (`--gpus-per-node=N` must match the number of agents). With only 1 GPU, only the first agent runs and the others never start.

### 2. Custom Number of Agents

```bash
sbatch --gpus-per-node=8 sweep_hyperqueue.sh 8
```

Submits 8 wandb agents with 8 GPUs. Always match `--gpus-per-node` to the agent count.

### 3. Custom Sweep ID and Experiment

```bash
sbatch sweep_hyperqueue.sh 4 user/project/sweep123 my_experiment/config
```

Runs 4 agents on your specific sweep and experiment.

## Monitoring Progress

### Watch Logs in Real-Time

```bash
tail -f /capstor/scratch/cscs/${USER}/wandb_logs/wandb_sweep_hq_<jobid>.log
```

### Check HyperQueue Status

While your job is running:

```bash
hq job list
hq job info <job-id>
hq worker list
```

### Check WandB Dashboard

Monitor sweep progress at [wandb.ai](https://wandb.ai):

- View all runs in your sweep
- See metrics and comparison plots
- Track agent activity in real-time

## Resuming Interrupted Jobs

If your job is interrupted (time limit, node failure, etc.), the journal file automatically saves the state. Simply resubmit with the same parameters:

```bash
sbatch sweep_hyperqueue.sh 8 user/project/sweep123 my_experiment/config
```

HyperQueue will:

- Detect the journal file from the previous run
- Identify which agents already completed
- Run only the incomplete agents
- Continue from where you left off

### Preserve Journal Across Runs (Optional)

To manually preserve state between different slurm job IDs:

```bash
# Save the journal before cleaning up
cp ~/.hq-journal-<old-jobid> ~/.hq-journal-backup

# Start a new job and tell it to use the old journal
export JOURNAL=~/.hq-journal-backup
sbatch sweep_hyperqueue.sh 8
```

## File Structure

```
sweep_hyperqueue.sh       # Main Slurm batch script
task_wandb_agent.sh       # Individual task script (run by HyperQueue)
/capstor/scratch/cscs/${USER}/wandb_logs/   # Output logs
  wandb_sweep_hq_*.log                      # HyperQueue orchestration log
  agent_<jobid>_<taskid>.log                # Per-agent logs
```

## Configuration

Edit `sweep_hyperqueue.sh` to customize:


| Parameter    | Default                                           | Change in                       |
| ------------ | ------------------------------------------------- | ------------------------------- |
| Account      | `a0091`                                           | Line 7 (`#SBATCH --account=`)   |
| Partition    | `normal`                                          | Line 8 (`#SBATCH --partition=`) |
| Time limit   | `23:00:00`                                        | Line 11 (`#SBATCH --time=`)     |
| GPU per task | `1`                                               | Line 17 (HyperQueue submit)     |
| Sweep ID     | `user/project/abcd1234`                           | Line 30 or CLI arg              |
| Experiment   | `auditory_decoding/poyo_neurosoft_8band_hp_sweep` | Line 31 or CLI arg              |


## Common Tasks

### Run a Sweep with 16 Agents for 48 Hours

Edit `sweep_hyperqueue.sh`:

```bash
#SBATCH --time=48:00:00
```

Then submit:

```bash
sbatch sweep_hyperqueue.sh 16 user/project/sweep123 my_experiment
```

### Check Completed vs. Pending Tasks

```bash
hq job info 0  # Replace 0 with your job ID
```

### Manually Stop a Running Sweep

```bash
hq server stop
```

This gracefully shuts down the server and workers, allowing HyperQueue to save state to the journal.

### Debug a Single Agent Task

Run the task script manually:

```bash
./task_wandb_agent.sh user/project/sweep123 my_experiment
```

## Troubleshooting

### Server Failed to Start

Check the log:

```bash
tail /capstor/scratch/cscs/${USER}/wandb_logs/wandb_sweep_hq_*.log
```

Ensure HyperQueue is installed. SLURM batch jobs use a minimal `PATH`, so either:

- Rely on `sweep_hyperqueue.sh` (it prepends `$HOME/.local/aarch64/bin`), or
- Add it to your shell profile:

```bash
export PATH="$HOME/.local/aarch64/bin:$PATH"
which hq
hq --version
```

### Workers Not Connecting

Verify workers started with `srun`:

```bash
hq worker list
```

If empty, check:

- GPU availability: `sinfo -o "%N %G"`
- SLURM errors in the log file

### Only one agent runs / other agents never start

Each wandb agent runs until the sweep finishes. With **1 GPU and 1 worker**, HyperQueue runs agents **sequentially**, so only agent 1 ever executes.

**Fix:** Match GPUs to agent count:

```bash
sbatch --gpus-per-node=4 sweep_hyperqueue.sh 4 suarezul/Foundry/abc123 auditory_decoding/foo
```

Check per-agent logs:

```bash
ls -lt /capstor/scratch/cscs/${USER}/wandb_logs/agent_<jobid>_*.log
tail -f /capstor/scratch/cscs/${USER}/wandb_logs/agent_<jobid>_1.log
```

### Tasks Hang or Don't Progress

Check WandB sweep status at wandb.ai — agents may be waiting for the sweep to accept new runs. Verify the sweep is still active and accepting agents.

### Resume Not Working

Manually check for journal files:

```bash
ls -la ~/.hq-journal-*
```

If the journal is corrupted, delete it and start fresh (this will lose progress):

```bash
rm ~/.hq-journal-*
sbatch sweep_hyperqueue.sh 4
```

## Performance Tips

1. **Match workers to available GPUs** — Each worker needs 1 GPU; don't submit more tasks than GPUs
2. **Use reasonable time limits** — Account for startup overhead (~30s) plus agent runtime
3. **Monitor first run** — For new sweep/experiment combos, do a short test run first
4. **Batch small tasks** — HyperQueue shines with 10+ agents; for just 1–2, use Slurm arrays instead

## Examples

### Bayesian Optimization Sweep (10 agents, 12 hours)

```bash
sbatch sweep_hyperqueue.sh 10 user/project/bayes_sweep auditory_decoding/bayesian_hp
```

### Grid Search (32 agents, 48 hours)

Edit `sweep_hyperqueue.sh` to change `--time=48:00:00`, then:

```bash
sbatch sweep_hyperqueue.sh 32 user/project/grid_sweep auditory_decoding/grid_search
```

### Resume Previous Run

```bash
sbatch sweep_hyperqueue.sh 20 user/project/sweep123 my_experiment
```

HyperQueue automatically detects and resumes from the previous journal.

## Further Reading

- [CSCS HyperQueue Guide](https://docs.cscs.ch/running/hyperqueue/#example-workflow)
- [HyperQueue Official Documentation](https://it4innovations.github.io/hyperqueue/)
- [WandB Sweep Documentation](https://docs.wandb.ai/guides/sweeps)

