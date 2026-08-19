---
name: run-experiment
description: >-
  Execute an experiment, generate analysis scripts, fetch results from WandB,
  produce figures, and fill in Results/Conclusions sections of an inbox
  experiment file. Use when the user says "run experiment", "analyze results",
  "the run is done", "fill in results", provides WandB run IDs, or wants to
  complete an existing experiment document.
disable-model-invocation: true
---

# Run Experiment

Execute analysis and fill results for a completed experiment run.

## Workflow

### Step 1: Identify Target Experiment

Scan `experiments/inbox/` for files with `Status: Draft` or `Status: In Progress`.

- If only one active experiment exists, confirm it with the user.
- If multiple exist, ask the user to select one:

```
question: "Which experiment to analyze?"
options: [dynamically populated from inbox files with Draft/In Progress status]
```

Update the file's status to `In Progress` if it was `Draft`.

### Step 2: Gather Run Information

Ask the user for WandB run details conversationally:

"Provide the WandB run ID(s) for this experiment. Format: `run-name (abc123de)`.
If it's a sweep, give me the sweep ID or group name."

If the experiment file already has WandB IDs in the Setup section, confirm
those are the runs to analyze.

### Step 3: Generate Analysis Script

Extract the experiment stem (target experiment filename without `.md`) and use it for all artifacts.

Create an analysis script named `YYYYMMDD-<initials>-<slug>.py` that:

- Accepts run ID(s) as constants at the top of the file
- Uses `wandb.Api()` to fetch metrics history
- Computes comparison tables between runs (or vs baseline)
- Generates relevant figures (training curves, bar charts, etc.)
- Saves figures to `analysis/figures/YYYYMMDD-<initials>-<slug>_*.png`
- Saves CSV tables/caches to `analysis/csv/YYYYMMDD-<initials>-<slug>_*.csv`
  (use `csv_dir` from `analysis/_wandb_utils.py`; do not write CSVs into `figures/`)
- Prints a summary metrics table to stdout
- Is self-contained (no imports from `foundry`)

The script must be runnable with:
```bash
uv run python analysis/YYYYMMDD-<initials>-<slug>.py
```

### Step 4: Execute Analysis

Run the analysis script and capture output:

```bash
uv run python analysis/YYYYMMDD-<initials>-<slug>.py
```

If the script fails, debug and fix it. Iterate until it produces clean output.

### Step 5: Interpretation Checkpoint (Interactive)

Present the metrics summary and any generated figures to the user.
Ask the user to validate the interpretation using the following structured
choices:

```
question: |
  Metrics summary:
  [paste stdout table from analysis script]
  
  Does this interpretation match your reading?
options:
  - "Yes, hypothesis confirmed"
  - "Partially confirmed (some metrics support, others don't)"
  - "Hypothesis refuted"
  - "Results are inconclusive — need additional runs"
  - "I interpret this differently (let me explain)"
```

If the user selects "differently" or "inconclusive", ask conversationally
for their interpretation and incorporate it.

### Step 6: Figure Selection (Interactive)

If multiple figures were generated, ask which to include:

```
question: "Which figures should appear in the experiment document?"
allow_multiple: true
options: [dynamically populated from generated figure filenames]
```

### Step 7: Future Directions

Ask conversationally:
"Based on these results, what open questions or next steps come to mind?
I'll format them as 'Notes for future experiments'."

### Step 8: Write Results

Update the experiment file in `experiments/inbox/` with:

**Status:** → `Completed`

**Results section:**
- Summary: narrative of what happened
- Metrics: table from analysis script output
- Analysis: reference to the script with run command
  ```bash
  uv run python analysis/YYYYMMDD-<initials>-<slug>.py
  ```
- Figures: embedded with relative paths to `../../analysis/figures/YYYYMMDD-<initials>-<slug>_*.png`

**Conclusions:** hypothesis verdict with supporting evidence

**Notes for future experiments:** bullet points from Step 7

Also update the Setup section with final WandB run details if they were
TBD before.

## Key Principles

- **Always generate a script.** Never hardcode metrics in the markdown.
  Results must be reproducible by re-running the analysis script.
- **Programmatic over manual.** Prefer `wandb.Api()` data extraction over
  reading values from the WandB UI manually.
- **User confirms interpretation.** Never write conclusions without the
  user validating the interpretation of results.
- **Figures are evidence.** Every claim in Conclusions should be backed by
  either a metric in the table or a referenced figure.
