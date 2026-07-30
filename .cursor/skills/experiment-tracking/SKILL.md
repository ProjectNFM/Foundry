---
name: experiment-tracking
description: >-
  Manage structured experiment hypothesis markdown files in the experiments/
  folder. Use when the user asks to create, update, or review experiment
  documents, log results, plan new experiments, or reference the experiment
  tree. Also use when the user mentions hypothesis files, experiment markdown,
  or the experiments/ directory.
---

# Experiment Tracking

This project tracks ML experiments as structured Markdown files in the
`experiments/` directory. Each file represents one hypothesis and its
associated experiments.

## File naming

Use kebab-case with a sequential numeric prefix:

```
experiments/
├── 001-overfit-single-batch.md
├── 002-overfit-single-session.md
├── 003-tokenizer-comparison.md
└── ...
```

## Markdown template

Every experiment file MUST follow this structure exactly:

```markdown
# <Short descriptive title>

**Status:** <Draft | In Progress | Completed | Abandoned>
**Date started:** YYYY-MM-DD
**Parent experiment:** [<title>](../experiments/<filename>.md) or "None (root)"
**Follow-up experiments:** [<title>](../experiments/<filename>.md), ... or "TBD"

## Background

Why this experiment matters. Reference prior work, related experiments, and
any context that motivated it. Keep it concise but sufficient for someone
unfamiliar to understand.

## Question

One clear question this experiment aims to answer.

## Hypothesis

A falsifiable prediction about the outcome.

## Experiment

### Setup

- **Model:** architecture, key hyperparameters
- **Data:** dataset(s), split strategy, any filtering
- **Task:** loss function, metrics
- **Training:** epochs, batch size, LR, hardware, etc.
- **WandB:** project, group, run name(s), run ID(s)

### Launch command

```bash
# The exact command(s) used to run the experiment
```

### Key config overrides

List any non-default Hydra overrides applied.

## Results

### Summary

Brief narrative of what happened.

### Metrics

| Metric | Value |
|--------|-------|
| ...    | ...   |

### Analysis

Describe how results were obtained. Always prefer programmatic extraction
over manual reading of logs.

**Analysis script:** `analysis/<script_name>.py`

```bash
# How to run the analysis
uv run python analysis/<script_name>.py
```

### Figures

Reference any generated plots:

![description](../analysis/figures/<filename>.png)

## Conclusions

What we learned. Was the hypothesis confirmed or refuted? Why?

## Notes for future experiments

Bullet list of ideas, observations, or open questions that should inform
follow-up work.
```

## Rules for the agent

1. **Always create an analysis script** in `analysis/` that fetches results via
   the wandb API rather than hardcoding numbers. The markdown should reference
   the script and explain how to run it.

2. **Link experiments to each other.** When a new experiment follows from a
   previous one, add it as a follow-up in the parent and set the parent link in
   the child. This creates a navigable tree.

3. **Start with what you know.** When planning a new experiment, fill in
   Background through Experiment fully. Leave Results, Conclusions, and Notes
   sections with `TBD` placeholders — they get filled after the run completes.

4. **Update status** as the experiment progresses: Draft -> In Progress ->
   Completed (or Abandoned).

5. **WandB run references** should include both the human-readable run name and
   the machine-readable run ID (the 8-char alphanumeric string), so analysis
   scripts can fetch data unambiguously.

6. **Analysis scripts** live in `analysis/` and should:
   - Accept the wandb run ID(s) as arguments or have them at the top as
     constants.
   - Use `wandb.Api()` to fetch metrics history.
   - Save any figures to `analysis/figures/`.
   - Print a summary table to stdout.
   - Be self-contained (no imports from the main `foundry` package).

7. **Figures directory:** analysis scripts that produce plots save them to
   `analysis/figures/`. The experiment markdown references them with relative
   paths.

8. **One hypothesis per file.** If an experiment tests multiple things, split
   into separate files that reference each other.

9. **Don't duplicate config.** Reference the Hydra experiment YAML and list only
   the non-default overrides. The full resolved config is in the run's `.hydra/`
   folder and on wandb.
