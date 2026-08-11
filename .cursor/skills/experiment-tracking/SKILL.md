---
name: experiment-tracking
description: >-
  Manage structured experiment hypothesis markdown files. Routes to specialized
  sub-skills for creation, execution, and archiving. Use when the user mentions
  experiments, hypothesis files, the experiments/ directory, or wants to plan,
  run, or archive experiments.
---

# Experiment Tracking

This project uses a 3-stage experiment pipeline with a staged inbox workflow.

## Sub-skills

| Stage | Skill | Trigger |
|-------|-------|---------|
| 1. Plan | [create-experiment](../create-experiment/SKILL.md) | "new experiment", "I want to test", "plan experiment" |
| 2a. Analyze runs | [run-experiment](../run-experiment/SKILL.md) | "run is done", "analyze results", WandB run IDs provided |
| 2b. Report sweep | [report-sweep](../report-sweep/SKILL.md) | WandB group and/or sweep ID(s), "report sweep", compare minipigs vs monkeys |
| 3. Archive | [archive-group](../archive-group/SKILL.md) | "archive", "group these", "clean inbox", "synthesize" |

When the user's intent maps to one of these stages, read and follow the
corresponding sub-skill immediately.

Prefer **report-sweep** over **run-experiment** when the user supplies a
group and/or sweep ID(s) and wants a species comparison report.
`report-sweep` auto-resolves one vs many sweeps — do not ask the user to
reformat IDs before invoking it.

## Directory Structure

```
experiments/
├── inbox/                        # Active/unclassified experiments (Stages 1 & 2)
│   └── YYYYMMDD-<initials>-<slug>.md
├── <NN>-<group-slug>/            # Archived groups (Stage 3 output)
│   ├── README.md                 # Group synthesis
│   └── <experiment-files>.md
└── _legacy/                      # Original 001–023 experiments (frozen reference)
```

## Shared Conventions

### File naming

Use date-prefix with contributor initials and kebab-case slug:
`YYYYMMDD-<initials>-<slug>.md`

The date is the experiment start date. The slug is 3–5 words max.

### Markdown template

Every experiment file MUST follow this structure:

```markdown
# <Short descriptive title>

**Status:** <Draft | In Progress | Completed | Abandoned>
**Date started:** YYYY-MM-DD
**Parent experiment:** [<title>](<relative-path>) or "None (root)"
**Follow-up experiments:** [<title>](<relative-path>), ... or "TBD"
**Tags:** <comma-separated keywords for thematic grouping>

## Background
## Question
## Hypothesis
## Experiment
### Setup
### Launch command
### Key config overrides
## Results
### Summary
### Metrics
### Analysis
### Figures
## Conclusions
## Notes for future experiments
```

### Rules

1. **Always create an analysis script** in `analysis/` that fetches results via
   the wandb API rather than hardcoding numbers.

2. **Link experiments to each other.** Parent/child links create a navigable DAG.

3. **Start with what you know.** Fill Background through Setup on creation.
   Leave Results onward as `TBD` until the run completes.

4. **Update status** as the experiment progresses:
   Draft → In Progress → Completed (or Abandoned).

5. **WandB run references** must include both the human-readable run name and
   the machine-readable run ID (8-char alphanumeric).

6. **Analysis scripts** live in `analysis/` and should:
   - Use `wandb.Api()` to fetch metrics history.
   - Save figures to `analysis/figures/`.
   - Print a summary table to stdout.
   - Be self-contained (no imports from the main `foundry` package).

7. **One hypothesis per file.** Split multi-hypothesis experiments into
   separate files that reference each other.

8. **Don't duplicate config.** Reference the Hydra experiment YAML and list
   only the non-default overrides.

9. **Archiving.** Completed experiments are moved from `inbox/` into thematic
   group directories. Update all relative links in the same commit.
