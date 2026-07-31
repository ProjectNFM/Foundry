---
name: create-experiment
description: >-
  Create a new experiment hypothesis file in experiments/inbox/ via interactive
  questioning. Use when the user wants to plan, design, or start a new
  experiment, or says "new experiment", "let's test", "I want to try",
  "create experiment", or "plan experiment".
disable-model-invocation: true
---

# Create Experiment

Plan and create a new experiment hypothesis file via interactive questioning.

## Workflow

### Step 1: Scan the Research Frontier

Read all experiments in `experiments/inbox/` and `experiments/*/` (archived
groups) to understand the current state:

- Extract titles, statuses, conclusions, and "Notes for future experiments"
- Identify the active research threads and their latest results
- Note any open questions from completed experiments

Also check `experiments/_legacy/` for historical context if relevant.

### Step 2: Grill-Me Interview

Use the `AskQuestion` tool for structured choices and conversational
follow-ups for open-ended answers. Conduct 4–6 rounds:

**Round 1 — Lineage:**
Ask which line of investigation this builds on. Populate options dynamically
from the existing experiment threads (group names + latest experiment in each).
Always include "Entirely new direction" as a final option.

**Round 2 — Core Intent:**
Ask what the ONE thing is they'd learn if this experiment succeeds. Options:
- "Whether [architectural change] improves [metric]"
- "Whether [data strategy] transfers to [task]"
- "Interaction between [factor A] and [factor B]"
- "Let me describe it in my own words"

If "own words" → ask conversationally: "Describe what you want to test in one
sentence. I'll help sharpen it into a falsifiable hypothesis."

**Round 3 — Variables:**
Ask for the independent variable (the thing being changed). Options:
- "Model architecture component"
- "Training hyperparameter"
- "Data preprocessing / tokenization"
- "Dataset composition or split"
- "Other (I'll specify)"

**Round 4 — Success Metrics:**
Ask how they'll know it worked. Options:
- "Validation loss (reconstruction)"
- "Downstream F1 / accuracy (linear probe)"
- "Downstream F1 / accuracy (finetuning)"
- "Qualitative (embedding analysis, visualization)"
- "Multiple metrics (I'll specify)"

**Round 5 — Baseline:**
Ask what the comparison point is. Auto-populate from the parent experiment's
best run if one was identified in Round 1. Options:
- "Previous experiment's best run: [auto-populated]"
- "Ablation (disable the component being tested)"
- "From-scratch baseline (no pretraining)"
- "I'll specify a custom baseline"

Skip rounds that are already obvious from context. Adapt questions based on
prior answers.

### Step 3: Validate Coherence

Before generating the file, verify:
- Does this logically follow from the parent experiment's conclusions?
- Is the hypothesis falsifiable with the proposed metric?
- Are there known confounds from earlier experiments?

If issues exist, raise them conversationally and suggest refinements.

### Step 4: Generate the Experiment File

Create `experiments/inbox/YYYYMMDD-<initials>-<slug>.md` with:

```markdown
# <Short descriptive title>

**Status:** Draft
**Date started:** YYYY-MM-DD
**Parent experiment:** [<title>](<relative-path>) or "None (root)"
**Follow-up experiments:** TBD
**Tags:** <inferred from lineage and variables>

## Background

<Written by the agent: why this experiment matters, referencing prior work
with links to parent/related experiments, summarizing their conclusions>

## Question

<One clear question distilled from the interview>

## Hypothesis

<A falsifiable prediction with predicted direction and magnitude if possible>

## Experiment

### Setup

- **Model:** <from context or TBD>
- **Data:** <from context or TBD>
- **Task:** <from context or TBD>
- **Training:** TBD
- **WandB:** TBD

### Launch command

```bash
# TBD — to be filled before running
```

### Key config overrides

TBD

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD
```

### Step 5: Update Parent Links

If a parent experiment was identified, update its "Follow-up experiments"
field to include a link to the newly created file.

## Key Principles

- **Never skip the interview.** Even if the user gives a one-liner, expand it
  through questioning. Vague hypotheses produce uninterpretable results.
- **Build on prior work.** The Background section must reference at least one
  existing experiment unless this is genuinely a new research direction.
- **One hypothesis per file.** If the interview reveals multiple hypotheses,
  propose splitting into separate experiment files that reference each other.
- **Initials come from the user.** If unknown, ask once and remember for the
  session.
