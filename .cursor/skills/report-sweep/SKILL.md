---
name: report-sweep
description: >-
  Fetch WandB sweep/group results, auto-resolve one or many sweep IDs,
  compare minipigs vs monkeys on validation metrics, and write a completed
  experiment report with Question, Hypothesis, Results, Conclusions, and
  next steps. Use when the user provides a WandB group and/or sweep ID(s),
  asks to report a sweep, compare species, or analyze auditory-decoding
  sweep results.
disable-model-invocation: true
---

# Report Sweep

Turn completed WandB sweep run(s) into a finished experiment markdown that
**compares minipigs and monkeys** on the same research question.

## Inputs

Accept whatever the user has — **do not require a fixed shape**. Ask only
for what cannot be inferred.

| Input | Required? | Notes |
|-------|-----------|-------|
| WandB **project** | Soft | Default `auditory_decoding` |
| WandB **group** | Soft | e.g. `NEUROSOFT_INTRASESSION_MULTISUBJ` |
| WandB **sweep ID(s)** | Soft | Zero, one, two, or more — any mix |
| WandB **entity** | No | `WANDB_ENTITY` / `default_entity()` |
| Contributor initials | Yes | For the inbox filename |

Need enough to uniquely locate the finished runs: typically a **group**,
and/or one-or-more **sweep IDs**. If neither is usable, ask once.

## Metrics (always report)

Use **max** validation summary values (`unwrap_summary_value(..., "max")`):

| Short name | WandB key pattern |
|------------|-------------------|
| F1 | `val/{task}_f1` |
| AUROC | `val/{task}_auroc` |
| Precision | `val/{task}_precision` |
| Recall | `val/{task}_recall` |
| Balanced accuracy | `val/{task}_balanced_acc` |

Infer `{task}` from run config / logged keys (common:
`neurosoft_acoustic_stim_8band`). Confirm if ambiguous.

## Workflow

### Step 1: Auto-resolve run set and species layout

Using `wandb.Api()`, figure out the layout from whatever was provided.
Do **not** assume one sweep or one ID per species.

**Gather candidate runs**

1. If sweep ID(s) given: load each via
   `api.sweep(f"{entity}/{project}/{sweep_id}")` and collect its runs.
2. If a group is given: also (or instead) fetch
   `api.runs(..., filters={"group": group})`.
3. If both are given: keep the intersection (runs in the group **and** in
   the provided sweep(s)), unless the user clearly wants the union —
   prefer intersection and note what was dropped.
4. Keep finished runs only (skip crashed/running unless the user asks).

**Label species** on each run, in order:

1. Tags: `minipigs` / `monkeys`
2. Else data config / hydra choice path containing `neurosoft_minipigs` or
   `neurosoft_monkeys`
3. Else sweep display name / config path hints (`minipigs`, `monkeys`)
4. Else mark `unknown` and resolve in the decision tree below

**Decide the layout** (inspect species counts × sweep membership):

| What you find | Action |
|---------------|--------|
| Both species in the collected runs (one or many sweeps) | Proceed — single comparative report |
| One sweep ID, both species inside it | Proceed |
| Two (or more) sweep IDs, each mostly one species | Treat as paired species sweeps; proceed |
| Multiple sweeps, mixed/overlapping species | Merge all finished runs; note sweep ID as a column if useful |
| Only one species found | Search the same **group** (and project) for the other species' finished runs / related sweeps. If found, include them and tell the user what was auto-added. If not found, ask for the missing sweep ID or confirm a single-species report |
| Group only, no sweep IDs | Discover sweeps/runs in the group; if multiple unrelated parameter grids exist, ask which factor/sweep to report (or report each as separate files only if they answer different questions) |
| Sweep only, no group | Use the sweep's runs; infer group from run metadata when present |
| Ambiguous / conflicting grids | Show a short inventory (sweep IDs, species counts, varied params) and ask which set to analyze |

Always state the resolved layout briefly before analysis, e.g.:

> Resolved: 2 sweeps (`abc123`=minipigs, `def456`=monkeys), group
> `NEUROSOFT_INTRASESSION_MULTISUBJ`, 36 finished runs.

### Step 2: Identify varied parameters

Union the sweep config `parameters` across all included sweeps, and
cross-check run configs:

- **Varied:** multiple values / a distribution in any included sweep, or
  values that actually differ across collected runs.
- **Fixed context:** single-value everywhere.
- If two species sweeps vary the **same** scientific knobs (possibly with
  different value grids), treat that shared factor as the experiment.
- If the grids differ materially, note the mismatch in Setup and still
  compare on the overlapping factors / best-per-species configs.
- Ignore fold indices in the scientific question (still report fold-wise
  or mean±std). Typical key: `hyperparameters.fold_number`.

Also locate matching Hydra YAML(s) under `configs/sweep/` when obvious
from sweep/display names — use them to refine intentional knobs.

### Step 3: Draft Question and Hypothesis

Derive both from the **varied** parameters before writing results.

**Question** — one clear comparative question, e.g.:

- "Does increasing `model.embed_dim` improve 8-band auditory decoding, and
  does the effect transfer across minipigs and monkeys?"

**Hypothesis** — falsifiable, with expected direction when sensible:

- Prefer a **shared-effect** or **species-interaction** hypothesis (e.g.
  "Larger embed_dim improves max val F1 for both species" or "Gains appear
  in minipigs but not monkeys").
- Do **not** invent magnitude claims without prior experiment context.

Present Question + Hypothesis to the user for a quick confirm/edit before
finalizing the markdown.

### Step 4: Generate analysis script

Create `analysis/<slug>_sweep_report.py` that:

- Takes `PROJECT`, optional `GROUP`, and `SWEEP_IDS` (empty / one / many;
  list or `{species: id}` when known) as constants reflecting the
  **resolved** layout from Step 1
- Re-fetches finished runs the same way Step 1 did (so the script alone
  reproduces the report)
- Extracts species, sweep id, varied HPs, fold, and the five max-val metrics
- Prints comparison tables to stdout (species as columns or a `Species`
  column — **not** separate report sections)
- Saves figures under `analysis/figures/<slug>_*.png` (at least one
  comparison figure for the primary metric, usually F1)
- Uses `analysis/_wandb_utils.py` helpers (`default_entity`,
  `unwrap_summary_value`, `figures_dir`)
- Remains runnable with:
  ```bash
  uv run python analysis/<slug>_sweep_report.py
  ```
- Does **not** import `foundry`

Preferred table shapes:

1. **Best config per species** (by max val F1), with all five metrics
2. **Full grid** (or top-k) with Species + varied params + metrics
3. If folds exist: mean ± std across folds for matched configs

### Step 5: Execute and interpret

Run the script, fix failures, then summarize:

- Which species performs better overall?
- Do the same HP settings win for both?
- Any species × parameter interaction?

Confirm interpretation with the user before writing Conclusions
(same spirit as `run-experiment`).

### Step 6: Write the experiment file

Create `experiments/inbox/YYYYMMDD-<initials>-<slug>.md` using the
standard experiment template from [experiment-tracking](../experiment-tracking/SKILL.md),
with these specifics:

**Status:** `Completed`

**Tags:** include `sweep`, species names, task/band, and varied-parameter
keywords.

**Setup must include:**

- Project, group, sweep ID(s)
- Species detection method
- Varied vs fixed parameters
- Metric key prefix (`val/{task}_*`)

**Results structure (species compared together):**

```markdown
## Results

### Summary
<Short narrative comparing minipigs vs monkeys>

### Metrics
#### Best configuration per species
| Species | <varied params...> | F1 | AUROC | Precision | Recall | Balanced acc | Run |
|---------|--------------------|----|-------|-----------|--------|--------------|-----|
| minipigs | ... | | | | | | name (id) |
| monkeys  | ... | | | | | | name (id) |

#### Full / top-k comparison
<Table with Species column — do not split into minipigs/monkeys subsections>

### Analysis
```bash
uv run python analysis/<slug>_sweep_report.py
```

### Figures
![...](../../analysis/figures/<slug>_....png)
```

**Conclusions:** verdict on the hypothesis with evidence from both species
(or explicitly note if the user confirmed a single-species report).

**Notes for future experiments:** brief next steps (ask the user; suggest
defaults if they have none).

## Anti-patterns

- Do **not** demand a fixed input shape (one vs two sweep IDs) — resolve it
- Do **not** create separate Results subsections titled "Minipigs" / "Monkeys"
- Do **not** hardcode metric numbers in markdown without an analysis script
- Do **not** treat fold as the scientific factor in Question/Hypothesis
- Do **not** silently drop a species when it can be recovered from the group
- Do **not** skip the user confirmation of Question/Hypothesis/Conclusions

## Relationship to other skills

| Skill | When to use instead |
|-------|---------------------|
| [run-experiment](../run-experiment/SKILL.md) | Non-sweep runs, or an inbox draft already exists and only needs filling |
| [create-experiment](../create-experiment/SKILL.md) | Planning before any sweep has run |
| [archive-group](../archive-group/SKILL.md) | Bundling several completed sweep reports into a thematic group |
