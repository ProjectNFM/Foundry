# Bug: `stage.skip: true` breaks data loading under snapshot launches

**Date:** 2026-08-19
**Affected jobs:** All DATA_SCALING and NEUROSOFT_8B_LOSO_PRETRAIN jobs submitted 2026-08-18
**Fix:** Unify staging and data-root rebasing inside `main.py`

## Symptom

Jobs fail immediately with:

```
ValueError: No recordings found at data/processed/kemp_sleep_edf_2013
```

Data exists on scratch, `stage_data` successfully copies 65 GB to `/tmp`, but
training never reads from it.

## Root cause

Two independent mechanisms interact badly:

1. **Snapshot launcher** — The Hydra/Slurm launcher creates a read-only
   git-archive snapshot and `cd`s into it. Because `data/` is gitignored, the
   snapshot has no `data/` directory. The relative config path
   `root: ./data/processed/` now resolves to a non-existent location.

2. **`stage.skip: true`** — The `stage_data` setup command in the submission
   script stages data to `$SLURM_TMPDIR/brainsets/processed/` and prints the
   new root, but nothing captures that output. Inside `main.py`,
   `_stage_data_if_needed()` checks `stage.skip` and returns early without
   updating `data.root`. The config keeps the broken relative path.

Before snapshots this was masked: the working directory was the live checkout
where `data/processed` was a symlink to scratch. The staged `/tmp` copy was
silently ignored — training read from scratch instead.

## Fix

The `stage.skip` workaround was not retained. Setting it to `false` would make
each task enter `stage_data()` a second time and would resolve the default
relative source path from inside the snapshot bundle.

Instead:

- `main.py` is the sole staging orchestrator and always updates `data.root`
  from the staging result;
- `stage.mode` explicitly selects `node_local` or `direct` data access;
- packed tasks serialize staging with a node-local file lock, then reuse files
  already present at the destination;
- launcher setup commands no longer invoke `stage_data`, so staging uses the
  exact snapshot code and the exact task overrides;
- source and archive roots come from the composed `stage` config; and
- Hydra task failures are re-raised so Submitit and Slurm report a failed job.

This removes the duplicated launcher/application staging paths rather than
requiring data to be staged twice.
