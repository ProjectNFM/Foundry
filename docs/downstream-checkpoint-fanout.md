# Checkpoint-to-downstream fan-out

Foundry compiles downstream runs from two declarations:

- A checkpoint-set JSONL registry contains one selected checkpoint per row.
  The checkpoint manifest remains authoritative; the registry records the
  selection and expected hashes/metadata used to reject stale or accidental
  inputs.
- A downstream YAML recipe defines target eligibility, experiment configs,
  regimes, fractions, target seeds, W&B metadata, expected matrix counts, and
  optional `fixed_overrides` copied into every compiled cell. Use fixed
  overrides to pin critical scientific settings that must not drift with a
  base config.

The compiler validates every checkpoint-manifest self-hash, checkpoint SHA-256,
source-selection-manifest self-hash, excluded target/species, source selection
and model seed, source recording identities and class summaries, forbidden-test
policy, zero target leakage, requested source fraction, positive finite source
validation, and duplicate scientific identities before expanding eligible
audit sessions. It writes deterministic JSONL cell vectors and an atomic lock
file backed by input, output, and compiler-content digests. Run validation
without writing output with `--check`:

```bash
python tools/compile_downstream_cells.py \
  --registry launch/checkpoint_sets/phase4a-mila-best.jsonl \
  --recipe configs/downstream_recipes/phase4a_full_finetuning.yaml \
  --audit docs/neurosoft-phase0-audit.json \
  --checkpoint-root /network/scratch/s/sobralm/foundry-checkpoints \
  --output-dir launch/phase4a \
  --check
```

The Phase 4A outputs are
`launch/phase4a/phase4a-downstream-minipigs.jsonl` and
`launch/phase4a/phase4a-downstream-monkeys.jsonl`. Changing transfer regime or
target fraction is a recipe change. New source mixtures, fractions, milestone
kinds, and model seeds are represented by new registry records.

The corrected Phase 4A matched-LR rerun uses
`configs/downstream_recipes/phase4a_full_finetuning_lr1p5e3.yaml` and emits
`phase4a-downstream-lr1p5e3-{minipigs,monkeys}.jsonl`. It pins
`hyperparameters.learning_rate=0.0015` inside every cell and uses new cell and
W&B identities. The original lists are retained only as provenance for the
invalidated `0.00025` downstream arm.

## Mila launch shape (later, not part of artifact compilation)

Production submission requires a clean committed repository and
`FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches`. The legacy
cluster partition is `long`. Run one normal Hydra multirun per species:

```bash
python main.py \
  experiment=auditory_decoding/neurosoft_conv_bigru_transfer_minipigs \
  hydra/launcher=slurm_default \
  hydra.launcher.partition=long \
  hydra.launcher.cell_list=launch/phase4a/phase4a-downstream-minipigs.jsonl \
  hydra.launcher.tasks_per_node=<benchmarked-value> \
  -m
```

Use the symmetric monkey config/list for monkeys. `tasks_per_node` packs that
many independent cell processes concurrently on one GPU. It neither runs a
sequential checkpoint worker nor shares a Python model instance. Do not choose
the production packing value from configuration alone: run a one-cell canary,
then compare valid-cell throughput at two and optionally four concurrent cells.

The controlled Mila launch lists are deliberately separate from the complete
matrix:

- `launch/phase4a/canaries/minipigs-tpn1.jsonl`
- `launch/phase4a/canaries/monkeys-tpn1.jsonl`
- `launch/phase4a/canaries/minipigs-tpn2.jsonl`
- `launch/phase4a/canaries/monkeys-tpn2.jsonl`

The one-cell lists use source/target seed 42. The two-cell lists hold the target
recording fixed and use two target seeds from source seed 43, so they benchmark
two previously unexecuted cells without rerunning a completed canary. Only
create four-cell lists after the two-cell measurements demonstrate credible
memory and throughput headroom.

SLURM requeue resumes a cell's own `last.ckpt`. Foundry validates the persisted
compiled-cell identity and skips source transfer on resume. Static packed
Submitit arrays do not provide a durable completion queue: a later manual
resubmission must use an explicitly filtered cell list (or a separate future
completion-planning step) to avoid rerunning completed cells.
