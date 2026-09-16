# Source-pretraining architecture viability

**Status:** Draft
**Date started:** 2026-09-16
**Parent experiment:** [Source-validation performance versus downstream transfer](20260915-MS-source-validation-downstream-trajectory.md)
**Follow-up experiments:** [Small-backbone checkpoint transfer](20260916-MS-small-backbone-transfer.md), [Large-backbone checkpoint transfer](20260916-MS-large-backbone-transfer.md), [Bias-free checkpoint transfer](20260916-MS-bias-free-transfer.md), [Shared-adapter checkpoint transfer](20260916-MS-shared-adapter-transfer.md), [Transfer failure-mode synthesis](20260916-MS-transfer-failure-mode-synthesis.md)
**Tags:** neurosoft, supervised-pretraining, architecture, capacity, adapter-bias, shared-adapter, checkpoint-age, 8band, minipigs, monkeys

## Background

The parent [source-validation versus downstream-transfer experiment](20260915-MS-source-validation-downstream-trajectory.md)
showed that the default `NeurosoftConvBiGRU` has a genuine early transfer
benefit that erodes with continued source training. At source step 100,
transfer exceeded matched scratch in both species; by step 10,000 that benefit
had disappeared and downstream optimization efficiency had deteriorated.

The related [adapter-bias perturbation experiment](20260915-MS-adapter-bias-perturbation.md)
showed that reliance on correctly matched recording-specific adapter biases
increases with source-checkpoint age. That perturbation was causal for source
predictions but did not establish that the bias caused downstream transfer
erosion. Its proposed follow-up was to retrain the complete source pipeline
without adapter biases.

This experiment implements and source-trains four interventions before any new
downstream matrix is launched:

1. an approximately ten-times-smaller transferable backbone;
2. an approximately ten-times-larger transferable backbone;
3. recording-specific input adapters without bias; and
4. one shared input adapter that right-pads the filtered channel sequence with
   zeros to a fixed width of 32 and deliberately ignores channel names.

The experiment is an implementation and viability gate. It asks whether all
four models learn non-degenerate source-task solutions under the matched Phase
4E recipe, characterizes their learning dynamics, and publishes the fixed
checkpoint manifests required by the linked downstream experiments. It does
not test downstream transfer.

## Question

Can the small-backbone, large-backbone, bias-free-adapter, and shared-padded-
adapter variants all be trained stably under the matched Phase 4E source
recipe and learn non-degenerate source-task solutions suitable for fixed-
checkpoint downstream evaluation?

## Hypothesis

All four variants will complete 10,000 source optimizer steps with finite
metrics and will show meaningful source-task learning from step 100 to step
10,000: subject-balanced source-validation cross-entropy will decrease,
supported macro-F1 will increase, and late-checkpoint predictions will not
collapse persistently to one class. The joint hypothesis is fully supported
only if every variant meets these common criteria in both species; otherwise
it is partially supported with a separate downstream-readiness decision for
each variant and species.

Differences in absolute source performance, learning speed, train-validation
gap, compute, or channel-count sensitivity are prespecified characterization
outcomes. They do not create separate hypothesis tests and a variant is not
rejected merely for underperforming the default source model.

## Experiment

### Setup

- **Model:** Four variants of `NeurosoftConvBiGRU`, compared with the existing
  default Phase 4E seed-42 source runs.
- **Data:** The existing same-species, target-subject-excluded full source
  pools: seven minipig exclusions and five monkey exclusions. Every new
  condition uses source-selection seed 42 and the exact corresponding
  `fraction-1.00/selection-42.json` manifest.
- **Task:** NeuroSoft eight-band acoustic-stimulus classification.
- **Source seed:** Model initialization seed 42 only. Source-selection and
  model seeds are deliberately paired as `(42, 42)` in every condition.
- **Training:** The final Phase 4E train-global-normalized source recipe:
  10,000 optimizer steps, validation every 100 steps, no early stopping,
  batch size 16, learning rate `2.5e-4`, weight decay `0.018`, and the existing
  precision fallback policy.
- **Fixed checkpoints used by this program:** Steps 100, 300, 1,000, 3,000,
  and 10,000.
- **Saved but currently excluded checkpoint:** Each run must retain its
  minimum-source-validation-loss checkpoint and manifest. It is excluded from
  this experiment's ordered analyses, all initial downstream registries, and
  all confirmation criteria.
- **Existing reference:** The 12 default-backbone Phase 4E source runs with
  source-selection/model seed `(42, 42)`.
- **WandB:** Planned groups
  `20260916-MS-PRETRAINING-ARCHITECTURE-VIABILITY-MINIPIGS` and
  `20260916-MS-PRETRAINING-ARCHITECTURE-VIABILITY-MONKEYS`; exact run names and
  eight-character run IDs will be recorded after launch.

The new source matrix is:

| Condition | Adapter | Transferable backbone | Minipig runs | Monkey runs | Total |
|---|---|---:|---:|---:|---:|
| Small backbone | Recording-specific, bias enabled | 51,066 (0.100631x) | 7 | 5 | 12 |
| Large backbone | Recording-specific, bias enabled | 5,084,598 (10.019781x) | 7 | 5 | 12 |
| Bias-free adapter | Recording-specific, no bias | 1x | 7 | 5 | 12 |
| Shared padded adapter | One shared `32 -> 64` biased projection | 1x | 7 | 5 | 12 |
| **Total new runs** |  |  | **28** | **20** | **48** |

Capacity is defined on the transferable `temporal_frontend + gru`, not on the
complete source model whose recording-specific adapter parameter count varies
with the source pool. The intended capacity configurations keep
`adapter_dim=64`, temporal kernel 64, stride 4, convolutional depth 1, and two
bidirectional GRU layers fixed. The default temporal/GRU widths of 128 contain
exactly 507,456 transferable parameters. Exact enumeration selected widths 38
for the small model (51,066 parameters, 0.100631x default) and 410 for the
large model (5,084,598 parameters, 10.019781x default).

The shared adapter takes the modality-filtered channels in their existing
order, right-pads missing trailing positions with zeros to exactly 32 channels,
and applies one `nn.Linear(32, 64, bias=True)` to every recording. It performs
no channel-name lookup, vocabulary construction, anatomical alignment, or
recording-specific parameter selection. Padding is applied after input
normalization and time padding is re-zeroed after the linear layer.

Every run must publish five fixed milestone manifests, one loss-selected
manifest, the normal final checkpoint, exact model and transferable parameter
counts, and compute metadata. The expected new fixed-checkpoint inventory is
`48 * 5 = 240` manifests. The expected saved-but-excluded loss-selected
inventory is 48 manifests.

#### Source analysis and readiness gate

The primary ordered analysis uses only the five fixed checkpoints. For each
condition and species, aggregate source metrics within excluded target subject
and estimate slopes against `ln(source_step)`. Bootstrap whole excluded target
subjects, preserving each subject's complete trajectory.

Each condition/species receives one readiness label:

- **Ready:** all runs and manifests are complete; metrics are finite;
  subject-balanced validation CE improves and supported F1 increases over the
  ordered trajectory; late predictions are not persistently single-class.
- **Ready with caveat:** training is finite and non-collapsed but source
  learning is weak, irregular, or concentrated in only part of the source
  pool. The condition may remain scientifically relevant for transfer.
- **Not ready:** numerical instability, implementation failure, corrupt or
  incomplete checkpoint provenance, or persistent task-learning collapse.

Characterization includes train and validation CE/F1 curves, train-validation
gaps, predicted-class counts, source performance by filtered channel count for
the shared adapter, parameter counts, FLOPs, throughput, memory, and wall time.
No target validation or test result enters this gate.

### Launch command

```bash
# Production launch must use the normal snapshot workflow from a clean
# committed repository. Generate and audit the immutable cells first.
export FOUNDRY_DATA_ROOT=/network/scratch/s/sobralm/brainsets/processed
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches
export FOUNDRY_CHECKPOINT_ROOT=/network/scratch/s/sobralm/foundry-checkpoints
# The RTX 8000 source reference used FP16 through the configured precision
# fallback.  Pin it here so all new conditions use the same hardware/precision
# regime, and avoid the node that failed before interpreter initialization.
export FOUNDRY_ENV_FILE=/home/mila/s/sobralm/Foundry/.venv/bin/activate

git status --short  # must print nothing

uv run python tools/generate_architecture_viability_source_cells.py
uv run python tools/audit_architecture_viability_parity.py --json

uv run python main.py \
  experiment=pretraining/neurosoft_conv_bigru_supervised_minipigs \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/20260916-MS-PRETRAINING-ARCHITECTURE-VIABILITY-MINIPIGS \
  'hydra.sweep.subdir=${run.name}' \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 hydra.launcher.timeout_min=480 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/architecture_viability/source-minipigs.jsonl \
  -m

uv run python main.py \
  experiment=pretraining/neurosoft_conv_bigru_supervised_monkeys \
  hydra.sweep.dir=/network/scratch/s/sobralm/runs/20260916-MS-PRETRAINING-ARCHITECTURE-VIABILITY-MONKEYS \
  'hydra.sweep.subdir=${run.name}' \
  hydra/launcher=slurm_default hydra.launcher.partition=long \
  hydra.launcher.gres=gpu:rtx8000:1 hydra.launcher.timeout_min=480 \
  +hydra.launcher.additional_parameters.exclude=cn-c004 \
  hydra.launcher.cell_list=launch/architecture_viability/source-monkeys.jsonl \
  -m
```

### Key config overrides

- Common: `trainer.max_steps=10000`, `trainer.val_check_interval=100`,
  `trainer.callbacks.early_stopping=null`, and validation-loss checkpoint
  selection retained for artifact creation only.
- Small/large: capacity-specific `model.temporal_channels` and
  `model.gru_hidden_size`, with `model.adapter_dim=64` and two GRU layers.
- Bias-free: planned `model.input_adapter_mode=per_session` and
  `model.input_adapter_bias=false`.
- Shared: planned `model.input_adapter_mode=shared_padded`,
  `model.input_adapter_bias=true`, and `model.shared_input_channels=32`.
- Every run and manifest must record `source_condition`, adapter mode, adapter
  bias, exact parameter counts, source-selection seed, model seed, excluded
  target subject, species, Git SHA, and snapshot bundle.

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD. The W&B-backed analysis script will be
`analysis/20260916-MS-pretraining-architecture-viability_analysis.py`.

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

Proceed only with condition/species pairs declared Ready or Ready with caveat.
The linked downstream experiment files prespecify the intended analyses but
must be revised rather than silently reinterpreted if this source-stage gate
reveals an implementation defect, collapse, or materially different model
than planned.
