# Plan: NeuralBench Integration via NeuralSet Runtime Adapter

**Status:** Agreed design; not yet implemented.

## Objective

Integrate NeuralBench into Foundry by building a runtime adapter that uses
NeuralSet dataloaders to feed NeuralBench-defined tasks into Foundry's
existing training stack (FoundryModule, Lightning, Hydra). This enables
evaluating Foundry models (POYO-EEG, EEGNet, baselines) on explicitly pinned
NeuralBench task--dataset pairs with the same splits, preprocessing, and task
definitions. A result is comparable to that specific NeuralBench protocol; it
is not a claim about the default task dataset, a pooled task score, or the
leaderboard unless each of those is separately reproduced.

NeuralBench remains the external task specification. Foundry remains the
training runtime and model host. NeuralSet provides the data at runtime.

## Agreed decisions

| Decision | Choice |
|----------|--------|
| Data strategy | NeuralSet bridge at runtime (not offline H5 conversion) |
| Training stack | Foundry's full pipeline: FoundryModule + Lightning + Hydra |
| Models | Foundry models are the adapter targets; NeuralBench EEGNet is run separately only as the POC reference |
| Config style | Hydra experiment configs (`experiment=neuralbench/p300`) |
| Adapter depth | Wrap NeuralSet output as `torch_brain.Data` objects to preserve full tokenization pipeline |
| Data caching | NeuralSet native caching on `/network/scratch` |
| Initial evaluation | Validation only; best-checkpoint test evaluation is deferred |
| Seed / eval | Pinned NeuralBench split seed and task settings; Foundry controls model/trainer seed |
| Dependency | Optional `uv` dependency group (`uv sync --group neuralbench`) |
| Version pinning | Strict (`neuralbench==X.Y.Z`) |
| Validation | Data-level checks in CI + reference model comparison milestone |
| WandB tracking | Dedicated project (`foundry-neuralbench`) |
| First task | `p3` / `Korczowski2014A` (Brain Invaders P300), at one pinned NeuralBench release |
| Pooled results | A later, explicitly defined Foundry pooling protocol; never implicit in a single-dataset run |
| Channel metadata | Complete, stable NeuralSet channel names are required; electrode positions are not required by POYO-EEG |

## Scope

### Candidate follow-on tasks (after the POC)

| NeuralBench task | Core dataset | Foundry equivalent |
|------------------|-------------|--------------------|
| P300 | Schreuder2010New is the task default; POC uses the alternate `Korczowski2014A` dataset | `BrainInvadersP300` |
| Motor Imagery | Stieger2021 + 17 extra datasets incl. PhysioNet | `PhysionetMI` |
| Sleep Stage | Kemp2000 | `KempSleepEDF2013` |

### Future expansion

The adapter infrastructure is intended to be reusable, but task onboarding is
not assumed to be configuration-only. Regression, retrieval, recording-level
metrics, or non-standard epochs may require a task-specific adapter contract.
No new Foundry H5 dataset class is required per task.

---

## Architecture

### Component overview

```
NeuralBench YAML task spec (splits, preprocessing, metrics, epoch def)
    │
    ▼
NeuralSet (Study → Events → Segmenter → SegmentDataset)
    │  produces PyTorch Dataset yielding {eeg: (C, T), label: ..., metadata: ...}
    │
    ▼
NeuralSetAdapter (foundry/data/neuralbench/adapter.py)
    │  wraps each NeuralSet sample as a torch_brain Data object
    │  with signal, intervals, channel IDs, session/subject metadata
    │
    ▼
NeuralBenchDataModule (foundry/data/neuralbench/datamodule.py)
    │  replaces NeuralDataModule for NeuralBench tasks
    │  uses NeuralSet's native split assignments
    │  uses IndexSampler (not window sampler) since epochs are pre-windowed
    │
    ▼
Foundry's standard pipeline (model.tokenize → collate → FoundryModule → trainer)
```

### 1. `NeuralSetAdapter` — bridge NeuralSet output to torch_brain Data

NeuralSet's `SegmentDataset` yields dictionaries of tensors keyed by extractor
name. For EEG tasks, the primary output is an `(n_channels, n_timepoints)`
tensor plus label information. Foundry's models expect `torch_brain.Data`
objects with signal arrays, interval-based labels, channel metadata, and
session/subject identifiers.

The adapter is a thin wrapper class:

```python
class NeuralSetAdapter(torch.utils.data.Dataset):
    """Wraps a NeuralSet SegmentDataset, converting each sample to torch_brain Data."""

    def __init__(
        self,
        segment_dataset,        # NeuralSet SegmentDataset
        task_config,            # NeuralBench task metadata (sampling rate, channels, etc.)
        split: str,             # "train", "valid", "test"
    ): ...

    def __len__(self) -> int:
        return len(self.segment_dataset)

    def __getitem__(self, idx: int) -> Data:
        raw = self.segment_dataset[idx]
        return self._to_torch_brain_data(raw)

    def _to_torch_brain_data(self, raw: dict) -> Data:
        # 1. Extract EEG tensor (C, T) → build RegularTimeSeries or equivalent
        # 2. Build Interval for the trial label (task target)
        # 3. Attach complete, canonical channel IDs
        # 4. Attach session.id, subject.id from NeuralSet metadata
        # 5. Return torch_brain.Data object
        ...
```

Key responsibilities:

- **Signal conversion**: NeuralSet provides preprocessed `(C, T)` tensors.
  Convert to `torch_brain`'s signal representation (timestamps + values or
  `RegularTimeSeries`). The signal spans `[0, duration]` for the epoch window.

- **Channel identity**: NeuralSet channel names are required for Foundry's
  channel-ID embeddings. Preserve them in a documented canonical form and
  fail with a useful diagnostic for empty or duplicate names. Do not invent
  names, add coordinates, or use an equidistant-layout fallback. The pooling
  protocol will later define whether same-named channels across datasets share
  an embedding or are namespaced.

- **Label conversion**: NeuralSet labels (int class index, regression target,
  or retrieval embedding) are converted to `torch_brain.Interval` objects with
  the appropriate attribute (e.g., `p300_trials` for P300, `sleep_stage` for
  sleep staging).

- **Session identity**: Construct synthetic session IDs from NeuralSet's
  subject/recording metadata (e.g., `"sub-001_run-0"`). This ensures
  Foundry's per-session logic (channel uniquification, session embeddings)
  works correctly.

### 2. `NeuralBenchDataModule` — parallel DataModule for NeuralBench tasks

NeuralBench tasks use pre-windowed epochs with fixed splits (not continuous
recordings with random window sampling). This fundamental difference means
`NeuralDataModule` cannot be directly reused. Instead, a parallel
`NeuralBenchDataModule` manages the NeuralSet pipeline and feeds adapted data
to Foundry's training loop.

```python
class NeuralBenchDataModule(LightningDataModule):
    """DataModule that uses NeuralSet to serve NeuralBench tasks."""

    def __init__(
        self,
        task: str,                    # NeuralBench task ID (e.g., "p3", "mi", "sleep_stage")
        dataset: str,                 # NeuralBench dataset (e.g., "Korczowski2014A")
        neuralbench_version: str,     # Pinned version for reproducibility
        cache_dir: str,               # NeuralSet cache path
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        tokenizer: Callable = None,
        task_configs: dict = None,
    ): ...

    def prepare_data(self):
        # NeuralSet download + cache preparation (runs once, single process)
        # neuralbench <modality> <task> --download --prepare
        ...

    def setup(self, stage=None):
        # 1. Load NeuralBench task config YAML (splits, preprocessing, epoch def)
        # 2. Build NeuralSet Study + EventTransforms + Segmenter per split
        # 3. Create SegmentDatasets for train/valid/test
        # 4. Wrap each in NeuralSetAdapter
        ...

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_adapter,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate,       # torch_brain collate (same as standard Foundry)
            num_workers=self.num_workers,
            ...
        )
```

Key differences from `NeuralDataModule`:

| Concern | NeuralDataModule | NeuralBenchDataModule |
|---------|-----------------|----------------------|
| Data source | Foundry H5 files via torch_brain Dataset | NeuralSet SegmentDataset |
| Sampling | Window-based (FastRandomFixedWindowSampler) | Index-based (epochs are pre-windowed) |
| Splits | Dataset's `get_sampling_intervals(split)` | NeuralSet's EventTransform split assignment |
| Preprocessing | Dataset-side transforms + recording-wide normalization | NeuralSet extractors (MNE pipeline) |
| Channel metadata | Stored in H5 | Canonical names from NeuralSet metadata |

The rest of the pipeline is identical: `model.tokenize()` is still applied as
a transform, `torch_brain.batching.collate` is still used for batching, and
`FoundryModule` handles loss/metrics/optimization.

### 3. Task configuration mapping

Each NeuralBench task defines metrics, loss, and evaluation protocol. These
must map to Foundry's task system (`configs/tasks/`).

Create NeuralBench-specific task configs under `configs/tasks/neuralbench/`:

```yaml
# configs/tasks/neuralbench/p300.yaml
p300:
  head:
    _target_: foundry.tasks.heads.ReadoutHead
    output_dim: 2
  loss:
    _target_: foundry.tasks.losses.CrossEntropyTaskLoss
  metrics:
    balanced_accuracy:
      _target_: torchmetrics.classification.MulticlassAccuracy
      num_classes: 2
      average: macro
  target_extractor:
    interval_key: p300_trials
    value_key: p300_trials.label
  monitor: val/p300/balanced_accuracy
  monitor_mode: max
```

For each NeuralBench task, define a matching Foundry task config that:
- Uses NeuralBench's declared loss and primary metric
- Matches the number of output classes / regression dimensions
- Sets the correct monitored quantity for early stopping / checkpoint selection

The YAML above is structural only. Before the POC comparison it must be
updated from the captured effective NeuralBench configuration, including loss
options, target representation, class weighting/sampling, and metric naming.

### 4. Hydra experiment configs

Each NeuralBench task gets a Hydra experiment config:

```yaml
# configs/experiment/neuralbench/p300.yaml
# @package _global_

defaults:
  - override /model: poyo_eeg
  - override /module: default
  - override /tasks: neuralbench/p300

data:
  _target_: foundry.data.neuralbench.datamodule.NeuralBenchDataModule
  task: p3
  dataset: Korczowski2014A
  neuralbench_version: "X.Y.Z"
  cache_dir: ${oc.env:NEURALSET_CACHE_DIR,/network/scratch/s/sobralm/neuralset-cache}
  batch_size: 64
  num_workers: 4

hyperparameters:
  sequence_length: 1.0    # NeuralBench P300 epoch duration
  latent_step: 0.02

run:
  name: neuralbench-p300-${model.name}

logger:
  wandb:
    project: foundry-neuralbench
    tags: [neuralbench, p300, ${model.name}]

trainer:
  max_epochs: 50
  callbacks:
    early_stopping:
      monitor: val/p300/balanced_accuracy
      mode: max
      patience: 10
```

Launch with standard Foundry workflow:

```bash
# Single run
uv run python main.py experiment=neuralbench/p300

# 3-seed protocol (matches NeuralBench)
uv run python main.py experiment=neuralbench/p300 seed=33,34,35 -m

# Sweep models
uv run python main.py experiment=neuralbench/p300 model=poyo_eeg,eegnet,shallow_cnn -m
```

### 5. Dependency management

Add NeuralBench/NeuralSet as an optional dependency group in `pyproject.toml`:

```toml
[dependency-groups]
neuralbench = ["neuralbench==X.Y.Z"]
```

With `uv`:

```bash
uv sync --group neuralbench
```

All NeuralBench-related code must be importable only when the dependency is
installed. Guard imports with try/except or lazy imports. The core Foundry
package must not break when neuralbench is absent.

---

## File structure

```
foundry/
  data/
    neuralbench/
      __init__.py
      adapter.py              # NeuralSetAdapter: NeuralSet sample → torch_brain Data
      datamodule.py           # NeuralBenchDataModule: LightningDataModule
      task_registry.py        # NeuralBench task ID → Foundry config mapping
      utils.py                # NeuralSet initialization, version checking

configs/
  experiment/
    neuralbench/
      p300.yaml               # P300 task experiment config
      motor_imagery.yaml      # Motor Imagery task experiment config
      sleep_stage.yaml        # Sleep Staging task experiment config
  tasks/
    neuralbench/
      p300.yaml               # Foundry task config for P300
      motor_imagery.yaml      # Foundry task config for MI
      sleep_stage.yaml        # Foundry task config for Sleep
tests/
  test_neuralbench_adapter.py     # Adapter unit tests
  test_neuralbench_datamodule.py  # DataModule integration tests
```

---

## Implementation phases

### Proof of concept — `p3` / `Korczowski2014A`

The first deliverable is a narrowly scoped feasibility proof, not a benchmark
campaign. It decides whether the live NeuralSet adapter is a viable direction
by proving both data-path compatibility and a single Foundry EEGNet validation
comparison against NeuralBench's EEGNet on the same pinned task--dataset pair.
The comparison uses validation only. It does not report a final test result,
pool datasets, or make a leaderboard claim.

#### POC phase 0 — Capture the NeuralBench reference contract

1. Add a strict `neuralbench==X.Y.Z` dependency group and lock the resolved
   environment, including NeuralSet and relevant source revisions.
2. Prepare `neuralbench eeg p3 --dataset Korczowski2014A` in a dedicated
   shared cache. Record the exact invocation and effective task configuration.
3. Capture split membership, segment count, label distribution, EEG shape,
   dtype, sampling rate, units, channel order/names, segment metadata, and
   preprocessing steps for each split.
4. Confirm the P3 contract actually selected by the release: epoch timing,
   baseline semantics, target encoding, class weighting/sampling, loss,
   validation metric, early-stopping rule, and random seeds.
5. Compare source subject/recording identifiers to Foundry's Brain Invaders
   inventory only as an identity audit; the POC runs from NeuralSet, not H5.

**Gate:** a checked-in provenance note (with no dataset tensors) unambiguously
identifies the release, task, dataset, reference command, and observed sample
contract. Any unknown sample field or preprocessing operation blocks the next
phase.

#### POC phase 1 — Minimal live adapter and model-forward proof

1. Implement `NeuralSetAdapter` in `foundry/data/neuralbench/adapter.py` to
   convert a NeuralSet segment into `torch_brain.Data` without altering its
   EEG values, channel order, labels, or timing.
2. Convert labels into the exact Foundry interval/attribute form needed by the
   P3 task config. Handle NeuralBench's actual encoded-label representation;
   do not assume integer labels.
3. Attach stable `subject.id`, `session.id`, and complete canonical
   `channels.id` values from NeuralSet metadata. Reject missing or duplicate
   channel names with a diagnostic.
4. Implement `NeuralBenchDataModule` in
   `foundry/data/neuralbench/datamodule.py`, with fixed index-based split
   datasets, `torch_brain.batching.collate`, and the same `set_tokenizer()`
   behavior as `NeuralDataModule`.
5. Supply the metadata interface that Foundry startup and lazy vocabularies
   require: channel IDs, session configuration/counts, and training-label
   counts. Do not rely on the H5-only `dm.dataset` contract without providing
   an equivalent adapter facade or making the required `main.py` contract
   change explicit.
6. Establish the tokenizer ordering explicitly: adapter output is transformed
   by `model.tokenize()` before collation, including when startup calls
   `setup("fit")` before the model exists.
7. Add focused fixtures/tests for signal preservation, timing, labels, channel
   IDs, split membership, collation, and lazy-vocabulary initialization.

**Gate:** batches from the live `p3` / `Korczowski2014A` adapter successfully
pass through both Foundry POYO-EEG and Foundry EEGNet. For selected samples,
the pre-tokenization Foundry signal, label, channel order, and segment timing
match NeuralSet's output within an explicitly recorded floating-point
tolerance.

#### POC phase 2 — Single-run EEGNet validation comparison

1. Add a Foundry P3 task config and Hydra experiment config for this exact
   task--dataset pair. Mirror every NeuralBench setting that affects validation:
   split assignment, preprocessing, target encoding, class weighting/sampler,
   loss (including its options), validation metric, model seed, trainer seed,
   epoch budget, and early-stopping/selection rule.
2. Before training, compare Foundry EEGNet and NeuralBench EEGNet architecture
   and all trainable/training hyperparameters. If they are not identical,
   label the result an *implementation comparison*, not a replication, and
   list every difference.
3. Run one NeuralBench EEGNet reference training invocation and one Foundry
   EEGNet invocation against the same prepared cache and pinned contract.
   Capture the selected validation score, epoch, learning-rate schedule, and
   per-split event/label counts from both runs.
4. Produce a short parity report that first rules out split, preprocessing,
   tensor, target, and metric discrepancies, then reports the absolute
   validation balanced-accuracy difference. Do not compare a Foundry score
   with a published aggregate or a differently seeded run.
5. Make the observed data checks repeatable in offline CI with mocked
   NeuralSet samples; full-cache/reference execution remains an optional
   integration check.

**Gate:** the adapter passes the two-model forward proof; Foundry and
NeuralBench use demonstrably equivalent data/task contracts; and the report
states whether the matched EEGNet run replicated the reference validation
result or identifies the remaining implementation-level difference. A score
alone never passes this gate.

### Phase 1 — Expand to Motor Imagery and Sleep Staging

Reuse the adapter infrastructure for two more tasks.

1. Run `neuralbench eeg mi --download --prepare` and
   `neuralbench eeg sleep_stage --download --prepare`. Inspect outputs.
2. Write task configs for MI and Sleep Staging
   (`configs/tasks/neuralbench/motor_imagery.yaml`,
   `configs/tasks/neuralbench/sleep_stage.yaml`).
3. Write experiment configs
   (`configs/experiment/neuralbench/motor_imagery.yaml`,
   `configs/experiment/neuralbench/sleep_stage.yaml`).
4. Resolve any task-specific adapter issues:
   - MI may use different channel montages across datasets.
   - Sleep Staging uses 30-second windows with multi-class labels.
   - Different split strategies (cross-subject for MI, predefined for Sleep).
5. Validate data and metric fidelity for both tasks (same protocol as
   the POC data- and metric-fidelity protocol).

**Gate:** All three tasks run end-to-end. Each produces metrics in the
`foundry-neuralbench` WandB project.

### Phase 2 — Generic task onboarding and documentation

Make adding new NeuralBench tasks low-friction.

1. Refactor any task-specific logic out of the adapter/datamodule into
   configuration. The goal: adding a new NeuralBench task requires only:
   - A Foundry task config YAML (loss, metrics, target mapping)
   - A Hydra experiment config YAML
   - Any required typed task-adapter contract beyond YAML
2. Build a task registry (`foundry/data/neuralbench/task_registry.py`) that
   maps NeuralBench task IDs to Foundry configurations.
3. Write developer documentation: how to add a new NeuralBench task to
   Foundry (step-by-step).
4. Add a CI job that validates the adapter against mocked NeuralSet data for
   all onboarded tasks.

**Gate:** A new contributor can onboard a task matching an existing adapter
contract with configuration and a validation script; unsupported task shapes
are identified before implementation.

---

## Technical risks and mitigations

### NeuralSet API instability

NeuralBench was released in May 2026 and the API may evolve. Strict version
pinning protects against breaking changes. If NeuralSet makes backward-
incompatible changes, the adapter pins to the old version until an explicit
upgrade is performed and validated.

### torch_brain Data format mismatch

NeuralSet produces preprocessed `(C, T)` tensors. torch_brain's `Data` object
expects a specific signal representation. The adapter must construct valid Data
objects that the tokenizer can process. Risk: if torch_brain's internal format
changes, the adapter breaks. Mitigation: comprehensive unit tests that verify
Data object structure, plus integration tests that pass adapted data through
the full tokenizer pipeline.

### Channel-name integrity

Foundry POYO-EEG needs valid channel IDs even though it does not need electrode
positions. NeuralSet datasets can expose absent, duplicated, auxiliary, or
inconsistently formatted names. Mitigation: record the observed names and
ordering in the POC contract, canonicalize only documented formatting changes,
and fail the run on missing or duplicate names rather than silently inventing
or dropping channels.

### Data download on compute nodes

NeuralSet downloads data on first use, while SLURM compute nodes may have no
internet access. Mitigation: prepare the version-pinned cache on a login node
before production job submission, record its provenance, and have jobs verify
rather than mutate it. The cache lives on shared network storage
(`/network/scratch`) so every process sees the same prepared data.

### NeuralSet caching and storage overhead

NeuralSet's caching system stores preprocessed tensors. For large benchmarks
(94 datasets), this can require significant storage. Mitigation: start with
only the 3 overlapping tasks. Document storage requirements per task. Use
NeuralSet's cache-sharing so multiple runs don't duplicate cached data.

### Foundry tokenizer compatibility

NeuralSet's preprocessed data may not include all the information Foundry's
tokenizers expect (e.g., raw signal for CWT tokenizer, specific sampling rates
for patch-based tokenizers). Mitigation: validate tokenizer compatibility for
each task during Phase 0 exploration. Some tokenizer variants may not work with
NeuralSet data; document which ones are supported.

---

## Non-goals

- Replacing Foundry's existing H5-based pipeline or torch_brain dependency.
- Running NeuralBench's reference models through Foundry.
- Claiming numerical equality with NeuralBench model scores (model implementation
  differences mean Foundry results are "evaluated under NeuralBench task protocol,"
  not reproductions of NeuralBench model results).
- Supporting all 36 NeuralBench tasks in the first implementation.
- Using NeuralBench's NeuralTrain for model training.
- Changing Foundry's model architecture to match NeuralBench expectations.

## References

- [NeuralBench paper (arXiv 2605.08495)](https://arxiv.org/abs/2605.08495)
- [NeuralSet paper (arXiv 2605.03169)](https://arxiv.org/abs/2605.03169)
- [NeuralBench GitHub](https://github.com/facebookresearch/neuroai/tree/main/neuralbench-repo)
- [NeuralSet GitHub](https://github.com/facebookresearch/neuroai)
- [NeuralBench task documentation](https://facebookresearch.github.io/neuroai/neuralbench/)
