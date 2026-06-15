# Task System & ClassificationMapping Design Discussion

**Branch:** `milo/kempsleep`
**Purpose:** Guide a design review discussion of the TaskMapping system.

---

## Part 1: Task System Wiring Overview

The task system is a declarative pipeline that connects YAML task definitions → experiment config → target extraction at tokenization → readout heads → losses → metrics → confusion matrices.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CONFIGURATION                                     │
│                                                                         │
│  configs/tasks/sleep_stage_5class.yaml                                  │
│       └──► TaskConfig.from_yaml() → TaskConfig dataclass                │
│                                                                         │
│  Experiment YAML (e.g. poyo_kemp_singlesess.yaml)                       │
│       └──► task_configs: [sleep_stage_5class]                           │
│       └──► data.task_type: sleep_stage  (for dataset transforms only)   │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        STARTUP (main.py)                                  │
│                                                                         │
│  _load_task_configs(cfg)                                                │
│       └──► Loads each YAML from configs/tasks/ → dict[str, TaskConfig]  │
│                                                                         │
│  _apply_auto_class_weights(cfg, datamodule, task_configs)               │
│       └──► Scans data, computes inverse-frequency weights               │
│       └──► Mutates task_configs[name].loss["class_weights"]             │
│                                                                         │
│  ModelClass(task_configs=...) → build_readout_router(task_configs)       │
│       └──► Per-task ReadoutHead with output_dim from mapping            │
│                                                                         │
│  NeuralDataModule(task_configs=..., tokenizer=model.tokenize)           │
│       └──► validate_task_mappings() at .setup()                         │
│       └──► _filter_intervals() removes intervals for excluded classes   │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  DATA PIPELINE (per sample)                               │
│                                                                         │
│  DataLoader → transform chain → model.tokenize(data)                    │
│       └──► extract_multitask_targets(task_configs, data)                │
│             └──► cfg.extractor(data)                                    │
│                   └──► TargetExtractor reads timestamps + values         │
│                   └──► classification_mapping.apply(raw_values)          │
│                         └──► Remaps to 0..N-1; removed → -1            │
│             └──► Produces: output_timestamps, target_values,            │
│                            task_index (1-based), target_weights          │
│       └──► Collated into batch via chain()/pad8()                       │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  TRAINING LOOP (FoundryModule)                            │
│                                                                         │
│  forward(batch) → backbone → ReadoutRouter → per-task predictions       │
│                                                                         │
│  _compute_task_losses():                                                │
│       └──► CrossEntropyTaskLoss(preds, targets, weights)                │
│            • ignore_index=-1 masks removed classes                       │
│            • class_weights buffer for imbalance correction               │
│            • Sequence-count weighted aggregation across tasks            │
│                                                                         │
│  metrics.update():                                                      │
│       └──► Masks target < 0 before updating                             │
│       └──► Softmax for classification; raw for regression               │
│       └──► Logged as val/{task_name}_{metric}                           │
│                                                                         │
│  ConfusionMatrixTracker (val only, only for tasks with mapping):         │
│       └──► Accumulates argmax preds + targets across epoch              │
│       └──► Renders W&B heatmap at epoch end via callback                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | File | Role |
|-----------|------|------|
| `TaskConfig` | `foundry/tasks/config.py` | Central dataclass bundling head, extractor, loss, metrics, mapping |
| `ClassificationMapping` | `foundry/tasks/classification_mapping.py` | Raw→mapped label contract, class removal, display names |
| `TargetExtractor` | `foundry/tasks/targets.py` | CPU-side target extraction from `Data`, applies mapping |
| `extract_multitask_targets` | `foundry/tasks/targets.py` | Orchestrates multi-task target collation |
| `ReadoutHead` / `MLPReadoutHead` | `foundry/tasks/heads.py` | Projection from embeddings to logits/values |
| `CrossEntropyTaskLoss` | `foundry/tasks/losses.py` | Classification loss with ignore_index, class weights |
| `MSETaskLoss` | `foundry/tasks/losses.py` | Regression loss |
| `classification_metrics` | `foundry/tasks/metrics.py` | Factory → MetricCollection (acc, f1, auroc, etc.) |
| `compute_class_weights_for_tasks` | `foundry/tasks/class_weights.py` | Duration-weighted inverse-frequency class weights |
| `validate_task_mappings` | `foundry/tasks/validation.py` | Startup check that mappings cover all raw labels |
| `ConfusionMatrixTracker` | `foundry/training/confusion_matrix.py` | Accumulates predictions, computes/renders confusion matrix |
| `ConfusionMatrixCallback` | `foundry/training/callbacks.py` | Lightning callback: logs confusion matrix at epoch end |
| `FoundryModule` | `foundry/training/module.py` | Lightning module: wires losses, metrics, confusion tracking |
| `NeuralDataModule` | `foundry/data/datamodules/base.py` | Data module: validates mappings, filters intervals, builds loaders |
| `filter_intervals_by_mapping` | `foundry/tasks/classification_mapping.py` | Removes sampling intervals for excluded classes |

### Key Invariants

1. **Task configs are experiment-owned** — experiments declare task config YAML names directly via `task_configs` list; `main.py` loads them without going through any dataset class.
2. **`task_index` is 1-based** (0 = padding) — the router subtracts 1 internally.
3. **Removed classes are handled at three layers**: interval filtering (sampler never draws them), label remapping to `-1` (extractor), `ignore_index=-1` (loss).
4. **Confusion matrices require `classification_mapping`** — plain `num_classes` tasks skip tracking.
5. **Hydra `instantiate` is deliberately avoided for the model** to keep task config dicts as plain data until losses/metrics are built.

---

## Part 2: ClassificationMapping Design Decisions

### Decision 1: Full Enumeration of Raw IDs

**What:** Every raw label ID that can appear in the data *must* be declared in `raw_to_mapped`. Undeclared IDs cause an immediate `ValueError`.

**Pros:**
- Catches dataset bugs and config drift immediately rather than silently corrupting training
- Makes the YAML config a complete, self-documenting specification of the label space
- Enables startup validation (`validate_task_mappings`) to verify data/config agreement

**Cons:**
- Fragile to upstream dataset changes (adding a new annotation class requires updating the mapping YAML)
- Verbose for datasets with many raw classes (e.g., 26-class acoustic stim would need all 26 declared)
- No "pass-through" mode for tasks that just want identity mapping without listing every ID

**Alternatives:**
- **Allowlist-only model**: Only list IDs you want to keep; everything else is implicitly removed. Simpler config, but loses the "self-documenting" property and can silently drop new classes.
- **Default behavior for undeclared IDs**: e.g. `default: null` (remove unlisted IDs) or `default: pass` (auto-map to sequential IDs). Reduces boilerplate but weakens the safety guarantee.
- **Schema-on-read**: Declare raw IDs at the dataset level (vocabulary), then the mapping just references names. This is the `suarez/task-class-adapter` approach.

---

### Decision 2: Contiguous Mapped IDs (0..N-1)

**What:** Mapped IDs must form a strict contiguous range starting at 0. Validated at construction time.

**Pros:**
- Guarantees direct use as tensor indices (no sparse-to-dense conversion needed)
- Simplifies `output_dim` derivation: just `len(mapped_ids)`
- Confusion matrix, metrics, and head all align without explicit size declarations

**Cons:**
- Requires pre-computation of the correct mapping when designing a new task (you can't just "remove class 2" and keep IDs 0,1,3,4)
- Makes dynamic subset experiments harder (removing a class requires re-numbering all subsequent classes)

**Alternatives:**
- **Sparse-to-dense remapping at model boundary**: Let mapped IDs be arbitrary; add a `sorted_mapped_ids` lookup table at the readout head. More flexible but adds a translation layer.
- **Automatic contiguous assignment**: Instead of the user specifying mapped IDs, just list which raw IDs to keep and in what order. The system auto-assigns 0..N-1. Less explicit but eliminates numbering errors.

---

### Decision 3: `None` for Class Removal (Producing `-1` Sentinel)

**What:** Raw IDs mapped to `None` in YAML produce `-1` in the target tensor. The loss uses `ignore_index=-1` to skip them.

**Pros:**
- Declarative and explicit — reading the YAML immediately shows which classes are excluded
- Defense-in-depth: even if interval filtering misses a sample, the loss safely ignores it
- Unified with the standard PyTorch `ignore_index` convention

**Cons:**
- `-1` targets in the batch consume memory and compute (forward pass still runs on those tokens)
- Relies on *all* downstream consumers (loss, metrics, confusion) correctly masking `-1`
- The "three-layer defense" (filtering + remapping + ignore_index) is arguably over-engineered for what should be a simple exclusion

**Alternatives:**
- **Drop at extraction**: Instead of producing `-1`, simply exclude those timestamps from the output of `extract_multitask_targets`. Zero memory cost, but loses the ability to audit what was excluded.
- **Separate "exclude" mask**: Return a boolean mask alongside targets rather than overloading the value space with `-1`. Cleaner semantics but requires all consumers to handle an extra tensor.
- **Interval-only filtering**: Rely entirely on removing intervals at the sampler level. Simpler, but single point of failure.

---

### Decision 4: Mapping Lives in Task YAML (Single Source of Truth)

**What:** The `classification_mapping` block is part of the task YAML config. There is no Python-side metadata, no experiment-level override.

**Pros:**
- Complete self-documentation: one file tells you everything about a task's label contract
- No Python code needed to understand or modify class definitions
- Static validation is straightforward (lint the YAML)
- Eliminates "where does this class definition come from?" confusion

**Cons:**
- **YAML proliferation**: To test a 3-class sleep staging variant, you create an entirely new YAML file (`sleep_stage_3class.yaml`), duplicating most of the config
- **No experiment-time flexibility**: Cannot quickly ablate over class subsets without creating N files
- **Tight coupling to one dataset**: The raw IDs in the mapping are specific to a single dataset's annotation scheme

**Alternatives:**
- **Two-layer system** (à la `suarez/task-class-adapter`): Base YAML defines the full task; experiment YAML selects which subset/grouping to use. More flexible for ablations, but harder to understand and validate.
- **Mapping inheritance/composition**: A base mapping + per-experiment diffs (e.g., "start from 5class, remove N1"). Avoids full duplication but adds complexity.
- **Runtime adaptation at startup**: Experiment config specifies `classes: [Wake, N2, N3, REM]`; system derives the mapping from a vocabulary. Most flexible, but least explicit.

---

### Decision 5: `TaskConfig.extractor` as a Lazy Property (Not Public `build_extractor()`)

**What:** The extractor is accessed via a `@property` on `TaskConfig` that auto-injects the `classification_mapping`. The raw `target_extractor` dict is still public but should not be used directly to create an extractor.

**Pros:**
- Impossible to forget to inject the mapping when getting an extractor through the intended path
- Lazy construction avoids import-time side effects
- Single correct access pattern for all consumers

**Cons:**
- The raw `target_extractor` dict is still accessible (could be misused)
- Property has side effects (creates and caches the extractor on first access via `object.__setattr__`)
- Hidden state: whether the extractor has been built or not is invisible from outside

**Alternatives:**
- **Private the raw dict**: Rename to `_target_extractor_spec` to discourage direct use. More Pythonic encapsulation but breaks serialization/Hydra patterns.
- **Factory method**: `cfg.build_extractor()` (explicit call). Clearer that construction is happening, but requires consumers to know to call it.
- **Eager construction**: Build the extractor in `__post_init__`. Simpler lifecycle, but forces import of `TargetExtractor` at config load time and prevents serialization.

---

### Decision 6: Mapping Drives `output_dim`, Class Names, and Metrics Automatically

**What:** When `classification_mapping` is present, `TaskConfig.output_dim` is derived from `mapping.num_classes`, `get_class_names()` returns mapping names, and `metrics.num_classes` is auto-injected (with conflict detection).

**Pros:**
- Single source of truth eliminates sync errors between head size, metrics, and display
- Declarative: add a class to the mapping → everything adjusts automatically
- Conflict detection catches accidental overrides early

**Cons:**
- Implicit behavior: `output_dim` doesn't appear in the YAML but determines model architecture
- Two code paths: tasks *with* mapping (derived) vs. tasks *without* (explicit `head.output_dim`). Maintaining both adds complexity.
- The auto-injection can surprise users who expect `output_dim` to come from `head`

**Alternatives:**
- **Always explicit**: Require `output_dim` in YAML even when a mapping is present; validate it matches. Redundant but discoverable.
- **Remove the non-mapping path**: Require all classification tasks to use a `ClassificationMapping` (even if it's just identity). Eliminates the dual code path.
- **Computed config layer**: A Hydra resolver that reads `${classification_mapping.num_classes}` so the YAML shows the dependency explicitly.

---

### Decision 7: Frozen Dataclass with Cached Derived Values

**What:** `ClassificationMapping` is `@dataclass(frozen=True)` with derived properties (`_num_classes`, `_kept_raw_ids`, `_removed_raw_ids`) computed in `__post_init__`.

**Pros:**
- Immutability guarantees safety across multi-threaded dataloader workers
- Caching avoids recomputation (relevant when `apply()` is called per sample)
- Hashable (can be used in sets/dicts if needed)

**Cons:**
- `object.__setattr__` in `__post_init__` is a workaround that looks surprising
- Cannot extend with mutable state (e.g., counting how often each class is seen) without creating a wrapper
- Harder to test with partial/mock objects

**Alternatives:**
- **Regular dataclass + `@cached_property`**: Drop `frozen=True`, use `@cached_property` for derived values. More conventional but loses immutability guarantees.
- **Named tuple**: Even more immutable, but no methods/properties.
- **Pydantic model**: Provides validation, serialization, and immutability via `model_config = ConfigDict(frozen=True)`. Heavier dependency but more standard.

---

### Decision 8: Many-to-One Merging (e.g., N3+N4 → class 3)

**What:** Multiple raw IDs can map to the same mapped ID, enabling class merging (e.g., N3 and N4 sleep stages both map to mapped class 3).

**Pros:**
- Declarative class merging without any custom code or data transforms
- The merged class name (`N3`) is defined in one place
- Class weight computation naturally handles the merged distribution

**Cons:**
- Merging is irreversible once the mapping is applied — cannot separate N3 from N4 downstream
- The YAML doesn't make the merging visually obvious unless you read carefully (just `3: 3` and `4: 3`)
- If you later want separate metrics for N3 vs N4, you need a different task config entirely

**Alternatives:**
- **Explicit merge groups**: A dedicated `merge` key in YAML (e.g., `merge: {N3: [3, 4]}`). More readable but adds complexity to the config schema.
- **Post-hoc grouping**: Keep all classes separate at the model level; merge only at evaluation/reporting time. Maximum flexibility but the model wastes capacity on distinctions that don't matter.
- **Hierarchical labels**: Model predicts both fine-grained (7-class) and coarse (5-class) simultaneously. Rich but architecturally more complex.

---

### Decision 9: Interval Filtering as a Separate Layer from Label Remapping

**What:** Removed classes are handled at two levels: (1) `filter_intervals_by_mapping` removes sampling intervals so the sampler never draws those windows, and (2) `mapping.apply()` produces `-1` for any that slip through.

**Pros:**
- Efficiency: filtered intervals mean fewer wasted forward passes
- Defense-in-depth: if filtering has gaps, the `-1`/`ignore_index` layer catches them
- Separation of concerns: sampler layer handles "what to draw", extraction layer handles "what to label"

**Cons:**
- Two layers doing "the same thing" (class exclusion) in different ways — conceptual overhead
- If someone only wires one layer, the system appears to work but may have subtle corruption
- The filtering layer depends on the interval object having the right `value_field` attribute — brittle for new datasets

**Alternatives:**
- **Filter-only (no `-1` sentinel)**: Guarantee filtering is perfect; crash if a removed class reaches extraction. Simpler but unforgiving.
- **Sentinel-only (no interval filtering)**: Let the sampler draw anything; always rely on `ignore_index`. Simpler to wire but wastes compute on excluded samples.
- **Unified exclusion layer**: A single `ExclusionPolicy` that handles both interval filtering and target masking in one coordinated component.

---

### Decision 10: Startup Validation (`validate_task_mappings`)

**What:** At `NeuralDataModule.setup()`, scan up to 5 recordings per task and verify every raw label ID in data is declared in `raw_to_mapped`.

**Pros:**
- Fails fast with a clear error message before any training step
- Catches config/data drift that would otherwise cause a mid-training crash in a dataloader worker
- Cheap (only 5 recordings scanned)

**Cons:**
- Only samples 5 recordings — could miss rare label IDs that appear in recording #6+
- Adds startup latency (minor: scanning a few recordings)
- Only validates label coverage, not that the mapping produces sensible training (e.g., all samples mapping to one class)

**Alternatives:**
- **Full dataset scan**: Check every recording. Slower but exhaustive.
- **Schema-level validation**: The dataset declares its raw vocabulary independently; the mapping is validated against the vocabulary, not the data. Faster and deterministic.
- **Runtime accumulation**: Track seen IDs during training; warn/error if a new one appears. Zero startup cost but deferred detection.

---

## Part 3: Summary of Architectural Tensions

### Flexibility vs. Explicitness

The current design strongly favors **explicitness** (every raw ID declared, contiguous mapping, single YAML source). This makes individual task configs easy to understand and validate, but imposes overhead for exploratory workflows (new YAML per class arrangement).

### Single-Dataset vs. Multi-Dataset

The mapping is implicitly coupled to one dataset's raw label vocabulary. If two datasets use different raw IDs for the same semantic class (e.g., "N3 sleep" = raw ID 3 in Kemp, raw ID 5 in another dataset), they cannot share a mapping and need separate task configs.

### Dual Code Path: Mapping vs. Legacy

Tasks with `classification_mapping` get a rich feature set (auto-derived `output_dim`, confusion matrix, interval filtering). Tasks without mapping use the older `head.output_dim` + `class_names` path and miss these features. This creates a two-tier system.

### Static Declaration vs. Runtime Adaptation

The `suarez/task-class-adapter` branch solves the flexibility problem with runtime adaptation (vocabulary + grouping presets). A potential unification could layer runtime adaptation *on top of* `ClassificationMapping`: the adaptation step produces an effective `ClassificationMapping` at startup, retaining all downstream integration benefits.

---

## Part 4: Open Questions for Discussion

1. **Should all classification tasks be required to use `ClassificationMapping`?** This eliminates the dual code path but forces even simple binary tasks to declare a mapping.

2. **How should we handle experiment-time class subsetting/grouping?** Options:
   - Multiple YAML files (current)
   - Experiment-level overrides that produce a mapping at startup
   - Mapping inheritance/composition

3. **Is YAML proliferation acceptable for this project's scale?** If there are only 5-10 class arrangements per dataset, it may be fine. At 50+, runtime adaptation becomes necessary.

4. **Should the `transform layer` (e.g., `PrepareSleepStages`) be aware of class removal?** Currently transforms are structural only; the mapping owns all label semantics. This is clean but means transforms must preserve all raw IDs even if they're destined for removal.

5. **Multi-dataset experiments**: If we train on Kemp + another sleep dataset, should there be a shared "canonical sleep mapping" or per-dataset mappings with a unification step?

6. **Should `ClassificationMapping` support reverse mapping** (mapped → raw) for interpretability/debugging? Currently only forward mapping is available.
