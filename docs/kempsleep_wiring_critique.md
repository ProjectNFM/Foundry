# Critique: Filtering & Remapping Wiring in `milo/kempsleep`

This document examines the flaws in how `ClassificationMapping` is currently wired into the data-to-training pipeline on the `milo/kempsleep` branch, and proposes improvements.

---

## Current Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           STARTUP (main.py)                                       │
│                                                                                  │
│  _load_task_configs(cfg)                                                         │
│       └──► TaskConfig (with classification_mapping from YAML)                    │
│                │                                                                 │
│                ├──► _apply_auto_class_weights() → class_weights.py               │
│                │        └── extractor.classification_mapping.kept_mask()          │
│                │        └── extractor.classification_mapping.apply()              │
│                │                                                                 │
│                ├──► ModelClass(task_configs=...) → build_readout_router()         │
│                │        └── cfg.output_dim (derived from mapping)                 │
│                │                                                                 │
│                └──► instantiate(cfg.data, tokenizer=model.tokenize)              │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│                           DATA LOADING                                            │
│                                                                                  │
│  NeuralDataModule._create_dataloader(split)                                      │
│       │                                                                          │
│       ├──► dataset.get_sampling_intervals(split)                                 │
│       │         ⚠️  NO FILTERING BY MAPPING HERE                                 │
│       │                                                                          │
│       └──► RandomFixedWindowSampler(sampling_intervals, ...)                     │
│                 └── May sample windows containing removed classes                │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│                       TOKENIZATION (per sample)                                   │
│                                                                                  │
│  model.tokenize(data)                                                            │
│       └──► extract_multitask_targets(task_configs, data)                         │
│                 └──► cfg.build_extractor() → TargetExtractor(mapping=...)         │
│                       └──► mapping.apply(raw_values)                              │
│                             └── Removed classes → -1                             │
│                             └── Undeclared IDs → ValueError (crash)              │
│                                                                                  │
│  ⚠️  The -1 values are packed into the batch tensors                             │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│                       TRAINING LOOP                                               │
│                                                                                  │
│  FoundryModule._shared_step(stage, batch)                                        │
│       │                                                                          │
│       ├──► _compute_task_losses(outputs, target_values, ...)                     │
│       │         └── CrossEntropyTaskLoss(preds, target, weights)                 │
│       │              ⚠️  No ignore_index — if target contains -1,               │
│       │                  CE indexes weight[-1] (last class!) silently            │
│       │                                                                          │
│       ├──► metrics[name].update(metric_preds, metric_target)                     │
│       │         ⚠️  Metrics receive -1 targets → undefined behavior             │
│       │                                                                          │
│       └──► confusion_trackers[name].update(pred_classes, target)                 │
│                  ⚠️  -1 in targets → counts[−1, pred] wraps to last row         │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Identified Flaws

### Flaw 1: `filter_intervals_by_mapping` Is Dead Code

The function exists, is tested, and is exported — but **nothing calls it** in the actual pipeline. `NeuralDataModule._create_dataloader()` passes raw `sampling_intervals` directly to the sampler without applying any class-based filtering.

**Impact:** If a task has `None` mappings (removed classes), windows containing those labels will be sampled. The `-1` values produced by `mapping.apply()` flow unchecked into loss and metrics.

**Why it's clunky:** The filtering utility and the datamodule live in the same codebase, but there's no explicit contract or hook point connecting them. The datamodule has no awareness of `ClassificationMapping` at all — it just passes through intervals.

---

### Flaw 2: No Safety Net at the Loss/Metrics Boundary

`CrossEntropyTaskLoss` does not set `ignore_index=-1`. If `-1` targets reach the loss:
- PyTorch's `F.cross_entropy` with `ignore_index` unset treats `-1` as a valid class index
- Negative indexing into the logits tensor: `logits[:, -1]` corresponds to the **last class**
- Training proceeds with silently corrupted gradients

Similarly, `torchmetrics` classifiers and `ConfusionMatrixTracker` receive `-1` values and either crash or silently corrupt aggregations (e.g., `counts[-1, p]` wraps around in the confusion matrix tensor).

---

### Flaw 3: Filtering and Remapping Are Separate Concerns With No Enforced Ordering

The current code assumes this ordering:
1. `filter_intervals_by_mapping` removes intervals with `None`-mapped labels (not wired)
2. `PrepareSleepStages` transform filters unknown stage 6 (hardcoded in transform)
3. `TargetExtractor` applies `mapping.apply()` on surviving values

But there's no mechanism guaranteeing this order. The `PrepareSleepStages` transform is injected via `get_required_transforms()`, while interval filtering would need to happen at the sampler level. These are two different layers with no coordination.

---

### Flaw 4: The Transform and the Mapping Duplicate Removal Logic

`PrepareSleepStages` hardcodes `ids != 6` (unknown stage removal), while `ClassificationMapping` has the capability to express this via `6: null` in `raw_to_mapped`. But the mapping currently doesn't declare ID 6 at all — which means if `PrepareSleepStages` ever fails to run, `mapping.apply()` would crash with "undeclared raw label ID: 6".

The system has two independent removal mechanisms for the same concern:
- Transform-level: hardcoded filtering before extraction
- Mapping-level: `None` → `-1` sentinel at extraction time

Neither is aware of the other, and neither is sufficient alone.

---

### Flaw 5: `build_extractor()` Is an Implicit Injection That Consumers Must Remember

`TaskConfig.build_extractor()` strips `_target_` and injects `classification_mapping`. This is the **only** correct way to get a wired extractor. But:
- It's a method you must know to call — the `target_extractor` dict on `TaskConfig` looks ready-to-use but is **incomplete** without the mapping
- The class weights code (`class_weights.py`) correctly calls `cfg.build_extractor()`, but there's nothing preventing someone from doing `TargetExtractor(**cfg.target_extractor)` and getting an extractor without the mapping
- If a new consumer is added (e.g., a dataset validator, an evaluation script), they must independently discover that `build_extractor()` is required

---

### Flaw 6: Class Weights Counting Reimplements Mapping Logic Inline

`_count_labels_for_task()` in `class_weights.py` has this pattern:

```python
if extractor.classification_mapping is not None:
    mapping = extractor.classification_mapping
    keep = mapping.kept_mask(values)
    selected = intervals.select_by_mask(keep)
    mapped = mapping.apply(values[keep])
    for label in np.unique(mapped):
        ...
```

This is essentially the same "filter then remap" pattern that should happen during data loading — but reimplemented separately in counting code. The pattern is repeated rather than abstracted.

---

### Flaw 7: Confusion Matrix Tracker Has No Guard Against Invalid Indices

`ConfusionMatrixTracker.update()` directly does:
```python
self._all_preds.append(preds.detach().cpu())
self._all_targets.append(targets.detach().cpu())
```

And `compute_confusion_matrix()` does:
```python
for t, p in zip(targets, preds):
    counts[t, p] += 1
```

If `t = -1`, this indexes `counts[-1, p]` — the **last row** of the matrix — silently corrupting the confusion matrix for the last class. There's no bounds check or sentinel filtering.

---

### Flaw 8: The Datamodule Doesn't Know Its Own Task Configs

**Status: Fixed.** `NeuralDataModule` now receives `task_configs` directly from `main.py` and uses them for both class weight computation (`compute_class_weights`) and interval filtering (`_filter_intervals`). The `TaskMixin` and `get_tasks_for_experiment` indirection have been removed — task configs are loaded directly from YAML by `main.py._load_task_configs()`.

---

## Proposed Improvements

### Improvement 1: Wire Interval Filtering Into the Datamodule

The datamodule should accept task configs (or at minimum, the relevant `ClassificationMapping`) and apply `filter_intervals_by_mapping` before constructing the sampler.

```python
class NeuralDataModule(LightningDataModule):
    def __init__(self, ..., task_configs: dict[str, TaskConfig] | None = None):
        self._task_configs = task_configs

    def _create_dataloader(self, split):
        sampling_intervals = self.dataset.get_sampling_intervals(split=split)

        # Apply mapping-based filtering
        if self._task_configs:
            for name, cfg in self._task_configs.items():
                if cfg.classification_mapping and cfg.classification_mapping.removed_raw_ids:
                    value_field = cfg.target_extractor["value_key"].split(".")[-1]
                    sampling_intervals = filter_intervals_by_mapping(
                        sampling_intervals, cfg.classification_mapping, value_field
                    )

        sampler = RandomFixedWindowSampler(sampling_intervals=sampling_intervals, ...)
```

This closes the gap between "the mapping knows which classes are removed" and "the sampler knows which windows to draw."

---

### Improvement 2: Add `ignore_index=-1` as a Defensive Layer

Even with filtering wired, add `ignore_index=-1` to `CrossEntropyTaskLoss` as a safety net:

```python
class CrossEntropyTaskLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index
        ...

    def forward(self, preds, targets, weights):
        return F.cross_entropy(
            preds, targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )
```

This ensures that if `-1` values ever leak through (due to incomplete filtering, a dataset bug, or a new data path), they are silently ignored rather than corrupting training.

---

### Improvement 3: Filter `-1` Targets in the Training Loop Before Metrics/Confusion

Add a mask in `_shared_step` that excludes `-1` targets:

```python
if name in metrics:
    valid_mask = target >= 0
    if valid_mask.any():
        metric_preds, metric_target = self._prepare_for_metrics(
            cfg, preds[valid_mask], target[valid_mask]
        )
        metrics[name].update(metric_preds, metric_target)

if stage == "val" and name in self._val_confusion_trackers:
    valid_mask = target >= 0
    if valid_mask.any():
        pred_classes = preds[valid_mask].argmax(dim=-1)
        self._val_confusion_trackers[name].update(pred_classes, target[valid_mask])
```

This is a defense-in-depth measure that ensures metrics never see invalid indices.

---

### Improvement 4: Unify Removal — Make the Mapping the Single Source of Truth

Instead of having `PrepareSleepStages` hardcode `ids != 6` and the mapping not declaring ID 6, choose one approach:

**Option A — Mapping declares all IDs, including those to remove:**
```yaml
classification_mapping:
  raw_to_mapped:
    0: 0    # Wake
    1: 1    # N1
    2: 2    # N2
    3: 3    # N3
    4: 3    # N4 → N3
    5: 4    # REM
    6: null # Unknown → removed
```

The transform `PrepareSleepStages` still converts intervals to timestamps (structural concern), but does **not** filter — that's the mapping's job. This makes the mapping the authoritative declaration of "what survives."

**Option B — The transform handles dataset-specific raw cleanup, the mapping handles task-specific semantics:**

Keep them separate but document the contract: transforms handle raw data normalization (interval→timestamps, artifact removal), and the mapping only operates on "clean" IDs. Then validate at startup that `mapping.raw_to_mapped.keys()` covers exactly the IDs that can survive the transform chain.

Option A is simpler and eliminates the implicit coupling.

---

### Improvement 5: Make `TargetExtractor` Impossible to Misuse

Instead of requiring callers to know about `build_extractor()`, make the `target_extractor` dict on `TaskConfig` non-public and provide only the built form:

```python
@dataclass
class TaskConfig:
    _target_extractor_spec: dict[str, Any]  # private
    classification_mapping: ClassificationMapping | None = None

    @cached_property
    def extractor(self) -> TargetExtractor:
        """Fully-wired extractor — the only public access point."""
        kwargs = dict(self._target_extractor_spec)
        kwargs.pop("_target_", None)
        if self.classification_mapping is not None:
            kwargs["classification_mapping"] = self.classification_mapping
        return TargetExtractor(**kwargs)
```

This eliminates the "you have to know to call `build_extractor()`" problem. The extractor is always correctly wired because there's no other way to get one.

---

### Improvement 6: Centralize the "Filter Then Remap" Pattern

Instead of reimplementing `kept_mask → select → apply` in every consumer, put it on the mapping itself:

```python
class ClassificationMapping:
    def filter_and_apply(
        self, values: np.ndarray, intervals=None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Filter removed labels and remap kept ones in one step.

        Returns:
            (mapped_values, keep_mask) where mapped_values only contains
            values for kept labels.
        """
        keep = self.kept_mask(values)
        mapped = self.apply(values[keep])
        return mapped, keep
```

Then `class_weights.py`, interval filtering, and any future consumer all call `mapping.filter_and_apply()` — one pattern, one place to debug.

---

### Improvement 7: Add Startup Validation

At model construction or training start, validate that the mapping covers the data:

```python
class TaskConfig:
    def validate_against_data(self, sample_values: np.ndarray) -> None:
        """Assert mapping covers all values that appear in actual data."""
        if self.classification_mapping is None:
            return
        unique_raw = set(sample_values.flat)
        declared = set(self.classification_mapping.raw_to_mapped.keys())
        undeclared = unique_raw - declared
        if undeclared:
            raise ValueError(
                f"Task '{self.name}': raw IDs {sorted(undeclared)} appear in data "
                f"but are not declared in classification_mapping.raw_to_mapped"
            )
```

Call this during `datamodule.setup()` on a scan of the first recording. This catches mapping/data mismatches before training starts — not halfway through epoch 1 with a `ValueError` inside the dataloader worker.

---

## Summary

| Flaw | Root Cause | Fix |
|------|-----------|-----|
| Filtering is dead code | Datamodule has no reference to task configs/mapping | Wire mapping into datamodule |
| No `ignore_index` | Loss assumes clean targets | Add `ignore_index=-1` |
| `-1` corrupts metrics/confusion | No guard in training loop | Mask invalid targets before metrics |
| Dual removal mechanisms | Transform and mapping both remove classes independently | Unify: mapping is the single authority |
| `build_extractor()` is easy to skip | Public `target_extractor` dict looks usable | Make extractor access property-based |
| filter+remap reimplemented per consumer | No single method for the combined operation | Add `filter_and_apply()` |
| No early validation | Mapping errors only surface at extraction time | Add startup data scan |

The core issue is that `ClassificationMapping` is well-designed as a **data structure** (validation, properties, immutability) but poorly integrated as a **pipeline component** (nothing automatically wires it into the stages that need it). The fixes above turn it from a passive configuration object into an active participant in the data flow.
