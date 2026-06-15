# Branch Comparison: Task Class Definition & Wiring

**Branches compared:**
- `suarez/task-class-adapter` — Runtime task adaptation with `TaskClassSchema` + `label_map`
- `milo/kempsleep` — Static declarative mapping with `ClassificationMapping`

---

## 1. Core Approach

| Aspect | `suarez/task-class-adapter` | `milo/kempsleep` |
|--------|---------------------------|------------------|
| Philosophy | **Runtime adaptation** — experiments declare which subset/grouping to use; base YAML stays maximal | **Declarative mapping** — each task YAML fully declares its raw→mapped contract |
| Central abstraction | `TaskClassSchema` (per-dataset vocabulary + filter semantics) + `_build_label_mapping()` | `ClassificationMapping` (complete raw→mapped enum + display names) |
| Where classes are defined | Two layers: base YAML (full task) + dataset-level `TASK_CLASS_SCHEMAS` dict | Single layer: `classification_mapping` block in task YAML |
| Remapping target | `TargetExtractor.label_map: dict[int, int]` | `TargetExtractor.classification_mapping: ClassificationMapping` |
| Config location | Experiment YAML (`data.classes`, `data.class_grouping`) | Task YAML (`classification_mapping.raw_to_mapped`) |
| When adaptation runs | Startup, per experiment | Load-time, per task definition |

---

## 2. How Classes Are Defined

### `suarez/task-class-adapter`

Classes are defined in **two separate locations**:

1. **Base task YAML** — contains the full superset definition:
   ```yaml
   # configs/tasks/neurosoft_acoustic_stim.yaml
   name: neurosoft_acoustic_stim
   head:
     output_dim: 26  # full vocabulary
   class_names: [stim_500Hz, stim_800Hz, ..., stim_8000Hz]
   ```

2. **Dataset class** — `TASK_CLASS_SCHEMAS` dict with a `TaskClassSchema`:
   ```python
   TaskClassSchema(
       vocabulary={"stim_500Hz": 4, "stim_800Hz": 5, ...},
       interval_filter_field="behavior_labels",
       interval_filter_mode="names",  # or "ids"
       grouping_presets={"3band": {"stim_500Hz": "low", ...}},
       group_order={"low": 0, "medium": 1, "high": 2},
       display_name_formatter=format_acoustic_stim_display_names,
   )
   ```

3. **Experiment config** — selects the active subset/grouping:
   ```yaml
   data:
     classes: [stim_500Hz, stim_800Hz, stim_1000Hz, ...]
     class_grouping: "3band"
   ```

### `milo/kempsleep`

Classes are defined in **a single location** — the task YAML:

```yaml
# configs/tasks/sleep_stage_5class.yaml
classification_mapping:
  raw_to_mapped:
    0: 0    # Wake
    1: 1    # N1
    2: 2    # N2
    3: 3    # N3
    4: 3    # N4 → N3 (merge)
    5: 4    # REM
  names:
    0: Wake
    1: N1
    2: N2
    3: N3
    4: REM
```

No experiment-level override for subsetting. To train on a different class arrangement, you create a new task YAML (e.g., `sleep_stage_3class.yaml`).

---

## 3. Wiring & Component Integration

### Target Extraction

| | `suarez/task-class-adapter` | `milo/kempsleep` |
|-|---------------------------|------------------|
| Mechanism | `TargetExtractor.label_map: dict[int, int]` | `TargetExtractor.classification_mapping: ClassificationMapping` |
| Injection point | `_adapt_task_config()` writes into `target_extractor["label_map"]` at runtime | `TaskConfig.build_extractor()` injects `classification_mapping` at build time |
| Unmapped value handling | `-1` sentinel → **raises `ValueError`** (strict) | `-1` sentinel (via `apply()`) — **no downstream filtering** |
| Shared extraction fn | No — each model has its own `_extract_targets` loop | Yes — `extract_multitask_targets()` is model-agnostic |

### Loss Function

| | `suarez/task-class-adapter` | `milo/kempsleep` |
|-|---------------------------|------------------|
| Loss type | `CrossEntropyTaskLoss` | `CrossEntropyTaskLoss` |
| Num classes source | Adapted `head.output_dim` | `cfg.output_dim` (derived from `mapping.num_classes`) |
| Class weights | `_apply_auto_class_weights()` with adapted config | `_apply_auto_class_weights()` with mapping-aware counting |
| `ignore_index` for `-1` | Not supported — relies on interval filtering to prevent `-1` reaching loss | Not supported — relies on upstream filtering |

### Metrics

| | `suarez/task-class-adapter` | `milo/kempsleep` |
|-|---------------------------|------------------|
| `num_classes` source | Written into `metrics["num_classes"]` by `_adapt_task_config()` | Auto-injected in `TaskConfig.from_dict()` from mapping |
| Confusion matrix | **Not implemented** | Full `ConfusionMatrixTracker` + `ConfusionMatrixCallback` (W&B heatmap) |
| Confusion matrix gate | N/A | Only tasks with `classification_mapping` get a tracker |

### Model / Readout Heads

| | `suarez/task-class-adapter` | `milo/kempsleep` |
|-|---------------------------|------------------|
| Head dim source | Adapted `head["output_dim"]` in task config dict | `cfg.output_dim` property (mapping-derived) |
| Router construction | Per-model head instantiation from adapted config | Shared `build_readout_router(task_configs, embed_dim)` |
| Sanity check | Explicit `assert model.router.heads[name].output_dim == task_cfg.output_dim` in `main.py` | Implicit — same `output_dim` feeds both head and metrics |

### Sampling / Interval Filtering

| | `suarez/task-class-adapter` | `milo/kempsleep` |
|-|---------------------------|------------------|
| Mechanism | `filter_sampling_intervals(intervals, schema, classes)` | `filter_intervals_by_mapping(intervals, mapping, value_field)` |
| Filter mode | Supports both name-based and ID-based filtering | ID-based only (via `kept_mask`) |
| Integration | Called in `NeuralDataModule._create_dataloader()` | **Exported but not yet wired** in production code path |
| Empty-filter handling | Returns empty interval + warning | Returns filtered interval (no special empty handling) |

### Class Weights

| | `suarez/task-class-adapter` | `milo/kempsleep` |
|-|---------------------------|------------------|
| Counting space | `label_map` keys → count raw IDs that survived adaptation | `mapping.kept_mask()` + `mapping.apply()` → count in mapped space |
| Removed class handling | Not applicable (subsetting removes them from intervals) | Explicit exclusion via `kept_mask` before counting |

---

## 4. Pros and Cons

### `suarez/task-class-adapter`

**Pros:**
- **Experiment-time flexibility** — can test different subsets/groupings without creating new YAML files per combination
- **Grouping presets** — named presets (e.g., "3band", "2band") make common ablation studies trivial
- **Base YAML stability** — the 26-class NeuroSoft task YAML never changes regardless of which experiment uses it
- **Name-based filtering** — interval filtering supports both string labels and integer IDs, covering diverse datasets
- **Display name formatting** — optional callable for pretty-printing class names (strip prefixes, etc.)
- **Explicit sanity check** — `main.py` asserts that model head dims match adapted configs

**Cons:**
- **Two-location class definition** — vocabulary lives in Python code (`TASK_CLASS_SCHEMAS`), not in config; requires knowing where to look
- **Dataset coupling** — each dataset class must define a `TaskClassSchema`; datasets without one cannot be adapted
- **No confusion matrix support** — branch does not implement confusion matrix tracking
- **Per-model target extraction** — no shared `extract_multitask_targets()` function; target extraction is duplicated
- **Single-task filtering assumption** — `_filter_intervals_by_classes()` uses the first available schema; multi-task experiments with different class spaces may break
- **No class removal semantics** — cannot map a raw ID to "don't supervise this"; must exclude via interval filtering
- **Config discoverability** — `data.classes` lives in experiment config while the vocabulary lives in Python; harder to validate statically

### `milo/kempsleep`

**Pros:**
- **Single source of truth** — `classification_mapping` in task YAML is the complete contract; no Python-side metadata needed
- **Self-documenting** — reading the YAML tells you exactly which raw IDs exist, how they map, and what names they get
- **Class removal support** — `None` mappings explicitly exclude classes from supervision (produces `-1`)
- **Strict validation** — enforces contiguous IDs, reachability, and full enumeration at construction time
- **Confusion matrix tracking** — full confusion matrix with W&B heatmap logging, gated on mapping presence
- **Shared target extraction** — `extract_multitask_targets()` works across all models identically
- **Property-driven wiring** — `TaskConfig.output_dim` and `TaskConfig.get_class_names()` derive from mapping; no manual sync
- **Auto-injection** — `metrics.num_classes` is injected automatically with conflict detection

**Cons:**
- **No experiment-time subsetting** — to try a 3-class subset, you create a new task YAML (`sleep_stage_3class.yaml`)
- **No grouping presets** — collapsing classes requires a new YAML with the desired `raw_to_mapped`
- **YAML proliferation risk** — each class arrangement variant needs its own task config file
- **Interval filtering not wired** — `filter_intervals_by_mapping` exists and is tested but is **not called** in the actual data pipeline
- **`-1` leakage risk** — if interval filtering is not wired upstream, removed-class labels (`-1`) could reach the loss with no `ignore_index`
- **No name-based filtering** — only supports integer ID field for interval filtering
- **Static-only** — if a user wants to ablate over class subsets, they must manually create N different YAML files

---

## 5. Shared Flaws / Gaps

Both implementations share several architectural weaknesses:

### 5.1 No `ignore_index` in Loss

Neither branch passes `ignore_index=-1` to `CrossEntropyLoss`. Both rely on upstream interval filtering to ensure unmapped/removed labels never reach the loss. If filtering fails or is misconfigured, `-1` targets will cause silent training corruption (negative index into weight tensor) or runtime errors.

### 5.2 Interval Filtering Not Robustly Wired

- **Suarez:** Filtering is implemented and called in `_create_dataloader()`, but only uses the **first available schema** — fragile for multi-task.
- **Milo:** `filter_intervals_by_mapping` is implemented and tested but **not actually called** in the live training pipeline.

In both cases, there is no safety net at the `TargetExtractor` level to silently drop invalid labels before they reach the loss.

### 5.3 No Unified Handling of `-1` Targets in the Training Loop

`FoundryModule._shared_step()` in both branches does not mask or filter `-1` values from the loss/metrics computation. If any slip through, they corrupt gradients.

### 5.4 Vocabulary / Mapping is Dataset-Specific, Not Task-Generic

Both approaches assume that raw label IDs are globally consistent per dataset. Neither handles the case where the same logical class (e.g., "N3 sleep") has different raw IDs across datasets in a multi-dataset experiment.

### 5.5 No Runtime Validation That Data Matches Mapping

Neither branch validates during training that the actual labels appearing in batches match the expected set. A dataset returning unexpected raw IDs (due to version changes or bugs) would only be caught when `TargetExtractor` raises on an unmapped value — a runtime crash rather than an early check.

### 5.6 Legacy Task Coexistence

Both branches must coexist with legacy tasks that use `head.output_dim` + `class_names` without a mapping/schema:
- **Suarez:** Tasks without `TASK_CLASS_SCHEMAS` pass through unchanged — no adaptation possible.
- **Milo:** Tasks without `classification_mapping` use the old path — no confusion matrix, no auto-injection.

This dual-path maintenance burden is present in both but is not addressed by either.

### 5.7 Class Weights Computed Before Training

Both compute inverse-frequency class weights at startup by scanning the full dataset. Neither handles the case where data augmentation or online sampling changes effective class distributions during training.

---

## 6. Summary Table

| Feature | `suarez/task-class-adapter` | `milo/kempsleep` |
|---------|---------------------------|------------------|
| Class definition location | Python code + YAML | YAML only |
| Experiment-time subsetting | Yes | No |
| Grouping / merging | Presets + custom dicts | Explicit `raw_to_mapped` |
| Class removal (`None`) | No | Yes |
| Validation strictness | Vocabulary name check only | Full contiguous + reachability |
| Interval filtering | Implemented + wired | Implemented + **not wired** |
| Confusion matrix | Not present | Full implementation |
| Shared target extraction | No (per-model) | Yes (`extract_multitask_targets`) |
| Display name control | Formatter function | `names` dict in YAML |
| YAML file count for N variants | 1 base + N experiments | N task YAMLs |
| `ignore_index` safety net | No | No |
| Multi-dataset compat | Schema per dataset | Mapping per task |

---

## 7. Recommendation Considerations

The two approaches are complementary rather than contradictory:

- **`milo/kempsleep`'s `ClassificationMapping`** excels as the **canonical, validated contract** for a finalized task definition. It is self-documenting, strict, and integrates cleanly with confusion matrices and metrics.

- **`suarez/task-class-adapter`'s runtime adaptation** excels for **exploratory experiments** where you want to test many class subsets/groupings from a single base task without YAML proliferation.

A unified solution could layer Suarez-style runtime adaptation **on top of** Milo-style `ClassificationMapping`, producing an effective mapping at startup that retains all validation and downstream integration benefits while supporting experiment-time flexibility.

**Critical gaps to address in either/both:**
1. Add `ignore_index=-1` to `CrossEntropyTaskLoss` as a safety net
2. Wire interval filtering into the live data pipeline (Milo branch)
3. Add a training-time assertion or mask for unexpected label values
4. Consolidate the legacy dual-path into a single code path
