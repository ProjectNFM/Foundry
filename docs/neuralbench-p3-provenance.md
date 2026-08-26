# NeuralBench P3 / Korczowski2014A — Reference Contract

**POC Phase 0 provenance note** — Captured 2026-08-20.

## Release

| Package      | Version |
|-------------|---------|
| neuralbench | 0.2.3   |
| neuralset   | 0.2.3   |
| neuralfetch | 0.2.3   |
| neuraltrain | 0.2.3   |

## Task & Dataset

| Field             | Value                                |
|-------------------|--------------------------------------|
| Device            | `eeg`                                |
| Task              | `p3`                                 |
| Dataset stem      | `korczowski2014a`                    |
| Study class       | `Korczowski2019BrainBi2014A`         |
| Data source       | Zenodo record 3266223 (MOABB)        |
| Reference command | `neuralbench eeg p3 --dataset korczowski2014a` |
| Output classes    | 2 (binary oddball: NonTarget / Target) |

## Sample Contract

### Tensor shapes and types (per sample, unbatched)

| Key                | Shape       | Dtype         | Description                         |
|--------------------|-------------|---------------|-------------------------------------|
| `neuro`            | (1, 16, 120)| float32       | EEG epoch: 16 channels × 120 samples |
| `target`           | (1, 2)      | int64         | One-hot encoded class label         |
| `subject_id`       | (1, 1)      | int64         | Integer-encoded subject index       |
| `channel_positions`| (1, 16, 3)  | float32       | 3D head-frame electrode positions   |

### EEG signal

| Property        | Value                |
|-----------------|----------------------|
| Sampling rate   | 120 Hz               |
| Epoch window    | −0.2 s to +0.8 s relative to stimulus (duration = 1.0 s) |
| Time samples    | 120 (= 1.0 s × 120 Hz) |
| Channels        | 16 EEG               |
| Dtype           | float32              |
| NaN / Inf       | None observed        |

### Channel order (16 channels, `channel_order="unique"`)

```
Fp1, Fp2, F3, AFz, F4, T7, Cz, T8, P7, P3, Pz, P4, P8, O1, Oz, O2
```

## Preprocessing Pipeline

Applied by `EegExtractor` / `MneRaw` per recording, then per segment:

1. **Channel selection**: `picks=eeg` (EEG channels only)
2. **Notch filter**: 50/60 Hz + harmonics (up to 300 Hz)
3. **Band-pass filter**: 0.1–75 Hz
4. **Resampling**: to 120 Hz
5. **Scaling**: `RobustScaler` per channel
6. **Per-segment baseline correction**: interval [0.0, 0.2] s relative to segment start
7. **Clamping**: ±20

Missing channels (across subjects) are zero-padded due to `channel_order="unique"`.

## Split Assignment

| Property          | Value                |
|-------------------|----------------------|
| Method            | `SklearnSplit`       |
| Split by          | `subject`            |
| Ratios            | 60% train / 20% val / 20% test |
| Valid random state | 33                  |
| Test random state  | 33                  |
| Stratification    | None                 |

All segments from the same subject belong to the same split (no data leakage).

### Split Statistics

| Split | Subjects | Segments | NonTarget (code=1) | Target (code=2) | Class ratio |
|-------|----------|----------|--------------------|-----------------|-------------|
| train | 38       | 35,270   | 29,391             | 5,879           | 5.00:1      |
| val   | 13       | 11,628   | 9,690              | 1,938           | 5.00:1      |
| test  | 13       | 14,124   | 11,770             | 2,354           | 5.00:1      |
| **total** | **64** | **61,022** | **50,851**       | **10,171**      | **5.00:1**  |

### Subject IDs per Split

**Train** (38 subjects):
1, 2, 3, 8, 10, 13, 14, 18, 21, 24, 25, 28, 30, 32, 34, 36, 37, 38, 39, 40,
42, 43, 44, 45, 46, 47, 51, 52, 53, 55, 56, 57, 58, 59, 60, 62, 63, 64

**Val** (13 subjects):
4, 5, 11, 12, 15, 17, 19, 20, 23, 35, 41, 48, 54

**Test** (13 subjects):
6, 7, 9, 16, 22, 26, 27, 29, 31, 33, 49, 50, 61

## Target Encoding

| Property        | Value                 |
|-----------------|-----------------------|
| Encoder         | `LabelEncoder`        |
| Event types     | `Stimulus`            |
| Event field     | `code`                |
| One-hot         | Yes (`return_one_hot=true`) |
| Aggregation     | `trigger`             |
| Classes         | code 1 → NonTarget [1,0], code 2 → Target [0,1] |

## Training Contract

| Property                | Value                        |
|-------------------------|------------------------------|
| Loss                    | `CrossEntropyLoss(label_smoothing=0.1)` |
| Class weights           | Computed (`compute_class_weights=true`) |
| Validation metric       | `val/bal_acc` (balanced accuracy) |
| Early-stopping mode     | `max`                        |
| Early-stopping patience | 10 epochs                    |
| Max epochs              | 40                           |
| Precision               | `32-true`                    |
| Gradient clip            | 1.0                         |
| Experiment seeds         | [33, 34, 35] (from grid)    |
| Data seed                | 33                           |
| Split seeds              | valid=33, test=33 (fixed)   |

### Metrics

| Log name           | TorchMetrics class | Config                                   |
|--------------------|--------------------|------------------------------------------|
| `acc`              | Accuracy           | task=multiclass, num_classes=2           |
| `f1_score_micro`   | F1Score            | task=multiclass, average=micro, num_classes=2 |
| `f1_score_macro`   | F1Score            | task=multiclass, average=macro, num_classes=2 |
| `bal_acc`          | Accuracy           | task=multiclass, num_classes=2, average=macro |
| `auroc`            | AUROC              | task=multiclass, num_classes=2           |
| `auprc`            | AveragePrecision   | task=multiclass, num_classes=2, average=macro |
| `confusion_matrix` | ConfusionMatrix    | task=multiclass, num_classes=2           |

## Foundry Identity Audit

Both Foundry and NeuralBench use the same underlying Korczowski Brain Invaders
2014A dataset (Zenodo record 3266223, MOABB paradigm).

| System      | Subject count | ID format                              |
|-------------|---------------|----------------------------------------|
| Foundry     | 64            | `sub001_0_0.h5` through `sub064_0_0.h5` |
| NeuralBench | 64            | `Korczowski2019BrainBi2014A/1` through `/64` |

**Mapping**: NeuralBench subject ID `N` corresponds to Foundry `sub{N:03d}_0_0.h5`.
All 64 subjects are present in both systems. The data originates from the same
Zenodo source; NeuralBench accesses it through MOABB's download layer while
Foundry preprocesses it through its own pipeline.

## Trigger Metadata

Each segment exposes trigger metadata with the following columns:

```
subject, session, task, run, study, stop, split, type, start,
timeline, duration, code, modality, description
```

Key observations:
- `type` is always `"Stimulus"` (the only event type in this task)
- `code` is 1 (NonTarget) or 2 (Target)
- `description` is `"NonTarget"` or `"Target"`
- `session` is always `"0"`, `run` is always `"0"`
- `modality` is always `None`

## Unknown Fields

**None.** All sample keys, preprocessing steps, and metadata fields are fully
identified and documented above. No blockers for proceeding to POC Phase 1.
