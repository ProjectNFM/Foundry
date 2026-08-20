# EEGNet Implementation Comparison: Foundry vs NeuralBench

**POC Phase 2, Step 2** — Pre-training architectural and hyperparameter audit.

**Verdict: Implementation Comparison** — These are two independent
implementations of EEGNet (Lawhern et al. 2018). They are structurally similar
but not weight-compatible. Results should be reported as an *implementation
comparison*, not a replication.

---

## 1. Model Architecture

### Source implementations

| Aspect | Foundry | NeuralBench |
|--------|---------|-------------|
| Class | `foundry.models.baselines.EEGNetEncoder` | `braindecode.models.EEGNet` (via neuraltrain) |
| Reference | Lawhern et al. J. Neural Eng. 2018 | Same paper |
| Framework | Custom PyTorch `nn.Module` | braindecode's modular EEGModel base |

### Block 1: Temporal + Spatial Filtering

| Parameter | Foundry | NeuralBench (braindecode) | Match? |
|-----------|---------|---------------------------|--------|
| Temporal Conv2d filters (F1) | 8 | 8 | Yes |
| Temporal kernel size | `(1, 64)` | `(1, 64)` | Yes |
| Temporal padding | `"same"` | `"same"` | Yes |
| Temporal bias | False | False | Yes |
| BatchNorm after temporal | `nn.BatchNorm2d(F1)` | `nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)` | **No** |
| Depthwise spatial Conv2d filters | F1×D = 16 | F1×D = 16 | Yes |
| Depthwise spatial kernel | `(num_channels, 1)` | `(n_chans, 1)` | Yes |
| Depthwise groups | F1 = 8 | F1 = 8 | Yes |
| Depthwise bias | False | False | Yes |
| Depthwise `max_norm` constraint | **None** | `max_norm=1` on weight | **No** |
| BatchNorm after spatial | `nn.BatchNorm2d(F1*D)` | `nn.BatchNorm2d(F1*D, momentum=0.01, eps=1e-3)` | **No** |
| Activation | ELU | ELU | Yes |
| Pooling | `AvgPool2d(1, 4)` | `AvgPool2d(1, 4)` | Yes |
| Dropout | `Dropout(0.5)` | `Dropout(0.25)` | **No** |

### Block 2: Separable Convolution

| Parameter | Foundry | NeuralBench (braindecode) | Match? |
|-----------|---------|---------------------------|--------|
| Separable Conv structure | Custom `SeparableConv2d` class | braindecode separable block | ~Yes |
| Depthwise kernel size | `(1, 16)` | `(1, 16)` | Yes |
| Pointwise filters (F2) | 16 | 16 (= F1×D) | Yes |
| Bias | False | False | Yes |
| BatchNorm | `nn.BatchNorm2d(F2)` default params | `nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3)` | **No** |
| Activation | ELU | ELU | Yes |
| Pooling | `AvgPool2d(1, 8)` | `AvgPool2d(1, 8)` | Yes |
| Dropout | `Dropout(0.5)` | `Dropout(0.25)` | **No** |

### Classifier Head

| Parameter | Foundry | NeuralBench (braindecode) | Match? |
|-----------|---------|---------------------------|--------|
| Structure | `ReadoutRouter` (multi-task compatible) | Single `nn.Linear` (via braindecode `LogSoftmax` or raw) | **Different** |
| Output dim | 2 (from task config) | 2 (from `brain_model_output_size`) | Yes |
| Final activation in model | None (logits) | None (logits, softmax applied at loss) | Yes |
| `max_norm` constraint on final layer | None | `norm_rate=0.25` if `final_layer_with_constraint=True` | See note |

**Note on final-layer constraint:** braindecode defaults `final_layer_with_constraint=False`
(deprecated in newer versions). When this triggers, a `max_norm=0.25` constraint is applied
to the classifier weight. In the NeuralBench effective config, no explicit override is set,
so the default `False` applies (no constraint on the final linear layer).

### Tokenization / Input Pipeline

| Aspect | Foundry | NeuralBench |
|--------|---------|-------------|
| Input to model | `(B, T, C)` from tokenizer, then normalized to `(B, 1, C, T)` in forward | `(B, 1, C, T)` from data pipeline directly |
| Tokenizer | `BaselineEEGModel.tokenize()` extracts signal from `Data` object | No tokenizer; raw `neuro` tensor passed |
| Channel padding | `pad2d()` for batch alignment | None (all samples have same shape) |
| Modality mask | Filters by supported modalities (EEG/ECoG/sEEG) | N/A (single modality) |

---

## 2. Training Hyperparameters

### Optimizer

| Parameter | Foundry | NeuralBench | Match? |
|-----------|---------|-------------|--------|
| Optimizer | AdamW | AdamW | Yes |
| Learning rate | 1e-4 | 1e-4 | Yes |
| Weight decay | **0.05** (set in experiment config) | **0.05** | Yes |
| Betas | PyTorch default (0.9, 0.999) | PyTorch default (0.9, 0.999) | Yes |
| Epsilon | PyTorch default (1e-8) | PyTorch default (1e-8) | Yes |

### Learning Rate Schedule

| Parameter | Foundry | NeuralBench | Match? |
|-----------|---------|-------------|--------|
| Scheduler | **ConstantLR** (no warmup/hold/decay in this config) | **OneCycleLR** | **No** |
| Max LR | 1e-4 (constant) | 1e-4 | N/A |
| Warmup fraction | N/A | `pct_start=0.1` (10% of total steps) | **No** |
| Anneal strategy | N/A | Cosine | **No** |
| Scheduler interval | step | step | Yes |

**Impact:** NeuralBench uses a one-cycle cosine policy that warms up for 10% of
training then anneals to near-zero. Foundry uses a flat learning rate throughout.
This is a significant training dynamics difference that may affect convergence
speed and final performance.

### Trainer

| Parameter | Foundry | NeuralBench | Match? |
|-----------|---------|-------------|--------|
| Max epochs | 40 | 40 | Yes |
| Precision | 32-true | 32-true | Yes |
| Gradient clip | 1.0 | 1.0 | Yes |
| Num sanity val steps | 0 | 0 | Yes |
| limit_train_batches | None | None | Yes |
| limit_val_batches | None | None | Yes |

### Early Stopping / Checkpointing

| Parameter | Foundry | NeuralBench | Match? |
|-----------|---------|-------------|--------|
| Monitor metric | `val/neuralbench_p300_bal_acc` | `val/bal_acc` | Yes (same metric, different log name) |
| Mode | max | max | Yes |
| Patience | 10 | 10 | Yes |
| Checkpoint selection | Best `val/neuralbench_p300_bal_acc` | Best `val/bal_acc` | Yes |

### Loss

| Parameter | Foundry | NeuralBench | Match? |
|-----------|---------|-------------|--------|
| Loss function | CrossEntropyLoss | CrossEntropyLoss | Yes |
| Label smoothing | 0.1 | 0.1 | Yes |
| Class weights | auto-computed (inverse-frequency) | `compute_class_weights=true` | Yes (same intent) |
| Target format | Integer class index (from argmax of one-hot) | One-hot (converted internally) | Yes (equivalent) |

### Batch / Data

| Parameter | Foundry | NeuralBench | Match? |
|-----------|---------|-------------|--------|
| Batch size | 64 | 64 | Yes |
| Train shuffle | True | True | Yes |
| Val shuffle | False | False | Yes |
| Num workers | 4 (experiment) | 10 (default) | Minor |
| Seed (data) | 33 | 33 | Yes |
| Seeds (grid) | 33, 34, 35 | 33, 34, 35 | Yes |

---

## 3. Data Contract

| Aspect | Foundry (via NeuralBenchDataModule) | NeuralBench | Match? |
|--------|-------------------------------------|-------------|--------|
| Input shape (per sample) | (16, 120) → (T=120, C=16) after tokenize | (1, 16, 120) | Yes (equivalent) |
| Sampling rate | 120 Hz | 120 Hz | Yes |
| Channels | 16, same order | 16, same order | Yes |
| Preprocessing | NeuralSet EegExtractor (same pipeline) | NeuralSet EegExtractor | Yes |
| Split method | NeuralSet split assignment | NeuralSet split assignment | Yes |
| Split seed | 33 | 33 | Yes |
| Train samples | 35,270 | 35,270 | Yes |
| Val samples | 11,628 | 11,628 | Yes |
| Class ratio | 5.00:1 (NonTarget:Target) | 5.00:1 | Yes |

---

## 4. Summary of Differences

### Critical (likely to affect val score)

| # | Difference | Foundry | NeuralBench | Expected Impact |
|---|-----------|---------|-------------|-----------------|
| 1 | **Dropout rate** | 0.5 | 0.25 | Higher regularization in Foundry; may reduce overfitting but also limit capacity |
| 2 | **LR schedule** | Constant | OneCycleLR (cosine) | Different convergence dynamics; one-cycle can reach higher peaks then decay |
| 3 | **BatchNorm parameters** | PyTorch defaults (momentum=0.1, eps=1e-5) | momentum=0.01, eps=1e-3 | Slower running-mean adaptation in NeuralBench; slightly different normalization |

### Moderate (may contribute to score delta)

| # | Difference | Foundry | NeuralBench | Expected Impact |
|---|-----------|---------|-------------|-----------------|
| 4 | **Spatial conv max_norm** | None | max_norm=1 | Weight constraint prevents spatial filter explosion; subtle effect |
| 5 | **Readout structure** | Multi-task ReadoutRouter | Single-task nn.Linear | Functionally equivalent for single-task, but extra routing logic |
| 6 | **Input pipeline** | tokenize → pad2d → forward | Direct tensor to forward | Same numerical path for uniform-shape data |

### Negligible

| # | Difference | Notes |
|---|-----------|-------|
| 7 | Num workers (4 vs 10) | No effect on model output |
| 8 | Log metric names | Cosmetic only |

---

## 5. Recommendations

1. **For the POC comparison**: run both systems with identical seeds and report
   the absolute validation balanced-accuracy difference. The dropout and LR
   schedule differences are the primary confounds.

2. **To close the gap** (optional future work):
   - Override Foundry `dropout_rate=0.25` to match braindecode defaults.
   - Implement a `OneCycleLR` option in FoundryModule or use a custom config.
   - Add `batch_norm_momentum=0.01, batch_norm_eps=1e-3` options to EEGNetEncoder.
   - Add `max_norm` constraint support to the spatial depthwise conv.

3. **Interpretation**: any validation score difference should first be attributed
   to the documented architectural/training differences above, not to a data
   contract mismatch. The data path is identical (same NeuralSet pipeline,
   same splits, same preprocessing).
