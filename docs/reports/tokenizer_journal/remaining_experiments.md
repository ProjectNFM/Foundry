# Remaining AJILE12 Experiments

Experiments to run once the token rate sweep results are in. These investigate the CWT+CNN hybrid architecture and the CWT training dynamics.

---

## Experiment Track 2: CWT + CNN Hybrid

**Question:** Does adding a Conv1d stack AFTER the CWT improve performance by combining CWT's sampling-rate-invariant frequency decomposition with CNN's ability to learn temporal patterns across frequency bins?

**Architecture change:**

Current `CWTEmbedding`:
```
CWT → (B, T, sources*2*freqs) → Linear → (B, T, embed_dim)
```

Proposed `CWTCNNEmbedding`:
```
CWT → (B, sources*2*freqs, T) → Conv1d stack → (B, num_filters, T) → Linear → (B, T, embed_dim)
```

This differs from the previously-tested "CWT-compressor" (which used strided convolutions to reduce token count). Here we keep the same token count and add conv layers to learn cross-frequency temporal patterns on top of the scalogram.

### Code changes

1. **New class `CWTCNNEmbedding`** in `foundry/models/embeddings/temporal/cwt.py`:
   - Composes `ContinuousCWTLayer` + `nn.Sequential` of Conv1d layers (same pattern as `ResampleCNNEmbedding.cnn`) + `nn.Linear`
   - Constructor params: same as `CWTEmbedding` plus `num_filters`, `kernel_size`, `num_conv_layers`, `activation`
   - Forward: CWT output `(B,S,2,F,T)` → reshape to `(B, S*2*F, T)` → conv stack → transpose → linear projection

2. **Register in exports:** Add `CWTCNNEmbedding` to `foundry/models/embeddings/temporal/__init__.py` and `foundry/models/embeddings/__init__.py`

3. **New tokenizer config:** `configs/model/tokenizer/per_channel_cwt_cnn.yaml`:
   ```yaml
   _target_: foundry.models.tokenizer.EEGTokenizer
   patch_duration: null
   embed_dim: ${model.embed_dim}
   channel_fusion: concat
   channel_emb_dim: 64
   channel_strategy:
     _target_: foundry.models.embeddings.PerChannelStrategy
     max_channels: ${hyperparameters.num_channels}
   temporal_embedding:
     _target_: foundry.models.embeddings.CWTCNNEmbedding
     embed_dim: ${eval:'${model.embed_dim} - ${model.tokenizer.channel_emb_dim}'}
     num_sources: 1
     num_freqs: 9
     min_freq: 0.5
     max_freq: 30.0
     freq_spacing: log
     target_token_rate: 100.0
     n_cycles: 2.5
     num_filters: 64
     kernel_size: 9
     num_conv_layers: 2
     activation: gelu
   ```

### Experiment config

- New YAML: `configs/experiment/tokenizer_explore/poyo_ajile_cwt_cnn_hybrid.yaml`
- Sweeps: `per_channel_cwt`, `per_channel_resample_cnn`, `per_channel_cwt_cnn` x 2 folds = **6 runs**
- Hyperparameters: LR=3e-4, WD=0.007, token_rate=100 (or whatever the token rate sweep determines is best)

---

## Experiment Track 3: CWT Training Dynamics

**Question:** Why do the CWT frequencies barely move during training? Is it because (a) gradients are vanishingly small, (b) the loss landscape is flat w.r.t. frequency params, or (c) the initialization is already near-optimal?

### Sub-experiment 3a: Gradient Monitoring

**Code changes:**

1. Extend `ParameterWatcherCallback` in `foundry/training/callbacks.py`:
   - Add `log_gradients: bool = False` parameter
   - In `on_train_batch_end`, if `log_gradients` and matched params have `.grad`, log:
     - `params/{name}/grad_norm`
     - `params/{name}/grad_mean`
     - `params/{name}/grad_max`
     - Per-element grad values for small tensors

2. Optionally extend `ContinuousCWTLayer.get_watched_params()` to also expose the raw unconstrained parameters (useful for understanding the softplus reparameterization's effect on gradient flow).

### Sub-experiment 3b: Separate Learning Rate for CWT Parameters

**Hypothesis:** The CWT frequency/cycle params need a higher LR than the backbone. The backbone's optimal LR (3e-4) may be too small for parameters that represent physical frequencies constrained through softplus.

**Code changes:**

1. Modify `configure_optimizers` in `foundry/training/task_modules.py`:
   - Accept an optional `cwt_lr_multiplier` (default 1.0) from hyperparameters
   - Create separate param groups: one for CWT params (matching `*cwt*freqs*` or `*cwt*n_cycles*`) with `lr * multiplier`, one for everything else

2. Add `cwt_lr_multiplier` to `hyperparameters` in experiment config

**Experiment config:**
- Sweeps `hyperparameters.cwt_lr_multiplier: "1,10,50,100"` x 2 folds = **8 runs**
- Uses per_channel_cwt tokenizer only
- Enables gradient logging

### Sub-experiment 3c: Different Frequency Initializations

**Question:** If we start CWT frequencies far from the default log-spaced initialization, do they converge to the same values?

**Code changes:** None -- frequency init is configurable via `min_freq`, `max_freq`, `num_freqs`, and `freq_spacing` in the tokenizer YAML.

**Configurations to test:**

| Name | min_freq | max_freq | freq_spacing | Rationale |
|------|----------|----------|-------------|-----------|
| Default | 0.5 | 30.0 | log | Current baseline |
| Wide | 0.1 | 100.0 | log | Covers gamma band |
| Narrow | 2.0 | 15.0 | log | Concentrated in alpha/beta |
| Linear | 0.5 | 30.0 | linear | Uniform spacing in Hz |

- 4 inits x 2 folds = **8 runs**

### Combined dynamics experiment

All of track 3 can share a single experiment YAML with sweeper params covering the LR multiplier and init variants. The gradient logging should be enabled for all runs in this group.

---

## Ordering and Dependencies

```
Token rate sweep (Track 1)
    ↓ results determine best token_rate
CWT+CNN hybrid (Track 2) — use best token_rate
CWT dynamics (Track 3) — can run in parallel with Track 2
```
