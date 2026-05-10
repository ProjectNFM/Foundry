# Current AJILE12 Experiments

All three experiments run as a single sweep via:

```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_token_rate_sweep -m
```

W&B group: `TOKEN_RATE_SWEEP`

---

## Experiment 1: Target Token Rate Scaling

**Question:** How does the output token rate (tokens per second) affect downstream
behavior classification performance, and do CWT and ResampleCNN scale differently
with increasing token count?

**Hypothesis:** A higher target token rate will lead to better results for both
tokenizers, since more tokens give the Perceiver backbone more temporal granularity
to work with. CWT may show better scaling than ResampleCNN because its frequency
decomposition produces richer per-token features at higher rates—each token carries
a full scalogram slice rather than a single waveform sample, so additional tokens
provide genuinely new time-frequency information rather than redundant oversampled
signal.

**Design:** Sweep `target_token_rate` over {100, 200, 400} Hz for `per_channel_cwt`
and `per_channel_resample_cnn`, each with 2 folds. All other hyperparameters are
held constant at the matched clean-comparison settings (LR=3e-4, WD=0.007,
embed_dim=256, concat fusion).

**Runs:** 2 tokenizers × 3 rates × 2 folds = 12 runs.

---

## Experiment 2: CWT + CNN Hybrid

**Question:** Does adding a convolutional stack after the CWT scalogram—before the
final linear projection—improve performance over either CWT or CNN alone?

**Hypothesis:** The CWT provides a structured, sampling-rate-invariant
time-frequency representation, but its final linear projection may not fully
exploit cross-frequency and local-temporal interactions in the scalogram. Adding
Conv1d layers after the CWT increases the model's capacity to extract richer
dynamics from the frequency decomposition, combining the CWT's inductive bias
(frequency structure, phase information) with the CNN's ability to learn
arbitrary temporal filters. This should yield higher performance than either
approach in isolation.

**Design:** The `CWTCNNEmbedding` module runs the standard CWT
(9 log-spaced frequencies, 0.5–30 Hz) then feeds the `(B, sources*2*freqs, T)`
scalogram through a 2-layer Conv1d stack (64 filters, kernel 9, GELU) before a
linear projection to `embed_dim`. It is swept across the same 3 token rates as
Experiment 1 so that the comparison is fair at each operating point.

**Runs:** 1 tokenizer × 3 rates × 2 folds = 6 runs (included in the same sweep).

---

## Experiment 3a: CWT Gradient Diagnostics

**Question:** Why do the CWT's learnable center frequencies and cycle counts barely
move during training? Is the loss landscape flat with respect to those parameters,
or are the gradients being attenuated before they reach the CWT layer?

**Hypothesis:** The CWT frequency and cycle-count parameters sit at the very bottom
of the network (first layer), and gradients must backpropagate through the
Perceiver backbone, the linear projection, and the FFT-based convolution before
reaching them. By the time gradients arrive at `freqs_unconstrained` and
`n_cycles_unconstrained`, they are attenuated to near-zero magnitude. This
explains why the learned frequencies are so consistent across runs and
tasks—the parameters are effectively frozen by vanishing gradients rather than
being at a true optimum.

**Design:** Gradient logging is enabled globally via `ParameterWatcherCallback`
with `log_gradients: true`. For every CWT parameter (matched by `*cwt`*) the
callback logs per-step:


| Metric                            | What it reveals                                       |
| --------------------------------- | ----------------------------------------------------- |
| `grad/norm`                       | Raw gradient magnitude reaching the param             |
| `grad_to_param_ratio`             | Gradient relative to param size—is it meaningful?     |
| `optimizer/exp_avg_norm`          | Adam momentum—has a consistent direction accumulated? |
| `optimizer/exp_avg_sq_norm`       | Adam variance—is the optimizer dampening updates?     |
| `optimizer/effective_step_norm`   | Actual update Adam applies per step                   |
| `optimizer/update_to_param_ratio` | The definitive diagnostic: relative change per step   |


If `update_to_param_ratio` for CWT params is orders of magnitude smaller than for
backbone parameters, the frequencies are effectively frozen and a higher CWT
learning rate (or separate param group) would be warranted.

**Runs:** No additional runs. These diagnostics are collected passively on all 18
runs in the sweep above.

---

## Summary


| #         | Tokenizer                       | Rates         | Folds | Runs   |
| --------- | ------------------------------- | ------------- | ----- | ------ |
| 1         | per_channel_cwt                 | 100, 200, 400 | 0, 1  | 6      |
| 1         | per_channel_resample_cnn        | 100, 200, 400 | 0, 1  | 6      |
| 2         | per_channel_cwt_cnn             | 100, 200, 400 | 0, 1  | 6      |
| 3a        | (gradient logging on all above) | —             | —     | 0      |
| **Total** |                                 |               |       | **18** |


## Results Summary

All 18 runs completed successfully (early stopping, patience=20).

### Experiment 1: Token Rate Scaling
- Performance is **surprisingly flat** across 100–400 Hz for both CWT and CNN.
- CWT maintains a small edge (~0.4–0.8% AUROC) at every rate.
- Neither tokenizer scales better than the other — both are flat.
- **Conclusion:** 100 Hz provides sufficient temporal resolution; 4× compute
  savings vs 400 Hz at negligible performance cost.

### Experiment 2: CWT+CNN Hybrid
- CWT+CNN **outperforms** both CWT and CNN at all token rates (~0.90 vs ~0.89 vs ~0.88 AUROC).
- However, CWT+CNN has ~15× more parameters (59,858 vs ~3,700–3,900) due to
  64 conv filters (vs 12 for CNN) operating on the 18-dim CWT output.
- Parameter-matched comparison is needed to isolate the architectural benefit.

### Experiment 3a: CWT Gradient Diagnostics
- **Confirmed:** CWT parameters are effectively frozen by gradient attenuation.
- `freqs update_to_param_ratio` ≈ 3–6 × 10⁻⁶ across all runs.
- `n_cycles update_to_param_ratio` ≈ 3–4 × 10⁻⁵ across all runs.
- Gradient norms are nonzero (0.01–0.28) but Adam's effective steps are tiny.
- Pattern is consistent across CWT and CWT+CNN, all rates, both folds.
- **Conclusion:** Frequencies reflect initialization, not task-specific adaptation.
  Experiment 3b (higher CWT LR) is warranted.

---

## Follow-up (next steps)

- **Experiment 3b:** Gradients are confirmed vanishingly small. Implement a
  separate `cwt_lr_multiplier` param group and sweep multipliers {10, 50, 100}
  to see if the frequencies move meaningfully and whether that improves
  performance.
- **Experiment 3c:** If the frequencies do move with higher LR, test different
  initializations (wide 0.1–100 Hz, narrow 2–15 Hz, linear spacing) to determine
  whether the model converges to a task-specific frequency set or stays near init.
- **Experiment 4:** Parameter-matched CWT+CNN vs CNN comparison (either increase
  CNN filters to 64, or reduce CWT+CNN filters to 12) to isolate the
  architectural contribution of CWT preprocessing.

