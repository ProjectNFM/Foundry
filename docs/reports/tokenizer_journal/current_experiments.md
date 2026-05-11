# Current AJILE12 Experiments

---

## Experiment 1: Target Token Rate Scaling — COMPLETED

**W&B group:** `TOKEN_RATE_SWEEP`

**Question:** How does the output token rate (tokens per second) affect downstream
behavior classification performance, and do CWT and ResampleCNN scale differently
with increasing token count?

**Design:** Sweep `target_token_rate` over {100, 200, 400} Hz for `per_channel_cwt`
and `per_channel_resample_cnn`, each with 2 folds. All other hyperparameters are
held constant at the matched clean-comparison settings (LR=3e-4, WD=0.007,
embed_dim=256, concat fusion).

**Runs:** 2 tokenizers × 3 rates × 2 folds = 12 runs.

**Results:**
- Performance is **surprisingly flat** across 100–400 Hz for both CWT and CNN.
- CWT maintains a small edge (~0.4–0.8% AUROC) at every rate.
- Neither tokenizer scales better than the other — both are flat.
- **Conclusion:** 100 Hz provides sufficient temporal resolution; 4× compute
  savings vs 400 Hz at negligible performance cost.

---

## Experiment 2: CWT + CNN Hybrid — COMPLETED

**W&B group:** `TOKEN_RATE_SWEEP`

**Question:** Does adding a convolutional stack after the CWT scalogram—before the
final linear projection—improve performance over either CWT or CNN alone?

**Design:** The `CWTCNNEmbedding` module runs the standard CWT
(9 log-spaced frequencies, 0.5–30 Hz) then feeds the `(B, sources*2*freqs, T)`
scalogram through a 2-layer Conv1d stack (64 filters, kernel 9, GELU) before a
linear projection to `embed_dim`. Swept across the same 3 token rates as
Experiment 1.

**Runs:** 1 tokenizer × 3 rates × 2 folds = 6 runs (included in the same sweep).

**Results:**
- CWT+CNN **outperforms** both CWT and CNN at all token rates (~0.90 vs ~0.89 vs ~0.88 AUROC).
- However, CWT+CNN has ~15× more parameters (59,858 vs ~3,700–3,900) due to
  64 conv filters (vs 12 for CNN) operating on the 18-dim CWT output.
- Parameter-matched comparison needed → Experiment 4.

---

## Experiment 3a: CWT Gradient Diagnostics — COMPLETED

**W&B group:** `TOKEN_RATE_SWEEP` (passive logging on all 18 runs)

**Question:** Why do the CWT's learnable center frequencies and cycle counts barely
move during training? Is the loss landscape flat with respect to those parameters,
or are the gradients being attenuated before they reach the CWT layer?

**Design:** Gradient logging enabled via `ParameterWatcherCallback` on all 18
TOKEN_RATE_SWEEP runs. Logged per-step: gradient norms, Adam optimizer state,
effective step size, and update-to-parameter ratio.

**Results:**
- **Confirmed:** CWT parameters are effectively frozen by gradient attenuation.
- `freqs update_to_param_ratio` ≈ 3–6 × 10⁻⁶ across all runs.
- `n_cycles update_to_param_ratio` ≈ 3–4 × 10⁻⁵ across all runs.
- Gradient norms are nonzero (0.01–0.28) but Adam's effective steps are tiny.
- **Conclusion:** Frequencies reflect initialization, not task-specific adaptation.
  → Experiment 3b (higher CWT LR) warranted.

---

## Experiment 3b: CWT Learning Rate Multiplier Sweep — RUNNING

```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_cwt_lr_sweep -m
```

**W&B group:** `CWT_LR_AND_PARAM_MATCH`

**Question:** If the CWT parameters receive a substantially higher learning rate
(via a separate Adam param group), do the frequencies move meaningfully and
does that improve downstream performance?

**Design:** Sweep `cwt_lr_multiplier` over {10, 50, 100} for `per_channel_cwt`
at `target_token_rate=200` Hz, 2 folds. Gradient diagnostics still enabled.

**Runs:** 1 tokenizer × 3 multipliers × 2 folds = 6 runs.

**Status:** Still running. 10x and 50x folds completed; 100x folds in progress.

---

## Experiment 4: Parameter-Matched CWT+CNN vs CNN — COMPLETED

```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_param_match -m
```

**W&B group:** `CWT_LR_AND_PARAM_MATCH`

**Question:** Does the CWT+CNN hybrid's advantage come from the CWT
preprocessing (architectural benefit) or from having more model capacity
(64 conv filters vs CNN's 12)?

**Design:** Compare `per_channel_cwt_cnn` (64 filters, 59,858 temporal params)
against `per_channel_resample_cnn_64f` (64 filters, 50,048 temporal params)
at `target_token_rate=200` Hz, 2 folds.

| Tokenizer            | Conv Filters | Temporal Params | Input to Conv1d |
| -------------------- | ------------ | --------------- | --------------- |
| CNN (12f, original)  | 12           | 3,924           | 1 channel       |
| CNN (64f)            | 64           | 50,048          | 1 channel       |
| CWT+CNN (64f)        | 64           | 59,858          | 18 channels     |

**Runs:** 2 tokenizers × 2 folds = 4 runs.

**Results:**

| Tokenizer     | Fold 0 | Fold 1 | Mean   | Std    |
| ------------- | ------ | ------ | ------ | ------ |
| CWT+CNN (64f) | 0.8914 | 0.9042 | 0.8978 | 0.0064 |
| CNN (64f)     | 0.8800 | 0.8932 | 0.8866 | 0.0066 |

- CWT+CNN retains a **~1.1% AUROC advantage** even with parameter-matched CNN.
- For context, the original CNN 12f scored ~0.884 at 200 Hz — scaling from
  12 to 64 filters (~13× params) yielded only ~0.3% improvement.
- **Conclusion:** The CWT+CNN advantage is at least partly architectural
  (CWT preprocessing helps), though some of the original ~2% gap was due to
  the capacity difference (it narrowed to ~1.1% with matching). Follow-up:
  test CWT+CNN at 12 filters → Experiment 5.

---

## Experiment 5: Low-Capacity CWT+CNN Hybrid — READY TO LAUNCH

```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_cwt_cnn_12f -m
```

**W&B group:** `CWT_CNN_12F`

**Question:** Does the CWT+CNN architectural advantage persist when the
convolutional stack is reduced to 12 filters, matching the original CNN's
capacity? If CWT+CNN 12f still outperforms CNN 12f, the benefit is purely
from the CWT preprocessing with zero capacity advantage.

**Design:** Run `per_channel_cwt_cnn_12f` (12 filters, 5,778 temporal params)
at `target_token_rate=200` Hz, 2 folds. The CNN 12f baseline (3,924 temporal
params) already exists from the TOKEN_RATE_SWEEP at 200 Hz with identical
hyperparameters (LR=3e-4, WD=0.007, embed_dim=256, concat fusion), so those
runs are reused directly.

| Tokenizer            | Conv Filters | Temporal Params | Input to Conv1d | Source           |
| -------------------- | ------------ | --------------- | --------------- | ---------------- |
| CNN (12f)            | 12           | 3,924           | 1 channel       | TOKEN_RATE_SWEEP |
| CWT+CNN (12f)        | 12           | 5,778           | 18 channels     | CWT_CNN_12F      |

**Runs:** 1 tokenizer × 2 folds = 2 new runs (+ 2 existing CNN 12f runs).

---

## Summary of All Experiments

| #    | Experiment                    | Status    | Group                   | Runs |
| ---- | ----------------------------- | --------- | ----------------------- | ---- |
| 1    | Token rate scaling            | Completed | TOKEN_RATE_SWEEP        | 12   |
| 2    | CWT+CNN hybrid (64f)          | Completed | TOKEN_RATE_SWEEP        | 6    |
| 3a   | CWT gradient diagnostics      | Completed | TOKEN_RATE_SWEEP        | 0    |
| 3b   | CWT LR multiplier sweep       | Running   | CWT_LR_AND_PARAM_MATCH | 6    |
| 4    | Parameter-matched CWT+CNN/CNN | Completed | CWT_LR_AND_PARAM_MATCH | 4    |
| 5    | Low-capacity CWT+CNN (12f)    | Ready     | CWT_CNN_12F             | 2    |

---

## Follow-up (pending results of 3b + 5)

- **Experiment 3c:** If the frequencies do move with higher LR, test different
  initializations (wide 0.1–100 Hz, narrow 2–15 Hz, linear spacing) to determine
  whether the model converges to a task-specific frequency set or stays near init.
- **Experiment 6:** Depending on Experiment 5 results, explore CWT+CNN with
  intermediate filter counts (e.g., 32) to map the capacity–performance curve.
