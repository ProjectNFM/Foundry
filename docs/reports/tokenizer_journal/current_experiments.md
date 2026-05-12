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

## Experiment 3b: CWT Learning Rate Multiplier Sweep — COMPLETED

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

**Results:**


| Multiplier | Fold 0 | Fold 1 | Mean   | Std    |
| ---------- | ------ | ------ | ------ | ------ |
| 1× (base)  | 0.8859 | 0.8914 | 0.8887 | 0.0039 |
| 10×        | 0.8813 | 0.8941 | 0.8877 | 0.0091 |
| 50×        | 0.8781 | 0.8923 | 0.8852 | 0.0100 |
| 100×       | 0.8771 | 0.8937 | 0.8854 | 0.0117 |


- Higher LR multipliers **successfully unfreeze** the CWT parameters
(update-to-param ratio increases ~100× from baseline), and frequencies/cycle
counts visibly drift from initialization.
- However, performance is **flat or slightly worse** — the baseline (1×) has
the highest mean AUROC.
- **Conclusion:** The bottleneck is not the specific frequency values but the
architecture of the representation (9 broad, overlapping wavelets at low
n_cycles). Repositioning the filters doesn't help because the basis remains
redundant. Future direction: more frequencies + higher n_cycles for a
selective, orthogonal spectral basis.

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


| Tokenizer           | Conv Filters | Temporal Params | Input to Conv1d |
| ------------------- | ------------ | --------------- | --------------- |
| CNN (12f, original) | 12           | 3,924           | 1 channel       |
| CNN (64f)           | 64           | 50,048          | 1 channel       |
| CWT+CNN (64f)       | 64           | 59,858          | 18 channels     |


**Runs:** 2 tokenizers × 2 folds = 4 runs.

**Results:**


| Tokenizer     | Fold 0 | Fold 1 | Mean   | Std    |
| ------------- | ------ | ------ | ------ | ------ |
| CWT+CNN (64f) | 0.8914 | 0.9042 | 0.8978 | 0.0064 |
| CNN (64f)     | 0.8800 | 0.8932 | 0.8866 | 0.0066 |


- CWT+CNN retains a **~1.1% AUROC advantage** even with parameter-matched CNN.
- For context, the original CNN 12f scored ~~0.884 at 200 Hz — scaling from
12 to 64 filters (~~13× params) yielded only ~0.3% improvement.
- **Conclusion:** The CWT+CNN advantage is at least partly architectural
(CWT preprocessing helps), though some of the original ~2% gap was due to
the capacity difference (it narrowed to ~1.1% with matching). Follow-up:
test CWT+CNN at 12 filters → Experiment 5.

---

## Experiment 5: Low-Capacity CWT+CNN Hybrid — COMPLETED

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


| Tokenizer     | Conv Filters | Temporal Params | Input to Conv1d | Source           |
| ------------- | ------------ | --------------- | --------------- | ---------------- |
| CNN (12f)     | 12           | 3,924           | 1 channel       | TOKEN_RATE_SWEEP |
| CWT+CNN (12f) | 12           | 5,778           | 18 channels     | CWT_CNN_12F      |


**Runs:** 1 tokenizer × 2 folds = 2 new runs (+ 2 existing CNN 12f runs).

**Results:**


| Tokenizer     | Fold 0 | Fold 1 | Mean   | Std    |
| ------------- | ------ | ------ | ------ | ------ |
| CNN (12f)     | 0.8798 | 0.8885 | 0.8841 | 0.0062 |
| CWT+CNN (12f) | 0.8842 | 0.8926 | 0.8884 | 0.0059 |


- CWT+CNN 12f outperforms CNN 12f by **~0.4% AUROC** — a smaller gap than the
  ~1.1% seen at 64 filters (Experiment 4).
- CWT+CNN 12f (0.888) essentially matches CWT-alone (0.889, from Experiment 3b),
  suggesting the 12-filter CNN stack adds negligible value on top of the CWT
  scalogram — the linear projection is already sufficient at this width.
- **Conclusion:** The CWT preprocessing provides a small architectural benefit
  at any capacity, but the convolutional stack needs sufficient width (>>12
  filters) to meaningfully exploit cross-frequency features from the scalogram.
  At 12 filters, the hybrid degenerates to CWT-like performance.

---

## Summary of All Experiments


| #   | Experiment                    | Status    | Group                   | Runs |
| --- | ----------------------------- | --------- | ----------------------- | ---- |
| 1   | Token rate scaling            | Completed | TOKEN_RATE_SWEEP        | 12   |
| 2   | CWT+CNN hybrid (64f)          | Completed | TOKEN_RATE_SWEEP        | 6    |
| 3a  | CWT gradient diagnostics      | Completed | TOKEN_RATE_SWEEP        | 0    |
| 3b  | CWT LR multiplier sweep       | Completed | CWT_LR_AND_PARAM_MATCH  | 6    |
| 4   | Parameter-matched CWT+CNN/CNN | Completed | CWT_LR_AND_PARAM_MATCH  | 4    |
| 5   | Low-capacity CWT+CNN (12f)    | Completed | CWT_CNN_12F             | 2    |
| 6   | CWT spectral resolution       | Ready     | CWT_SPECTRAL_RESOLUTION | 12   |


---

## Experiment 6: CWT Spectral Resolution — Tighter Bands with More Frequencies — READY TO LAUNCH

```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_cwt_spectral_resolution -m
```

**W&B group:** `CWT_SPECTRAL_RESOLUTION`

**Question:** Does increasing the number of CWT frequency bins while raising
`n_cycles` (narrower, more selective bandpass filters) produce a better spectral
decomposition for downstream classification — especially when the CWT
frequencies are allowed to adapt via a 100× learning rate multiplier?

**Motivation:** Experiments 3a–3b established that (a) CWT parameters are
effectively frozen by gradient attenuation, and (b) unfreezing them with a
higher LR does not improve performance *given only 9 broad wavelets*. The
joint interpretation identified the root cause: with only 9 wavelets at
`n_cycles ≈ 2.5`, each wavelet is a very broad bandpass filter with substantial
spectral overlap between adjacent bins. Repositioning these broad, overlapping
filters cannot help because the representation remains a redundant, smeared
basis regardless of exact center frequencies. The architectural fix is to
increase *both* the number of frequency bins and the cycle count, producing a
set of narrow, selective, minimally-overlapping bandpass filters — a clean
spectral basis from which downstream layers can extract sharp spectral
contrasts. With a denser, tighter spectral basis, the 100× LR multiplier may
now allow the model to *meaningfully* rearrange frequencies to task-relevant
positions — something that was pointless with only 9 broad overlapping filters.

**Hypothesis:** Performance improvement requires the *combination* of more
frequencies and higher `n_cycles`. More broad wavelets alone (24f + nc2.5)
just adds redundancy. Narrower wavelets alone (9f + nc7) improve selectivity
but can't resolve frequencies between the 9 widely-spaced bins. The
interaction (24f + nc7) should yield the clearest gain, confirming the
overlap/redundancy hypothesis from Experiment 3b.

**Design:** 2×2 factorial crossing `num_freqs` ∈ {9, 24} with
`n_cycles` ∈ {2.5, 7.0}, evaluated on both CWT-only and CWT+CNN architectures.
All CWT configs use `cwt_lr_multiplier=100` so the learnable frequencies and
cycle counts can adapt during training. CNN baselines (12f and 64f) are
included in the same sweep for direct comparison.

All configs use `target_token_rate=200` Hz, `freq_range=0.5–30` Hz (log
spacing), and the standard hyperparameters (LR=3e-4, WD=0.007, embed_dim=256,
concat fusion, 2 folds).

The frequency range is kept at 0.5–30 Hz for all configs to isolate the
effect of spectral resolution from the effect of covering a wider band.

**CWT LR multiplier:** All CWT and CWT+CNN configs use `cwt_lr_multiplier=100`
(separate Adam param group at LR=3e-2 for CWT freqs/n_cycles, LR=3e-4 for
everything else). The 9f-nc2.5 CWT baseline is reused from Experiment 3b
(`CWT_LR_AND_PARAM_MATCH`, which ran `per_channel_cwt` at 100× LR, 200 Hz,
2 folds). The CWT+CNN 9f-64F-nc2.5 and CNN 64f baselines are reused from
Experiment 4 (same group). CNN 12f is reused from `TOKEN_RATE_SWEEP` at
200 Hz.

**Overlap analysis at 10 Hz (representative mid-range bin):**


| Config      | Adjacent freq spacing | -3dB bandwidth | Overlap          |
| ----------- | --------------------- | -------------- | ---------------- |
| 9f, nc=2.5  | ~5.8 Hz               | ~3.0 Hz        | Moderate overlap |
| 9f, nc=7    | ~5.8 Hz               | ~1.1 Hz        | Minimal overlap  |
| 24f, nc=2.5 | ~1.8 Hz               | ~3.0 Hz        | Heavy overlap    |
| 24f, nc=7   | ~1.8 Hz               | ~1.1 Hz        | Minimal overlap  |


**Note on low-frequency wavelets:** At 0.5 Hz with `n_cycles=7`, the wavelet
envelope (σ ≈ 2.2 s) extends well beyond the 1-second window. The effective
frequency selectivity at the lowest bins is window-limited regardless of
`n_cycles`. The tighter bands matter most at mid-to-high frequencies (≥ 3 Hz)
where wavelets fit within the window. A frequency-scaled `n_cycles` variant
(proportional to frequency) could address this in a follow-up, but a uniform
`n_cycles=7` is simpler and sufficient for the initial test.

### CWT-only configurations


| Config              | num_freqs | n_cycles | Temporal Params | Config YAML                 | Source                 |
| ------------------- | --------- | -------- | --------------- | --------------------------- | ---------------------- |
| CWT-9f-nc2.5 (base) | 9         | 2.5      | 3,666           | `per_channel_cwt`           | CWT_LR_AND_PARAM_MATCH |
| CWT-9f-nc7          | 9         | 7.0      | 3,666           | `per_channel_cwt_9f_nc7`    | New                    |
| CWT-24f-nc2.5       | 24        | 2.5      | 9,456           | `per_channel_cwt_24f_nc2p5` | New                    |
| CWT-24f-nc7         | 24        | 7.0      | 9,456           | `per_channel_cwt_24f_nc7`   | New                    |


`n_cycles` does not affect parameter count (same `num_freqs` → same feat_dim
→ same linear projection). The 9f→24f jump adds ~~6k temporal params (2.6×);
this is unavoidable without adding an intermediate bottleneck (which would be
a confounding architecture change). The increase is noted but modest relative
to the full model (~~6M backbone params).

### CWT+CNN configurations (capacity-matched)


| Config                      | num_freqs | n_cycles | num_filters | Temporal Params | Config YAML                         | Source                 |
| --------------------------- | --------- | -------- | ----------- | --------------- | ----------------------------------- | ---------------------- |
| CWT+CNN-9f-64F-nc2.5 (base) | 9         | 2.5      | 64          | 59,858          | `per_channel_cwt_cnn`               | CWT_LR_AND_PARAM_MATCH |
| CWT+CNN-9f-64F-nc7          | 9         | 7.0      | 64          | 59,858          | `per_channel_cwt_cnn_9f_nc7`        | New                    |
| CWT+CNN-24f-48F-nc2.5       | 24        | 2.5      | 48          | 51,024          | `per_channel_cwt_cnn_24f_48F_nc2p5` | New                    |
| CWT+CNN-24f-48F-nc7         | 24        | 7.0      | 48          | 51,024          | `per_channel_cwt_cnn_24f_48F_nc7`   | New                    |


With 24 freqs, `num_filters` is reduced from 64 to 48 to keep total temporal
params in the same ballpark (~51k vs ~60k, within ~15%). This is a
conservative capacity match — any improvement from higher spectral resolution
cannot be attributed to extra capacity.

Parameter breakdown for CWT+CNN-24f-48F:

- CWT learnable: 24 freqs + 24 n_cycles = 48
- Conv1: `in=48, out=48, k=9` → 20,784
- Conv2: `in=48, out=48, k=9` → 20,784
- Linear: `48 → 192` → 9,408
- **Total: 51,024**

### Baselines (reused from previous experiments)


| Config         | Temporal Params | Source                                   |
| -------------- | --------------- | ---------------------------------------- |
| CWT 9f nc2.5   | 3,666           | CWT_LR_AND_PARAM_MATCH, Exp 3b (100× LR) |
| CWT+CNN 9f 64F | 59,858          | CWT_LR_AND_PARAM_MATCH, Exp 4            |
| CNN 64f        | 50,048          | CWT_LR_AND_PARAM_MATCH, Exp 4            |
| CNN 12f        | 3,924           | TOKEN_RATE_SWEEP 200 Hz                  |


### Runs

- 6 new configs × 2 folds = **12 new runs** (all CWT configs with 100× LR multiplier)
- 4 baselines × 2 folds = 8 reused runs
- **Total compared: 20 runs**

### Predicted outcomes


| Outcome                           | Interpretation                                                                                                          |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 24f+nc7 >> 9f+nc2.5 (interaction) | Overlap/redundancy hypothesis confirmed. Narrow, dense spectral basis is the right architecture.                        |
| 9f+nc7 >> 9f+nc2.5 alone          | Selectivity matters even without more bins — 9 narrow wavelets suffice.                                                 |
| 24f+nc2.5 >> 9f+nc2.5 alone       | More frequencies help even when broad — the downstream layers can separate overlapping bands.                           |
| All configs ≈ baseline            | The CWT bottleneck is not spectral resolution; the value is purely the inductive bias of *any* frequency decomposition. |


---

## Follow-up (pending results of 5 + 6)

- **Experiment 6b — Frequency-scaled n_cycles:** If Experiment 6 confirms that
higher `n_cycles` helps, test a variant where `n_cycles` scales with frequency
(e.g., `n_cycles = freq * 0.23`, giving ~2.5 at 10 Hz and ~7 at 30 Hz). This
avoids the window-limited selectivity issue at low frequencies while
maintaining tight bands where they matter.
- **Experiment 6c — Extended frequency range:** Test 24f at 0.5–50 Hz to cover
low-gamma (30–50 Hz) activity, which may be relevant for movement-related
behavior classification. Compare against 24f at 0.5–30 Hz to isolate the
range effect from the resolution effect.
- **Experiment 7:** Depending on Experiment 5 results, explore CWT+CNN with
intermediate filter counts (e.g., 32) to map the capacity–performance curve.

