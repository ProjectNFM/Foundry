# Cross-Species One-Band vs Uniform Cotraining

## Main Result

Adding one frequency band from the auxiliary species was more effective than
adding the same number of auxiliary trials sampled uniformly across all bands.
Across 16 matched comparisons, one-band cotraining outperformed uniform
cotraining in 10 cases, tied in 1, and underperformed in 5. Its mean balanced
accuracy advantage over the uniform control was 0.78 percentage points (pp).

This is a composition effect, not a claim that cotraining always improves the
target-only model. For monkey decoding, one-band cotraining improved balanced
accuracy by 1.16 pp on average, compared with 0.16 pp for uniform cotraining.
For minipig decoding, the average changes were -0.06 pp and -0.60 pp,
respectively: one-band data was less harmful, but did not improve the baseline
on average.

| Target | One-band mean delta | Uniform mean delta | One-band advantage |
| --- | ---: | ---: | ---: |
| monkeys | +1.16 pp | +0.16 pp | +1.01 pp |
| minipigs | -0.06 pp | -0.60 pp | +0.55 pp |
| combined | +0.55 pp | -0.22 pp | +0.78 pp |

These are unweighted means across the listed band conditions. The target-only
balanced-accuracy baselines were 42.03% for monkeys and 32.85% for minipigs.

The strongest target-specific comparisons were:

| Target | Auxiliary condition | N | One-band delta | Uniform delta | Advantage |
| --- | --- | ---: | ---: | ---: | ---: |
| monkeys | minipig mid treble | 2,672 | +3.07 pp | -0.59 pp | +3.67 pp |
| minipigs | monkey high treble | 3,645 | +1.20 pp | -0.45 pp | +1.65 pp |

![One-band and matched-volume controls](figures/oneband_vs_uniform_bandmatched_delta.png)

The complete values and exact auxiliary sample counts are in
[`oneband_vs_uniform_bandmatched_delta.csv`](data/oneband_vs_uniform_bandmatched_delta.csv).

## Comparison Design

Each target species has three conditions:

1. **Target only:** train using only the target species.
2. **One band:** add all available training trials from one frequency band of
   the other species.
3. **Uniform same-N:** add the same number of trials from the other species,
   distributed as uniformly as possible across all eight bands.

The one-band and uniform conditions therefore differ in auxiliary-data
composition, not auxiliary-data volume. Both are evaluated on the same target
species validation set.

Shared settings:

- Causal within-session split, seed 42.
- Eight acoustic frequency-band classes.
- 0.5 s signal windows sampled at 2 kHz.
- Per-channel convolutional patches and learned channel embeddings.
- Source-namespaced channels and sessions; shared POYO backbone and readout.
- Recordings with fewer than eight neural channels excluded.
- Checkpoint and early-stopping monitor: `val/loss`, patience 20.
- Source-specific validation metrics under `val/minipigs/...` and
  `val/monkeys/...`.

## Reproduction

Run the target-only baselines:

```bash
uv run python main.py experiment=auditory_decoding/poyo_neurosoft_freqband8_minipigs_causal
uv run python main.py experiment=auditory_decoding/poyo_neurosoft_freqband8_monkeys_causal
```

Run all one-band auxiliary conditions for monkey validation:

```bash
uv run python main.py -m \
  experiment=auditory_decoding/poyo_neurosoft_freqband8_monkeys_plus_minipig_band_causal \
  auxiliary_band=low_bass,mid_bass,low_mids,midrange,high_mids,low_treble,mid_treble,high_treble
```

Use `poyo_neurosoft_freqband8_minipigs_plus_monkey_band_causal` for minipig
validation. Replace `band_causal` with `uniform_causal` to run the corresponding
same-N controls; each uniform config resolves the correct count from
`auxiliary_band`.

## Interpretation

The result suggests that *which* auxiliary examples are added matters more
than their count alone. Concentrating auxiliary data in one band can produce a
more useful regularization signal than spreading the same budget uniformly
across heterogeneous cross-species data. The effect is band-dependent and is
not uniformly positive.

These are exploratory single-seed results. Several values use each run's best
validation balanced accuracy rather than a fixed checkpoint selected once for
all metrics. The next required control is to repeat the target-only, strongest
one-band, and matched uniform conditions over multiple seeds and report
session-level uncertainty from checkpoints selected by `val/loss`.
