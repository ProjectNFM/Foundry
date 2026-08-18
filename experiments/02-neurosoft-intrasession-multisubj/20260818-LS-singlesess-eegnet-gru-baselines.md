# Session-Level EEGNet / GRU vs POYO (8-band)

**Status:** Completed
**Date started:** 2026-08-18
**Parent experiment:** [Intrasession Optimal-HP Training Paradigm Baselines](20260727-LS-intrasession-opt-baselines.md)
**Follow-up experiments:** TBD
**Tags:** neurosoft, 8band, intrasession, singlesession, baseline, eegnet, gru, poyo, auditory_decoding, minipigs, monkeys

## Background

The [opt-HP paradigm baselines](20260727-LS-intrasession-opt-baselines.md)
ranked POYO pooling as single-session < single-subject < multi-subject
(minipigs) and put multi-subject clearly on top for monkeys after
excluding pathological sessions. [Capacity](20260805-LS-model-capacity.md)
then raised the multi-subject ceiling (best fold-0 F1 **0.394 / 0.538**).

Those comparisons are all within POYO. This report adds **non-foundation**
session-level baselines — EEGNet and a bidirectional GRU — from
`NEUROSOFT_INTRASESSION_SINGLESESS`. Each EEGNet/GRU/POYO-session run
trains and evaluates on **one recording**, so those three are
protocol-matched. Transformer POYO typically needs more training data
than CNN/RNN baselines, so that matched session comparison is an unfair
low-data test; single-subject and
multi-subject POYO are the fairer high-data references, even though they
are not protocol-matched to session EEGNet/GRU.

## Question

How do session-averaged EEGNet and GRU max-val metrics compare to
(1) opt-HP **single-session** POYO (mean across sessions), (2) opt-HP
**single-subject** POYO (mean across subjects), and (3) the **best
multi-subject** POYO (reduced-capacity fold-0 winners)?

## Hypothesis

Transformer-based POYO typically needs more training data than CNN/RNN
models, so comparing it to EEGNet/GRU at the **single-session**
(low-data) regime is unfair.
Given enough data — single-subject or, especially, **multi-subject**
pooling — POYO should **outperform** those traditional session-level
baselines on **both F1 and AUROC**.

## Experiment

### Setup

- **Project:** `auditory_decoding`
- **Species detection:** WandB run tags (`minipigs` / `monkeys`)
- **Task / metric prefix:** `val/neurosoft_acoustic_stim_8band_*`
  (report **max** summary / history values)
- **Split:** `intrasession-block` (fixed)

| Condition | Source | Aggregation |
|-----------|--------|-------------|
| EEGNet | group `NEUROSOFT_INTRASESSION_SINGLESESS`, tags `baselines`+`eegnet` | mean±std of per-session fold-means (excl. outliers) |
| GRU | same group, tags `baselines`+`gru` | same |
| POYO session | opt-HP sweeps `hiyb4224` / `h5gf9jn1` ([20260727](20260727-LS-intrasession-opt-baselines.md)) | same |
| POYO subject | opt-HP sweeps `4k9zt970` / `aycfxm9b` | mean±std of per-subject fold-means |
| POYO multi (best) | capacity fold-0 winners `ncx1been` / `zrvjtixp` ([20260805](20260805-LS-model-capacity.md)) | single pooled run |

**Session outliers:** missing F1 or fold-mean F1 ≥ 0.99 (same rule as
[20260727](20260727-LS-intrasession-opt-baselines.md)). Two monkey
sessions (`sub-02_ses-04/05` RH) are excluded; none for minipigs.
Finished primary session runs: 41×3 folds × 3 models (minipigs) and
27×3 × 3 (monkeys). POYO subject: 7 minipigs + 6 monkeys (3 folds).

**Session-level model configs:**

| Model | Config | Notes |
|-------|--------|--------|
| POYO-EEG | opt-HP singlesess (resample CNN; species HPs from parent) | `bs=128`, patience 50 |
| EEGNet | `configs/experiment/auditory_decoding/eegnet_neurosoft_8band_intrasession_singlesess.yaml` | opt-HP: F1=8, D=2, F2=16; `lr=0.015`, `wd=0.018`, `bs=16`, patience 20 |
| GRU | `configs/experiment/auditory_decoding/gru_neurosoft_8band_intrasession_singlesess.yaml` | opt-HP: 2-layer bidirectional, hidden 128; `lr=0.0015`, `wd=0.018`, `bs=16`, patience 20 |

Repo YAMLs for EEGNet/GRU are minipigs-templated; monkey runs in the
group use the same architectures with `neurosoft_monkeys` data / tags.
EEGNet/GRU HPs were selected by a prior hyperparameter search (values
above). Non-opt POYO in the same group (45 minipig runs) is excluded
from the primary tables.

### Launch command

```bash
# EEGNet / GRU (session grid via Hydra sweeper on recording_ids)
uv run python main.py experiment=auditory_decoding/eegnet_neurosoft_8band_intrasession_singlesess -m
uv run python main.py experiment=auditory_decoding/gru_neurosoft_8band_intrasession_singlesess -m

# POYO references (already reported)
wandb agent <entity>/auditory_decoding/hiyb4224   # session minipigs
wandb agent <entity>/auditory_decoding/h5gf9jn1   # session monkeys
wandb agent <entity>/auditory_decoding/4k9zt970   # subject minipigs
wandb agent <entity>/auditory_decoding/aycfxm9b   # subject monkeys
# best multi: ncx1been / zrvjtixp from capacity sweep ov9f1g0n / 104ze4mt
```

### Key config overrides

See YAMLs above. POYO uses species-optimal tokenizer / lr / WD / dropout
from the [HP search](20260717-LS-intrasession-multisubj-hp.md); EEGNet
and GRU use HPs selected by a prior search (architecture, `lr`, `wd`
in the session YAMLs).

## Results

### Summary

On the **matched session protocol**, EEGNet and GRU beat opt-HP POYO on
mean max-val F1 for both species (minipigs EEGNet 0.578 vs POYO 0.287;
monkeys GRU 0.571 vs POYO 0.394). They also beat **single-subject** POYO
on F1. Versus the **best multi-subject** POYO, session EEGNet (minipigs)
and session GRU (monkeys) are still higher on F1, but best multi-subject
POYO has the highest AUROC in both species — by a wide margin in monkeys
(0.892 vs ~0.56–0.59). That F1/AUROC split is the main result: pooled
POYO ranks well, but does not convert those scores into competitive
hard-label F1.

Session-level F1 and AUROC can disagree, especially in monkeys: several
high-F1 sessions have AUROC near 0.35. Minipig session AUROC is more
consistent with the F1 ranking (EEGNet 0.771, GRU 0.737, POYO 0.606).

### Metrics

#### Session EEGNet / GRU vs POYO session, subject, and best multi

Aggregation: EEGNet, GRU, and POYO session = mean±std **across sessions**
of fold-mean max-val metrics (excl. outliers). POYO subject = mean±std
**across subjects**. POYO multi (best) = capacity fold-0 winner (no unit
std).

| Species | Condition | n | F1 | AUROC | Precision | Recall | Balanced acc |
|---------|-----------|---|----|-------|-----------|--------|--------------|
| minipigs | EEGNet (session) | 41 | 0.5775±0.1981 | 0.7711±0.1256 | 0.6480±0.1699 | 0.5894±0.1904 | 0.5894±0.1904 |
| minipigs | GRU (session) | 41 | 0.5100±0.2552 | 0.7370±0.1487 | 0.5525±0.2618 | 0.5339±0.2297 | 0.5339±0.2297 |
| minipigs | POYO session | 41 | 0.2866±0.2061 | 0.6061±0.1369 | 0.3280±0.2226 | 0.3292±0.1833 | 0.3292±0.1833 |
| minipigs | POYO subject | 7 | 0.3203±0.1124 | 0.7054±0.0908 | 0.3752±0.1154 | 0.3274±0.1033 | 0.3274±0.1033 |
| minipigs | POYO multi (best) | 1 | 0.3936 | 0.8009 | 0.4049 | 0.3954 | 0.3954 |
| monkeys | EEGNet (session) | 25 | 0.5033±0.2116 | 0.5636±0.2427 | 0.6190±0.1854 | 0.5263±0.1970 | 0.5263±0.1970 |
| monkeys | GRU (session) | 25 | 0.5706±0.2611 | 0.5851±0.2751 | 0.6200±0.2556 | 0.6016±0.2360 | 0.6016±0.2360 |
| monkeys | POYO session | 25 | 0.3944±0.2630 | 0.5021±0.2597 | 0.4443±0.2700 | 0.4385±0.2362 | 0.4385±0.2362 |
| monkeys | POYO subject | 6 | 0.3710±0.1364 | 0.5878±0.2215 | 0.4465±0.1396 | 0.4105±0.1189 | 0.4105±0.1189 |
| monkeys | POYO multi (best) | 1 | 0.5382 | 0.8916 | 0.5356 | 0.5485 | 0.5485 |

POYO session / subject numbers reproduce
[20260727](20260727-LS-intrasession-opt-baselines.md); POYO multi (best)
reproduces [20260805](20260805-LS-model-capacity.md).

#### Δ F1 / AUROC: session EEGNet or GRU minus each POYO reference

| Species | Baseline | − POYO session | − POYO subject | − POYO multi (best) |
|---------|----------|----------------|----------------|---------------------|
| minipigs | EEGNet | +0.291 / +0.165 | +0.257 / +0.066 | +0.184 / −0.030 |
| minipigs | GRU | +0.223 / +0.131 | +0.190 / +0.032 | +0.116 / −0.064 |
| monkeys | EEGNet | +0.109 / +0.062 | +0.132 / −0.024 | −0.035 / −0.328 |
| monkeys | GRU | +0.176 / +0.083 | +0.200 / −0.003 | +0.032 / −0.307 |

#### Excluded single-session outliers

| Species | Model | Session | F1 | Reason |
|---------|-------|---------|----|--------|
| monkeys | EEGNet, GRU, POYO | `sub-02_ses-04_task-AcousStim_acq-RH_desc-raw` | 1.00 | F1 ≥ 0.99 |
| monkeys | EEGNet, GRU | `sub-02_ses-05_task-AcousStim_acq-RH_desc-raw` | 1.00 | F1 ≥ 0.99 |
| monkeys | POYO | `sub-02_ses-05_task-AcousStim_acq-RH_desc-raw` | missing | missing F1 |

### Analysis

```bash
uv run python analysis/20260818-LS-singlesess-eegnet-gru-baselines.py
# optional: reuse cached CSVs
uv run python analysis/20260818-LS-singlesess-eegnet-gru-baselines.py --cached
```

### Figures

![F1: session EEGNet/GRU vs POYO session, subject, and best multi](../../analysis/figures/20260818-LS-singlesess-eegnet-gru-baselines_f1_by_model.png)

![AUROC: session EEGNet/GRU vs POYO session, subject, and best multi](../../analysis/figures/20260818-LS-singlesess-eegnet-gru-baselines_auroc_by_model.png)

![Per-session F1: EEGNet and GRU vs best multi-subject POYO](../../analysis/figures/20260818-LS-singlesess-eegnet-gru-baselines_f1_vs_poyo_multi.png)

![Per-session AUROC: EEGNet and GRU vs best multi-subject POYO](../../analysis/figures/20260818-LS-singlesess-eegnet-gru-baselines_auroc_vs_poyo_multi.png)

Each point is one recording. X is that session’s **history-max**
`val_session/` F1 or AUROC on the best multi-subject POYO run (fold 0:
`ncx1been` / `zrvjtixp`). Points above the identity line beat
multi-subject POYO on that same session.

**Supplementary — per-session bars (excl. outliers):** EEGNet / GRU /
session POYO, plus **single-subject POYO** and **best multi-subject
POYO** `val_session/` scores for the same recording (solid purple /
green; subject POYO is a fold-mean across the three subject-level
runs).

![Per-session F1](../../analysis/figures/20260818-LS-singlesess-eegnet-gru-baselines_supp_f1_per_session.png)

![Per-session AUROC](../../analysis/figures/20260818-LS-singlesess-eegnet-gru-baselines_supp_auroc_per_session.png)

## Conclusions

The hypothesis that high-data POYO should **outperform** session
EEGNet/GRU on **both** F1 and AUROC is **partial**: **refuted on F1**,
**supported on AUROC** for multi-subject POYO. Session-level POYO is
the expected unfair low-data loss.

- **Matched session protocol (low data):** EEGNet and GRU outperform
  session POYO on mean F1 and AUROC for both species — the expected
  unfair comparison for a transformer trained in a low-data regime.
- **Vs single-subject POYO:** session EEGNet/GRU still win on F1; AUROC
  is mixed (minipigs baselines slightly ahead; monkeys ≈ tied).
  Single-subject pooling is not a high enough data regime for POYO to
  outperform the traditional baselines.
- **Vs best multi-subject POYO:** session EEGNet (minipigs, ΔF1 +0.184)
  and session GRU (monkeys, ΔF1 +0.032) remain higher on **mean** F1, but
  best multi-subject POYO wins **pooled** AUROC — slightly in minipigs
  (0.801 vs EEGNet 0.771) and by a large margin in monkeys (0.892 vs
  ~0.56–0.59). Matched per-session scatters (`val_session/` on the same
  recordings) tell a session-wise story: on F1, most minipig sessions
  still sit above identity (EEGNet 35/41, GRU 26/41) while monkeys
  favor pooled POYO vs EEGNet (8/25 above) and are split vs GRU
  (14/25); on AUROC, minipig baselines win most sessions (38/41, 28/41)
  even though pooled POYO has the higher species AUROC, and monkeys
  track the identity line more closely (10/25, 16/25 above).
- **Caveats:** session vs pooled protocols differ; EEGNet/GRU HPs were
  tuned but patience is 20 vs POYO 50; monkey session F1 can look
  strong while AUROC is near chance (see supplementary per-session
  AUROC).

**Interpretation (F1 vs AUROC).** High AUROC with lower F1 means POYO
ranks positives above negatives well, but the default hard-decision
threshold (typically 0.5) is a poor operating point — or the scores
are uncalibrated. AUROC is threshold-free ranking; F1 is computed
after converting probabilities into class labels. In an 8-class
problem that default cut is rarely optimal, so False Positives /
False Negatives inflate even when ranking is strong.

Pooling still improves POYO’s **ranking**. It has not yet produced a
multi-subject F1 that beats these session-level CNN/RNN baselines.

## Notes for future experiments

- **Tune classification thresholds:** search per-class probability
  thresholds (cross-validation) to maximize F1 instead of using 0.5.
- **Calibrate scores:** if probabilities cluster at the extremes or
  the middle, apply Platt scaling or isotonic regression before
  thresholding.
- **Class imbalance:** F1 depends on precision/recall; for rare
  bands, report Precision-Recall AUC (PR-AUC) as a complement to
  AUROC.
- Train **EEGNet and GRU at single-subject and multi-subject** pooling
  so the architecture comparison is protocol-matched to the POYO thread.
- Inspect monkey sessions with **high F1 and low AUROC** (possible
  collapse / class imbalance artifacts) before using session F1 as a
  scoreboard.
- Treat session-level EEGNet (minipigs) and GRU (monkeys) as a
  **practical F1 ceiling** that pooled POYO still needs to beat, after
  threshold/calibration work.
