# Foundry Research Roadmap

**Last updated:** 2026-08-19  
**Scope:** Research synthesis and priority order, not a single experiment hypothesis

## North Star

The project aims to identify a tokenizer and pretraining objective that learn
general neural representations from brain signals with minimal assumptions
about their structure. The eventual model should accept signals with arbitrary
duration, sampling rate, and channel count across a broad range of neural
recordings, while producing representations that carry useful biological and
task-relevant meaning.

Generality is not enough by itself. The model should be maximally useful to
neuroscientists, especially when labels and machine-learning expertise are
limited. It must therefore demonstrate competitive absolute performance, not
merely outperform other methods in an artificial frozen-feature setting while
remaining practically weak.

The following downstream structures should all remain first-class goals:

- window- or trial-level classification;
- dense temporal labeling and event detection;
- continuous regression;
- channel- or site-level prediction.

The primary transfer goal is strong zero-shot intersubject performance. Label
efficiency should initially be studied through conventional linear probing and
fine-tuning. Unlabeled subject adaptation and true in-context learning remain
important medium-term directions, but they should not drive the immediate
architecture.

## Emerging Representation Target

A single fixed-size vector is unlikely to preserve arbitrary fine-grained
information as duration and channel count grow. Timestamps tell the model where
information occurred, but cannot restore information discarded by a finite
bottleneck. Real-time processing requires bounded incremental computation, not
necessarily one fixed number of latents for an entire recording.

The emerging long-term representation should therefore be hierarchical:

1. **Channel-local, time-resolved representations** preserve localized neural
   information and support channel/site outputs.
2. **Fixed-rate fused temporal representations** provide a stable interface for
   event detection, dense labeling, regression, and real-time processing.
3. **Pooled global representations or bounded memory** summarize long-range
   context for window- and recording-level tasks.

Chunked or streaming processing can extend this hierarchy to arbitrary duration.

### Signal-first input contract

The first implementation should rely mainly on the signal and context inferred
from it. Rich acquisition metadata should remain optional rather than becoming
a prerequisite for use.

The minimal API should require:

```python
representation = model.encode(
    signal,                 # [batch, channels, samples]
    sampling_rate=512.0,    # or explicit timestamps
    channel_mask=None,
    sample_mask=None,
)
```

Physical time is non-negotiable: identical discrete samples cannot reveal their
absolute frequency scale without timestamps or a sampling rate. Channel names,
coordinates, reference, device, and subject identifiers can be accepted through
an optional metadata container later.

A future structured output could expose:

```python
representation.content
representation.content_timestamps
representation.channel_content
representation.context
representation.coverage
```

The current per-window normalization discards scale and offset. Even in a
signal-first design, the context path should eventually see the signal before
normalization and retain amplitude, robust scale, spectral-envelope, and
channel-relation information instead of deleting it permanently.

### Measurement invariance

The desired invariance is not simply "all altered views produce identical
representations." Different transformations have different semantics:

| Change | Desired behavior | Limitation |
|---|---|---|
| Channel reordering | Exact permutation invariance/equivariance | Requires channel masks/order to move consistently |
| Resampling | Agreement over the shared physical bandwidth | Downsampling genuinely removes high-frequency information |
| Re-referencing | Canonicalization or predictable equivariance | Some references remove information |
| Channel removal | Consistency over shared observations and graceful uncertainty | Removed channels may contain unique neural information |
| Signal augmentation | Invariance only when the transform is biologically safe | A transform useful for one task can destroy another task's signal |

Repeated tasks and different subjects should not be forced to produce identical
content. Task labels are imperfect proxies for brain activity. Their
representations may be semantically related, while retaining genuine variation
in strategy, attention, latency, physiology, and pathology.

## What the Existing Results Establish

### Reconstruction loss is a poor representation metric

The [information-leak experiment](./05-pretraining-parameter-exploration/20260812-MS-channel-encoder-leak-fix-impact.md)
increased pretraining validation loss from 0.0576 to 0.2838 after fixing both
shortcuts, while downstream transfer changed only slightly. Likewise, the
[masking sweep](./05-pretraining-parameter-exploration/20260811-MS-masking-parameter-sweep.md) made
reconstruction monotonically harder from mask ratio 0.5 to 0.9 without improving
motor-imagery or P300 transfer.

Consequences:

- pretraining loss should be treated as an optimization diagnostic, not the
  model-selection objective;
- further mask-ratio sweeps under the same raw-MSE formulation are low priority;
- downstream transfer, label efficiency, and nuisance decodability should drive
  representation decisions.

### P300 shows evidence of harmful transfer

The cleaned from-scratch baseline reported CWT-dynamic POYO at approximately
0.364 F1 and EEGNet at approximately 0.386 F1 under the current intersubject
protocol. The architecture gap is modest and noisy, but the training dynamics
are striking: POYO nearly memorizes the training set.

Pretrained POYO is slightly worse than scratch POYO throughout training on both
validation F1 and AUROC, while attaining better training F1. This rules against
the simplest catastrophic-forgetting and threshold-calibration explanations.
The leading interpretation is that pretraining makes subject-, session-, or
recording-specific structure easier to exploit while reducing transfer of the
target/non-target distinction to unseen subjects.

This remains a hypothesis about the representation, not yet a demonstrated
mechanism. It motivates subject/session probes before a major architecture
redesign.

### Current channel embeddings do not guarantee content-context separation

The current [`RelativeChannelEncoder`](../foundry/models/relative_channel_encoder.py)
pools temporal signal tokens, attends across channels, and produces an embedding
that is concatenated into every temporal token by the
[`EEGTokenizer`](../foundry/models/tokenizer.py). All information is then mixed
through one backbone. There is no separate content output, context output,
information bottleneck, or downstream routing constraint.

The [dynamic-channel embedding analysis](./_legacy/019-dynamic-channel-embedding-analysis.md)
found that these embeddings did not organize by electrode type and appeared to
capture signal properties such as brain state or amplitude/frequency structure.
Channel embeddings are therefore a possible seed for a future context stream,
but they are not currently an explicit factorization.

### Data-scaling conclusions remain provisional

The [data-scaling series](./02-data-scaling/README.md) suggested that the
three-dataset B2 mixture was a sweet spot, but several design issues weaken that
conclusion:

- only one pretraining seed was used per condition;
- many reported differences were much smaller than downstream fold variance;
- fixed optimizer steps did not imply matched example exposure across datasets;
- sampling was effectively weighted by recording hours rather than the
  channel-hour quantity used to describe the conditions;
- all dynamic-channel runs were affected by the later-discovered reconstruction
  leak.

No more broad data-composition scaling should be prioritized until the benchmark,
tokenizer, and objective are stable.

### Multi-length pretraining remains exploratory

The [multi-length experiment](./05-pretraining-parameter-exploration/20260811-MS-multi-length-pretraining.md)
is already running and should be completed and analyzed. Its interpretation is
limited by two implementation-level confounds:

- enumerating windows at every requested duration creates many more short-window
  batches, rather than sampling durations uniformly per batch;
- keeping 20 latent time bins across 1-, 2-, 5-, and 10-second windows changes
  effective temporal resolution from 50 ms to 500 ms, confounding duration with
  compression.

The result may indicate whether the direction is promising, but should not be
treated as a clean test of temporal-scale diversity.

## Benchmark Blind Spots

### Brain Invaders P300

The current intrasession control is not clean. Event intervals are assigned to
folds before selected events are expanded into one-second epochs. Rapid adjacent
stimuli can therefore produce training and validation windows that share raw
samples. Recording-wide normalization also uses statistics from data later
assigned to validation.

The intersubject split does keep subjects separate, so this overlap does not
directly explain intersubject negative transfer. It does invalidate the claim
that the small intrasession advantage proves cross-subject variability is
unimportant.

Published Brain Invaders results are usually within-session accuracy or AUROC,
whereas the Foundry result is unseen-subject positive-class F1. These are not
directly comparable. A canonical xDAWN/Riemannian baseline and an overlap-free
within-subject protocol are needed. See the
[MOABB benchmark table](https://moabb.neurotechx.com/docs/paper_results.html)
and [BI2014a description](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BI2014a.html).

### PhysioNet motor imagery

No obvious subject overlap was found: recordings belonging to one subject appear
to receive a common fold assignment. Nevertheless, F1 around 0.88 across very
different architectures is suspicious enough to audit.

Possible shortcuts or optimism include:

- extracting approximately four seconds instead of MOABB's standard three-second
  task window, potentially exposing transition or cue-offset information;
- selecting and reporting the best validation checkpoint on the same partition;
- tuning on fold 0 and including that easier fold in the reported mean;
- run or annotation structure that may correlate with the label.

The official MOABB benchmark is within-session and therefore not directly
comparable to pooled training over many subjects, but its substantially lower
reported values reinforce the need for controls. See the
[MOABB PhysionetMI documentation](https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html)
and [PhysioNet dataset description](https://physionet.org/content/eegmmidb/1.0.0/).

### NeuroSoft

The active NeuroSoft work should also be interpreted cautiously:

- 0.5-second labeled windows are passed to a model configured for two seconds,
  causing most of the model input to be padding rather than real context;
- the current LOSO held-out subject is used both for checkpoint selection and
  reported performance, rather than serving as an untouched test subject;
- downstream hyperparameters were selected under intrasession evaluation and
  reused for LOSO.

These issues should be corrected before NeuroSoft is promoted to a core transfer
benchmark.

## Priority Roadmap

The ordering below is based on information gained per unit of engineering and
compute, and on dependency: later experiments are only interpretable if earlier
gates pass.

| Priority | Direction | Cost | Decision enabled |
|---:|---|---|---|
| 0 | Harvest in-flight work | Very low | Avoid discarding already-paid-for evidence |
| 1 | Repair and freeze benchmarks | Low engineering, little GPU | Establish whether downstream gains are real |
| 2 | Audit existing representations | Low | Determine whether nuisance/context entanglement is the failure |
| 3 | Run a coarse label-efficiency pilot | Moderate downstream compute | Test practical utility without new pretraining |
| 4 | Resolve the CWT question with a controlled comparison | Moderate pretraining compute | Choose a defensible working tokenizer |
| 5 | Compare genuinely different objectives | Higher | Move beyond raw-MSE limitations |
| 6 | Build explicit content-context streams | Highest | Address demonstrated, rather than assumed, entanglement |

### Priority 0: Harvest in-flight work

Complete and analyze:

- multi-length downstream evaluation;
- leak-fixed iEEG pretraining transfer;
- NeuroSoft intrasession scratch baselines;
- NeuroSoft LOSO scratch baselines.

Record their limitations explicitly. Do not launch automatic follow-ups until
the relevant benchmark defects are resolved.

### Priority 1: Benchmark integrity sprint

This is the immediate priority and highest-leverage low-hanging fruit.

#### P300 deliverables

1. Split continuous blocks before producing one-second event windows.
2. Assert that no raw sample interval overlaps across train, validation, and
   test.
3. Fit normalization on training data only.
4. Plot target and non-target ERPs per subject and channel to verify that the
   expected physiological signal survives preprocessing.
5. Reproduce a canonical xDAWN/Riemannian baseline alongside EEGNet.
6. Maintain separate overlap-free within-subject and held-out-subject protocols.
7. Report AUROC, average precision, F1, class recall, and confusion matrices.

#### PhysioNet deliverables

1. Compare three-second canonical epochs against the current longer epochs.
2. Verify imagined-only run selection and T1/T2 label interpretation explicitly.
3. Generate manifests containing subjects, recordings, runs, labels, and trial
   intervals for every partition.
4. Assert no duplicate or overlapping examples across partitions.
5. Run shuffled-label, pre-cue-only, post-task-only, and run-ID-only controls.
6. Reserve an untouched test set and stop selecting the reported result on the
   same validation fold.

#### Exit criterion

A benchmark can be frozen when label shuffling reaches chance, obvious
non-neural controls fail, canonical baselines behave credibly, split manifests
show no leakage, and reported outcomes come from untouched subjects or sessions.

### Priority 2: Representation and transfer audit

Use existing checkpoints; do not pretrain new models.

Compare random, leak-fixed pretrained, and fine-tuned POYO representations with
controlled probes for:

- downstream task labels;
- subject identity;
- session or run identity;
- dataset identity;
- channel or montage properties where available.

Additionally:

- require strict checkpoint transfer;
- compare the best pretraining checkpoint with the final checkpoint;
- measure pretrained versus scratch training and validation trajectories;
- measure representation drift during fine-tuning when practical.

#### Decision gate

If pretraining greatly improves subject/session decoding while failing to
improve task decoding, nuisance entanglement is demonstrated and content-context
separation becomes well motivated. If not, prioritize the tokenizer or objective
before changing the representation structure.

### Priority 3: Coarse data-efficiency pilot

Use cleaned benchmarks and existing checkpoints. Do not begin with a large
label-fraction grid.

The first pilot should use three coarse budgets and vary the number of labeled
training subjects while retaining all trials for included subjects. This tests
whether pretraining can replace population diversity in unseen-subject transfer.
Randomly retaining a global fraction of windows is not sufficient because it
may expose the model to every subject and session.

At each budget compare:

- random frozen POYO plus a linear head;
- pretrained frozen POYO plus a linear head;
- scratch POYO trained end-to-end;
- pretrained POYO fine-tuned end-to-end;
- EEGNet trained on the identical labeled subset.

Use identical subsets across models and multiple subset seeds. Report the area
under the label-efficiency curve, the labels or subjects required to reach 90%
of full-data performance, and pretrained-minus-scratch performance at each
budget.

#### Decision gate

If pretraining does not help at coarse low-data budgets, postpone an exhaustive
sample-efficiency suite and address the representation/objective. If a clear
advantage emerges, expand both subject-count and within-subject trial-count
curves.

### Priority 4: Controlled tokenizer decision

Do not run another broad tokenizer sweep. The existing comparison is confounded:
the CWT-CNN temporal tokenizer has roughly 15 times the parameters of the current
ResampleCNN tokenizer, as well as different spectral priors.

The lowest-cost discriminating experiment is to add a parameter-matched
raw/resample CNN while keeping fixed:

- backbone and latent structure;
- output token rate;
- pretraining data, schedule, and optimizer steps;
- masking and leak fixes;
- channel-embedding mode;
- downstream protocols.

Before training, calculate the effective temporal support and boundary fraction
of each CWT band. With 2.5 cycles, a 0.5 Hz wavelet extends far beyond a one- or
two-second window. Unsupported low-frequency bands should be removed or tested
only with suitably long context.

#### Decision gate

- If capacity-matched raw convolution matches CWT, prefer the simpler tokenizer.
- If CWT retains a stable advantage on cleaned benchmarks and across seeds, keep
  it as the working tokenizer without claiming universal optimality.
- If results remain task-dependent, retain two tokenizer families and make
  tokenizer choice an explicit part of the model interface.

### Priority 5: Objective comparison

After the benchmark and tokenizer are stable, compare raw waveform MSE with one
genuinely different objective rather than another masking hyperparameter.
Promising families include latent prediction, teacher-student consistency, or a
hybrid objective that does not make high-variance background waveform error the
sole training signal.

The comparison should be judged by:

- frozen task probes;
- fine-tuning;
- label-efficiency curves;
- subject/session nuisance probes;
- robustness to measurement transformations;
- multiple pretraining seeds.

### Priority 6: Explicit content-context architecture

Keep this as a gated medium-term direction. A future architecture could expose:

- a time-resolved content stream encouraged to remain stable under
  measurement-preserving transformations;
- per-channel and global context streams preserving montage, quality, scale,
  subject, and recording information;
- downstream access to content alone or to content conditioned on context.

Reconstruction may use both streams, but context must be bandwidth- or
time-limited so that it cannot store the full waveform. Subject and session
decodability should be measured separately in each stream.

Start this work only after the representation audit shows that context
entanglement is a material cause of transfer failure. Before a full dual-stream
redesign, test cheaper interventions such as transformation consistency and
retaining pre-normalization statistics.

## Deferred Adaptation Roadmap

Two distinct adaptation problems should remain visible but are not immediate
priorities.

### Known task with unlabeled target-subject context

A task-specific head and possibly backbone receive minutes or hours of unlabeled
signal from a new subject. This could range from context-vector inference to
self-supervised test-time adaptation. Offline transductive adaptation and causal
online adaptation must be reported separately.

### New task and new subject with a few labeled examples

Initially evaluate ordinary prototype classifiers, linear heads, adapters, and
fine-tuning. True gradient-free in-context learning would later require
episodic training across many subjects and sufficiently diverse tasks; the
current small task suite is unlikely to support a strong new-task ICL claim.

For now, sample efficiency should remain within conventional linear-probe and
fine-tuning protocols.

## Immediate Work Package

The next concrete sprint should contain only:

1. P300 split repair, overlap assertions, ERP sanity plots, and a canonical
   baseline.
2. PhysioNet split/run audit, three-second-window comparison, and negative
   controls.
3. Subject/session/task probes on one existing leak-fixed checkpoint.
4. Analysis of all already-running multi-length and NeuroSoft jobs, with their
   confounds stated explicitly.

This work requires almost no new pretraining and can invalidate or support
several much larger research branches.

## Operating Principles for Future Experiments

1. **Freeze benchmarks before comparing representations.** A moving or leaky
   task cannot select a tokenizer or objective.
2. **Use matched controls.** Match architecture, parameter count, token rate,
   data exposure, optimizer steps, and downstream subsets wherever possible.
3. **Use multiple seeds when differences are small.** One pretraining seed is
   insufficient when effects are below downstream fold variance.
4. **Separate validation from reporting.** Hyperparameter selection, early
   stopping, and final testing should not use the same subjects.
5. **Prefer decision experiments over sweeps.** Each experiment should
   distinguish between specific competing explanations.
6. **Measure nuisance information directly.** Do not infer content-context
   entanglement only from downstream failure.
7. **Judge utility in absolute terms.** Frozen features are informative, but a
   useful foundation model must also approach credible task-specific baselines
   after reasonable adaptation.
8. **Keep the user-facing contract simple.** Signal, time, and masks should be
   sufficient; richer metadata and adaptation should be optional enhancements.

For production launches, follow the repository launch policy: start from a
clean committed repository, set
`FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches`, use the
`long` Slurm partition unless deliberately overridden, and record both the job
ID and immutable snapshot bundle path.
