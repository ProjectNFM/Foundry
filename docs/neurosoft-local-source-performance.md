# Local NeuroSoft source-pretraining performance

Source-pretraining normalization is fitted from the effective selected train
intervals only. Metadata discovery still constructs and validates the dataset,
source selection, parent pool, audit binding, task mappings, and leakage policy,
but explicitly defers normalization to the final datamodule.

The formula and float64 sum/squared-sum accumulation policy are unchanged.
Streaming and eager reference results are expected to agree after the existing
float32 artifact conversion within `rtol=1e-6`; regression fixtures enforce
that tolerance for both per-channel and recording-global statistics.

## Normalization cache

The minipig and monkey source-pretraining recipes enable a shared cache. Its
location is selected in this order:

1. `data.input_normalization.cache.directory`;
2. the `FOUNDRY_NORMALIZATION_CACHE` environment variable;
3. `<data.root>/.foundry_normalization_cache`.

Each entry is keyed by the effective train-interval hash, source selection
manifest hash, fraction-manifest hashes, normalization mode and numerical
policy, supported modalities, recording and channel metadata, and backing HDF5
path/size/mtime/ctime identity. In-memory fixtures use a content hash. Artifacts are
hash-verified on every load; mismatched, incomplete, or corrupt entries are
logged, rejected, and atomically recomputed under a per-key file lock. Every
run independently emits its immutable NPZ and JSON artifacts with cache status,
cache identity, train-interval hash, and source/fraction provenance.

To clear the cache safely, stop local runs using it and remove only the chosen
cache directory. Do not remove the processed-data directory. It is also safe to
point a run at a new empty directory:

```bash
export FOUNDRY_NORMALIZATION_CACHE=/path/on/shared-storage/foundry-normalization
```

Changing `trainer.max_steps` or the validation schedule does not change the
cache key. Changing a split, source/fraction manifest, normalization parameter,
channel layout, or backing data artifact does.

## RTX 8000 precision

The source recipes request `bf16-mixed` but explicitly opt into
`run.unsupported_bf16_fallback=16-mixed`. On a Turing/RTX 8000 GPU the resolver
logs and records requested BF16, effective FP16, GPU name, and compute
capability. On BF16-capable A100/Clariden hardware the request remains BF16.
Recipes without this opt-in fail with an actionable error on unsupported CUDA
hardware instead of silently changing precision. `trainer.precision=32-true`
remains available for correctness diagnostics.

For a controlled local comparison, use the benchmark config with the same
manifest, batch size, worker count, cache, and validation schedule:

```bash
python main.py \
  experiment=pretraining/neurosoft_conv_bigru_supervised_minipigs_rtx8000_benchmark \
  source_manifest=manifests/neurosoft_supervised/v1/source_volume/minipigs/target-sub-06/fraction-1.00/selection-42.json \
  run.seed=42 trainer.precision=16-mixed

python main.py \
  experiment=pretraining/neurosoft_conv_bigru_supervised_minipigs_rtx8000_benchmark \
  source_manifest=manifests/neurosoft_supervised/v1/source_volume/minipigs/target-sub-06/fraction-1.00/selection-42.json \
  run.seed=42 trainer.precision=32-true
```

The run writes `precision_benchmark.json` containing startup and training time,
post-warmup median and p95 step time, peak allocated/reserved GPU memory,
effective precision, GPU capability, and a finite-loss check. GPU utilization
can be sampled alongside the run with `nvidia-smi`; it is intentionally not
polled from every training step.
