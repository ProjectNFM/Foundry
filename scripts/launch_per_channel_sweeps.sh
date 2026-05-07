#!/usr/bin/env bash
# Launch the three per-channel tokenizer sweeps sequentially.
#
# Each sweep runs up to 200 trials (enforced by run_cap in the sweep
# config) and uses the EffectiveBatchSizeCallback to auto-tune batch
# sizes per GPU.
#
# Usage:
#   bash scripts/launch_per_channel_sweeps.sh              # auto-detect GPUs
#   bash scripts/launch_per_channel_sweeps.sh 0,1,2,3      # specify GPUs
set -euo pipefail

GPUS="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SWEEP_CONFIGS=(
    configs/sweep/per_channel_cwt_sweep.yaml
    configs/sweep/per_channel_cwt_compressor_sweep.yaml
    configs/sweep/per_channel_resample_cnn_sweep.yaml
)

GPU_ARGS=()
if [[ -n "$GPUS" ]]; then
    GPU_ARGS=(--gpus "$GPUS")
fi

cd "$PROJECT_ROOT"

for cfg in "${SWEEP_CONFIGS[@]}"; do
    echo ""
    echo "================================================================"
    echo "  Starting sweep: $cfg"
    echo "  $(date)"
    echo "================================================================"
    echo ""

    uv run python scripts/wandb_sweep.py "$cfg" "${GPU_ARGS[@]}"

    echo ""
    echo "  Finished: $cfg  ($(date))"
    echo "================================================================"
done

echo ""
echo "All three per-channel sweeps complete."
