#!/usr/bin/env bash
#
# Launch a W&B sweep for per-channel-per-timepoint tokenizer LR/WD tuning.
# Starts 8 parallel agents, one per GPU, sharing a single sweep ID.
#
# Usage:
#   # Create a new sweep and launch agents:
#   bash scripts/launch_tokenizer_wandb_sweep.sh
#
#   # Resume an existing sweep:
#   bash scripts/launch_tokenizer_wandb_sweep.sh <sweep-id>

set -euo pipefail

SWEEP_CONFIG="configs/sweeps/tokenizer_lr_wd_bayes.yaml"
NUM_GPUS="${NUM_GPUS:-8}"
COUNT_PER_AGENT="${COUNT_PER_AGENT:-16}"
PROJECT="${WANDB_PROJECT:-foundry}"

cd "$(dirname "$0")/.."

normalize_sweep_id() {
    local raw_id="$1"

    # Accept full W&B sweep URLs and normalize to entity/project/sweep_id.
    raw_id="${raw_id#https://wandb.ai/}"
    raw_id="${raw_id#http://wandb.ai/}"
    raw_id="${raw_id#wandb.ai/}"
    raw_id="${raw_id//\/sweeps\//\/}"

    # Strip hidden control characters and surrounding whitespace.
    raw_id="$(printf "%s" "$raw_id" | tr -d '\r\n' | xargs)"
    printf "%s" "$raw_id"
}

is_valid_sweep_id() {
    local candidate="$1"
    [[ "$candidate" =~ ^[^/[:space:]]+$ ]] \
        || [[ "$candidate" =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]] \
        || [[ "$candidate" =~ ^[^/[:space:]]+/[^/[:space:]]+/[^/[:space:]]+$ ]]
}

if [[ $# -ge 1 ]]; then
    SWEEP_ID="$(normalize_sweep_id "$1")"
    echo "Resuming existing sweep: $SWEEP_ID"
else
    echo "Creating new W&B sweep from $SWEEP_CONFIG ..."
    SWEEP_OUTPUT="$(wandb sweep --project "$PROJECT" "$SWEEP_CONFIG" 2>&1 | tee /dev/stderr)"
    SWEEP_ID="$(
        printf "%s\n" "$SWEEP_OUTPUT" \
            | sed -nE 's@.*wandb agent[[:space:]]+([^[:space:]]+).*@\1@p' \
            | tail -n1
    )"
    SWEEP_ID="$(normalize_sweep_id "$SWEEP_ID")"
    if [[ -z "$SWEEP_ID" ]]; then
        echo "ERROR: Failed to parse sweep ID from wandb output."
        exit 1
    fi
    echo "Created sweep: $SWEEP_ID"
fi

if ! is_valid_sweep_id "$SWEEP_ID"; then
    echo "ERROR: Invalid sweep ID format: '$SWEEP_ID'"
    echo "Expected one of: sweep, project/sweep, or entity/project/sweep"
    exit 1
fi

PIDS=()

cleanup() {
    echo ""
    echo "Shutting down agents..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait
    echo "All agents stopped."
}
trap cleanup EXIT INT TERM

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Starting agent on GPU $gpu_id (up to $COUNT_PER_AGENT runs) ..."
    CUDA_VISIBLE_DEVICES="$gpu_id" \
        wandb agent --count "$COUNT_PER_AGENT" "$SWEEP_ID" &
    PIDS+=($!)
done

echo ""
echo "All $NUM_GPUS agents launched. Total budget: $((NUM_GPUS * COUNT_PER_AGENT)) runs."
echo "Sweep dashboard: https://wandb.ai/$SWEEP_ID"
echo "Press Ctrl+C to stop all agents."

wait
