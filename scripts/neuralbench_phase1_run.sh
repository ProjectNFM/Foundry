#!/bin/bash
# Phase 1: Run NeuralBench reference EEGNet and Foundry EEGNet for MI and Sleep Stage.
#
# Prerequisites:
#   - Data downloaded and prepared: neuralbench eeg motor_imagery --dataset schalk2004bci2000 --download -p
#   - Data downloaded and prepared: neuralbench eeg sleep_stage --download -p
#   - Label verification: uv run python scripts/neuralbench_phase1_verify.py
#
# Usage:
#   bash scripts/neuralbench_phase1_run.sh [mi|sleep|nb_mi|nb_sleep|all]

set -euo pipefail
cd "$(dirname "$0")/.."

# Source environment if available
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

export RANK=0

case "${1:-all}" in
    mi)
        echo "=== Foundry EEGNet — Motor Imagery ==="
        time uv run python main.py experiment=neuralbench/mi_eegnet_comparison \
            hydra/launcher=local_gpu
        ;;
    sleep)
        echo "=== Foundry EEGNet — Sleep Stage ==="
        time uv run python main.py experiment=neuralbench/sleep_stage_eegnet_comparison \
            hydra/launcher=local_gpu
        ;;
    nb_mi)
        echo "=== NeuralBench Reference EEGNet — Motor Imagery ==="
        time uv run neuralbench --grid --force \
            --model eegnet --dataset schalk2004bci2000 eeg motor_imagery
        ;;
    nb_sleep)
        echo "=== NeuralBench Reference EEGNet — Sleep Stage ==="
        time uv run neuralbench --grid --force \
            --model eegnet eeg sleep_stage
        ;;
    all)
        echo "=== Running all Phase 1 comparisons ==="
        echo ""
        echo "--- NeuralBench Reference EEGNet — Motor Imagery ---"
        time uv run neuralbench --grid --force \
            --model eegnet --dataset schalk2004bci2000 eeg motor_imagery
        echo ""
        echo "--- NeuralBench Reference EEGNet — Sleep Stage ---"
        time uv run neuralbench --grid --force \
            --model eegnet eeg sleep_stage
        echo ""
        echo "--- Foundry EEGNet — Motor Imagery ---"
        time uv run python main.py experiment=neuralbench/mi_eegnet_comparison \
            hydra/launcher=local_gpu
        echo ""
        echo "--- Foundry EEGNet — Sleep Stage ---"
        time uv run python main.py experiment=neuralbench/sleep_stage_eegnet_comparison \
            hydra/launcher=local_gpu
        ;;
    *)
        echo "Usage: $0 [mi|sleep|nb_mi|nb_sleep|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Done ==="
