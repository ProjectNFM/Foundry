#!/bin/bash

# Individual task script for HyperQueue
# This script runs a single wandb agent
# HQ_TASK_ID is set by HyperQueue for each task

SWEEP_ID=${1:-"user/project/abcd1234"}
EXPERIMENT=${2:-"auditory_decoding/experiment1234"}

PROJECT_DIR="${FOUNDRY_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
cd "${PROJECT_DIR}" || {
    echo "$(date): ERROR: failed to cd to ${PROJECT_DIR}"
    exit 1
}

# Load WandB credentials if present
if [[ -f .env ]]; then
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
fi

# Per-agent log file (HyperQueue sets HQ_TASK_ID for each array task)
LOG_DIR="${LOG_DIR:-/capstor/scratch/cscs/${USER}/wandb_logs}"
if [[ -n "${HQ_TASK_ID:-}" ]]; then
    AGENT_LOG="${LOG_DIR}/agent_${SLURM_JOB_ID:-local}_${HQ_TASK_ID}.log"
    mkdir -p "${LOG_DIR}"
    exec >> "${AGENT_LOG}" 2>&1
    echo "$(date): Logging to ${AGENT_LOG}"
fi

echo "$(date): Starting wandb agent task ${HQ_TASK_ID} on $(hostname)"
echo "  Project: ${PROJECT_DIR}"
echo "  Sweep ID: ${SWEEP_ID}"
echo "  Experiment: ${EXPERIMENT}"

# Run the wandb agent (wandb >=0.24 no longer supports: wandb agent ID -- python ...)
export WANDB_SWEEP_EXPERIMENT="${EXPERIMENT}"
uv run python -m foundry.wandb_sweep_agent_worker "${SWEEP_ID}"

AGENT_EXIT_CODE=$?
echo "$(date): Completed wandb agent task ${HQ_TASK_ID} with exit code ${AGENT_EXIT_CODE}"

exit ${AGENT_EXIT_CODE}