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
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-<unset>}"
echo "  HQ_RESOURCE_VALUES_gpus_nvidia=${HQ_RESOURCE_VALUES_gpus_nvidia:-<unset>}"
echo "  HQ_RESOURCE_VALUES_slots=${HQ_RESOURCE_VALUES_slots:-<unset>}"

# Per-GPU MPS exposes the physical GPU as logical device 0. Keep the pipe
# (which selects the GPU) and force visible devices to 0.
if [[ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "  MPS remap: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# Run the wandb agent (wandb >=0.24 no longer supports: wandb agent ID -- python ...)
# --no-sync: 16 packed agents must not race uv lock/venv installs.
export WANDB_SWEEP_EXPERIMENT="${EXPERIMENT}"
uv run --no-sync python -m foundry.tools.wandb_sweep_agent_worker "${SWEEP_ID}"

AGENT_EXIT_CODE=$?
echo "$(date): Completed wandb agent task ${HQ_TASK_ID} with exit code ${AGENT_EXIT_CODE}"

exit ${AGENT_EXIT_CODE}