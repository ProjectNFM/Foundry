#!/bin/bash

# HyperQueue-based sweep script for wandb agents
# This script uses HyperQueue as a meta-scheduler to efficiently run multiple wandb agents
#
# Usage:
#   sbatch sweep_hyperqueue.sh <num_agents> [sweep_id] [experiment]
#
# Example:
#   sbatch --gpus-per-node=4 sweep_hyperqueue.sh 4
#   sbatch --gpus-per-node=8 sweep_hyperqueue.sh 8 user/project/sweep123 auditory_decoding/my_sweep

#SBATCH --account=a0091
#SBATCH --partition=normal
#SBATCH --job-name=hq_wandb_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=450G  # full GH200 node (~4×72 cores)
#SBATCH --cpus-per-task=288   # full GH200 node (~4×72 cores)
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/wandb_logs/wandb_sweep_hq_%j.log

# Configuration
LOG_DIR="/capstor/scratch/cscs/${USER}/wandb_logs"
NUM_AGENTS=${1:-4}
SWEEP_ID=${2:-"user/project/abcd1234"}
EXPERIMENT=${3:-"auditory_decoding/experiment1234"}

# HyperQueue: SLURM batch jobs use a minimal PATH, so add the CSCS
# aarch64 install location explicitly (see docs.cscs.ch/running/hyperqueue).
HQ_DIR="${HQ_DIR:-${HOME}/.local/aarch64/bin}"
export PATH="${HQ_DIR}:${PATH}"
if ! command -v hq &>/dev/null; then
    echo "[$(date)] ERROR: hq not found in PATH (looked in ${HQ_DIR})"
    echo "Install: https://github.com/It4innovations/hyperqueue/releases"
    echo "  wget .../hq-*-linux-arm64-linux.tar.gz"
    echo "  mkdir -p ${HQ_DIR} && tar -xvzf hq-*.tar.gz -C ${HQ_DIR}"
    exit 1
fi

# Set up unique directories for this job
export HQ_SERVER_DIR=~/.hq-server-${SLURM_JOBID}
export JOURNAL=~/.hq-journal-${SLURM_JOBID}

echo "========================================"
echo "Starting HyperQueue WandB Sweep"
echo "========================================"
echo "Job ID: ${SLURM_JOBID}"
echo "Number of agents: ${NUM_AGENTS}"
echo "Sweep ID: ${SWEEP_ID}"
echo "Experiment: ${EXPERIMENT}"
echo "Start time: $(date)"
echo "========================================"

# Navigate to the project directory.
# SLURM copies the batch script to /var/spool/slurmd/...; use SLURM_SUBMIT_DIR
# (the directory from which sbatch was invoked) instead of $0/BASH_SOURCE.
PROJECT_DIR="${FOUNDRY_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
cd "${PROJECT_DIR}" || exit 1
export FOUNDRY_ROOT="${PROJECT_DIR}"

TASK_SCRIPT="${PROJECT_DIR}/task_wandb_agent.sh"
if [[ ! -f "${TASK_SCRIPT}" ]]; then
    echo "[$(date)] ERROR: task script not found at ${TASK_SCRIPT}"
    echo "Submit from the Foundry repo root, or set FOUNDRY_ROOT."
    exit 1
fi
echo "Project directory: ${PROJECT_DIR}"

# Ensure logs directory exists
mkdir -p "${LOG_DIR}"
export LOG_DIR

# Each wandb agent is long-running (pulls trials until the sweep ends). We need
# one GPU + one HyperQueue worker per agent so they run in parallel. With a
# single worker, only agent 1 ever runs.
GPUS_AVAILABLE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}"
if [[ "${NUM_AGENTS}" -gt "${GPUS_AVAILABLE}" ]]; then
    echo "[$(date)] ERROR: ${NUM_AGENTS} agents requested but only ${GPUS_AVAILABLE} GPU(s) allocated."
    echo "Submit with matching GPUs, e.g.:"
    echo "  sbatch --gpus-per-node=${NUM_AGENTS} sweep_hyperqueue.sh ${NUM_AGENTS} ..."
    exit 1
fi
echo "GPUs allocated: ${GPUS_AVAILABLE}, agents: ${NUM_AGENTS}"

# Start HyperQueue server
echo "[$(date)] Starting HyperQueue server..."
hq server start --journal="${JOURNAL}" &
SERVER_PID=$!

# Wait for the server to be ready
echo "[$(date)] Waiting for HyperQueue server to be ready..."
if ! hq server wait --timeout=120; then
    echo "[$(date)] ERROR: HyperQueue server failed to start"
    kill ${SERVER_PID} 2>/dev/null || true
    exit 1
fi

echo "[$(date)] HyperQueue server started successfully (PID: ${SERVER_PID})"

# Start one HyperQueue worker per GPU so agents can run in parallel
echo "[$(date)] Starting ${NUM_AGENTS} HyperQueue workers (one per GPU)..."
WORKER_PIDS=()
for worker_id in $(seq 1 "${NUM_AGENTS}"); do
    gpu_id=$((worker_id - 1))
    CUDA_VISIBLE_DEVICES="${gpu_id}" srun --overlap --ntasks=1 --gpus-per-task=1 \
        hq worker start &
    WORKER_PIDS+=($!)
done

# Give workers time to connect
sleep 5
if ! hq worker list | grep -q RUNNING; then
    echo "[$(date)] WARNING: No RUNNING workers detected; check hq worker list"
fi
hq worker list

# Submit wandb agent tasks
echo "[$(date)] Submitting ${NUM_AGENTS} wandb agent tasks..."
hq submit \
    --resource "gpus/nvidia=1" \
    --array "1-${NUM_AGENTS}" \
    "${TASK_SCRIPT}" \
    "${SWEEP_ID}" \
    "${EXPERIMENT}"

if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: Failed to submit tasks to HyperQueue"
    hq server stop
    exit 1
fi

echo "[$(date)] Successfully submitted ${NUM_AGENTS} tasks"

# Wait for all tasks to complete
echo "[$(date)] Waiting for all wandb agents to complete..."
hq job wait all

WAIT_EXIT_CODE=$?
echo "[$(date)] Job wait completed with exit code: ${WAIT_EXIT_CODE}"

# Shutdown HyperQueue
echo "[$(date)] Stopping HyperQueue server..."
hq server stop

# Clean up
echo "[$(date)] Cleaning up temporary files..."
rm -rf "${HQ_SERVER_DIR}" 2>/dev/null || true
rm -f "${JOURNAL}" 2>/dev/null || true

echo ""
echo "========================================"
echo "HyperQueue WandB Sweep Completed"
echo "========================================"
echo "End time: $(date)"
echo "Exit code: ${WAIT_EXIT_CODE}"
echo "========================================"

exit ${WAIT_EXIT_CODE}