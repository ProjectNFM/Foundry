#!/bin/bash

# HyperQueue-based sweep script for wandb agents
# This script uses HyperQueue as a meta-scheduler to efficiently run multiple wandb agents
#
# Usage:
#   sbatch cscs_sweep_hyperqueue.sh <num_agents> [sweep_id] [experiment]
#
# Alps GH200 nodes are exclusive (4 GPUs × 72 cores × ~120 GB HBM). Small
# models (GRU/EEGNet) often use ~15% of one GPU, so pack several wandb
# agents per GPU. num_agents must be a multiple of the allocated GPU count.
#
# Example (1 agent per GPU, original behaviour):
#   sbatch --exclusive --mem=450G --cpus-per-task=288 --gpus-per-node=4 \
#     jobs/cscs_sweep_hyperqueue.sh 4 user/project/sweep123 auditory_decoding/my_sweep
#
# Example (4 agents per GPU — recommended for GRU/EEGNet at ~15% HBM):
#   sbatch --exclusive --mem=450G --cpus-per-task=288 --gpus-per-node=4 \
#     jobs/cscs_sweep_hyperqueue.sh 16 user/project/sweep123 auditory_decoding/my_sweep

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

TASK_SCRIPT="${PROJECT_DIR}/jobs/cscs_task_wandb_agent.sh"
if [[ ! -f "${TASK_SCRIPT}" ]]; then
    echo "[$(date)] ERROR: task script not found at ${TASK_SCRIPT}"
    echo "Submit from the Foundry repo root, or set FOUNDRY_ROOT."
    exit 1
fi
echo "Project directory: ${PROJECT_DIR}"

# Ensure logs directory exists
mkdir -p "${LOG_DIR}"
export LOG_DIR

# Each wandb agent is long-running (pulls trials until the sweep ends). Start
# one HyperQueue worker per physical GPU and pack several agents onto that
# worker when NUM_AGENTS > GPU count (fractional gpus/nvidia request).
GPUS_AVAILABLE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}"
TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-288}"

if (( NUM_AGENTS < GPUS_AVAILABLE )); then
    echo "[$(date)] ERROR: ${NUM_AGENTS} agents for ${GPUS_AVAILABLE} GPU(s)."
    echo "Request at least one agent per GPU, e.g.:"
    echo "  sbatch --gpus-per-node=${NUM_AGENTS} ${0} ${NUM_AGENTS} ..."
    exit 1
fi
if (( NUM_AGENTS % GPUS_AVAILABLE != 0 )); then
    echo "[$(date)] ERROR: num_agents (${NUM_AGENTS}) must be a multiple of GPU count (${GPUS_AVAILABLE})."
    exit 1
fi
if (( TOTAL_CPUS % GPUS_AVAILABLE != 0 )); then
    echo "[$(date)] ERROR: ${TOTAL_CPUS} CPUs do not divide evenly across ${GPUS_AVAILABLE} GPU(s)."
    exit 1
fi

AGENTS_PER_GPU=$((NUM_AGENTS / GPUS_AVAILABLE))
CORES_PER_GPU=$((TOTAL_CPUS / GPUS_AVAILABLE))
if (( CORES_PER_GPU % AGENTS_PER_GPU != 0 )); then
    echo "[$(date)] ERROR: ${CORES_PER_GPU} cores/GPU do not divide evenly across ${AGENTS_PER_GPU} agents/GPU."
    echo "Pick an agents-per-GPU that divides 72 (e.g. 1, 2, 3, 4, 6, 8)."
    exit 1
fi
CPUS_PER_AGENT=$((CORES_PER_GPU / AGENTS_PER_GPU))

echo "GPUs allocated: ${GPUS_AVAILABLE}"
echo "Agents: ${NUM_AGENTS} (${AGENTS_PER_GPU} per GPU)"
echo "CPUs per GPU / agent: ${CORES_PER_GPU} / ${CPUS_PER_AGENT}"
echo "HQ slot request per agent: 1 (${AGENTS_PER_GPU} slots/GPU)"

MPS_PREFIX="/tmp/${USER}/slurm-${SLURM_JOBID:-local}/nvidia"
stop_mps() {
    if [[ "${AGENTS_PER_GPU}" -le 1 ]]; then
        return 0
    fi
    echo "[$(date)] Stopping MPS daemons..."
    for gpu_id in $(seq 0 $((GPUS_AVAILABLE - 1))); do
        echo quit | CUDA_MPS_PIPE_DIRECTORY="${MPS_PREFIX}-mps-${gpu_id}" \
            CUDA_MPS_LOG_DIRECTORY="${MPS_PREFIX}-log-${gpu_id}" \
            nvidia-cuda-mps-control 2>/dev/null || true
    done
}

# NVIDIA MPS lets several small training processes share one GH200 GPU
# instead of time-slicing. One daemon per GPU, as recommended by CSCS.
start_mps() {
    if [[ "${AGENTS_PER_GPU}" -le 1 ]]; then
        echo "[$(date)] One agent per GPU; skipping MPS"
        return 0
    fi
    if ! command -v nvidia-cuda-mps-control &>/dev/null; then
        echo "[$(date)] WARNING: nvidia-cuda-mps-control not found; packing without MPS"
        return 0
    fi

    echo "[$(date)] Starting one MPS daemon per GPU..."
    export CUDA_DEVICE_MAX_CONNECTIONS=8
    export CUDA_DEVICE_MAX_COPY_CONNECTIONS=8
    pkill --uid "$(id -un)" '^nvidia-cuda-mps-' 2>/dev/null || true

    for gpu_id in $(seq 0 $((GPUS_AVAILABLE - 1))); do
        mkdir -p "${MPS_PREFIX}-mps-${gpu_id}" "${MPS_PREFIX}-log-${gpu_id}"
        CUDA_MPS_PIPE_DIRECTORY="${MPS_PREFIX}-mps-${gpu_id}" \
        CUDA_MPS_LOG_DIRECTORY="${MPS_PREFIX}-log-${gpu_id}" \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
            nvidia-cuda-mps-control -d
    done

    for gpu_id in $(seq 0 $((GPUS_AVAILABLE - 1))); do
        pid_file="${MPS_PREFIX}-mps-${gpu_id}/nvidia-cuda-mps-control.pid"
        if ! timeout 60 bash -c "until [[ -f \"${pid_file}\" ]]; do sleep 1; done"; then
            echo "[$(date)] ERROR: MPS daemon for GPU ${gpu_id} did not start"
            stop_mps
            exit 1
        fi
    done
    echo "[$(date)] MPS daemons ready"
}

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

start_mps
trap stop_mps EXIT

# One worker per physical GPU. Do not pass --gpus-per-task: srun would
# reset CUDA_VISIBLE_DEVICES to 0,1,2,3 and HQ would advertise 4 GPUs per
# worker. The wrapper pins a single GPU *after* srun starts. Packing uses
# a logical `slots` resource so HQ does not rewrite CUDA_VISIBLE_DEVICES.
WORKER_SCRIPT="${PROJECT_DIR}/jobs/cscs_hq_worker.sh"
if [[ ! -f "${WORKER_SCRIPT}" ]]; then
    echo "[$(date)] ERROR: worker script not found at ${WORKER_SCRIPT}"
    hq server stop
    exit 1
fi
chmod +x "${WORKER_SCRIPT}"

echo "[$(date)] Starting ${GPUS_AVAILABLE} HyperQueue workers (one per GPU)..."
WORKER_PIDS=()
MPS_ARG=""
if [[ "${AGENTS_PER_GPU}" -gt 1 ]]; then
    MPS_ARG="${MPS_PREFIX}"
fi
for gpu_id in $(seq 0 $((GPUS_AVAILABLE - 1))); do
    core_start=$((gpu_id * CORES_PER_GPU))
    core_end=$((core_start + CORES_PER_GPU - 1))
    core_list="$(seq -s, "${core_start}" "${core_end}")"
    srun --overlap --ntasks=1 \
        "${WORKER_SCRIPT}" \
        "${gpu_id}" \
        "${core_list}" \
        "${AGENTS_PER_GPU}" \
        "${MPS_ARG}" &
    WORKER_PIDS+=($!)
done

# Wait until workers register; do not submit tasks into an empty cluster.
WORKERS_READY=0
for _ in $(seq 1 12); do
    sleep 5
    if hq worker list | grep -q RUNNING; then
        WORKERS_READY=1
        break
    fi
done
hq worker list
if [[ "${WORKERS_READY}" -ne 1 ]]; then
    echo "[$(date)] ERROR: HyperQueue workers failed to start; not submitting agents."
    echo "Check srun errors above. Typical cause: worker command not found or GPU bind failed."
    hq server stop
    exit 1
fi

# Submit wandb agent tasks
echo "[$(date)] Submitting ${NUM_AGENTS} wandb agent tasks..."
hq submit \
    --cpus="${CPUS_PER_AGENT}" \
    --resource "slots=1" \
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
rm -rf "${MPS_PREFIX}-mps-"* "${MPS_PREFIX}-log-"* 2>/dev/null || true

echo ""
echo "========================================"
echo "HyperQueue WandB Sweep Completed"
echo "========================================"
echo "End time: $(date)"
echo "Exit code: ${WAIT_EXIT_CODE}"
echo "========================================"

exit ${WAIT_EXIT_CODE}
