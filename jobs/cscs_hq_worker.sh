#!/bin/bash
# Start one HyperQueue worker pinned to a single physical GPU.
#
# Must run *inside* srun: Slurm otherwise resets CUDA_VISIBLE_DEVICES to
# 0,1,2,3, HQ advertises four GPUs, and packed tasks get a device they
# cannot actually use (RuntimeError: No CUDA GPUs are available).
#
# Args: <gpu_id> <comma-separated-cores> <slots> [mps_prefix]

set -euo pipefail

gpu_id="${1:?gpu id required}"
core_list="${2:?core list required}"
slots="${3:?slot count required}"
mps_prefix="${4:-}"

HQ_DIR="${HQ_DIR:-${HOME}/.local/aarch64/bin}"
export PATH="${HQ_DIR}:${PATH}"
if ! command -v hq &>/dev/null; then
    echo "ERROR: hq not found in PATH (looked in ${HQ_DIR})" >&2
    exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Alps sets ROCR_VISIBLE_DEVICES even on NVIDIA nodes; HQ would then
# advertise gpus/amd=4 and confuse scheduling.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

if [[ -n "${mps_prefix}" ]]; then
    export CUDA_MPS_PIPE_DIRECTORY="${mps_prefix}-mps-${gpu_id}"
    export CUDA_MPS_LOG_DIRECTORY="${mps_prefix}-log-${gpu_id}"
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
    export CUDA_DEVICE_MAX_COPY_CONNECTIONS="${CUDA_DEVICE_MAX_COPY_CONNECTIONS:-8}"
    # Per-GPU MPS daemons are started with CUDA_VISIBLE_DEVICES=$gpu_id, so
    # they expose that physical GPU as logical device 0. Clients that keep
    # CUDA_VISIBLE_DEVICES=1/2/3 see an empty device list and fail with
    # "No CUDA GPUs are available". GPU 0 only worked by coincidence.
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
fi

echo "$(date): HQ worker gpu=${gpu_id} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} slots=${slots} MPS=${CUDA_MPS_PIPE_DIRECTORY:-off}"

# Do not advertise gpus/nvidia — HQ would rewrite CUDA_VISIBLE_DEVICES on
# each task. Pinning stays in this process environment and is inherited.
exec hq worker start \
    --detect-resources=none \
    --cpus="[[${core_list}]]" \
    --resource "slots=sum(${slots})"
