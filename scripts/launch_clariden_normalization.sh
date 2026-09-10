#!/usr/bin/env bash
set -euo pipefail

: "${CSCS_ACCOUNT:?Set CSCS_ACCOUNT before launching}"
: "${FOUNDRY_CLARIDEN_EDF:?Set FOUNDRY_CLARIDEN_EDF to an absolute EDF path}"
: "${FOUNDRY_ENV_FILE:?Set FOUNDRY_ENV_FILE to the application env file}"
: "${FOUNDRY_SNAPSHOT_ROOT:?Set FOUNDRY_SNAPSHOT_ROOT to shared Clariden storage}"

if [[ -n "$(git status --short)" ]]; then
  echo "Refusing to launch: commit all repository changes first." >&2
  exit 1
fi

configs=(
  auditory_decoding/eegnet_neurosoft_global_zscore_minipigs
  auditory_decoding/neurosoft_conv_bigru_global_zscore_minipigs
  auditory_decoding/eegnet_neurosoft_global_zscore_monkeys
  auditory_decoding/neurosoft_conv_bigru_global_zscore_monkeys
)

for experiment in "${configs[@]}"; do
  python main.py \
    "experiment=${experiment}" \
    hydra/launcher=slurm_clariden \
    -m
done
