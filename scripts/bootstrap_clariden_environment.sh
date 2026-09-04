#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/shared/path/to/venv" >&2
  exit 2
fi

foundry_target_venv=$1
if [[ ${foundry_target_venv} != /* ]]; then
  echo "The target venv path must be absolute." >&2
  exit 2
fi
if [[ $(uname -m) != aarch64 ]]; then
  echo "Run this script inside the Clariden ARM64 container." >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "The pinned container must provide uv." >&2
  exit 2
fi

uv venv --python "$(command -v python)" "${foundry_target_venv}"
VIRTUAL_ENV="${foundry_target_venv}" uv sync --active --frozen

"${foundry_target_venv}/bin/python" - <<'PY'
import platform
import sys

import foundry
import hydra
import torch

print(f"python={sys.version}")
print(f"executable={sys.executable}")
print(f"platform={platform.platform()}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"foundry={foundry.__file__}")
print(f"hydra={hydra.__file__}")
PY
