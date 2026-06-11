"""Verify legacy modality-registry constructs are fully removed (issue 09)."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_LEGACY_PATTERNS = [
    "ModalitySpec",
    "register_modality",
    "MultitaskReadout",
    "prepare_for_multitask_readout",
    "MappedCrossEntropyLoss",
    "resolve_readout_specs",
]

_EXCLUDE_PREFIXES = ("docs/",)


def _git_grep(pattern: str) -> list[str]:
    result = subprocess.run(
        ["git", "grep", "-n", pattern],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 1:
        return []
    lines = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        path = line.split(":", 1)[0]
        if any(path.startswith(prefix) for prefix in _EXCLUDE_PREFIXES):
            continue
        lines.append(line)
    return lines


def test_modalities_module_deleted():
    assert not (REPO_ROOT / "foundry/data/datasets/modalities.py").exists()
