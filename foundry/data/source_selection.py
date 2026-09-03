"""Deterministic reconstruction of compact NeuroSoft source selections."""

from __future__ import annotations

import hashlib

import numpy as np

SOURCE_SELECTION_IMPLEMENTATION = "neurosoft-classwise-permutation-v1"


def select_class_indices(
    available_indices: list[int],
    *,
    canonical_recording_id: str,
    class_id: int,
    seed: int,
    count: int,
) -> list[int]:
    """Select a deterministic prefix of a class-specific live index stream."""
    if count < 0 or count > len(available_indices):
        raise ValueError(
            f"Requested {count} examples from {len(available_indices)} available "
            f"for {canonical_recording_id!r} class {class_id}"
        )
    if count == 0:
        return []
    if count == len(available_indices):
        return sorted(available_indices)

    recording_digest = hashlib.sha256(
        canonical_recording_id.encode("utf-8")
    ).digest()
    recording_words = np.frombuffer(recording_digest[:16], dtype="<u4")
    generator = np.random.default_rng(
        np.random.SeedSequence(
            [seed, class_id, *(int(word) for word in recording_words)]
        )
    )
    permutation = generator.permutation(len(available_indices))
    return sorted(
        available_indices[position] for position in permutation[:count]
    )
