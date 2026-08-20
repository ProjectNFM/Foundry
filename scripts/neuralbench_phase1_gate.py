"""POC Phase 1 Gate: NeuralBench adapter forward-pass and fidelity proof.

Validates that:
1. NeuralBenchDataModule sets up correctly for p3/Korczowski2014A
2. Adapted samples produce valid torch_brain Data objects
3. Batches pass through both Foundry POYO-EEG and Foundry EEGNet
4. Pre-tokenization signal, label, channel order, and timing match
   NeuralSet's output within documented floating-point tolerance

Usage:
    uv run python scripts/neuralbench_phase1_gate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not passed:
        print("         *** GATE BLOCKED ***")


def print_summary() -> bool:
    n_pass = sum(1 for _, p, _ in CHECKS if p)
    n_fail = sum(1 for _, p, _ in CHECKS if not p)
    print(f"\n{'='*72}")
    print(f"Phase 1 Gate: {n_pass} passed, {n_fail} failed")
    if n_fail > 0:
        print("GATE: BLOCKED — fix failures above before proceeding to Phase 2")
        for name, passed, detail in CHECKS:
            if not passed:
                print(f"  - {name}: {detail}")
    else:
        print("GATE: PASSED")
    print(f"{'='*72}")
    return n_fail == 0


# ---------------------------------------------------------------------------
# Phase 1 Gate
# ---------------------------------------------------------------------------

def main() -> bool:
    print("=" * 72)
    print("NeuralBench POC Phase 1 — Adapter & Forward-Pass Gate")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. DataModule setup
    # ------------------------------------------------------------------
    print("\n[1/6] Setting up NeuralBenchDataModule...")
    t0 = time.time()

    from foundry.data.neuralbench import NeuralBenchDataModule

    dm = NeuralBenchDataModule(
        task="p3",
        dataset="korczowski2014a",
        cache_dir="/network/scratch/s/sobralm/neuralset-data",
        batch_size=4,
        num_workers=0,
    )
    dm.setup("fit")
    setup_time = time.time() - t0

    check(
        "DataModule.setup() completes",
        dm._train_adapter is not None and dm._val_adapter is not None,
        f"took {setup_time:.1f}s",
    )
    check(
        "Train adapter has expected sample count",
        len(dm._train_adapter) == 35270,
        f"got {len(dm._train_adapter)} (expected 35270)",
    )
    check(
        "Val adapter has expected sample count",
        len(dm._val_adapter) == 11628,
        f"got {len(dm._val_adapter)} (expected 11628)",
    )
    check(
        "Session IDs populated",
        len(dm.get_recording_ids()) == 64,
        f"got {len(dm.get_recording_ids())} sessions (expected 64)",
    )
    check(
        "Channel IDs populated",
        len(dm.get_channel_ids()) == 64 * 16,
        f"got {len(dm.get_channel_ids())} channel IDs (expected {64*16})",
    )

    # ------------------------------------------------------------------
    # 2. Adapter sample fidelity (pre-tokenization)
    # ------------------------------------------------------------------
    print("\n[2/6] Checking adapter sample fidelity...")

    from torch_brain.data import Interval, RegularTimeSeries

    sample_idx = 0
    raw_sample, raw_data = dm._train_adapter._get_sample_data(sample_idx)
    adapted = dm._train_adapter._to_torch_brain_data(sample_idx)

    # Signal shape and dtype
    check(
        "Adapted signal is RegularTimeSeries",
        isinstance(adapted.eeg, RegularTimeSeries),
    )
    sig = np.asarray(adapted.eeg.signal)
    check(
        "Signal shape is (120, 16)",
        sig.shape == (120, 16),
        f"got {sig.shape}",
    )
    check(
        "Signal dtype is float32",
        sig.dtype == np.float32,
        f"got {sig.dtype}",
    )
    check(
        "Sampling rate is 120 Hz",
        adapted.eeg.sampling_rate == 120.0,
        f"got {adapted.eeg.sampling_rate}",
    )

    # Signal value preservation
    raw_neuro = raw_data["neuro"]
    if isinstance(raw_neuro, torch.Tensor):
        raw_neuro = raw_neuro.numpy()
    raw_signal = raw_neuro.squeeze(0).T  # (T, C)
    max_diff = float(np.max(np.abs(sig - raw_signal)))
    check(
        "Signal values match NeuralSet within float32 tolerance",
        max_diff < 1e-6,
        f"max abs diff = {max_diff:.2e}",
    )

    # Domain / timing
    domain_start = float(adapted.domain.start[0])
    domain_end = float(adapted.domain.end[0])
    check(
        "Domain spans [0.0, 1.0]",
        abs(domain_start) < 1e-9 and abs(domain_end - 1.0) < 1e-9,
        f"got [{domain_start}, {domain_end}]",
    )

    # Label
    check(
        "Trial interval present",
        hasattr(adapted, "p300_trials")
        and isinstance(adapted.p300_trials, Interval),
    )
    trial = adapted.p300_trials
    label = np.asarray(trial.targets)[0]
    raw_target = raw_data["target"]
    if isinstance(raw_target, torch.Tensor):
        raw_target = raw_target.numpy()
    expected_label = "NonTarget" if np.argmax(raw_target.flatten()) == 0 else "Target"
    check(
        "Label matches NeuralSet",
        label == expected_label,
        f"got '{label}', expected '{expected_label}'",
    )

    # Channel IDs
    channel_ids = list(adapted.channels.id)
    check(
        "16 channel IDs present",
        len(channel_ids) == 16,
        f"got {len(channel_ids)}",
    )
    expected_order = [
        "Fp1", "Fp2", "F3", "AFz", "F4", "T7", "Cz", "T8",
        "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2",
    ]
    bare_names = [cid.rsplit("/", 1)[-1] for cid in channel_ids]
    check(
        "Channel order matches NeuralBench contract",
        bare_names == expected_order,
        f"got {bare_names}",
    )

    # Session ID
    session_id = str(adapted.session.id)
    check(
        "Session ID is non-empty and contains prefix",
        len(session_id) > 0 and session_id.startswith("nb/p3/"),
        f"got '{session_id}'",
    )

    # ------------------------------------------------------------------
    # 3. Multiple-sample consistency
    # ------------------------------------------------------------------
    print("\n[3/6] Multi-sample consistency...")

    labels_seen = set()
    sessions_seen = set()
    n_train = len(dm._train_adapter)
    probe_indices = [0, 1, 2, n_train // 4, n_train // 2, 3 * n_train // 4, n_train - 1]
    for i in probe_indices:
        d = dm._train_adapter._to_torch_brain_data(i)
        labels_seen.add(np.asarray(d.p300_trials.targets)[0])
        sessions_seen.add(str(d.session.id))

    check(
        "Both label classes observed across probed train samples",
        labels_seen == {"NonTarget", "Target"},
        f"seen: {labels_seen}",
    )
    check(
        "Multiple sessions seen across probed train samples",
        len(sessions_seen) > 1,
        f"seen {len(sessions_seen)} sessions",
    )

    # ------------------------------------------------------------------
    # 4. Task config and EEGNet forward pass
    # ------------------------------------------------------------------
    print("\n[4/6] EEGNet forward pass...")

    from foundry.tasks.config import TaskConfig

    tasks_dir = Path(__file__).resolve().parent.parent / "configs" / "tasks"
    tc = TaskConfig.from_yaml(tasks_dir / "p300_binary.yaml")
    task_configs = {tc.name: tc}

    from foundry.models.baselines import EEGNetEncoder

    num_channels = 16
    num_samples = 120  # 1.0s × 120 Hz
    eegnet = EEGNetEncoder(
        task_configs=task_configs,
        num_channels=num_channels,
        num_samples=num_samples,
        F1=8,
        D=2,
        F2=16,
        kernel_length=32,
        dropout_rate=0.5,
    )
    eegnet.eval()

    dm_eegnet = NeuralBenchDataModule(
        task="p3",
        dataset="korczowski2014a",
        cache_dir="/network/scratch/s/sobralm/neuralset-data",
        batch_size=4,
        num_workers=0,
    )
    dm_eegnet._task_configs = task_configs
    dm_eegnet.setup("fit")
    dm_eegnet.set_tokenizer(eegnet.tokenize)

    loader = dm_eegnet.train_dataloader()
    eegnet_batch = next(iter(loader))

    try:
        with torch.no_grad():
            eegnet_out = eegnet(**eegnet_batch)
        eegnet_ok = True
        eegnet_detail = f"output keys: {sorted(eegnet_out.keys())}"
    except Exception as e:
        eegnet_ok = False
        eegnet_detail = str(e)

    check("EEGNet forward pass succeeds", eegnet_ok, eegnet_detail)

    # ------------------------------------------------------------------
    # 5. POYO-EEG forward pass
    # ------------------------------------------------------------------
    print("\n[5/6] POYO-EEG forward pass...")

    from foundry.models.poyo_eeg import POYOEEGModel
    from foundry.models.tokenizer import EEGTokenizer
    from foundry.models.embeddings.channel import PerChannelStrategy
    from foundry.models.embeddings.temporal.patch_linear import (
        PatchLinearEmbedding,
    )

    embed_dim = 64
    patch_samples = 12
    channel_strategy = PerChannelStrategy(max_channels=num_channels)
    temporal_emb = PatchLinearEmbedding(
        embed_dim=embed_dim,
        num_input_channels=1,
        patch_samples=patch_samples,
    )
    tokenizer = EEGTokenizer(
        channel_strategy=channel_strategy,
        temporal_embedding=temporal_emb,
        embed_dim=embed_dim,
        patch_duration=0.1,
    )

    poyo = POYOEEGModel(
        tokenizer=tokenizer,
        task_configs=task_configs,
        embed_dim=embed_dim,
        sequence_length=1.0,
        latent_step=0.1,
        num_latents_per_step=4,
        depth=2,
        dim_head=32,
        cross_heads=2,
        self_heads=2,
        ffn_dropout=0.0,
        lin_dropout=0.0,
        atn_dropout=0.0,
    )
    poyo.eval()

    dm_poyo = NeuralBenchDataModule(
        task="p3",
        dataset="korczowski2014a",
        cache_dir="/network/scratch/s/sobralm/neuralset-data",
        batch_size=4,
        num_workers=0,
    )
    dm_poyo._task_configs = task_configs
    dm_poyo.setup("fit")

    # Initialize vocabs
    session_ids = dm_poyo.get_recording_ids()
    channel_ids = dm_poyo.get_channel_ids()
    poyo.initialize_vocabs({
        "session_ids": session_ids,
        "channel_ids": channel_ids,
    })

    dm_poyo.set_tokenizer(poyo.tokenize)

    poyo_loader = dm_poyo.train_dataloader()
    poyo_batch = next(iter(poyo_loader))

    non_forward_keys = {
        "target_values", "target_weights", "session_id", "absolute_start",
    }
    poyo_inputs = {
        k: v for k, v in poyo_batch.items() if k not in non_forward_keys
    }

    try:
        with torch.no_grad():
            poyo_out = poyo(**poyo_inputs)
        poyo_ok = True
        poyo_detail = f"output type: {type(poyo_out).__name__}"
    except Exception as e:
        poyo_ok = False
        poyo_detail = str(e)

    check("POYO-EEG forward pass succeeds", poyo_ok, poyo_detail)

    # ------------------------------------------------------------------
    # 6. Cross-split consistency
    # ------------------------------------------------------------------
    print("\n[6/6] Cross-split consistency...")

    train_sessions = set()
    val_sessions = set()
    for i in range(min(100, len(dm._train_adapter))):
        d = dm._train_adapter._to_torch_brain_data(i)
        train_sessions.add(str(d.session.id))
    for i in range(min(100, len(dm._val_adapter))):
        d = dm._val_adapter._to_torch_brain_data(i)
        val_sessions.add(str(d.session.id))

    overlap = train_sessions & val_sessions
    check(
        "No session overlap between train and val (subject-level split)",
        len(overlap) == 0,
        f"overlap: {overlap}" if overlap else "disjoint",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    return print_summary()


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
