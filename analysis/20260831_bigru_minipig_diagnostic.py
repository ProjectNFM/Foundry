"""Diagnostic script to find why Conv-BiGRU fails on minipig session.

Tests:
1. Data inspection: class distribution, signal statistics
2. Overfit test: can the model memorize 16 examples?
3. Gradient analysis: are gradients flowing through all components?
4. Component isolation: conv-only (no GRU) vs full model
5. Comparison: EEGNet on same data
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_brain.batching import collate
from torch_brain.datasets.dataset import DatasetIndex

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foundry.data.datasets.neurosoft import NeurosoftMinipigs2026
from foundry.data.samplers import FastRandomFixedWindowSampler
from foundry.tasks.config import TaskConfig
from foundry.tasks.classification_mapping import filter_intervals_by_mapping
from foundry.models.neurosoft_conv_bigru import NeurosoftConvBiGRU
from foundry.seed import set_seed

DATA_ROOT = "./data/processed/"
RECORDING_ID = "sub-06_ses-02_task-AcousStim_acq-LH_desc-raw"
TASK_YAML = Path(__file__).resolve().parent.parent / "configs" / "tasks" / "neurosoft_acoustic_stim_8band.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_task_config():
    return TaskConfig.from_yaml(TASK_YAML)


def create_dataset(task_config, split_type="intrasession-causal"):
    ds = NeurosoftMinipigs2026(
        root=DATA_ROOT,
        split_type=split_type,
        task_type="acoustic_stim",
        recording_ids=[RECORDING_ID],
    )
    return ds


def get_session_channel_count(ds):
    rec = ds.get_recording(RECORDING_ID)
    signal = rec.ecog.signal if hasattr(rec, "ecog") and rec.ecog is not None else rec.eeg.signal
    ch_types = rec.channels.type.astype(str) if hasattr(rec.channels, "type") else None
    supported = {"eeg", "ecog", "seeg", "ieeg"}
    if ch_types is not None:
        keep = np.isin(np.char.lower(ch_types), list(supported))
        n_supported = int(keep.sum())
    else:
        n_supported = signal.shape[1]
    return n_supported


def inspect_data(ds, task_config):
    """Report class distribution and signal statistics."""
    print("\n" + "=" * 70)
    print("DATA INSPECTION")
    print("=" * 70)

    rec = ds.get_recording(RECORDING_ID)
    signal_source = rec.ecog if hasattr(rec, "ecog") and rec.ecog is not None else rec.eeg
    signal = np.asarray(signal_source.signal, dtype=np.float32)
    print(f"\nRecording: {RECORDING_ID}")
    print(f"  Raw signal shape: {signal.shape} (samples x channels)")
    print(f"  Signal range: [{signal.min():.4f}, {signal.max():.4f}]")
    print(f"  Signal mean: {signal.mean():.6f}, std: {signal.std():.6f}")
    print(f"  Per-channel std range: [{signal.std(axis=0).min():.6f}, {signal.std(axis=0).max():.6f}]")

    ch_types = rec.channels.type.astype(str) if hasattr(rec.channels, "type") else None
    if ch_types is not None:
        print(f"  Channel types: {dict(Counter(ch_types))}")
        supported = {"eeg", "ecog", "seeg", "ieeg"}
        keep = np.isin(np.char.lower(ch_types), list(supported))
        n_supported = int(keep.sum())
        print(f"  Supported channels: {n_supported} / {len(ch_types)}")
        signal_supported = signal[:, keep]
        print(f"  Supported signal shape: {signal_supported.shape}")
        print(f"  Supported signal range: [{signal_supported.min():.4f}, {signal_supported.max():.4f}]")
        print(f"  Supported signal mean: {signal_supported.mean():.6f}, std: {signal_supported.std():.6f}")
        per_ch_std = signal_supported.std(axis=0)
        print(f"  Per-channel std: {per_ch_std}")

    for split in ["train", "valid", "test"]:
        intervals = ds.get_sampling_intervals(split)
        if RECORDING_ID in intervals:
            session_intervals = intervals[RECORDING_ID]
            class_mapping = task_config.class_mapping

            # Inspect raw interval structure
            if split == "train":
                print(f"\n  Raw interval attributes: {dir(session_intervals)}")
                if hasattr(session_intervals, "behavior_labels"):
                    raw_labels = session_intervals.behavior_labels
                    print(f"  Raw behavior_labels type: {type(raw_labels)}")
                    if hasattr(raw_labels, '__len__'):
                        print(f"  Raw behavior_labels length: {len(raw_labels)}")
                        unique_labels = set(raw_labels) if len(raw_labels) < 10000 else set(list(raw_labels)[:1000])
                        print(f"  Sample raw labels: {list(unique_labels)[:20]}")

            if class_mapping is not None:
                filtered = filter_intervals_by_mapping(
                    session_intervals, class_mapping, "behavior_labels"
                )
            else:
                filtered = session_intervals
            n_windows = len(filtered)

            # Count labels from the filtered intervals
            labels = []
            if hasattr(filtered, "behavior_labels"):
                raw_labels = filtered.behavior_labels
                mapping_dict = class_mapping.mapping if class_mapping else {}
                for label in raw_labels:
                    label_str = str(label)
                    if label_str in mapping_dict:
                        labels.append(mapping_dict[label_str])
                    elif class_mapping is None:
                        labels.append(label_str)

            label_counts = Counter(labels)
            total = sum(label_counts.values())
            print(f"\n  {split} split: {n_windows} intervals, {total} labeled trials")
            for cls_name in task_config.class_mapping.class_names:
                count = label_counts.get(cls_name, 0)
                pct = 100 * count / total if total > 0 else 0
                print(f"    {cls_name}: {count} ({pct:.1f}%)")


def create_model(task_config, n_channels, variant="compact"):
    """Create a Conv-BiGRU model."""
    session_configs = {RECORDING_ID: n_channels}
    tc = {task_config.name: task_config}

    if variant == "compact":
        model = NeurosoftConvBiGRU(
            task_configs=tc,
            session_configs=session_configs,
            num_samples=1000,
            adapter_dim=32,
            temporal_channels=64,
            temporal_kernel_samples=64,
            temporal_stride=4,
            conv_depth=1,
            dropout_rate=0.0,  # no dropout for diagnostic
            gru_hidden_size=64,
            gru_num_layers=1,
        )
    elif variant == "full":
        model = NeurosoftConvBiGRU(
            task_configs=tc,
            session_configs=session_configs,
            num_samples=1000,
            adapter_dim=64,
            temporal_channels=128,
            temporal_kernel_samples=64,
            temporal_stride=4,
            conv_depth=1,
            dropout_rate=0.0,
            gru_hidden_size=128,
            gru_num_layers=2,
        )
    elif variant == "tiny":
        model = NeurosoftConvBiGRU(
            task_configs=tc,
            session_configs=session_configs,
            num_samples=1000,
            adapter_dim=n_channels,  # identity-scale adapter
            temporal_channels=32,
            temporal_kernel_samples=64,
            temporal_stride=4,
            conv_depth=1,
            dropout_rate=0.0,
            gru_hidden_size=32,
            gru_num_layers=1,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_balanced_batch(ds, model, task_config, n_per_class=2):
    """Get a balanced batch with n_per_class examples per class."""
    intervals = ds.get_sampling_intervals("train")
    session_intervals = intervals[RECORDING_ID]
    class_mapping = task_config.class_mapping
    if class_mapping is not None:
        session_intervals = filter_intervals_by_mapping(
            session_intervals, class_mapping, "behavior_labels"
        )

    # The intervals object is array-like: intervals.start, .end, .behavior_labels
    # are all arrays indexed together
    if hasattr(session_intervals, "behavior_labels"):
        raw_labels = session_intervals.behavior_labels
        starts = session_intervals.start
        ends = session_intervals.end
    else:
        raise ValueError("No behavior_labels found on intervals")

    mapping_dict = class_mapping.mapping
    class_bins = {name: [] for name in class_mapping.class_names}
    for i, label in enumerate(raw_labels):
        label_str = str(label)
        mapped = mapping_dict.get(label_str)
        if mapped is not None and mapped in class_bins:
            class_bins[mapped].append(i)

    selected_indices = []
    for cls_name, idx_list in class_bins.items():
        if len(idx_list) >= n_per_class:
            selected_indices.extend(idx_list[:n_per_class])
        elif len(idx_list) > 0:
            selected_indices.extend(idx_list)
        else:
            print(f"  WARNING: class {cls_name} has 0 training intervals")

    print(f"\n  Balanced batch: {len(selected_indices)} intervals from {len(class_bins)} classes")
    for cls_name, idx_list in class_bins.items():
        used = min(len(idx_list), n_per_class)
        print(f"    {cls_name}: {used} selected (of {len(idx_list)} available)")

    items = []
    for i in selected_indices:
        start = float(starts[i])
        end = float(ends[i])
        mid = (start + end) / 2
        window_start = mid - 0.25
        window_end = mid + 0.25
        idx = DatasetIndex(
            recording_id=RECORDING_ID,
            start=window_start,
            end=window_end,
        )
        data = ds[idx]
        tokenized = model.tokenize(data)
        items.append(tokenized)

    batch = collate(items)
    return batch


def overfit_test(model, batch, n_steps=500, lr=0.001, label=""):
    """Try to overfit the model on a fixed batch."""
    print(f"\n{'=' * 70}")
    print(f"OVERFIT TEST: {label}")
    print(f"{'=' * 70}")

    total, trainable = count_params(model)
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch_device = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_device[k] = v.to(DEVICE)
        else:
            batch_device[k] = v

    target_values = batch_device.pop("target_values")
    target_weights = batch_device.pop("target_weights")
    session_ids = batch_device.pop("session_id")
    abs_start = batch_device.pop("absolute_start")

    task_name = list(model.task_configs.keys())[0]
    task_idx = 1  # task indices are 1-based

    losses = []
    accs = []

    for step in range(n_steps):
        optimizer.zero_grad()
        outputs = model(**batch_device)
        logits = outputs[task_name]
        targets = target_values[task_name].to(DEVICE).long()
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        loss.backward()

        if step % 100 == 0 or step == n_steps - 1:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                valid = targets != -1
                correct = (preds[valid] == targets[valid]).float().mean().item()
                accs.append(correct)

            grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    component = name.split(".")[0]
                    if component not in grad_norms:
                        grad_norms[component] = []
                    grad_norms[component].append(param.grad.norm().item())

            grad_summary = {k: f"{np.mean(v):.6f}" for k, v in grad_norms.items()}
            print(f"  Step {step:4d}: loss={loss.item():.4f}, acc={correct:.3f}, "
                  f"grad_norms={grad_summary}")

        losses.append(loss.item())
        optimizer.step()

    final_loss = losses[-1]
    final_acc = accs[-1] if accs else 0
    converged = final_loss < 0.1 and final_acc > 0.95

    print(f"\n  RESULT: {'PASS' if converged else 'FAIL'} "
          f"(final loss={final_loss:.4f}, acc={final_acc:.3f})")
    return converged, losses, accs


def gradient_analysis(model, batch, n_steps=20, lr=0.001, label=""):
    """Analyze gradient flow through the model components."""
    print(f"\n{'=' * 70}")
    print(f"GRADIENT ANALYSIS: {label}")
    print(f"{'=' * 70}")

    model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch_device = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_device[k] = v.to(DEVICE)
        else:
            batch_device[k] = v

    target_values = batch_device.pop("target_values")
    target_weights = batch_device.pop("target_weights")
    session_ids = batch_device.pop("session_id")
    abs_start = batch_device.pop("absolute_start")

    task_name = list(model.task_configs.keys())[0]

    for step in range(n_steps):
        optimizer.zero_grad()
        outputs = model(**batch_device)
        logits = outputs[task_name]
        targets = target_values[task_name].to(DEVICE).long()
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        loss.backward()

        if step == 0 or step == n_steps - 1:
            print(f"\n  Step {step}:")
            print(f"    Loss: {loss.item():.4f}")

            # Per-parameter gradient analysis
            for name, param in model.named_parameters():
                if param.grad is not None:
                    g = param.grad
                    print(f"    {name}: shape={list(param.shape)}, "
                          f"grad_norm={g.norm():.6f}, "
                          f"grad_mean={g.mean():.8f}, "
                          f"grad_std={g.std():.8f}, "
                          f"param_norm={param.norm():.4f}, "
                          f"ratio={g.norm()/max(param.norm(), 1e-8):.6f}")

            # Check output logit statistics
            with torch.no_grad():
                print(f"\n    Logit stats: mean={logits.mean():.4f}, "
                      f"std={logits.std():.4f}, "
                      f"range=[{logits.min():.4f}, {logits.max():.4f}]")
                probs = F.softmax(logits, dim=-1)
                print(f"    Prob stats: mean={probs.mean():.4f}, "
                      f"std={probs.std():.4f}, "
                      f"max_prob={probs.max():.4f}")
                print(f"    Predicted classes: {logits.argmax(dim=-1).tolist()}")
                print(f"    Target classes: {targets.tolist()}")

        optimizer.step()


def conv_only_test(task_config, n_channels, batch, n_steps=500, lr=0.001):
    """Test with conv frontend only (no GRU) using global average pooling."""
    print(f"\n{'=' * 70}")
    print(f"CONV-ONLY TEST (no GRU, global avg pool)")
    print(f"{'=' * 70}")

    from foundry.models.neurosoft_conv_bigru import SessionInputAdapter, _SeparableTemporalBlock
    from foundry.models.readout import build_readout_router

    class ConvOnlyModel(nn.Module):
        def __init__(self, task_configs, session_configs, adapter_dim=32,
                     temporal_channels=64, dropout_rate=0.0):
            super().__init__()
            self._task_configs = TaskConfig.normalize_task_configs(task_configs)
            self.session_adapter = SessionInputAdapter(session_configs, adapter_dim)
            self.temporal_block = _SeparableTemporalBlock(
                adapter_dim, temporal_channels,
                kernel_size=64, stride=4, padding=30, dropout_rate=dropout_rate,
            )
            self.embedding_dim = temporal_channels
            self.router = build_readout_router(self._task_configs, self.embedding_dim)

        @property
        def task_configs(self):
            return self._task_configs

        def forward(self, *, input_values, task_index, input_session_ids,
                    input_channel_counts=None, input_seq_len=None, **_):
            B, C_pad, T_pad = input_values.shape
            session_ids = [self.session_adapter._as_session_id(s) for s in input_session_ids]
            if input_channel_counts is None:
                input_channel_counts = [self.session_adapter.channel_counts[s] for s in session_ids]
            seq_lens = torch.as_tensor(
                [T_pad] * B if input_seq_len is None else input_seq_len,
                dtype=torch.long, device=input_values.device,
            )
            channel_counts = torch.as_tensor(input_channel_counts, dtype=torch.long, device=input_values.device)

            x = self.session_adapter(
                input_values, input_session_ids=session_ids,
                input_channel_counts=channel_counts, input_seq_len=seq_lens,
            )
            feature_lengths = self.temporal_block.output_length(seq_lens)
            x = self.temporal_block(x, input_lengths=seq_lens)
            # Global average pooling
            mask = torch.arange(x.shape[-1], device=x.device).unsqueeze(0) < feature_lengths.unsqueeze(1)
            embedding = (x * mask.unsqueeze(1).float()).sum(dim=-1) / feature_lengths.unsqueeze(1).float()

            batch_size, n_out = task_index.shape
            routed = embedding.unsqueeze(1).expand(batch_size, n_out, -1).reshape(-1, self.embedding_dim)
            flat_index = task_index.reshape(-1)
            valid = flat_index > 0
            return self.router(routed[valid], (flat_index[valid] - 1).long())

    session_configs = {RECORDING_ID: n_channels}
    tc = {task_config.name: task_config}
    model = ConvOnlyModel(tc, session_configs)
    total, trainable = count_params(model)
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    return overfit_test(model, batch, n_steps=n_steps, lr=lr, label="Conv-only (no GRU)")


def adapter_only_test(task_config, n_channels, batch, n_steps=500, lr=0.001):
    """Test with just the adapter + global pool (no conv, no GRU)."""
    print(f"\n{'=' * 70}")
    print(f"ADAPTER-ONLY TEST (linear adapter → global avg pool → readout)")
    print(f"{'=' * 70}")

    from foundry.models.neurosoft_conv_bigru import SessionInputAdapter
    from foundry.models.readout import build_readout_router

    class AdapterOnlyModel(nn.Module):
        def __init__(self, task_configs, session_configs, adapter_dim=32):
            super().__init__()
            self._task_configs = TaskConfig.normalize_task_configs(task_configs)
            self.session_adapter = SessionInputAdapter(session_configs, adapter_dim)
            self.embedding_dim = adapter_dim
            self.router = build_readout_router(self._task_configs, self.embedding_dim)

        @property
        def task_configs(self):
            return self._task_configs

        def forward(self, *, input_values, task_index, input_session_ids,
                    input_channel_counts=None, input_seq_len=None, **_):
            B, C_pad, T_pad = input_values.shape
            session_ids = [self.session_adapter._as_session_id(s) for s in input_session_ids]
            if input_channel_counts is None:
                input_channel_counts = [self.session_adapter.channel_counts[s] for s in session_ids]
            seq_lens = torch.as_tensor(
                [T_pad] * B if input_seq_len is None else input_seq_len,
                dtype=torch.long, device=input_values.device,
            )
            channel_counts = torch.as_tensor(input_channel_counts, dtype=torch.long, device=input_values.device)

            x = self.session_adapter(
                input_values, input_session_ids=session_ids,
                input_channel_counts=channel_counts, input_seq_len=seq_lens,
            )
            # Global average pooling over time
            mask = torch.arange(T_pad, device=x.device).unsqueeze(0) < seq_lens.unsqueeze(1)
            embedding = (x * mask.unsqueeze(1).float()).sum(dim=-1) / seq_lens.unsqueeze(1).float()

            batch_size, n_out = task_index.shape
            routed = embedding.unsqueeze(1).expand(batch_size, n_out, -1).reshape(-1, self.embedding_dim)
            flat_index = task_index.reshape(-1)
            valid = flat_index > 0
            return self.router(routed[valid], (flat_index[valid] - 1).long())

    session_configs = {RECORDING_ID: n_channels}
    tc = {task_config.name: task_config}
    model = AdapterOnlyModel(tc, session_configs)
    total, trainable = count_params(model)
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    return overfit_test(model, batch, n_steps=n_steps, lr=lr, label="Adapter-only")


def eegnet_overfit_test(task_config, n_channels, batch_items, n_steps=500, lr=0.001):
    """Test EEGNet overfit on the same data for comparison."""
    print(f"\n{'=' * 70}")
    print(f"EEGNET OVERFIT TEST (same data, for comparison)")
    print(f"{'=' * 70}")

    from foundry.models.baselines import EEGNetEncoder

    tc = {task_config.name: task_config}
    model = EEGNetEncoder(
        task_configs=tc,
        num_channels=n_channels,
        num_samples=1000,
        F1=8, D=2, F2=16,
        kernel_length=64,
        dropout_rate=0.0,
    )
    total, trainable = count_params(model)
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    # Re-tokenize for EEGNet format (needs (B, C, T) input)
    items = []
    for item in batch_items:
        new_item = {}
        for k, v in item.items():
            if k in ("input_session_ids", "input_channel_counts", "input_seq_len",
                      "session_id", "absolute_start"):
                continue
            new_item[k] = v
        items.append(new_item)

    batch = collate(items)
    return overfit_test(model, batch, n_steps=n_steps, lr=lr, label="EEGNet")


def check_adapter_output_statistics(model, batch):
    """Check what the session adapter produces."""
    print(f"\n{'=' * 70}")
    print(f"ADAPTER OUTPUT ANALYSIS")
    print(f"{'=' * 70}")

    model = model.to(DEVICE)
    model.eval()

    batch_device = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_device[k] = v.to(DEVICE)
        else:
            batch_device[k] = v

    with torch.no_grad():
        input_values = batch_device["input_values"]
        input_session_ids = batch_device["input_session_ids"]
        input_channel_counts = batch_device.get("input_channel_counts")
        input_seq_len = batch_device.get("input_seq_len")

        print(f"\n  Input signal:")
        print(f"    Shape: {input_values.shape}")
        print(f"    Range: [{input_values.min():.4f}, {input_values.max():.4f}]")
        print(f"    Mean: {input_values.mean():.6f}, Std: {input_values.std():.6f}")

        B, C_pad, T_pad = input_values.shape
        session_ids = [model.session_adapter._as_session_id(s) for s in input_session_ids]
        if input_channel_counts is None:
            input_channel_counts = [model.session_adapter.channel_counts[s] for s in session_ids]
        seq_lens = torch.as_tensor(
            [T_pad] * B if input_seq_len is None else input_seq_len,
            dtype=torch.long, device=input_values.device,
        )
        channel_counts = torch.as_tensor(input_channel_counts, dtype=torch.long, device=input_values.device)

        adapted = model.session_adapter(
            input_values, input_session_ids=session_ids,
            input_channel_counts=channel_counts, input_seq_len=seq_lens,
        )
        print(f"\n  After session adapter (Linear({int(channel_counts[0])} → {model.adapter_dim})):")
        print(f"    Shape: {adapted.shape}")
        print(f"    Range: [{adapted.min():.4f}, {adapted.max():.4f}]")
        print(f"    Mean: {adapted.mean():.6f}, Std: {adapted.std():.6f}")
        print(f"    Per-dim std: min={adapted.std(dim=(0,2)).min():.6f}, "
              f"max={adapted.std(dim=(0,2)).max():.6f}")

        # After temporal frontend
        x = adapted
        feature_lengths = seq_lens
        for i, block in enumerate(model.temporal_frontend):
            next_feature_lengths = block.output_length(feature_lengths)
            x = block(x, input_lengths=feature_lengths)
            feature_lengths = next_feature_lengths
            print(f"\n  After temporal block {i}:")
            print(f"    Shape: {x.shape}")
            print(f"    Range: [{x.min():.4f}, {x.max():.4f}]")
            print(f"    Mean: {x.mean():.6f}, Std: {x.std():.6f}")

        # After GRU
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        x_gru = x.transpose(1, 2)
        packed = pack_padded_sequence(
            x_gru, feature_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = model.gru(packed)
        x_gru, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=x_gru.shape[1]
        )
        print(f"\n  After GRU:")
        print(f"    Shape: {x_gru.shape}")
        print(f"    Range: [{x_gru.min():.4f}, {x_gru.max():.4f}]")
        print(f"    Mean: {x_gru.mean():.6f}, Std: {x_gru.std():.6f}")

        # Masked mean pool
        time_index = torch.arange(x_gru.shape[1], device=x_gru.device).unsqueeze(0)
        mask = time_index < feature_lengths.unsqueeze(1)
        embedding = (x_gru * mask.unsqueeze(-1)).sum(dim=1) / feature_lengths.unsqueeze(1).to(x_gru.dtype)
        print(f"\n  Final embedding (after masked mean pool):")
        print(f"    Shape: {embedding.shape}")
        print(f"    Range: [{embedding.min():.4f}, {embedding.max():.4f}]")
        print(f"    Mean: {embedding.mean():.6f}, Std: {embedding.std():.6f}")

        # Check if embeddings are similar across different classes
        print(f"\n  Embedding similarity across batch items:")
        norms = embedding.norm(dim=-1)
        print(f"    Norms: {norms.tolist()}")
        cos_sim = F.cosine_similarity(embedding.unsqueeze(0), embedding.unsqueeze(1), dim=-1)
        print(f"    Mean cosine similarity: {cos_sim.mean():.4f}")
        print(f"    Min cosine similarity: {cos_sim.min():.4f}")


def main():
    set_seed(42)

    print("=" * 70)
    print("Conv-BiGRU Minipig Diagnostic")
    print("=" * 70)

    task_config = load_task_config()
    ds = create_dataset(task_config)

    # 1. Data inspection
    n_channels = get_session_channel_count(ds)
    print(f"\nSupported channels: {n_channels}")
    inspect_data(ds, task_config)

    # 2. Create model and get batch
    model = create_model(task_config, n_channels, variant="compact")
    batch = get_balanced_batch(ds, model, task_config, n_per_class=2)

    # 3. Check adapter output statistics
    check_adapter_output_statistics(model, batch)

    # 4. Gradient analysis
    model_grad = create_model(task_config, n_channels, variant="compact")
    gradient_analysis(model_grad, batch, n_steps=10, lr=0.001, label="Compact BiGRU")

    # 5. Overfit test - compact BiGRU
    model_overfit = create_model(task_config, n_channels, variant="compact")
    converged_compact, _, _ = overfit_test(
        model_overfit, batch, n_steps=500, lr=0.001, label="Compact BiGRU (0 dropout)"
    )

    # 6. Overfit test - conv only (no GRU)
    conv_only_test(task_config, n_channels, batch, n_steps=500, lr=0.001)

    # 7. Overfit test - adapter only
    adapter_only_test(task_config, n_channels, batch, n_steps=500, lr=0.001)

    # 8. Overfit test - try higher learning rate
    model_highlr = create_model(task_config, n_channels, variant="compact")
    overfit_test(model_highlr, batch, n_steps=500, lr=0.01, label="Compact BiGRU (lr=0.01)")

    # 9. Overfit test - try even higher learning rate
    model_higherlr = create_model(task_config, n_channels, variant="compact")
    overfit_test(model_higherlr, batch, n_steps=500, lr=0.1, label="Compact BiGRU (lr=0.1)")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
