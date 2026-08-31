"""Test signal scaling as the fix for Conv-BiGRU minipig failure.

The initial diagnostic found that all embeddings are identical (cosine sim=1.0)
because the raw ECoG signal (~1e-4 V) is overwhelmed by the Linear adapter's
bias. This script tests three fixes:

1. Input scaling: multiply signal by 1e4 to bring to ~unit scale
2. Bias removal: use bias=False on the adapter Linear
3. Input z-scoring: per-channel z-normalization of the input

Also compares monkey session signal scale to understand why monkey works.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_brain.batching import collate
from torch_brain.datasets.dataset import DatasetIndex

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foundry.data.datasets.neurosoft import NeurosoftMinipigs2026, NeurosoftMonkeys2026
from foundry.tasks.config import TaskConfig
from foundry.tasks.classification_mapping import filter_intervals_by_mapping
from foundry.models.neurosoft_conv_bigru import (
    NeurosoftConvBiGRU, SessionInputAdapter, _SeparableTemporalBlock,
)
from foundry.models.readout import build_readout_router
from foundry.seed import set_seed

DATA_ROOT = "./data/processed/"
MINIPIG_RID = "sub-06_ses-02_task-AcousStim_acq-LH_desc-raw"
MONKEY_RID = "sub-01_ses-04_task-AcousStim_acq-RH_desc-raw"
TASK_YAML = Path(__file__).resolve().parent.parent / "configs" / "tasks" / "neurosoft_acoustic_stim_8band.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_task_config():
    return TaskConfig.from_yaml(TASK_YAML)


def get_channel_count(ds, rid):
    rec = ds.get_recording(rid)
    signal_source = rec.ecog if hasattr(rec, "ecog") and rec.ecog is not None else rec.eeg
    ch_types = rec.channels.type.astype(str) if hasattr(rec.channels, "type") else None
    supported = {"eeg", "ecog", "seeg", "ieeg"}
    if ch_types is not None:
        keep = np.isin(np.char.lower(ch_types), list(supported))
        return int(keep.sum())
    return signal_source.signal.shape[1]


def get_balanced_batch(ds, rid, model, task_config, n_per_class=2):
    intervals = ds.get_sampling_intervals("train")
    session_intervals = intervals[rid]
    class_mapping = task_config.class_mapping
    filtered = filter_intervals_by_mapping(session_intervals, class_mapping, "behavior_labels")
    mapping_dict = class_mapping.mapping
    raw_labels = filtered.behavior_labels
    starts = filtered.start
    ends = filtered.end

    class_bins = {name: [] for name in class_mapping.class_names}
    for i, label in enumerate(raw_labels):
        mapped = mapping_dict.get(str(label))
        if mapped is not None and mapped in class_bins:
            class_bins[mapped].append(i)

    selected = []
    for cls_name, idx_list in class_bins.items():
        n = min(len(idx_list), n_per_class)
        selected.extend(idx_list[:n])

    items = []
    for i in selected:
        start, end = float(starts[i]), float(ends[i])
        mid = (start + end) / 2
        idx = DatasetIndex(recording_id=rid, start=mid - 0.25, end=mid + 0.25)
        data = ds[idx]
        tokenized = model.tokenize(data)
        items.append(tokenized)

    return collate(items), items


def overfit(model, batch, n_steps=500, lr=0.001, label=""):
    print(f"\n{'=' * 70}")
    print(f"OVERFIT: {label}")
    print(f"{'=' * 70}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total:,}")

    model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch_dev = {}
    for k, v in batch.items():
        batch_dev[k] = v.to(DEVICE) if isinstance(v, torch.Tensor) else v

    tv = batch_dev.pop("target_values")
    tw = batch_dev.pop("target_weights")
    batch_dev.pop("session_id")
    batch_dev.pop("absolute_start")

    task_name = list(model.task_configs.keys())[0]

    for step in range(n_steps):
        optimizer.zero_grad()
        outputs = model(**batch_dev)
        logits = outputs[task_name]
        targets = tv[task_name].to(DEVICE).long()
        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        loss.backward()

        if step % 100 == 0 or step == n_steps - 1:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                valid = targets != -1
                acc = (preds[valid] == targets[valid]).float().mean().item()
            adapter_gnorm = np.mean([
                p.grad.norm().item() for n, p in model.named_parameters()
                if "session_adapter" in n and "weight" in n and p.grad is not None
            ]) if any("session_adapter" in n and "weight" in n for n, _ in model.named_parameters()) else 0
            print(f"  Step {step:4d}: loss={loss.item():.4f} acc={acc:.3f} "
                  f"adapter_w_grad={adapter_gnorm:.6f}")

        optimizer.step()

    converged = loss.item() < 0.1 and acc > 0.95
    print(f"  -> {'PASS' if converged else 'FAIL'} final_loss={loss.item():.4f} acc={acc:.3f}")
    return converged


class ScaledBiGRU(NeurosoftConvBiGRU):
    """BiGRU with input signal scaling."""
    def __init__(self, scale_factor=1e4, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor

    def forward(self, *, input_values, **kwargs):
        return super().forward(input_values=input_values * self.scale_factor, **kwargs)

    def tokenize(self, data):
        result = super().tokenize(data)
        return result


class NoBiasAdapterBiGRU(nn.Module):
    """BiGRU with bias=False on the session adapter."""
    def __init__(self, *, task_configs, session_configs, **kwargs):
        super().__init__()
        self._task_configs = TaskConfig.normalize_task_configs(task_configs)
        adapter_dim = kwargs.get("adapter_dim", 32)
        self.adapter_dim = adapter_dim
        temporal_channels = kwargs.get("temporal_channels", 64)
        gru_hidden_size = kwargs.get("gru_hidden_size", 64)
        dropout_rate = kwargs.get("dropout_rate", 0.0)

        self.session_adapter = nn.ModuleDict({
            str(sid): nn.Linear(n_ch, adapter_dim, bias=False)
            for sid, n_ch in session_configs.items()
        })
        self.channel_counts = {str(k): int(v) for k, v in session_configs.items()}

        first_padding = (64 - 4) // 2
        self.temporal_frontend = nn.ModuleList([
            _SeparableTemporalBlock(
                adapter_dim, temporal_channels,
                kernel_size=64, stride=4, padding=first_padding,
                dropout_rate=dropout_rate,
            )
        ])
        self.gru = nn.GRU(
            input_size=temporal_channels, hidden_size=gru_hidden_size,
            num_layers=1, bidirectional=True, batch_first=True,
        )
        self.embedding_dim = gru_hidden_size * 2
        self.readout_dropout = nn.Dropout(dropout_rate)
        self.router = build_readout_router(self._task_configs, self.embedding_dim)

    @property
    def task_configs(self):
        return self._task_configs

    def forward(self, *, input_values, task_index, input_session_ids,
                input_channel_counts=None, input_seq_len=None, **_):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        B, C_pad, T_pad = input_values.shape

        session_ids = []
        for s in input_session_ids:
            if torch.is_tensor(s):
                s = s.item()
            session_ids.append(str(s))

        if input_seq_len is None:
            seq_lens = torch.full((B,), T_pad, dtype=torch.long, device=input_values.device)
        else:
            seq_lens = torch.as_tensor(input_seq_len, dtype=torch.long, device=input_values.device)

        if input_channel_counts is None:
            input_channel_counts = [self.channel_counts[s] for s in session_ids]
        ch_counts = torch.as_tensor(input_channel_counts, dtype=torch.long, device=input_values.device)

        out = input_values.new_zeros(B, self.adapter_dim, T_pad)
        for i in range(B):
            sid = session_ids[i]
            c = int(ch_counts[i])
            t = int(seq_lens[i])
            adapted = self.session_adapter[sid](input_values[i, :c].transpose(0, 1)).transpose(0, 1)
            adapted[:, t:] = 0
            out[i] = adapted

        x = out
        feature_lengths = seq_lens
        for block in self.temporal_frontend:
            next_lengths = block.output_length(feature_lengths)
            x = block(x, input_lengths=feature_lengths)
            feature_lengths = next_lengths

        x = x.transpose(1, 2)
        packed = pack_padded_sequence(x, feature_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        x, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.shape[1])

        time_idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        mask = time_idx < feature_lengths.unsqueeze(1)
        embedding = (x * mask.unsqueeze(-1)).sum(dim=1) / feature_lengths.unsqueeze(1).to(x.dtype)
        embedding = self.readout_dropout(embedding)

        batch_size, n_out = task_index.shape
        routed = embedding.unsqueeze(1).expand(batch_size, n_out, -1).reshape(-1, self.embedding_dim)
        flat_index = task_index.reshape(-1)
        valid = flat_index > 0
        return self.router(routed[valid], (flat_index[valid] - 1).long())


class ZNormBiGRU(NeurosoftConvBiGRU):
    """BiGRU with per-channel z-normalization before the adapter."""
    def __init__(self, channel_means, channel_stds, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("channel_means", torch.tensor(channel_means, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer("channel_stds", torch.tensor(channel_stds, dtype=torch.float32).view(1, -1, 1))

    def forward(self, *, input_values, **kwargs):
        B, C_pad, T = input_values.shape
        C = self.channel_means.shape[1]
        normalized = input_values.clone()
        normalized[:, :C, :] = (input_values[:, :C, :] - self.channel_means) / (self.channel_stds + 1e-8)
        return super().forward(input_values=normalized, **kwargs)


def check_monkey_signal_scale():
    """Compare signal scale between minipig and monkey sessions."""
    print("\n" + "=" * 70)
    print("SIGNAL SCALE COMPARISON: MINIPIG vs MONKEY")
    print("=" * 70)

    ds_mp = NeurosoftMinipigs2026(
        root=DATA_ROOT, split_type="intrasession-causal",
        task_type="acoustic_stim", recording_ids=[MINIPIG_RID],
    )
    ds_mk = NeurosoftMonkeys2026(
        root=DATA_ROOT, split_type="intrasession-causal",
        task_type="acoustic_stim", recording_ids=[MONKEY_RID],
    )

    for name, ds, rid in [("Minipig", ds_mp, MINIPIG_RID), ("Monkey", ds_mk, MONKEY_RID)]:
        rec = ds.get_recording(rid)
        signal_source = rec.ecog if hasattr(rec, "ecog") and rec.ecog is not None else rec.eeg
        signal = np.asarray(signal_source.signal, dtype=np.float32)
        ch_types = rec.channels.type.astype(str) if hasattr(rec.channels, "type") else None
        supported = {"eeg", "ecog", "seeg", "ieeg"}
        if ch_types is not None:
            keep = np.isin(np.char.lower(ch_types), list(supported))
            signal = signal[:, keep]
        print(f"\n  {name} ({rid[:20]}...):")
        print(f"    Shape: {signal.shape}")
        print(f"    Range: [{signal.min():.6f}, {signal.max():.6f}]")
        print(f"    Mean: {signal.mean():.8f}, Std: {signal.std():.8f}")
        print(f"    Per-channel std: min={signal.std(axis=0).min():.8f}, "
              f"max={signal.std(axis=0).max():.8f}, "
              f"median={np.median(signal.std(axis=0)):.8f}")
        print(f"    Abs mean: {np.abs(signal).mean():.8f}")


def run_full_training_test(model, ds, rid, task_config, n_epochs=50, lr=0.0015, label=""):
    """Run a mini training loop with train/val splits to check generalization."""
    from foundry.data.samplers import FastRandomFixedWindowSampler
    print(f"\n{'=' * 70}")
    print(f"MINI TRAINING: {label}")
    print(f"{'=' * 70}")

    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total:,}")

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.018)
    task_name = list(model.task_configs.keys())[0]
    class_mapping = task_config.class_mapping

    def make_loader(split):
        intervals = ds.get_sampling_intervals(split)
        filtered_intervals = {
            rid: filter_intervals_by_mapping(intervals[rid], class_mapping, "behavior_labels")
        }
        sampler = FastRandomFixedWindowSampler(
            sampling_intervals=filtered_intervals,
            window_length=0.5, drop_short=True,
            generator=torch.Generator().manual_seed(42),
        )
        return torch.utils.data.DataLoader(
            ds, sampler=sampler, batch_size=16, num_workers=0,
            collate_fn=collate,
        )

    if hasattr(ds, "set_tokenizer"):
        ds.set_tokenizer(model.tokenize if hasattr(model, "tokenize") else None)
    else:
        from torch_brain.transforms import Compose
        tokenizer = model.tokenize if hasattr(model, "tokenize") else None
        if tokenizer:
            ds.transform = Compose([tokenizer])
    train_loader = make_loader("train")
    val_loader = make_loader("valid")

    best_val_f1 = 0
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        train_correct, train_total = 0, 0
        for batch in train_loader:
            batch_dev = {}
            for k, v in batch.items():
                batch_dev[k] = v.to(DEVICE) if isinstance(v, torch.Tensor) else v
            tv = batch_dev.pop("target_values")
            tw = batch_dev.pop("target_weights")
            batch_dev.pop("session_id", None)
            batch_dev.pop("absolute_start", None)

            optimizer.zero_grad()
            outputs = model(**batch_dev)
            logits = outputs[task_name]
            targets = tv[task_name].to(DEVICE).long()
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            with torch.no_grad():
                valid_mask = targets != -1
                preds = logits.argmax(dim=-1)
                train_correct += (preds[valid_mask] == targets[valid_mask]).sum().item()
                train_total += valid_mask.sum().item()

        model.eval()
        val_correct, val_total = 0, 0
        n_predicted_classes = set()
        with torch.no_grad():
            for batch in val_loader:
                batch_dev = {}
                for k, v in batch.items():
                    batch_dev[k] = v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                tv = batch_dev.pop("target_values")
                tw = batch_dev.pop("target_weights")
                batch_dev.pop("session_id", None)
                batch_dev.pop("absolute_start", None)

                outputs = model(**batch_dev)
                logits = outputs[task_name]
                targets = tv[task_name].to(DEVICE).long()
                valid_mask = targets != -1
                preds = logits.argmax(dim=-1)
                val_correct += (preds[valid_mask] == targets[valid_mask]).sum().item()
                val_total += valid_mask.sum().item()
                n_predicted_classes.update(preds[valid_mask].tolist())

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        train_loss = np.mean(train_losses)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_acc={val_acc:.3f} n_val_classes={len(n_predicted_classes)}")

    return val_acc, len(n_predicted_classes)


def main():
    set_seed(42)

    task_config = load_task_config()

    # Compare signal scales
    check_monkey_signal_scale()

    # Load minipig data
    ds = NeurosoftMinipigs2026(
        root=DATA_ROOT, split_type="intrasession-causal",
        task_type="acoustic_stim", recording_ids=[MINIPIG_RID],
    )
    n_ch = get_channel_count(ds, MINIPIG_RID)
    tc = {task_config.name: task_config}
    sc = {MINIPIG_RID: n_ch}

    # Get balanced batch for overfit tests
    ref_model = NeurosoftConvBiGRU(
        task_configs=tc, session_configs=sc, num_samples=1000,
        adapter_dim=32, temporal_channels=64, temporal_kernel_samples=64,
        temporal_stride=4, conv_depth=1, dropout_rate=0.0,
        gru_hidden_size=64, gru_num_layers=1,
    )
    batch, raw_items = get_balanced_batch(ds, MINIPIG_RID, ref_model, task_config, n_per_class=2)

    # ---- FIX 1: Input scaling (1e4) ----
    print("\n\n" + "#" * 70)
    print("# FIX 1: SCALE INPUT BY 1e4")
    print("#" * 70)
    model_scaled = ScaledBiGRU(
        scale_factor=1e4,
        task_configs=tc, session_configs=sc, num_samples=1000,
        adapter_dim=32, temporal_channels=64, temporal_kernel_samples=64,
        temporal_stride=4, conv_depth=1, dropout_rate=0.0,
        gru_hidden_size=64, gru_num_layers=1,
    )
    overfit(model_scaled, batch, n_steps=500, lr=0.001,
            label="Compact BiGRU + input*1e4, lr=0.001")

    # ---- FIX 2: No adapter bias ----
    print("\n\n" + "#" * 70)
    print("# FIX 2: ADAPTER bias=False")
    print("#" * 70)
    model_nobias = NoBiasAdapterBiGRU(
        task_configs=tc, session_configs=sc,
        adapter_dim=32, temporal_channels=64, gru_hidden_size=64,
    )
    overfit(model_nobias, batch, n_steps=500, lr=0.001,
            label="Compact BiGRU no-bias adapter, lr=0.001")

    # ---- FIX 3: Z-normalization ----
    print("\n\n" + "#" * 70)
    print("# FIX 3: PER-CHANNEL Z-NORMALIZATION")
    print("#" * 70)
    rec = ds.get_recording(MINIPIG_RID)
    signal = np.asarray(rec.ecog.signal, dtype=np.float32)
    ch_types = rec.channels.type.astype(str)
    keep = np.isin(np.char.lower(ch_types), ["eeg", "ecog", "seeg", "ieeg"])
    signal = signal[:, keep]
    ch_means = signal.mean(axis=0).tolist()
    ch_stds = signal.std(axis=0).tolist()
    model_znorm = ZNormBiGRU(
        channel_means=ch_means, channel_stds=ch_stds,
        task_configs=tc, session_configs=sc, num_samples=1000,
        adapter_dim=32, temporal_channels=64, temporal_kernel_samples=64,
        temporal_stride=4, conv_depth=1, dropout_rate=0.0,
        gru_hidden_size=64, gru_num_layers=1,
    )
    overfit(model_znorm, batch, n_steps=500, lr=0.001,
            label="Compact BiGRU + z-norm, lr=0.001")

    # ---- FIX 1 at production LR ----
    print("\n\n" + "#" * 70)
    print("# FIX 1 AT PRODUCTION LR (0.0015)")
    print("#" * 70)
    model_scaled2 = ScaledBiGRU(
        scale_factor=1e4,
        task_configs=tc, session_configs=sc, num_samples=1000,
        adapter_dim=32, temporal_channels=64, temporal_kernel_samples=64,
        temporal_stride=4, conv_depth=1, dropout_rate=0.0,
        gru_hidden_size=64, gru_num_layers=1,
    )
    overfit(model_scaled2, batch, n_steps=500, lr=0.0015,
            label="Compact BiGRU + input*1e4, lr=0.0015")

    # ---- Check embedding similarity after scaling ----
    print("\n\n" + "#" * 70)
    print("# EMBEDDING ANALYSIS WITH SCALING")
    print("#" * 70)
    model_check = ScaledBiGRU(
        scale_factor=1e4,
        task_configs=tc, session_configs=sc, num_samples=1000,
        adapter_dim=32, temporal_channels=64, temporal_kernel_samples=64,
        temporal_stride=4, conv_depth=1, dropout_rate=0.0,
        gru_hidden_size=64, gru_num_layers=1,
    )
    model_check = model_check.to(DEVICE)
    model_check.eval()
    with torch.no_grad():
        batch_dev = {}
        for k, v in batch.items():
            batch_dev[k] = v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        batch_dev.pop("target_values"); batch_dev.pop("target_weights")
        batch_dev.pop("session_id"); batch_dev.pop("absolute_start")
        input_vals = batch_dev["input_values"] * 1e4
        print(f"  Scaled input: range=[{input_vals.min():.4f}, {input_vals.max():.4f}], "
              f"std={input_vals.std():.4f}")
        embedding = model_check.encode(
            input_values=input_vals,
            input_session_ids=batch_dev["input_session_ids"],
            input_channel_counts=batch_dev.get("input_channel_counts"),
            input_seq_len=batch_dev.get("input_seq_len"),
        )
        cos_sim = F.cosine_similarity(embedding.unsqueeze(0), embedding.unsqueeze(1), dim=-1)
        print(f"  Embedding mean cosine sim: {cos_sim.mean():.4f}")
        print(f"  Embedding min cosine sim: {cos_sim.min():.4f}")
        print(f"  Embedding std: {embedding.std():.4f}")

    # ---- Full mini training with scaling ----
    print("\n\n" + "#" * 70)
    print("# MINI TRAINING: SCALED vs UNSCALED")
    print("#" * 70)
    model_unscaled = NeurosoftConvBiGRU(
        task_configs=tc, session_configs=sc, num_samples=1000,
        adapter_dim=32, temporal_channels=64, temporal_kernel_samples=64,
        temporal_stride=4, conv_depth=1, dropout_rate=0.3,
        gru_hidden_size=64, gru_num_layers=1,
    )
    run_full_training_test(model_unscaled, ds, MINIPIG_RID, task_config,
                           n_epochs=50, lr=0.0015, label="Baseline (no scaling)")

    model_scaled_train = ScaledBiGRU(
        scale_factor=1e4,
        task_configs=tc, session_configs=sc, num_samples=1000,
        adapter_dim=32, temporal_channels=64, temporal_kernel_samples=64,
        temporal_stride=4, conv_depth=1, dropout_rate=0.3,
        gru_hidden_size=64, gru_num_layers=1,
    )
    run_full_training_test(model_scaled_train, ds, MINIPIG_RID, task_config,
                           n_epochs=50, lr=0.0015, label="Scaled (input*1e4)")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
