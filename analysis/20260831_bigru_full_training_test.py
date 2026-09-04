"""Full training test: scaled vs unscaled BiGRU on minipig session.

Confirms the signal-scaling root cause by running 50-epoch training loops
with and without input scaling on the actual train/val splits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_brain.batching import collate
from torch_brain.transforms import Compose

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foundry.data.datasets.neurosoft import NeurosoftMinipigs2026
from foundry.data.samplers import FastRandomFixedWindowSampler
from foundry.tasks.config import TaskConfig
from foundry.tasks.classification_mapping import filter_intervals_by_mapping
from foundry.models.neurosoft_conv_bigru import NeurosoftConvBiGRU
from foundry.seed import set_seed

DATA_ROOT = "./data/processed/"
MINIPIG_RID = "sub-06_ses-02_task-AcousStim_acq-LH_desc-raw"
TASK_YAML = Path(__file__).resolve().parent.parent / "configs" / "tasks" / "neurosoft_acoustic_stim_8band.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ScaledBiGRU(NeurosoftConvBiGRU):
    def __init__(self, scale_factor=1e4, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor

    def forward(self, *, input_values, **kwargs):
        return super().forward(input_values=input_values * self.scale_factor, **kwargs)

    def encode(self, *, input_values, **kwargs):
        return super().encode(input_values=input_values * self.scale_factor, **kwargs)


def run_training(model, ds, task_config, n_epochs=50, lr=0.0015, label=""):
    print(f"\n{'=' * 70}")
    print(f"TRAINING: {label}")
    print(f"{'=' * 70}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total:,}, LR: {lr}, Epochs: {n_epochs}")

    ds.transform = Compose([model.tokenize])

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.018)
    task_name = list(model.task_configs.keys())[0]
    class_mapping = task_config.class_mapping

    def make_loader(split):
        intervals = ds.get_sampling_intervals(split)
        filtered = {
            MINIPIG_RID: filter_intervals_by_mapping(
                intervals[MINIPIG_RID], class_mapping, "behavior_labels"
            )
        }
        sampler = FastRandomFixedWindowSampler(
            sampling_intervals=filtered,
            window_length=0.5, drop_short=True,
            generator=torch.Generator().manual_seed(42),
        )
        return DataLoader(
            ds, sampler=sampler, batch_size=16, num_workers=0,
            collate_fn=collate,
        )

    train_loader = make_loader("train")
    val_loader = make_loader("valid")

    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        train_losses, train_correct, train_total = [], 0, 0
        n_train_classes = set()
        for batch in train_loader:
            batch_dev = {}
            for k, v in batch.items():
                batch_dev[k] = v.to(DEVICE) if isinstance(v, torch.Tensor) else v
            tv = batch_dev.pop("target_values")
            batch_dev.pop("target_weights")
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
                n_train_classes.update(preds[valid_mask].tolist())

        model.eval()
        val_correct, val_total = 0, 0
        n_val_classes = set()
        with torch.no_grad():
            for batch in val_loader:
                batch_dev = {}
                for k, v in batch.items():
                    batch_dev[k] = v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                tv = batch_dev.pop("target_values")
                batch_dev.pop("target_weights")
                batch_dev.pop("session_id", None)
                batch_dev.pop("absolute_start", None)

                outputs = model(**batch_dev)
                logits = outputs[task_name]
                targets = tv[task_name].to(DEVICE).long()
                valid_mask = targets != -1
                preds = logits.argmax(dim=-1)
                val_correct += (preds[valid_mask] == targets[valid_mask]).sum().item()
                val_total += valid_mask.sum().item()
                n_val_classes.update(preds[valid_mask].tolist())

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == n_epochs - 1 or patience_counter == 0:
            print(f"  Epoch {epoch:3d}: loss={np.mean(train_losses):.4f} "
                  f"train_acc={train_acc:.3f}({len(n_train_classes)}cls) "
                  f"val_acc={val_acc:.3f}({len(n_val_classes)}cls) "
                  f"best={best_val_acc:.3f}@{best_epoch}")

        if patience_counter >= 40:
            print(f"  Early stopped at epoch {epoch}")
            break

    print(f"\n  FINAL: best_val_acc={best_val_acc:.3f} at epoch {best_epoch}, "
          f"predicted {len(n_val_classes)} classes")
    return best_val_acc, len(n_val_classes)


def main():
    set_seed(42)

    task_config = TaskConfig.from_yaml(TASK_YAML)
    tc = {task_config.name: task_config}

    ds = NeurosoftMinipigs2026(
        root=DATA_ROOT, split_type="intrasession-causal",
        task_type="acoustic_stim", recording_ids=[MINIPIG_RID],
    )

    rec = ds.get_recording(MINIPIG_RID)
    ch_types = rec.channels.type.astype(str)
    keep = np.isin(np.char.lower(ch_types), ["eeg", "ecog", "seeg", "ieeg"])
    n_ch = int(keep.sum())
    sc = {MINIPIG_RID: n_ch}

    print(f"Minipig session: {n_ch} supported channels")
    print(f"Device: {DEVICE}")

    # Baseline: no scaling
    model_base = NeurosoftConvBiGRU(
        task_configs=tc, session_configs=sc, num_samples=1000,
        adapter_dim=32, temporal_channels=64, temporal_kernel_samples=64,
        temporal_stride=4, conv_depth=1, dropout_rate=0.3,
        gru_hidden_size=64, gru_num_layers=1,
    )
    run_training(model_base, ds, task_config, n_epochs=50, lr=0.0015,
                 label="Baseline compact BiGRU (NO scaling)")

    # Fix: input scaling
    set_seed(42)
    model_scaled = ScaledBiGRU(
        scale_factor=1e4,
        task_configs=tc, session_configs=sc, num_samples=1000,
        adapter_dim=32, temporal_channels=64, temporal_kernel_samples=64,
        temporal_stride=4, conv_depth=1, dropout_rate=0.3,
        gru_hidden_size=64, gru_num_layers=1,
    )
    run_training(model_scaled, ds, task_config, n_epochs=50, lr=0.0015,
                 label="Compact BiGRU + input*1e4")

    # Fix: input scaling with production recipe (full model)
    set_seed(42)
    model_scaled_full = ScaledBiGRU(
        scale_factor=1e4,
        task_configs=tc, session_configs=sc, num_samples=1000,
        adapter_dim=64, temporal_channels=128, temporal_kernel_samples=64,
        temporal_stride=4, conv_depth=1, dropout_rate=0.3,
        gru_hidden_size=128, gru_num_layers=2,
    )
    run_training(model_scaled_full, ds, task_config, n_epochs=50, lr=0.0015,
                 label="Full BiGRU (510K params) + input*1e4")

    print("\n" + "=" * 70)
    print("ALL TRAINING TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
