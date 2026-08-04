"""Brain Invaders EEGNet Reprocessed Data HP Search Analysis.

Fetches runs from WandB group BI_P300_HP_EEGNET_REPROCESSED and compares
with the original HP search results. Investigates training dynamics to
diagnose why performance remains low despite data reprocessing.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

ENTITY = "poyo-eeg"
PROJECT = "foundry_finetuning"
GROUP_NEW = "BI_P300_HP_EEGNET_REPROCESSED"
GROUP_OLD = "BI_P300_HP_SEARCH"

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PREFIX = "029_bi_eegnet_reproc"


def unwrap(val, key="max"):
    if hasattr(val, "get"):
        return float(val.get(key, 0.0))
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def fetch_runs(group: str) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group})

    records = []
    for run in runs:
        params = {
            "run_id": run.id,
            "run_name": run.name,
            "group": group,
            "state": run.state,
        }

        # Extract lr from config (more reliable than name parsing)
        try:
            params["lr"] = float(
                run.config.get("hyperparameters", {}).get("learning_rate", 0)
            )
        except (TypeError, ValueError):
            lr_match = re.search(r"lr([\deE.+-]+)", run.name)
            params["lr"] = float(lr_match.group(1)) if lr_match else None

        # Print all available summary keys for first run (debug)
        if not records:
            print(f"\n  Available summary keys for {run.name}:")
            for k in sorted(run.summary.keys()):
                if not k.startswith("_"):
                    print(f"    {k}: {run.summary[k]}")

        # Val and train metrics
        for metric in [
            "p300_binary_f1",
            "p300_binary_auroc",
            "p300_binary_acc",
            "p300_binary_loss",
            "p300_binary_precision",
            "p300_binary_recall",
            "p300_binary_balanced_acc",
            "p300_binary_cohen_kappa",
        ]:
            val_key = f"val/{metric}"
            train_key = f"train/{metric}"
            params[f"val_{metric}"] = unwrap(
                run.summary.get(val_key),
                "max" if "loss" not in metric else "min",
            )
            params[f"train_{metric}"] = unwrap(
                run.summary.get(train_key),
                "max" if "loss" not in metric else "min",
            )

        params["epoch"] = run.summary.get("epoch", 0)
        params["train_loss"] = unwrap(run.summary.get("train/loss"), "min")
        params["val_loss"] = unwrap(run.summary.get("val/loss"), "min")

        # Dataset size info
        params["train_samples"] = run.summary.get(
            "train/samples_per_epoch", None
        )
        params["val_samples"] = run.summary.get("val/samples_per_epoch", None)

        records.append(params)

    return pd.DataFrame(records)


def fetch_training_curves(group: str) -> dict:
    """Fetch epoch-level training curves for all runs in group."""
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group})

    curves = {}
    for run in runs:
        try:
            lr = float(
                run.config.get("hyperparameters", {}).get("learning_rate", 0)
            )
            lr_label = f"{lr:.0e}"
        except (TypeError, ValueError):
            lr_label = run.name

        try:
            df_hist = run.history(
                keys=[
                    "epoch",
                    "train/p300_binary_f1",
                    "val/p300_binary_f1",
                    "train/p300_binary_loss",
                    "val/p300_binary_loss",
                    "train/loss",
                    "val/loss",
                    "train/p300_binary_recall",
                    "val/p300_binary_recall",
                    "train/p300_binary_precision",
                    "val/p300_binary_precision",
                ],
                samples=10000,
            )
            if not df_hist.empty:
                curves[f"lr={lr_label} ({run.id[:8]})"] = df_hist
        except Exception as e:
            print(f"  Warning: could not fetch history for {run.name}: {e}")

    return curves


def print_summary(df_new: pd.DataFrame, df_old_eegnet: pd.DataFrame):
    print("\n" + "=" * 100)
    print("BRAIN INVADERS EEGNET REPROCESSED — RESULTS SUMMARY")
    print("=" * 100)

    # New results
    print(f"\n{'─' * 100}")
    print(f"  REPROCESSED ({len(df_new)} runs)")
    print(f"{'─' * 100}")
    cols = [
        "run_name",
        "lr",
        "train_p300_binary_f1",
        "val_p300_binary_f1",
        "train_p300_binary_recall",
        "val_p300_binary_recall",
        "train_p300_binary_precision",
        "val_p300_binary_precision",
        "epoch",
    ]
    available_cols = [c for c in cols if c in df_new.columns]
    print(
        df_new.sort_values("val_p300_binary_f1", ascending=False)[
            available_cols
        ].to_string(index=False, float_format="%.4f")
    )

    # Compare best old vs new
    print(f"\n{'=' * 100}")
    print("COMPARISON: OLD (pre-reprocessing) vs NEW (reprocessed)")
    print("=" * 100)

    if not df_old_eegnet.empty:
        best_old = df_old_eegnet.sort_values(
            "val_p300_binary_f1", ascending=False
        ).iloc[0]
        best_new = df_new.sort_values(
            "val_p300_binary_f1", ascending=False
        ).iloc[0]

        print(
            f"\n  Best OLD EEGNet: val_f1={best_old.get('val_p300_binary_f1', 'N/A'):.4f}, "
            f"lr={best_old.get('lr', 'N/A')}"
        )
        print(
            f"  Best NEW EEGNet: val_f1={best_new.get('val_p300_binary_f1', 'N/A'):.4f}, "
            f"lr={best_new.get('lr', 'N/A')}"
        )

    # Train vs Val comparison
    print(f"\n{'=' * 100}")
    print("TRAIN vs VAL METRICS (diagnosing underfitting vs overfitting)")
    print("=" * 100)
    for _, row in df_new.sort_values(
        "val_p300_binary_f1", ascending=False
    ).iterrows():
        train_f1 = row.get("train_p300_binary_f1", "N/A")
        val_f1 = row.get("val_p300_binary_f1", "N/A")
        train_loss = row.get("train_loss", "N/A")
        val_loss = row.get("val_loss", "N/A")
        print(f"\n  lr={row.get('lr', 'N/A'):.0e}:")
        print(
            f"    Train F1={train_f1:.4f}  Val F1={val_f1:.4f}  "
            f"Gap={train_f1 - val_f1:.4f}"
            if isinstance(train_f1, float)
            else ""
        )
        print(
            f"    Train Loss={train_loss:.4f}  Val Loss={val_loss:.4f}"
            if isinstance(train_loss, float)
            else ""
        )
        print(
            f"    Train Recall={row.get('train_p300_binary_recall', 'N/A'):.4f}  "
            f"Val Recall={row.get('val_p300_binary_recall', 'N/A'):.4f}"
            if isinstance(row.get("train_p300_binary_recall"), float)
            else ""
        )
        print(f"    Epochs: {row.get('epoch', 'N/A')}")


def plot_training_curves(curves: dict):
    """Plot train/val F1, loss, recall, precision over epochs for each LR."""
    n_runs = len(curves)
    if n_runs == 0:
        print("No training curves to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = [
        ("train/p300_binary_f1", "val/p300_binary_f1", "F1 Score"),
        ("train/p300_binary_loss", "val/p300_binary_loss", "Task Loss"),
        (
            "train/p300_binary_recall",
            "val/p300_binary_recall",
            "Recall (Target)",
        ),
        (
            "train/p300_binary_precision",
            "val/p300_binary_precision",
            "Precision (Target)",
        ),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, n_runs))

    for ax, (train_key, val_key, title) in zip(axes.flat, metrics):
        for (label, df_curve), color in zip(curves.items(), colors):
            epochs = df_curve.get("epoch", pd.Series(range(len(df_curve))))
            if train_key in df_curve.columns:
                train_vals = df_curve[train_key].dropna()
                if not train_vals.empty:
                    ax.plot(
                        epochs.iloc[train_vals.index],
                        train_vals,
                        "-",
                        color=color,
                        alpha=0.5,
                        label=f"{label} (train)",
                    )
            if val_key in df_curve.columns:
                val_vals = df_curve[val_key].dropna()
                if not val_vals.empty:
                    ax.plot(
                        epochs.iloc[val_vals.index],
                        val_vals,
                        "--",
                        color=color,
                        alpha=0.8,
                        label=f"{label} (val)",
                    )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "EEGNet on Reprocessed Brain Invaders P300 — Training Dynamics",
        fontsize=14,
    )
    plt.tight_layout()
    path = FIGURES_DIR / f"{PREFIX}_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


def plot_lr_comparison(df_new: pd.DataFrame):
    """Bar chart of train vs val F1 across learning rates."""
    df_sorted = df_new.sort_values("lr")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # F1 comparison
    ax = axes[0]
    x = np.arange(len(df_sorted))
    width = 0.35
    train_f1 = df_sorted["train_p300_binary_f1"].fillna(0).values
    val_f1 = df_sorted["val_p300_binary_f1"].fillna(0).values
    lr_labels = [f"{lr:.0e}" for lr in df_sorted["lr"]]

    ax.bar(
        x - width / 2,
        train_f1,
        width,
        label="Train F1",
        color="#2196F3",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2, val_f1, width, label="Val F1", color="#FF9800", alpha=0.8
    )

    for i, (t, v) in enumerate(zip(train_f1, val_f1)):
        ax.text(i - width / 2, t + 0.01, f"{t:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(lr_labels)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("F1 Score")
    ax.set_title("Train vs Val F1 by Learning Rate")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)

    # Recall + Precision
    ax = axes[1]
    train_recall = df_sorted["train_p300_binary_recall"].fillna(0).values
    val_recall = df_sorted["val_p300_binary_recall"].fillna(0).values
    train_prec = df_sorted["train_p300_binary_precision"].fillna(0).values
    val_prec = df_sorted["val_p300_binary_precision"].fillna(0).values

    w = 0.2
    ax.bar(
        x - 1.5 * w,
        train_recall,
        w,
        label="Train Recall",
        color="#4CAF50",
        alpha=0.8,
    )
    ax.bar(
        x - 0.5 * w,
        val_recall,
        w,
        label="Val Recall",
        color="#8BC34A",
        alpha=0.8,
    )
    ax.bar(
        x + 0.5 * w,
        train_prec,
        w,
        label="Train Precision",
        color="#9C27B0",
        alpha=0.8,
    )
    ax.bar(
        x + 1.5 * w,
        val_prec,
        w,
        label="Val Precision",
        color="#E1BEE7",
        alpha=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(lr_labels)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Score")
    ax.set_title("Recall & Precision by Learning Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)

    plt.suptitle("EEGNet Reprocessed: LR Comparison", fontsize=14)
    plt.tight_layout()
    path = FIGURES_DIR / f"{PREFIX}_lr_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_old_vs_new(df_new: pd.DataFrame, df_old_eegnet: pd.DataFrame):
    """Compare best old EEGNet vs new reprocessed EEGNet results."""
    if df_old_eegnet.empty:
        print("No old EEGNet data to compare.")
        return

    best_old = df_old_eegnet.sort_values(
        "val_p300_binary_f1", ascending=False
    ).iloc[0]
    best_new = df_new.sort_values("val_p300_binary_f1", ascending=False).iloc[0]

    metrics = [
        "val_p300_binary_f1",
        "val_p300_binary_auroc",
        "val_p300_binary_precision",
        "val_p300_binary_recall",
    ]
    labels = ["F1", "AUROC", "Precision", "Recall"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics))
    width = 0.35

    old_vals = [best_old.get(m, 0) for m in metrics]
    new_vals = [best_new.get(m, 0) for m in metrics]

    ax.bar(
        x - width / 2,
        old_vals,
        width,
        label=f"Old (lr={best_old.get('lr', '?')})",
        color="#F44336",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        new_vals,
        width,
        label=f"New (lr={best_new.get('lr', '?')})",
        color="#4CAF50",
        alpha=0.8,
    )

    for i, (o, n) in enumerate(zip(old_vals, new_vals)):
        ax.text(i - width / 2, o + 0.01, f"{o:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, n + 0.01, f"{n:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("EEGNet: Before vs After Data Reprocessing (Best Runs)")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = FIGURES_DIR / f"{PREFIX}_old_vs_new.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def check_dataset_sizes():
    """Check the actual data files to verify reprocessing."""
    import h5py

    data_dirs = [
        Path.home()
        / ".cache"
        / "torch_brain"
        / "korczowski_brain_invaders_2014a",
        Path(
            "/network/scratch/s/sobralm/processed_data/korczowski_brain_invaders_2014a"
        ),
    ]

    for data_dir in data_dirs:
        if not data_dir.exists():
            continue

        print(f"\n{'=' * 100}")
        print(f"DATA VERIFICATION: {data_dir}")
        print(f"{'=' * 100}")

        h5_files = sorted(data_dir.glob("*.h5"))
        print(f"Found {len(h5_files)} HDF5 files")

        total_trials = 0
        trial_durations = []
        for h5f in h5_files[:5]:
            try:
                with h5py.File(h5f, "r") as f:
                    if "p300_trials" in f:
                        starts = f["p300_trials/start"][:]
                        ends = f["p300_trials/end"][:]
                        n_trials = len(starts)
                        durations = ends - starts
                        total_trials += n_trials
                        trial_durations.extend(durations.tolist())
                        print(f"\n  {h5f.name}: {n_trials} trials")
                        print(
                            f"    Duration range: [{durations.min():.4f}, {durations.max():.4f}]s"
                        )
                        print(f"    Mean duration: {durations.mean():.4f}s")
                        print(
                            f"    Trials >= 1.0s: {(durations >= 1.0).sum()} / {n_trials}"
                        )
                        print(
                            f"    Trials >= 0.5s: {(durations >= 0.5).sum()} / {n_trials}"
                        )
                        print(
                            f"    Trials < 0.5s: {(durations < 0.5).sum()} / {n_trials}"
                        )

                        if "targets" in f["p300_trials"]:
                            targets = f["p300_trials/targets"][:]
                            unique, counts = np.unique(
                                targets, return_counts=True
                            )
                            print(
                                f"    Label distribution: {dict(zip(unique.tolist() if hasattr(unique, 'tolist') else unique, counts.tolist()))}"
                            )

                    if "brainset" in f:
                        if "derived_version" in f["brainset"].attrs:
                            print(
                                f"    Derived version: {f['brainset'].attrs['derived_version']}"
                            )
            except Exception as e:
                print(f"  Error reading {h5f.name}: {e}")

        if trial_durations:
            td = np.array(trial_durations)
            print(f"\n  TOTAL across first 5 files: {total_trials} trials")
            print("  Overall duration stats:")
            print(f"    Mean: {td.mean():.4f}s, Median: {np.median(td):.4f}s")
            print(f"    Min: {td.min():.4f}s, Max: {td.max():.4f}s")
            print(
                f"    Would survive 1.0s window: {(td >= 1.0).sum()} / {len(td)} ({100 * (td >= 1.0).mean():.1f}%)"
            )

        return  # only check first found dir


if __name__ == "__main__":
    # 1) Check data files on disk
    check_dataset_sizes()

    # 2) Fetch new reprocessed runs
    print("\n\nFetching reprocessed runs...")
    df_new = fetch_runs(GROUP_NEW)
    print(f"Fetched {len(df_new)} runs from {GROUP_NEW}")

    # 3) Fetch old EEGNet runs for comparison
    print("Fetching old HP search runs for comparison...")
    df_old = fetch_runs(GROUP_OLD)
    df_old_eegnet = df_old[
        df_old["run_name"].str.contains("eegnet", case=False)
    ]
    print(f"Fetched {len(df_old)} old runs ({len(df_old_eegnet)} EEGNet)")

    # 4) Summary
    print_summary(df_new, df_old_eegnet)

    # 5) Training curves
    print("\nFetching training curves...")
    curves = fetch_training_curves(GROUP_NEW)
    plot_training_curves(curves)

    # 6) LR comparison
    plot_lr_comparison(df_new)

    # 7) Old vs new comparison
    plot_old_vs_new(df_new, df_old_eegnet)

    print(f"\n{'=' * 100}")
    print("DONE. All figures saved to:", FIGURES_DIR)
    print(f"{'=' * 100}")
