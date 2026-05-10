"""Shared utilities for W&B experiment analysis scripts.

Provides W&B fetching, data processing, and plotting primitives that individual
per-experiment scripts build on.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# ---------------------------------------------------------------------------
# Matplotlib defaults (applied on import)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.figsize": (7, 4),
    }
)

# ---------------------------------------------------------------------------
# W&B project coordinates
# ---------------------------------------------------------------------------
WANDB_ENTITY = "poyo-eeg"
WANDB_PROJECT = "foundry"

# ---------------------------------------------------------------------------
# Metric constants
# ---------------------------------------------------------------------------
BEHAVIOR_METRIC = "val/ajile_active_behavior_auroc"
POSE_METRIC = "val/ajile_pose_estimation_r2"

HISTORY_KEYS_BEHAVIOR = [
    "val/ajile_active_behavior_auroc",
    "val/ajile_active_behavior_f1",
    "val/ajile_active_behavior_acc",
    "train/loss",
    "val/loss",
]
HISTORY_KEYS_POSE = [
    "val/ajile_pose_estimation_r2",
    "val/ajile_pose_estimation_mae",
    "train/loss",
    "val/loss",
]

# ---------------------------------------------------------------------------
# Experiment group definitions
# ---------------------------------------------------------------------------


@dataclass
class ExperimentGroup:
    name: str
    task: str  # "behavior" or "pose"
    description: str
    primary_metric: str
    metric_direction: str  # "max" or "min"
    history_keys: list[str] = field(default_factory=list)


EXPERIMENT_GROUPS: dict[str, ExperimentGroup] = {
    "CWT_VS_CNN_BEHAVIOR": ExperimentGroup(
        name="CWT_VS_CNN_BEHAVIOR",
        task="behavior",
        description="CWT vs ResampleCNN (dim256 & dim512) — Behavior",
        primary_metric=BEHAVIOR_METRIC,
        metric_direction="max",
        history_keys=HISTORY_KEYS_BEHAVIOR,
    ),
    "CWT_VS_CNN_POSE": ExperimentGroup(
        name="CWT_VS_CNN_POSE",
        task="pose",
        description="CWT vs ResampleCNN — Pose Estimation",
        primary_metric=POSE_METRIC,
        metric_direction="max",
        history_keys=HISTORY_KEYS_POSE,
    ),
    "AJILE12_TOKENIZER_ABLATION": ExperimentGroup(
        name="AJILE12_TOKENIZER_ABLATION",
        task="behavior",
        description="Full tokenizer ablation — Behavior",
        primary_metric=BEHAVIOR_METRIC,
        metric_direction="max",
        history_keys=HISTORY_KEYS_BEHAVIOR,
    ),
    "AJILE12_TOKENIZER_ABLATION_POSE": ExperimentGroup(
        name="AJILE12_TOKENIZER_ABLATION_POSE",
        task="pose",
        description="Full tokenizer ablation — Pose Estimation",
        primary_metric=POSE_METRIC,
        metric_direction="max",
        history_keys=HISTORY_KEYS_POSE,
    ),
    "AJILE_NEW_ALLTOKS": ExperimentGroup(
        name="AJILE_NEW_ALLTOKS",
        task="behavior",
        description="Extended tokenizer sweep (lower LR) — Behavior",
        primary_metric=BEHAVIOR_METRIC,
        metric_direction="max",
        history_keys=HISTORY_KEYS_BEHAVIOR,
    ),
}

# ---------------------------------------------------------------------------
# Display-name mapping for tokenizer configs
# ---------------------------------------------------------------------------
TOKENIZER_DISPLAY_NAMES = {
    "per_channel_cwt": "Per-Ch CWT",
    "per_channel_resample_cnn": "Per-Ch CNN",
    "per_channel_per_timepoint_linear": "Per-Ch Linear",
    "per_channel_cwt_add": "Per-Ch CWT (add)",
    "per_channel_resample_cnn_add": "Per-Ch CNN (add)",
    "per_channel_cwt_dim512": "Per-Ch CWT (512)",
    "per_channel_resample_cnn_dim512": "Per-Ch CNN (512)",
    "per_channel_per_timepoint_linear_dim512": "Per-Ch Linear (512)",
    "spatial_session_cwt": "Spatial CWT",
    "spatial_session_resample_cnn": "Spatial CNN",
    "spatial_session_per_timepoint_linear": "Spatial Linear",
    "spatial_session_per_timepoint_identity": "Spatial Identity",
    "spatial_session_cwt_common": "Spatial CWT (common)",
    "spatial_session_cwt_dim512": "Spatial CWT (512)",
    "spatial_session_resample_cnn_dim512": "Spatial CNN (512)",
    "spatial_session_mlp_cwt": "Spatial MLP-CWT",
    "spatial_session_mlp_resample_cnn": "Spatial MLP-CNN",
    "spatial_session_mlp_per_timepoint_identity": "Spatial MLP-Identity",
    "per_channel_resample_cnn_64f": "Per-Ch CNN (64f)",
}

# ---------------------------------------------------------------------------
# Unified color palette — Okabe-Ito colorblind-friendly
# Keyed by temporal embedding family, inferred from tokenizer_label strings
# ---------------------------------------------------------------------------
PALETTE_TEMPORAL = {
    "cwt": "#0072B2",  # blue
    "cnn": "#D55E00",  # vermillion
    "linear": "#009E73",  # bluish green
    "identity": "#CC79A7",  # reddish purple
}

# Legacy palette (kept for backward compatibility)
PALETTE = {
    "CWTEmbedding": PALETTE_TEMPORAL["cwt"],
    "ResampleCNNEmbedding": PALETTE_TEMPORAL["cnn"],
    "PerTimepointLinearEmbedding": PALETTE_TEMPORAL["linear"],
    "PerTimepointIdentityEmbedding": PALETTE_TEMPORAL["identity"],
}


def get_color(label: str) -> str:
    """Return a color for a tokenizer label by matching its temporal family.

    Works on tokenizer_label strings (e.g. 'Per-Ch CWT', 'Spatial CNN')
    or tokenizer keys (e.g. 'per_channel_cwt', 'spatial_session_resample_cnn').
    """
    s = label.lower()
    if "cwt" in s:
        return PALETTE_TEMPORAL["cwt"]
    if "cnn" in s or "resample" in s:
        return PALETTE_TEMPORAL["cnn"]
    if "identity" in s:
        return PALETTE_TEMPORAL["identity"]
    if "linear" in s:
        return PALETTE_TEMPORAL["linear"]
    return "#888888"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_tokenizer_name(display_name: str) -> str:
    """Extract tokenizer key from a run's displayName."""
    name = display_name
    for prefix in ["ajile_behavior_", "ajile_pose_estimation_"]:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return re.sub(r"_fold\d+$", "", name)


def parse_fold(display_name: str) -> int:
    match = re.search(r"fold(\d+)", display_name)
    return int(match.group(1)) if match else -1


def extract_best_metric(
    summary_json: str, metric: str, direction: str
) -> float | None:
    """Pull the best value of a metric from W&B summaryMetrics JSON."""
    summary = json.loads(summary_json)
    val = summary.get(metric)
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get("max" if direction == "max" else "min")
    return float(val)


def extract_temporal_embedding(config_json: str) -> str:
    config = json.loads(config_json)
    model = config.get("model", {}).get("value", {})
    tokenizer = model.get("tokenizer", {})
    temporal = tokenizer.get("temporal_embedding", {})
    target = temporal.get("_target_", "unknown")
    return target.rsplit(".", 1)[-1]


def extract_channel_strategy(config_json: str) -> str:
    config = json.loads(config_json)
    model = config.get("model", {}).get("value", {})
    tokenizer = model.get("tokenizer", {})
    strategy = tokenizer.get(
        "channel_strategy", tokenizer.get("channel_fusion", {})
    )
    if isinstance(strategy, dict):
        target = strategy.get("_target_", "unknown")
        return target.rsplit(".", 1)[-1]
    return str(strategy)


def extract_embed_dim(config_json: str) -> int:
    config = json.loads(config_json)
    model = config.get("model", {}).get("value", {})
    return model.get("embed_dim", 256)


# ---------------------------------------------------------------------------
# W&B data fetching
# ---------------------------------------------------------------------------


def fetch_group_runs(api: wandb.Api, group_name: str) -> list[dict]:
    """Fetch all runs for a W&B group."""
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"group": group_name},
    )
    return [
        {
            "id": run.id,
            "name": run.name,
            "display_name": run.display_name,
            "state": run.state,
            "group": group_name,
            "summary_metrics": json.dumps(run.summary._json_dict),
            "config": json.dumps(run.config),
        }
        for run in runs
    ]


def build_summary_dataframe(
    runs: list[dict], group: ExperimentGroup
) -> pd.DataFrame:
    """Build a tidy dataframe from fetched runs."""
    rows = []
    for run in runs:
        if run["state"] not in ("finished", "running"):
            continue
        best_val = extract_best_metric(
            run["summary_metrics"], group.primary_metric, group.metric_direction
        )
        if best_val is None:
            continue
        tokenizer = parse_tokenizer_name(run["display_name"])
        rows.append(
            {
                "run_id": run["id"],
                "display_name": run["display_name"],
                "group": run["group"],
                "tokenizer": tokenizer,
                "tokenizer_label": TOKENIZER_DISPLAY_NAMES.get(
                    tokenizer, tokenizer
                ),
                "fold": parse_fold(run["display_name"]),
                "temporal_embedding": extract_temporal_embedding(run["config"]),
                "channel_strategy": extract_channel_strategy(run["config"]),
                "embed_dim": extract_embed_dim(run["config"]),
                "best_metric": best_val,
                "metric_name": group.primary_metric,
                "state": run["state"],
            }
        )
    return pd.DataFrame(rows)


def fetch_training_curves(
    api: wandb.Api, run_id: str, keys: list[str], samples: int = 500
) -> pd.DataFrame:
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    return run.history(keys=keys + ["epoch", "_step"], samples=samples)


# ---------------------------------------------------------------------------
# Fetching / caching pipeline (used by each per-experiment script)
# ---------------------------------------------------------------------------


def load_or_fetch_group(
    group: ExperimentGroup,
    api: wandb.Api | None,
    cache_dir: Path,
    *,
    plot_only: bool = False,
) -> pd.DataFrame | None:
    """Load cached CSV or fetch from W&B.  Returns None if unavailable."""
    cache_path = cache_dir / f"{group.name}.csv"

    print(f"\n{'=' * 60}")
    print(f"Group: {group.name} — {group.description}")
    print(f"{'=' * 60}")

    if plot_only and cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"  Loaded {len(df)} runs from cache")
        return df

    if api is None:
        print("  Skipping (no API and no cache)")
        return None

    print("  Fetching runs from W&B...")
    raw_runs = fetch_group_runs(api, group.name)
    print(f"  Found {len(raw_runs)} runs")

    df = build_summary_dataframe(raw_runs, group)
    df.to_csv(cache_path, index=False)
    print(f"  Saved {len(df)} processed runs to {cache_path}")

    raw_path = cache_dir / f"{group.name}_raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw_runs, f, indent=2)

    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_group_bar_chart(
    df: pd.DataFrame,
    group: ExperimentGroup,
    output_dir: Path,
) -> Path:
    """Grouped bar chart: mean +/- std of primary metric per tokenizer."""
    agg = (
        df.groupby("tokenizer_label")["best_metric"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=group.metric_direction != "max")
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)

    colors = [get_color(label) for label in agg["tokenizer_label"]]

    fig, ax = plt.subplots(figsize=(max(7, len(agg) * 0.9), 4.5))
    x = np.arange(len(agg))
    bars = ax.bar(
        x,
        agg["mean"],
        yerr=agg["std"],
        capsize=4,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, (_, row) in zip(bars, agg.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + row["std"] + 0.005,
            f"{row['mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    metric_label = group.primary_metric.split("/")[-1].replace("_", " ").title()
    ax.set_ylabel(metric_label)
    ax.set_title(f"{group.description}\n(mean ± std over folds)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        agg["tokenizer_label"], rotation=35, ha="right", fontsize=8
    )

    seen = set()
    legend_handles = []
    for name, key in [
        ("CWT", "cwt"),
        ("CNN", "cnn"),
        ("Linear", "linear"),
        ("Identity", "identity"),
    ]:
        c = PALETTE_TEMPORAL[key]
        if c not in seen and any(
            get_color(label) == c for label in agg["tokenizer_label"]
        ):
            legend_handles.append(mpatches.Patch(color=c, label=name))
            seen.add(c)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fname = f"{group.name.lower()}_bar.pdf"
    path = output_dir / fname
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_training_curves(
    api: wandb.Api,
    df: pd.DataFrame,
    group: ExperimentGroup,
    output_dir: Path,
    max_runs: int = 8,
) -> Path | None:
    """Training curves (val metric vs epoch) for each tokenizer, fold 0."""
    subset = df[df["fold"] == 0].head(max_runs)
    if subset.empty:
        return None

    metric_key = group.primary_metric
    fig, ax = plt.subplots(figsize=(7, 4))

    for _, row in subset.iterrows():
        try:
            hist = fetch_training_curves(api, row["run_id"], [metric_key])
        except Exception as e:
            print(
                f"  Warning: could not fetch history for {row['display_name']}: {e}"
            )
            continue
        if hist.empty or metric_key not in hist.columns:
            continue

        color = get_color(row["tokenizer_label"])
        ax.plot(
            hist["epoch"] if "epoch" in hist.columns else hist.index,
            hist[metric_key],
            label=row["tokenizer_label"],
            color=color,
            alpha=0.8,
        )

    metric_label = metric_key.split("/")[-1].replace("_", " ").title()
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{group.description} — Training Curves (fold 0)")
    ax.legend(fontsize=7, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fname = f"{group.name.lower()}_curves.pdf"
    path = output_dir / fname
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def generate_summary_table(
    all_dfs: dict[str, pd.DataFrame], output_dir: Path
) -> Path:
    """Generate a CSV summary table across all groups."""
    rows = []
    for group_name, df in all_dfs.items():
        group = EXPERIMENT_GROUPS[group_name]
        agg = df.groupby("tokenizer_label")["best_metric"].agg(
            ["mean", "std", "count"]
        )
        agg["std"] = agg["std"].fillna(0)
        for tok, row in agg.iterrows():
            rows.append(
                {
                    "group": group_name,
                    "task": group.task,
                    "tokenizer": tok,
                    "metric": group.primary_metric,
                    "mean": round(row["mean"], 4),
                    "std": round(row["std"], 4),
                    "n_folds": int(row["count"]),
                }
            )
    summary = pd.DataFrame(rows)
    path = output_dir / "summary_table.csv"
    summary.to_csv(path, index=False)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Shared argparse builder
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("docs/figures")
DEFAULT_CACHE_DIR = Path("docs/figures/data")


def make_base_parser(description: str) -> argparse.ArgumentParser:
    """Create an ArgumentParser with the standard --output-dir / --cache-dir / --plot-only / --no-curves flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated figures (default: docs/figures)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for cached W&B data (default: docs/figures/data)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reuse cached CSVs instead of fetching from W&B",
    )
    parser.add_argument(
        "--no-curves",
        action="store_true",
        help="Skip training curve plots (faster)",
    )
    return parser


def setup_dirs_and_api(args: argparse.Namespace) -> wandb.Api | None:
    """Create output/cache directories and return a W&B API (or None if --plot-only)."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    return None if args.plot_only else wandb.Api()
