"""Analyze leak-fixed iEEG pretraining and the subsequent Neurosoft benchmark.

Fetches two MAE pretraining runs from WandB and, optionally, their future
Neurosoft 8-band acoustic-stimulus benchmark. The benchmark group must contain
run names with one of these condition tokens: ``kochi_only``, ``kochi_b2``, or
``no_pretrain``. This keeps the analysis independent of Foundry internals and
avoids hard-coded metrics.

Usage:
    uv run python analysis/038_ieeg_leak_fixed_pretraining.py
    uv run python analysis/038_ieeg_leak_fixed_pretraining.py \\
        --neurosoft-group NEUROSOFT_PRETRAIN_COMPARISON
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb


FIGURES_DIR = Path(__file__).resolve().parent / "figures"
PRETRAIN_PROJECT = "foundry_pretraining"
PRETRAIN_GROUP = "IEEG_LEAK_FIXED_PRETRAIN"
PRETRAIN_RUNS = {
    "kochi_only": "pretrain_ieeg_kochi_fixed",
    "kochi_b2": "pretrain_ieeg_kochi_b2_fixed",
}
NEUROSOFT_PROJECT = "auditory_decoding"
NEUROSOFT_METRIC = "val/neurosoft_acoustic_stim_8band_f1"
CONDITION_TOKENS = ("kochi_only", "kochi_b2", "no_pretrain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--neurosoft-group",
        help="Optional WandB group containing the three Neurosoft conditions.",
    )
    return parser.parse_args()


def fetch_pretraining(api: wandb.Api) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch reconstruction-loss histories and a best-loss summary."""
    entity = api.default_entity
    history_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    for condition, name in PRETRAIN_RUNS.items():
        runs = list(
            api.runs(
                f"{entity}/{PRETRAIN_PROJECT}",
                filters={"group": PRETRAIN_GROUP, "display_name": name},
            )
        )
        if not runs:
            print(f"[WARN] No pretraining run found: {name}")
            continue

        run = runs[0]
        history = run.history(keys=["val/loss"], samples=50_000, pandas=True)
        losses = history.get("val/loss", pd.Series(dtype=float)).dropna()
        summary_records.append(
            {
                "condition": condition,
                "run_name": run.name,
                "run_id": run.id,
                "state": run.state,
                "best_val_loss": losses.min() if not losses.empty else None,
                "last_step": run.summary.get("_step"),
            }
        )
        for _, row in history.dropna(subset=["val/loss"]).iterrows():
            history_records.append(
                {
                    "condition": condition,
                    "run_name": run.name,
                    "step": row.get("_step"),
                    "val_loss": row["val/loss"],
                }
            )

    return pd.DataFrame(history_records), pd.DataFrame(summary_records)


def plot_pretraining(history: pd.DataFrame) -> Path | None:
    """Save the pretraining validation-loss curves, when runs are available."""
    if history.empty:
        return None

    FIGURES_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition, subset in history.groupby("condition"):
        subset = subset.sort_values("step")
        ax.plot(subset["step"], subset["val_loss"], label=condition, linewidth=1.8)
    ax.set(xlabel="Step", ylabel="Validation reconstruction loss")
    ax.set_title("Leak-fixed iEEG pretraining")
    ax.legend(title="Source configuration")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = FIGURES_DIR / "038_ieeg_leak_fixed_pretraining_loss.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def condition_from_name(run_name: str) -> str | None:
    """Map future benchmark names to the documented initialization conditions."""
    lowered = run_name.lower()
    return next((token for token in CONDITION_TOKENS if token in lowered), None)


def fetch_neurosoft(api: wandb.Api, group: str) -> pd.DataFrame:
    """Fetch best Neurosoft F1 per run from the supplied benchmark group."""
    entity = api.default_entity
    records: list[dict[str, object]] = []
    for run in api.runs(
        f"{entity}/{NEUROSOFT_PROJECT}", filters={"group": group}
    ):
        condition = condition_from_name(run.name)
        if condition is None:
            continue
        history = run.history(keys=[NEUROSOFT_METRIC], samples=50_000, pandas=True)
        scores = history.get(NEUROSOFT_METRIC, pd.Series(dtype=float)).dropna()
        if scores.empty:
            continue
        records.append(
            {
                "condition": condition,
                "run_name": run.name,
                "run_id": run.id,
                "best_f1": scores.max(),
                "state": run.state,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    api = wandb.Api()
    history, pretrain_summary = fetch_pretraining(api)

    print("\nPretraining summary")
    print(pretrain_summary.to_string(index=False) if not pretrain_summary.empty else "No runs found.")
    figure = plot_pretraining(history)
    if figure:
        print(f"Saved {figure}")

    if not args.neurosoft_group:
        return

    neurosoft = fetch_neurosoft(api, args.neurosoft_group)
    print("\nNeurosoft benchmark summary")
    if neurosoft.empty:
        print("No matching Neurosoft runs found.")
        return
    summary = neurosoft.groupby("condition")["best_f1"].agg(["mean", "std", "count"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
