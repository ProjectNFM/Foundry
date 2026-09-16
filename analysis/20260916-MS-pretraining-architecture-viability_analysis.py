"""Initial W&B inventory for the architecture-viability source experiment.

The hypothesis analysis will be added after the immutable source cell lists
exist. This scaffold already fetches the planned groups through ``wandb.Api``
and records the exact run names and IDs rather than hardcoding results.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb


ROOT = Path(__file__).resolve().parents[1]
STEM = "20260916-MS-pretraining-architecture-viability"
PROJECT = "neurosoft_supervised_pretraining"
GROUPS = (
    "20260916-MS-PRETRAINING-ARCHITECTURE-VIABILITY-MINIPIGS",
    "20260916-MS-PRETRAINING-ARCHITECTURE-VIABILITY-MONKEYS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity", default=os.getenv("WANDB_ENTITY", "poyo-eeg")
    )
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--group", action="append", dest="groups")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = tuple(args.groups or GROUPS)
    api = wandb.Api()
    rows: list[dict[str, object]] = []
    for group in groups:
        for run in api.runs(
            f"{args.entity}/{args.project}", filters={"group": group}
        ):
            config = dict(run.config or {})
            rows.append(
                {
                    "group": group,
                    "run_name": run.name,
                    "run_id": run.id,
                    "state": run.state,
                    "source_condition": (
                        config.get("run", {}).get("source_condition")
                        if isinstance(config.get("run"), dict)
                        else config.get("source_condition")
                    ),
                    "source_target_subject": (
                        config.get("run", {}).get("source_target_subject")
                        if isinstance(config.get("run"), dict)
                        else config.get("source_target_subject")
                    ),
                    "seed": config.get("run", {}).get("seed")
                    if isinstance(config.get("run"), dict)
                    else config.get("seed"),
                }
            )
    if not rows:
        raise SystemExit(f"No W&B runs found for groups: {groups}")
    frame = pd.DataFrame(rows).sort_values(["group", "run_name"])
    csv_path = ROOT / "analysis" / "csv" / f"{STEM}_run_inventory.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    summary = frame.groupby(["group", "state"]).size().unstack(fill_value=0)
    print(summary.to_string())
    axis = summary.plot.bar(
        figsize=(10, 5), ylabel="Runs", title="W&B run inventory"
    )
    axis.figure.tight_layout()
    figure = ROOT / "analysis" / "figures" / f"{STEM}_run_inventory.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
