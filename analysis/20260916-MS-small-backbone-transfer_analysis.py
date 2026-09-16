"""Initial W&B inventory for the small-backbone transfer experiment."""

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
STEM = "20260916-MS-small-backbone-transfer"
PROJECT = "neurosoft_supervised_pretraining"
GROUPS = (
    "20260916-MS-SMALL-BACKBONE-TRANSFER-MINIPIGS",
    "20260916-MS-SMALL-BACKBONE-TRANSFER-MONKEYS",
    "20260916-MS-SMALL-BACKBONE-SCRATCH-MINIPIGS",
    "20260916-MS-SMALL-BACKBONE-SCRATCH-MONKEYS",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity", default=os.getenv("WANDB_ENTITY", "poyo-eeg")
    )
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--group", action="append", dest="groups")
    args = parser.parse_args()
    groups = tuple(args.groups or GROUPS)
    api = wandb.Api()
    rows = [
        {
            "group": group,
            "run_name": run.name,
            "run_id": run.id,
            "state": run.state,
        }
        for group in groups
        for run in api.runs(
            f"{args.entity}/{args.project}", filters={"group": group}
        )
    ]
    if not rows:
        raise SystemExit(f"No W&B runs found for groups: {groups}")
    frame = pd.DataFrame(rows).sort_values(["group", "run_name"])
    csv_path = ROOT / "analysis" / "csv" / f"{STEM}_run_inventory.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    summary = frame.groupby(["group", "state"]).size().unstack(fill_value=0)
    print(summary.to_string())
    summary.plot.bar(figsize=(10, 5), ylabel="Runs", title="W&B run inventory")
    plt.tight_layout()
    figure = ROOT / "analysis" / "figures" / f"{STEM}_run_inventory.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
