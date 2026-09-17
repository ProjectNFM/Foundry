"""Reproducible source-readiness analysis for the 60-run architecture matrix."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Any
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[1]
STEM = "20260916-MS-pretraining-architecture-viability"
PROJECT = "neurosoft_supervised_pretraining"
TASK = "neurosoft_acoustic_stim_8band"
GROUPS = {
    "minipigs": "20260916-MS-PRETRAINING-ARCHITECTURE-VIABILITY-B128-FIX1-MINIPIGS",
    "monkeys": "20260916-MS-PRETRAINING-ARCHITECTURE-VIABILITY-B128-FIX1-MONKEYS",
}
CONDITIONS = (
    "reference_backbone",
    "small_backbone",
    "large_backbone",
    "bias_free",
    "shared_padded",
)
EXPECTED_RUNS = {"minipigs": 35, "monkeys": 25}
MILESTONES = np.array([100, 300, 1000, 3000, 10000])
VAL_LOSS = "val/loss"
VAL_F1 = f"val/{TASK}_supported_f1"
LABELS = {
    "reference_backbone": "Reference",
    "small_backbone": "Small",
    "large_backbone": "Large",
    "bias_free": "Bias-free",
    "shared_padded": "Shared padded",
}


def nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def scalar(summary: Any, key: str, aggregate: str = "max") -> float:
    value = summary.get(key, summary.get(f"{key}.{aggregate}"))
    if isinstance(value, dict):
        value = value.get(aggregate)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity", default=os.getenv("WANDB_ENTITY", "poyo-eeg")
    )
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cached-inputs",
        action="store_true",
        help="Re-evaluate W&B-derived CSV caches without re-fetching history.",
    )
    return parser.parse_args()


def validation_history(run: Any) -> pd.DataFrame:
    history = run.history(
        keys=["trainer/global_step", VAL_LOSS, VAL_F1],
        samples=20000,
        pandas=True,
    )
    required = {"trainer/global_step", VAL_LOSS, VAL_F1}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"{run.name}: missing {sorted(missing)}")
    history = history.dropna(subset=list(required)).copy()
    history["trainer/global_step"] = pd.to_numeric(
        history["trainer/global_step"]
    )
    return history.sort_values("trainer/global_step")


def fixed_milestones(history: pd.DataFrame, run_name: str) -> pd.DataFrame:
    records = []
    observed = history["trainer/global_step"].to_numpy(dtype=float)
    for milestone in MILESTONES:
        index = int(np.argmin(abs(observed - (milestone - 1))))
        row = history.iloc[index]
        if abs(float(row["trainer/global_step"]) - (milestone - 1)) > 2:
            raise ValueError(
                f"{run_name}: no validation event for fixed step {milestone}"
            )
        records.append(
            {
                "milestone_step": int(milestone),
                "logged_global_step": int(row["trainer/global_step"]),
                "val_loss": float(row[VAL_LOSS]),
                "val_supported_f1": float(row[VAL_F1]),
            }
        )
    return pd.DataFrame(records)


def extract(
    api: Any, entity: str, project: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows, frames, total = [], [], 0
    for species, group in GROUPS.items():
        group_runs = list(
            api.runs(f"{entity}/{project}", filters={"group": group})
        )
        print(f"Fetched {len(group_runs):2d} {species} runs from {group}")
        if len(group_runs) != EXPECTED_RUNS[species]:
            raise RuntimeError(
                f"{species}: expected {EXPECTED_RUNS[species]} runs"
            )
        for run in group_runs:
            total += 1
            config = dict(run.config or {})
            summary = (
                run.summary._json_dict
                if hasattr(run.summary, "_json_dict")
                else dict(run.summary)
            )
            condition = nested(config, "run", "source_condition") or config.get(
                "source_condition"
            )
            subject = nested(
                config, "run", "source_target_subject"
            ) or config.get("source_target_subject")
            milestones = fixed_milestones(validation_history(run), run.name)
            provenance_ok = all(
                bool(config.get(key))
                for key in (
                    "provenance.bundle_id",
                    "provenance.git_sha",
                    "provenance.manifest_path",
                )
            )
            run_rows.append(
                {
                    "species": species,
                    "group": run.group,
                    "run_name": run.name,
                    "run_id": run.id,
                    "state": run.state,
                    "condition": condition,
                    "subject": subject,
                    "source_seed": nested(config, "run", "seed")
                    or config.get("seed"),
                    "optimizer_steps": scalar(
                        summary, "compute/optimizer_steps"
                    ),
                    "total_parameters": scalar(
                        summary, "compute/total_parameters"
                    ),
                    "wall_time_s": scalar(
                        summary, "compute/elapsed_wall_time_s"
                    ),
                    "peak_memory_gb": scalar(
                        summary, "compute/peak_memory_allocated_gb"
                    ),
                    "provenance_config_complete": provenance_ok,
                }
            )
            (
                milestones["species"],
                milestones["condition"],
                milestones["subject"],
                milestones["run_id"],
            ) = species, condition, subject, run.id
            frames.append(milestones)
            print(f"  [{total:02d}/60] {run.name}")
    return pd.DataFrame(run_rows), pd.concat(frames, ignore_index=True)


def bootstrap_slope(
    values: pd.DataFrame, metric: str, replicates: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    matrix = (
        values.pivot(index="subject", columns="milestone_step", values=metric)
        .reindex(columns=MILESTONES)
        .to_numpy()
    )
    indices = rng.integers(0, len(matrix), size=(replicates, len(matrix)))
    trajectories = matrix[indices].mean(axis=1)
    x = np.log(MILESTONES)
    centered_x = x - x.mean()
    slopes = trajectories @ centered_x / np.dot(centered_x, centered_x)
    return (
        float(np.mean(slopes)),
        float(np.quantile(slopes, 0.025)),
        float(np.quantile(slopes, 0.975)),
    )


def summarize(
    milestones: pd.DataFrame, runs: pd.DataFrame, reps: int, seed: int
) -> pd.DataFrame:
    rng, rows = np.random.default_rng(seed), []
    for (species, condition), values in milestones.groupby(
        ["species", "condition"], sort=True
    ):
        values = values.sort_values(["subject", "milestone_step"])
        group = runs[(runs.species == species) & (runs.condition == condition)]
        loss_slope, loss_low, loss_high = bootstrap_slope(
            values, "val_loss", reps, rng
        )
        f1_slope, f1_low, f1_high = bootstrap_slope(
            values, "val_supported_f1", reps, rng
        )
        early, late = (
            values[values.milestone_step == 100],
            values[values.milestone_step == 10000],
        )
        finite = bool(
            np.isfinite(
                values[["val_loss", "val_supported_f1"]].to_numpy()
            ).all()
        )
        complete = (
            len(group) == (7 if species == "minipigs" else 5)
            and set(group.state) == {"finished"}
            and (group.optimizer_steps >= 10000).all()
            and group.provenance_config_complete.all()
        )
        # The experiment's viability criterion is non-degenerate source-task
        # learning.  Validation CE describes the onset/severity of fitting and
        # overfitting; it is deliberately not a monotonic gate criterion.
        improving = late.val_supported_f1.mean() > early.val_supported_f1.mean()
        nontrivial = late.val_supported_f1.mean() > 0.05
        readiness = (
            "Ready"
            if complete and finite and improving and nontrivial
            else "Ready with caveat"
            if complete and finite and nontrivial
            else "Not ready"
        )
        rows.append(
            {
                "species": species,
                "condition": condition,
                "n_subjects": len(group),
                "complete_finite_provenance": complete and finite,
                "val_loss_step100_mean": early.val_loss.mean(),
                "val_loss_step10000_mean": late.val_loss.mean(),
                "val_loss_change": late.val_loss.mean() - early.val_loss.mean(),
                "val_loss_slope_log_step": loss_slope,
                "val_loss_slope_ci_low": loss_low,
                "val_loss_slope_ci_high": loss_high,
                "val_f1_step100_mean": early.val_supported_f1.mean(),
                "val_f1_step10000_mean": late.val_supported_f1.mean(),
                "val_f1_change": late.val_supported_f1.mean()
                - early.val_supported_f1.mean(),
                "val_f1_slope_log_step": f1_slope,
                "val_f1_slope_ci_low": f1_low,
                "val_f1_slope_ci_high": f1_high,
                "prediction_count_audit": "requires checkpoint/output audit",
                "readiness": readiness,
            }
        )
    return pd.DataFrame(rows).sort_values(["species", "condition"])


def figures(
    milestones: pd.DataFrame, runs: pd.DataFrame, directory: Path
) -> None:
    palette = dict(zip(CONDITIONS, plt.get_cmap("tab10").colors))
    for metric, ylabel, suffix in (
        (
            "val_loss",
            "Subject-balanced validation cross-entropy",
            "val_loss_trajectories",
        ),
        (
            "val_supported_f1",
            "Subject-balanced validation supported macro-F1",
            "val_supported_f1_trajectories",
        ),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
        for axis, species in zip(axes, ("minipigs", "monkeys"), strict=True):
            for condition in CONDITIONS:
                values = milestones[
                    (milestones.species == species)
                    & (milestones.condition == condition)
                ]
                grouped = values.groupby("milestone_step")[metric]
                mean, sem = grouped.mean(), grouped.sem().fillna(0)
                axis.plot(
                    mean.index,
                    mean,
                    marker="o",
                    label=LABELS[condition],
                    color=palette[condition],
                )
                axis.fill_between(
                    mean.index,
                    mean - sem,
                    mean + sem,
                    color=palette[condition],
                    alpha=0.16,
                )
            axis.set_xscale("log")
            axis.set_xticks(MILESTONES)
            axis.set_xticklabels([str(x) for x in MILESTONES])
            axis.set(
                title=species.title(),
                xlabel="Fixed source checkpoint step",
                ylabel=ylabel,
            )
            axis.grid(alpha=0.25)
        axes[1].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(directory / f"{STEM}_{suffix}.png", dpi=180)
        plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for axis, metric, ylabel in zip(
        axes,
        ("wall_time_s", "peak_memory_gb"),
        ("Wall time (s)", "Peak allocated GPU memory (GiB)"),
        strict=True,
    ):
        runs.pivot_table(
            index="condition", columns="species", values=metric, aggfunc="mean"
        ).reindex(CONDITIONS).rename(index=LABELS).plot.bar(ax=axis)
        axis.set(ylabel=ylabel, xlabel="", title=ylabel)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(directory / f"{STEM}_compute.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    csv_dir, figure_dir = (
        ROOT / "analysis" / "csv",
        ROOT / "analysis" / "figures",
    )
    csv_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if args.cached_inputs:
        runs = pd.read_csv(csv_dir / f"{STEM}_run_inventory.csv")
        milestones = pd.read_csv(csv_dir / f"{STEM}_fixed_milestones.csv")
        if len(runs) != 60 or len(milestones) != 300:
            raise RuntimeError(
                "Cached W&B inputs do not contain the expected 60 runs / 300 milestones"
            )
        print(
            "Re-evaluating cached W&B inputs (60 runs, 300 fixed milestones)."
        )
    else:
        runs, milestones = extract(wandb.Api(), args.entity, args.project)
    summary = summarize(milestones, runs, args.bootstrap_replicates, args.seed)
    runs.to_csv(csv_dir / f"{STEM}_run_inventory.csv", index=False)
    milestones.to_csv(csv_dir / f"{STEM}_fixed_milestones.csv", index=False)
    summary.to_csv(csv_dir / f"{STEM}_readiness_summary.csv", index=False)
    figures(milestones, runs, figure_dir)
    print("\n=== Architecture-viability readiness summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFigures:")
    for path in sorted(figure_dir.glob(f"{STEM}_*.png")):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
