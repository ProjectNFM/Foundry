"""Analyze the Phase 4D transfer recipe and LR screen from W&B.

The script uses the immutable Phase-4D cell lists as the expected design,
audits each W&B run against its compiled provenance, fetches validation
histories for the time-to-90%-of-peak endpoint, and reports matched
subject-balanced transfer-minus-scratch comparisons.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import wandb

from _wandb_utils import (
    csv_dir,
    default_entity,
    figures_dir,
    unwrap_summary_value,
)


ROOT = Path(__file__).resolve().parents[1]
PROJECT = "neurosoft_supervised_pretraining"
TASK = "neurosoft_acoustic_stim_8band"
VALIDATION_F1 = f"val/{TASK}_supported_f1"
TEST_F1 = f"test/{TASK}_supported_f1"
PREFIX = "20260911-MS-transfer-lr-warmup-screen"
CONDITION_ORDER = [
    "transfer_uniform",
    "transfer_discriminative",
    "transfer_adapter_warmup_uniform",
    "transfer_adapter_warmup_discriminative",
    "scratch_uniform",
    "scratch_adapter_warmup",
]
TRANSFER_CONDITIONS = CONDITION_ORDER[:4]
LR_ORDER = [0.0003, 0.0015, 0.003]
CONDITION_LABELS = {
    "transfer_uniform": "Transfer\nuniform",
    "transfer_discriminative": "Transfer\ndiscriminative",
    "transfer_adapter_warmup_uniform": "Transfer + adapter warmup\nuniform",
    "transfer_adapter_warmup_discriminative": (
        "Transfer + adapter warmup\ndiscriminative"
    ),
    "scratch_uniform": "Scratch\nuniform",
    "scratch_adapter_warmup": "Scratch + adapter warmup",
}


def nested(value: dict[str, Any], *keys: str) -> Any:
    dotted = ".".join(keys)
    if dotted in value:
        return value[dotted]
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def scalar(value: Any, unwrap_key: str = "max") -> float | None:
    if value is None:
        return None
    try:
        value = unwrap_summary_value(value, unwrap_key)
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def summary_scalar(
    summary: Any, key: str, unwrap_key: str = "max"
) -> float | None:
    """Read a W&B summary metric stored as ``metric.max``/``metric.min``."""
    for candidate in (f"{key}.{unwrap_key}", key):
        try:
            value = summary.get(candidate)
        except AttributeError:
            value = None
        result = scalar(value, unwrap_key)
        if result is not None:
            return result
    return None


def load_cells() -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    for species in ("minipigs", "monkeys"):
        path = (
            ROOT
            / "launch/phase4d"
            / f"phase4d-transfer-lr-warmup-{species}.jsonl"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing compiled cell list {path}; compile Phase 4D first"
            )
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["cell_id"] in cells:
                raise RuntimeError(f"Duplicate Phase-4D cell: {row['cell_id']}")
            cells[str(row["cell_id"])] = row
    return cells


def validate_run(run: Any, expected: dict[str, Any]) -> dict[str, Any]:
    config = dict(run.config or {})
    observed_cell = nested(config, "run", "cell_id")
    if observed_cell != expected["cell_id"]:
        raise RuntimeError(
            f"{run.id}: cell mismatch: observed={observed_cell!r}, "
            f"expected={expected['cell_id']!r}"
        )
    observed_lr = scalar(
        nested(config, "hyperparameters", "learning_rate"), "max"
    )
    if observed_lr is None or not np.isclose(
        observed_lr, expected["base_learning_rate"]
    ):
        raise RuntimeError(f"{run.id}: base LR mismatch")
    observed_regime = nested(config, "run", "pretrained_transfer_regime")
    if observed_regime != expected.get("transfer_regime"):
        raise RuntimeError(f"{run.id}: transfer regime mismatch")
    observed_warmup = nested(config, "hyperparameters", "adapter_warmup_steps")
    if int(observed_warmup or 0) != int(expected["adapter_warmup_steps"]):
        raise RuntimeError(f"{run.id}: adapter warmup mismatch")
    return {
        "condition": expected["condition_id"],
        "species": expected["species"],
        "subject": expected["target_subject"],
        "recording": expected["target_recording"],
        "base_lr": expected["base_learning_rate"],
        "adapter_warmup_steps": expected["adapter_warmup_steps"],
        "source": "pretrained" if expected["checkpoint_id"] else "scratch",
        "cell_id": expected["cell_id"],
        "run_id": str(run.id),
        "run_name": str(run.name or ""),
        "state": str(run.state),
        "test_supported_f1": summary_scalar(run.summary or {}, TEST_F1),
        "best_val_supported_f1": summary_scalar(
            run.summary or {}, VALIDATION_F1
        ),
    }


def stable_endpoint(history: pd.DataFrame) -> dict[str, Any]:
    step_column = (
        "trainer/global_step" if "trainer/global_step" in history else "_step"
    )
    if VALIDATION_F1 not in history or step_column not in history:
        raise RuntimeError(
            "validation history lacks optimizer step and supported F1"
        )
    values = history[[step_column, VALIDATION_F1]].dropna().copy()
    values = values.sort_values(step_column).drop_duplicates(
        step_column, keep="last"
    )
    if len(values) < 3:
        raise RuntimeError("need at least three validation evaluations")
    values["smoothed_f1"] = (
        values[VALIDATION_F1].rolling(3, min_periods=3).median()
    )
    values = values.dropna(subset=["smoothed_f1"]).reset_index(drop=True)
    threshold = 0.9 * float(values["smoothed_f1"].max())
    stable = values["smoothed_f1"].ge(threshold)
    crossing = (
        stable
        & stable.shift(-1, fill_value=False)
        & stable.shift(-2, fill_value=False)
    )
    if crossing.any():
        row = values.loc[crossing.idxmax()]
        return {"stable_step": float(row[step_column]), "censored": False}
    return {
        "stable_step": float(values[step_column].iloc[-1]),
        "censored": True,
    }


def fetch_one(
    api: Any, entity: str, expected: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        run = api.run(f"{entity}/{PROJECT}/{expected['wandb_run_id']}")
        record = validate_run(run, expected)
    except Exception as exc:
        record = {
            "condition": expected["condition_id"],
            "species": expected["species"],
            "subject": expected["target_subject"],
            "recording": expected["target_recording"],
            "base_lr": expected["base_learning_rate"],
            "adapter_warmup_steps": expected["adapter_warmup_steps"],
            "source": "pretrained" if expected["checkpoint_id"] else "scratch",
            "cell_id": expected["cell_id"],
            "run_id": str(expected["wandb_run_id"]),
            "run_name": str(expected["run_name"]),
            "state": "fetch_error",
            "test_supported_f1": None,
            "best_val_supported_f1": None,
            "analysis_eligible": False,
            "analysis_error": str(exc),
        }
        return record, expected

    record["analysis_eligible"] = False
    record["analysis_error"] = "missing test supported macro-F1 summary"
    if record["test_supported_f1"] is None:
        return record, expected

    try:
        history = run.history(
            keys=["_step", "trainer/global_step", VALIDATION_F1],
            samples=10_000,
            pandas=True,
        )
        endpoint = stable_endpoint(history)
        record.update(endpoint)
        record["analysis_eligible"] = True
        record["analysis_error"] = ""
    except Exception as exc:
        record["analysis_error"] = str(exc)
    return record, expected


def bootstrap_interval(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(
        axis=1
    )
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def format_lr(value: float) -> str:
    return f"{value:g}"


def plot_paired_forest(summary: pd.DataFrame) -> None:
    """Plot readable horizontal paired estimates with one facet per species."""
    metric_specs = [
        (
            "f1_delta",
            "Transfer minus matched scratch F1",
            "supported macro-F1",
        ),
        (
            "stable_step_saved",
            "Steps saved versus matched scratch",
            "optimizer steps",
        ),
    ]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 14),
        sharex="col",
        constrained_layout=True,
    )
    for column, (metric, title, xlabel) in enumerate(metric_specs):
        plotted = summary[summary.metric.eq(metric)].copy()
        limit = float(
            np.nanmax(
                np.abs(plotted[["ci_low", "ci_high"]].to_numpy(dtype=float))
            )
            * 1.12
        )
        for row, species in enumerate(("minipigs", "monkeys")):
            axis = axes[row, column]
            species_data = plotted[plotted.species.eq(species)].copy()
            species_data["condition_order"] = species_data.condition.map(
                {
                    condition: index
                    for index, condition in enumerate(TRANSFER_CONDITIONS)
                }
            )
            species_data["lr_order"] = species_data.base_lr.map(
                {lr: index for index, lr in enumerate(LR_ORDER)}
            )
            species_data = species_data.sort_values(
                ["condition_order", "lr_order"]
            )
            positions = np.arange(len(species_data))
            axis.errorbar(
                species_data["mean"],
                positions,
                xerr=[
                    species_data["mean"] - species_data["ci_low"],
                    species_data["ci_high"] - species_data["mean"],
                ],
                fmt="o",
                color="#2878b5",
                ecolor="#2878b5",
                capsize=3,
            )
            axis.axvline(0, color="black", linewidth=0.8)
            axis.set_xlim(-limit, limit)
            axis.set_yticks(positions)
            axis.set_yticklabels(
                [
                    f"{CONDITION_LABELS[row.condition]}  |  LR {format_lr(row.base_lr)}"
                    for row in species_data.itertuples()
                ],
                fontsize=8,
            )
            axis.grid(axis="x", alpha=0.2)
            axis.set_title(species.capitalize())
            if row == 1:
                axis.set_xlabel(xlabel)
        axes[0, column].set_title(f"{title}\nMinipigs", fontsize=11)
        axes[1, column].set_title(f"{title}\nMonkeys", fontsize=11)
    fig.suptitle(
        "Paired transfer effects with bootstrap 95% intervals", fontsize=15
    )
    fig.savefig(
        figures_dir(__file__) / f"{PREFIX}_paired_effects.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def draw_heatmap(
    axis: Any,
    data: pd.DataFrame,
    value_column: str,
    row_order: list[str],
    norm: Any,
    cmap: str,
    value_format: str,
) -> Any:
    matrix = data.pivot_table(
        index="condition",
        columns="base_lr",
        values=value_column,
        aggfunc="mean",
    ).reindex(index=row_order, columns=LR_ORDER)
    image = axis.imshow(
        matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, norm=norm
    )
    axis.set_xticks(range(len(LR_ORDER)))
    axis.set_xticklabels([format_lr(lr) for lr in LR_ORDER])
    axis.set_yticks(range(len(row_order)))
    axis.set_yticklabels(
        [CONDITION_LABELS[condition] for condition in row_order]
    )
    axis.set_xlabel("Base learning rate")
    for y_index, row in enumerate(matrix.to_numpy(dtype=float)):
        for x_index, value in enumerate(row):
            if np.isfinite(value):
                normalized = norm(value) if norm is not None else 0.5
                text_color = "white" if normalized > 0.62 else "black"
                axis.text(
                    x_index,
                    y_index,
                    value_format.format(value),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )
    axis.tick_params(length=0)
    for edge in axis.spines.values():
        edge.set_visible(False)
    return image


def plot_effect_heatmaps(summary: pd.DataFrame) -> None:
    """Plot paired F1 and speed effects as species-faceted heatmaps."""
    metric_specs = [
        ("f1_delta", "Δ test F1", "{:+.3f}"),
        ("stable_step_saved", "Steps saved", "{:+.0f}"),
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(15, 10), constrained_layout=True, squeeze=False
    )
    for row, (metric, label, value_format) in enumerate(metric_specs):
        plotted = summary[summary.metric.eq(metric)].copy()
        maximum = float(
            np.nanmax(
                np.abs(plotted[["ci_low", "ci_high"]].to_numpy(dtype=float))
            )
        )
        norm = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
        for column, species in enumerate(("minipigs", "monkeys")):
            axis = axes[row, column]
            image = draw_heatmap(
                axis,
                plotted[plotted.species.eq(species)],
                "mean",
                TRANSFER_CONDITIONS,
                norm,
                "RdBu_r",
                value_format,
            )
            axis.set_title(f"{species.capitalize()} — {label}")
            colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colorbar.ax.set_ylabel(label, rotation=270, labelpad=14)
    fig.suptitle(
        "Transfer effects versus matched scratch baselines", fontsize=15
    )
    fig.savefig(
        figures_dir(__file__) / f"{PREFIX}_effect_heatmaps.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_absolute_heatmaps(condition_summary: pd.DataFrame) -> None:
    """Plot absolute test F1 for every recipe, species, and learning rate."""
    value_column = "mean_test_supported_f1"
    minimum = float(condition_summary[value_column].min())
    maximum = float(condition_summary[value_column].max())
    norm = plt.Normalize(vmin=minimum, vmax=maximum)
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 7), constrained_layout=True, squeeze=False
    )
    for column, species in enumerate(("minipigs", "monkeys")):
        axis = axes[0, column]
        image = draw_heatmap(
            axis,
            condition_summary[condition_summary.species.eq(species)],
            value_column,
            CONDITION_ORDER,
            norm,
            "viridis",
            "{:.3f}",
        )
        axis.axhline(len(TRANSFER_CONDITIONS) - 0.5, color="white", linewidth=2)
        axis.set_title(species.capitalize())
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.ax.set_ylabel(
            "mean test supported macro-F1", rotation=270, labelpad=14
        )
    fig.suptitle(
        "Absolute test performance by recipe and learning rate", fontsize=15
    )
    fig.savefig(
        figures_dir(__file__) / f"{PREFIX}_absolute_performance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_subject_effects(paired: pd.DataFrame) -> None:
    """Show subject-level paired F1 effects rather than only aggregate intervals."""
    colors = {
        lr: color
        for lr, color in zip(LR_ORDER, ("#2166ac", "#4dac26", "#b2182b"))
    }
    figure, axes = plt.subplots(
        1, 2, figsize=(16, 8), sharex=True, constrained_layout=True
    )
    all_values = paired["f1_delta"].to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(all_values)) * 1.18)
    for axis, species in zip(axes, ("minipigs", "monkeys")):
        species_data = paired[paired.species.eq(species)]
        for y_index, condition in enumerate(TRANSFER_CONDITIONS):
            for lr_index, lr in enumerate(LR_ORDER):
                subset = species_data[
                    species_data.condition.eq(condition)
                    & np.isclose(species_data.base_lr, lr)
                ]
                if subset.empty:
                    continue
                jitter = np.linspace(-0.07, 0.07, len(subset))
                y = y_index + (lr_index - 1) * 0.13
                axis.scatter(
                    subset.f1_delta,
                    y + jitter,
                    color=colors[lr],
                    alpha=0.75,
                    s=28,
                    edgecolor="white",
                    linewidth=0.4,
                )
                axis.scatter(
                    subset.f1_delta.mean(),
                    y,
                    color=colors[lr],
                    marker="D",
                    edgecolor="black",
                    linewidth=0.8,
                    s=42,
                    zorder=3,
                )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlim(-limit, limit)
        axis.set_yticks(range(len(TRANSFER_CONDITIONS)))
        axis.set_yticklabels([CONDITION_LABELS[c] for c in TRANSFER_CONDITIONS])
        axis.invert_yaxis()
        axis.set_title(species.capitalize())
        axis.set_xlabel("Subject-level transfer minus scratch test F1")
        axis.grid(axis="x", alpha=0.2)
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"LR {format_lr(lr)}",
                markerfacecolor=colors[lr],
                markersize=7,
            )
            for lr in LR_ORDER
        ],
        loc="lower left",
        frameon=False,
    )
    figure.suptitle("Subject-level paired F1 effects", fontsize=15)
    figure.savefig(
        figures_dir(__file__) / f"{PREFIX}_subject_paired_effects.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_tradeoff(summary: pd.DataFrame) -> None:
    """Show the accuracy/speed trade-off for each recipe and learning rate."""
    wide = summary.pivot_table(
        index=["species", "condition", "base_lr"],
        columns="metric",
        values="mean",
    ).reset_index()
    colors = {
        condition: color
        for condition, color in zip(
            TRANSFER_CONDITIONS, plt.get_cmap("tab10").colors
        )
    }
    markers = {lr: marker for lr, marker in zip(LR_ORDER, ("o", "s", "^"))}
    x_limit = float(
        np.nanmax(np.abs(wide.f1_delta.to_numpy(dtype=float))) * 1.25
    )
    y_values = wide.stable_step_saved.to_numpy(dtype=float)
    y_padding = float((y_values.max() - y_values.min()) * 0.08)
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 7), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, species in zip(axes, ("minipigs", "monkeys")):
        species_data = wide[wide.species.eq(species)]
        for row in species_data.itertuples():
            axis.scatter(
                row.f1_delta,
                row.stable_step_saved,
                color=colors[row.condition],
                marker=markers[row.base_lr],
                s=90,
                edgecolor="black",
                linewidth=0.5,
            )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(species.capitalize())
        axis.set_xlabel("Δ test F1 versus matched scratch")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Steps saved versus matched scratch")
    axes[0].set_xlim(-x_limit, x_limit)
    axes[0].set_ylim(y_values.min() - y_padding, y_values.max() + y_padding)
    condition_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=CONDITION_LABELS[c].replace("\n", " "),
            markerfacecolor=colors[c],
            markersize=8,
        )
        for c in TRANSFER_CONDITIONS
    ]
    lr_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[lr],
            color="black",
            linestyle="",
            label=f"LR {format_lr(lr)}",
            markersize=8,
        )
        for lr in LR_ORDER
    ]
    axes[1].legend(
        handles=condition_handles + lr_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
    )
    fig.suptitle(
        "Accuracy–speed trade-off relative to matched scratch", fontsize=15
    )
    fig.savefig(
        figures_dir(__file__) / f"{PREFIX}_tradeoff.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_absolute_tradeoff(condition_summary: pd.DataFrame) -> None:
    """Show absolute F1 and convergence, including the scratch controls."""
    colors = {
        condition: color
        for condition, color in zip(
            CONDITION_ORDER, plt.get_cmap("tab10").colors
        )
    }
    markers = {lr: marker for lr, marker in zip(LR_ORDER, ("o", "s", "^"))}
    x_values = condition_summary.mean_test_supported_f1.to_numpy(dtype=float)
    y_values = condition_summary.mean_stable_step.to_numpy(dtype=float)
    x_padding = float((x_values.max() - x_values.min()) * 0.08)
    y_padding = float((y_values.max() - y_values.min()) * 0.08)
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 8), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, species in zip(axes, ("minipigs", "monkeys")):
        species_data = condition_summary[condition_summary.species.eq(species)]
        for row in species_data.itertuples():
            axis.scatter(
                row.mean_test_supported_f1,
                row.mean_stable_step,
                color=colors[row.condition],
                marker=markers[row.base_lr],
                s=90,
                edgecolor="black",
                linewidth=0.5,
            )
        axis.set_title(species.capitalize())
        axis.set_xlabel("Mean test supported macro-F1")
        axis.grid(alpha=0.2)
        axis.invert_yaxis()
    axes[0].set_ylabel(
        "Mean stable validation endpoint (optimizer steps)\nlower is better"
    )
    axes[0].set_xlim(x_values.min() - x_padding, x_values.max() + x_padding)
    axes[0].set_ylim(y_values.max() + y_padding, y_values.min() - y_padding)
    condition_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=CONDITION_LABELS[condition].replace("\n", " "),
            markerfacecolor=colors[condition],
            markersize=8,
        )
        for condition in CONDITION_ORDER
    ]
    lr_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[lr],
            color="black",
            linestyle="",
            label=f"LR {format_lr(lr)}",
            markersize=8,
        )
        for lr in LR_ORDER
    ]
    axes[1].legend(
        handles=condition_handles + lr_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
    )
    fig.suptitle(
        "Absolute accuracy–convergence trade-off\n"
        "higher F1 and earlier convergence are better",
        fontsize=15,
    )
    fig.savefig(
        figures_dir(__file__) / f"{PREFIX}_absolute_tradeoff.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entity", default=default_entity(), required=default_entity() is None
    )
    args = parser.parse_args()
    cells = load_cells()
    api = wandb.Api(timeout=120)
    expected_by_run_id = {
        str(row["wandb_run_id"]): row for row in cells.values()
    }

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [
            pool.submit(
                lambda row=row: fetch_one(api, args.entity, row),
            )
            for row in expected_by_run_id.values()
        ]
        for future in as_completed(futures):
            record, _ = future.result()
            records.append(record)
    frame = pd.DataFrame(records)
    if len(frame) != len(cells) or frame.duplicated("cell_id").any():
        raise RuntimeError(
            "W&B results do not cover the compiled Phase-4D matrix"
        )

    out_csv = csv_dir(__file__)
    frame.to_csv(out_csv / f"{PREFIX}_runs.csv", index=False)

    coverage = frame.groupby(["species", "condition"], as_index=False).agg(
        expected_cells=("cell_id", "size"),
        eligible_cells=("analysis_eligible", "sum"),
        finished_state=(
            "state",
            lambda values: int((values == "finished").sum()),
        ),
        test_metric_cells=(
            "test_supported_f1",
            lambda values: int(values.notna().sum()),
        ),
    )
    coverage["ineligible_cells"] = (
        coverage.expected_cells - coverage.eligible_cells
    )
    coverage.to_csv(out_csv / f"{PREFIX}_coverage.csv", index=False)

    eligible = frame[frame.analysis_eligible].copy()
    subject = eligible.groupby(
        ["species", "condition", "base_lr", "subject"], as_index=False
    ).agg(
        test_supported_f1=("test_supported_f1", "mean"),
        stable_step=("stable_step", "mean"),
        censored=("censored", "mean"),
    )
    scratch = subject[subject.condition.str.startswith("scratch")].rename(
        columns={
            "condition": "scratch_condition",
            "test_supported_f1": "scratch_f1",
            "stable_step": "scratch_step",
        }
    )
    transfer = subject[subject.condition.str.startswith("transfer")].copy()
    transfer["scratch_condition"] = np.where(
        transfer.condition.str.contains("adapter_warmup"),
        "scratch_adapter_warmup",
        "scratch_uniform",
    )
    paired = transfer.merge(
        scratch[
            [
                "species",
                "base_lr",
                "subject",
                "scratch_condition",
                "scratch_f1",
                "scratch_step",
            ]
        ],
        on=["species", "base_lr", "subject", "scratch_condition"],
        how="inner",
    )
    paired["f1_delta"] = paired.test_supported_f1 - paired.scratch_f1
    paired["stable_step_saved"] = paired.scratch_step - paired.stable_step

    rng = np.random.default_rng(20260911)
    summaries = []
    for keys, group in paired.groupby(["species", "condition", "base_lr"]):
        species, condition, base_lr = keys
        for metric in ("f1_delta", "stable_step_saved"):
            values = group[metric].to_numpy(dtype=float)
            low, high = bootstrap_interval(values, rng)
            summaries.append(
                {
                    "species": species,
                    "condition": condition,
                    "base_lr": base_lr,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "n_subjects": len(values),
                }
            )
    summary = pd.DataFrame(summaries)
    subject.to_csv(out_csv / f"{PREFIX}_subject_summary.csv", index=False)
    summary.to_csv(out_csv / f"{PREFIX}_paired_summary.csv", index=False)

    condition_summary = eligible.groupby(
        ["species", "condition", "base_lr"], as_index=False
    ).agg(
        n_cells=("cell_id", "size"),
        mean_test_supported_f1=("test_supported_f1", "mean"),
        mean_best_val_supported_f1=("best_val_supported_f1", "mean"),
        mean_stable_step=("stable_step", "mean"),
    )
    condition_summary.to_csv(
        out_csv / f"{PREFIX}_condition_summary.csv", index=False
    )

    plot_paired_forest(summary)
    plot_effect_heatmaps(summary)
    plot_absolute_heatmaps(condition_summary)
    plot_subject_effects(paired)
    plot_tradeoff(summary)
    plot_absolute_tradeoff(condition_summary)
    print("Coverage by species and condition:")
    print(coverage.to_string(index=False))
    print("\nEligible condition means:")
    print(condition_summary.to_string(index=False))
    print("\nPaired transfer-minus-scratch summaries:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
