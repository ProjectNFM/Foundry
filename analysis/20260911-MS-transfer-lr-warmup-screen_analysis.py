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
        "test_supported_f1": scalar((run.summary or {}).get(TEST_F1)),
        "best_val_supported_f1": scalar((run.summary or {}).get(VALIDATION_F1)),
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
    run = api.run(f"{entity}/{PROJECT}/{expected['wandb_run_id']}")
    if run.state != "finished":
        raise RuntimeError(f"{run.id}: run state is {run.state}")
    record = validate_run(run, expected)
    if record["test_supported_f1"] is None:
        raise RuntimeError(f"{run.id}: missing test supported macro-F1 summary")
    history = run.history(
        keys=["_step", "trainer/global_step", VALIDATION_F1],
        samples=10_000,
        pandas=True,
    )
    endpoint = stable_endpoint(history)
    record.update(endpoint)
    return record, expected


def bootstrap_interval(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(
        axis=1
    )
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


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

    subject = frame.groupby(
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
    out_csv = csv_dir(__file__)
    frame.to_csv(out_csv / f"{PREFIX}_runs.csv", index=False)
    subject.to_csv(out_csv / f"{PREFIX}_subject_summary.csv", index=False)
    summary.to_csv(out_csv / f"{PREFIX}_paired_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for axis, metric, title, ylabel in [
        (
            axes[0],
            "f1_delta",
            "Transfer minus matched scratch F1",
            "supported macro-F1",
        ),
        (
            axes[1],
            "stable_step_saved",
            "Steps saved versus matched scratch",
            "optimizer steps",
        ),
    ]:
        plotted = summary[summary.metric.eq(metric)]
        positions = np.arange(len(plotted))
        axis.errorbar(
            positions,
            plotted["mean"],
            yerr=[
                plotted["mean"] - plotted["ci_low"],
                plotted["ci_high"] - plotted["mean"],
            ],
            fmt="o",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [
                f"{row.species}\n{row.condition}\n{row.base_lr:g}"
                for row in plotted.itertuples()
            ],
            rotation=70,
            ha="right",
            fontsize=7,
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    fig.savefig(figures_dir(__file__) / f"{PREFIX}_paired_effects.png", dpi=160)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
