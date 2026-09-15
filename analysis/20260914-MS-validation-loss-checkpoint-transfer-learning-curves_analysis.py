"""Render the Phase 4E downstream learning-curve figure set from W&B.

The analysis reads the immutable compiled Phase 4E cell lists, fetches their
test summaries and validation histories, and combines them with the completed
train-global-z-score EEGNet baseline.  It never uses target test results to
select a source checkpoint or recipe.

Outputs are subject-balanced (seed -> recording -> subject -> species) with
20,000-draw non-parametric bootstrap confidence intervals over subjects.

Usage:
    uv run python analysis/20260914-MS-validation-loss-checkpoint-transfer-learning-curves_analysis.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
PREFIX = "20260914-MS-validation-loss-checkpoint-transfer-learning-curves"
PROJECT = "neurosoft_supervised_pretraining"
TASK = "neurosoft_acoustic_stim_8band"
TEST_F1 = f"test/{TASK}_supported_f1"
VALIDATION_F1 = f"val/{TASK}_supported_f1"
FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.00)
SPECIES = ("minipigs", "monkeys")
PHASE4E_GROUPS = {
    "minipigs": {
        "transfer_validation_loss": "PHASE4E_VALIDATION_LOSS_TRANSFER_MINIPIGS",
        "scratch_matched": "PHASE4E_VALIDATION_LOSS_SCRATCH_MINIPIGS",
    },
    "monkeys": {
        "transfer_validation_loss": "PHASE4E_VALIDATION_LOSS_TRANSFER_MONKEYS",
        "scratch_matched": "PHASE4E_VALIDATION_LOSS_SCRATCH_MONKEYS",
    },
}
GLOBAL_EEGNET_GROUPS = {
    "minipigs": "NORM_GLOBAL_EEGNET_MINIPIGS_PROD_OFFLINE_16_20260902",
    "monkeys": "NORM_GLOBAL_EEGNET_MONKEYS_PROD_OFFLINE_16_20260902",
}
MODEL_COLORS = {
    "EEGNet (global z-score)": "#2878B5",
    "GRU scratch": "#77B978",
    "GRU pretrained": "#147A46",
}
LINESTYLES = {"minipigs": "-", "monkeys": "--"}
SPECIES_LABELS = {"minipigs": "Minipigs", "monkeys": "Monkeys"}


def scalar(summary: Any, key: str) -> float | None:
    """Return a finite W&B scalar, accepting both flattened and dict values."""
    try:
        value = summary.get(key)
    except AttributeError:
        value = None
    for unwrap_key in ("max", "min"):
        try:
            result = float(unwrap_summary_value(value, unwrap_key))
        except (TypeError, ValueError):
            continue
        if np.isfinite(result):
            return result
    return None


def nested(value: dict[str, Any], *keys: str) -> Any:
    dotted = ".".join(keys)
    if dotted in value:
        return value[dotted]
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def subject_from_recording(recording: str) -> str:
    match = re.search(r"(sub-\d+)", recording)
    if not match:
        raise ValueError(
            f"Could not derive subject from recording {recording!r}"
        )
    return match.group(1)


def stable_endpoint(history: pd.DataFrame) -> dict[str, float | bool]:
    """Use the established 3-point-median, stable-90%-of-own-peak endpoint."""
    step_key = (
        "trainer/global_step" if "trainer/global_step" in history else "_step"
    )
    if step_key not in history or VALIDATION_F1 not in history:
        raise ValueError("validation history lacks required step/F1 columns")
    values = history[[step_key, VALIDATION_F1]].dropna().copy()
    values = values.sort_values(step_key).drop_duplicates(step_key, keep="last")
    if len(values) < 3:
        raise ValueError("need at least three validation evaluations")
    values["smoothed_f1"] = (
        values[VALIDATION_F1].rolling(3, min_periods=3).median()
    )
    values = values.dropna(subset=["smoothed_f1"]).reset_index(drop=True)
    threshold = 0.9 * float(values.smoothed_f1.max())
    stable = values.smoothed_f1.ge(threshold)
    crossing = (
        stable
        & stable.shift(-1, fill_value=False)
        & stable.shift(-2, fill_value=False)
    )
    if crossing.any():
        return {
            "stable_step": float(values.loc[crossing.idxmax(), step_key]),
            "censored": False,
        }
    return {"stable_step": float(values[step_key].iloc[-1]), "censored": True}


def load_phase4e_cells() -> list[dict[str, Any]]:
    """Load the complete immutable design, including the audit cells."""
    rows: list[dict[str, Any]] = []
    for species in SPECIES:
        path = (
            ROOT
            / "launch"
            / "phase4e"
            / f"phase4e-transfer-learning-curves-{species}.jsonl"
        )
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cell = json.loads(line)
            rows.append(
                {
                    "phase": "phase4e",
                    "condition": cell["condition_id"],
                    "species": species,
                    "recording_id": cell["target_recording"],
                    "subject": cell["target_subject"],
                    "fraction": float(cell["target_fraction"]),
                    "target_seed": int(cell["target_finetuning_seed"]),
                    "source_seed": (
                        int(cell["source_selection_seed"])
                        if cell["condition_id"] == "transfer_validation_loss"
                        else None
                    ),
                    "cell_id": cell["cell_id"],
                    "run_id": cell["wandb_run_id"],
                    "wandb_group": cell["wandb_group"],
                }
            )
    frame = pd.DataFrame(rows)
    if frame.cell_id.duplicated().any() or len(frame) != 3060:
        raise RuntimeError(
            "Compiled Phase 4E design is not the expected 3,060 unique cells"
        )
    return rows


def fetch_phase4e_run(
    api: Any, entity: str, row: dict[str, Any]
) -> dict[str, Any]:
    """Fetch one expected Phase 4E summary and validation endpoint."""
    record = dict(row)
    try:
        run = api.run(f"{entity}/{PROJECT}/{row['run_id']}")
        record["run_name"] = str(run.name or "")
        record["state"] = str(run.state)
        record["test_f1"] = scalar(run.summary or {}, f"{TEST_F1}.max")
        history = run.history(
            keys=["_step", "trainer/global_step", VALIDATION_F1],
            samples=10_000,
            pandas=True,
        )
        record.update(stable_endpoint(history))
        record["analysis_error"] = ""
    except Exception as exc:  # retain failures in the audit table
        record.update(
            {
                "state": "fetch_error",
                "test_f1": None,
                "stable_step": None,
                "censored": None,
                "analysis_error": str(exc),
            }
        )
    return record


def eegnet_recording(config: dict[str, Any], name: str) -> str | None:
    for path in (
        ("neurosoft", "recording_id"),
        ("data", "dataset_kwargs", "recording_ids"),
    ):
        value = nested(config, *path)
        if isinstance(value, list) and value:
            return str(value[0])
        if value:
            return str(value)
    match = re.search(
        r"(sub-\d+_ses-[^_]+_task-AcousStim_acq-[^_]+_desc-raw)", name
    )
    return match.group(1) if match else None


def eegnet_fraction(config: dict[str, Any], name: str) -> float | None:
    for path in (
        ("neurosoft", "training_fraction_requested"),
        ("neurosoft", "training_fraction"),
        ("data", "training_fraction"),
    ):
        value = nested(config, *path)
        if value is not None:
            return float(value)
    match = re.search(r"_f(0?\.\d+|1(?:\.0+)?)_", name)
    return float(match.group(1)) if match else None


def eegnet_seed(config: dict[str, Any], name: str) -> int | None:
    for path in (("neurosoft", "model_seed"), ("run", "seed")):
        value = nested(config, *path)
        if value is not None:
            return int(value)
    match = re.search(r"_s(\d+)$", name)
    return int(match.group(1)) if match else None


def fetch_global_eegnet(
    api: Any, entity: str, start: int = 0, limit: int | None = None
) -> pd.DataFrame:
    """Fetch canonical global-z-score EEGNet runs and their validation endpoints."""
    selected_runs: list[tuple[Any, dict[str, Any], str]] = []
    for species, group in GLOBAL_EEGNET_GROUPS.items():
        runs = list(
            api.runs(
                f"{entity}/{PROJECT}",
                filters={"group": group},
                per_page=500,
                lazy=False,
            )
        )
        candidates: list[tuple[Any, dict[str, Any]]] = []
        for run in runs:
            config = dict(run.config or {})
            name = str(run.name or "")
            recording = eegnet_recording(config, name)
            fraction, seed = (
                eegnet_fraction(config, name),
                eegnet_seed(config, name),
            )
            if recording and fraction in FRACTIONS and seed in (42, 43, 44):
                candidates.append(
                    (
                        run,
                        {
                            "species": species,
                            "recording_id": recording,
                            "subject": subject_from_recording(recording),
                            "fraction": fraction,
                            "target_seed": seed,
                        },
                    )
                )
        # Prefer non-retry completed runs, then earliest created run per planned cell.
        candidates.sort(
            key=lambda item: (
                "retry" in set(getattr(item[0], "tags", []) or []),
                str(getattr(item[0], "created_at", "")),
                item[0].id,
            )
        )
        canonical: dict[tuple[Any, ...], tuple[Any, dict[str, Any]]] = {}
        for run, parsed in candidates:
            key = (
                parsed["species"],
                parsed["recording_id"],
                parsed["fraction"],
                parsed["target_seed"],
            )
            if key not in canonical and run.state == "finished":
                canonical[key] = (run, parsed)
        selected = list(canonical.values())[
            start : None if limit is None else start + limit
        ]
        selected_runs.extend((run, parsed, group) for run, parsed in selected)

    def fetch_one(
        run: Any, parsed: dict[str, Any], group: str
    ) -> dict[str, Any]:
        record = {
            "phase": "global_eegnet",
            "condition": "global_eegnet",
            **parsed,
            "source_seed": None,
            "cell_id": None,
            "run_id": run.id,
            "wandb_group": group,
            "run_name": str(run.name or ""),
            "state": str(run.state),
            "test_f1": scalar(run.summary or {}, f"{TEST_F1}.max"),
        }
        try:
            history = run.history(
                keys=["_step", "trainer/global_step", VALIDATION_F1],
                samples=10_000,
                pandas=True,
            )
            record.update(stable_endpoint(history))
            record["analysis_error"] = ""
        except Exception as exc:
            record.update(
                {
                    "stable_step": None,
                    "censored": None,
                    "analysis_error": str(exc),
                }
            )
        return record

    with ThreadPoolExecutor(max_workers=16) as pool:
        records = list(pool.map(lambda item: fetch_one(*item), selected_runs))
    return pd.DataFrame(records)


def interval(
    values: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(
        axis=1
    )
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize_subjects(
    subject: pd.DataFrame, metric: str, group_columns: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, data in subject.groupby(group_columns, sort=True):
        values = data[metric].dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        low, high = interval(values, np.random.default_rng(20_260_914))
        payload = dict(
            zip(
                group_columns,
                key if isinstance(key, tuple) else (key,),
                strict=True,
            )
        )
        payload.update(
            {
                "metric": metric,
                "n_subjects": len(values),
                "mean": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
            }
        )
        rows.append(payload)
    return pd.DataFrame(rows)


def phase4e_units(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average source seeds before target-seed pairing, then form paired units."""
    base = runs[runs.phase.eq("phase4e")].copy()
    transfer = base[base.condition.eq("transfer_validation_loss")]
    scratch = base[base.condition.eq("scratch_matched")]
    key = ["species", "recording_id", "subject", "fraction", "target_seed"]
    transfer = transfer.groupby(key, as_index=False).agg(
        test_f1=("test_f1", "mean"),
        stable_step=("stable_step", "mean"),
        censored=("censored", "mean"),
    )
    scratch = scratch.groupby(key, as_index=False).agg(
        test_f1=("test_f1", "mean"),
        stable_step=("stable_step", "mean"),
        censored=("censored", "mean"),
    )
    paired = transfer.merge(
        scratch,
        on=key,
        suffixes=("_transfer", "_scratch"),
        validate="one_to_one",
    )
    paired["f1_advantage"] = paired.test_f1_transfer - paired.test_f1_scratch
    paired["steps_saved"] = (
        paired.stable_step_scratch - paired.stable_step_transfer
    )
    recording = paired.groupby(
        ["species", "recording_id", "subject", "fraction"], as_index=False
    ).agg(
        f1_advantage=("f1_advantage", "mean"),
        steps_saved=("steps_saved", "mean"),
    )
    subject = recording.groupby(
        ["species", "subject", "fraction"], as_index=False
    ).mean(numeric_only=True)
    return recording, subject


def absolute_subject_units(runs: pd.DataFrame) -> pd.DataFrame:
    """Construct comparable condition-specific subject units for absolute plots."""
    phase4e = runs[runs.phase.eq("phase4e")].copy()
    transfer = phase4e[phase4e.condition.eq("transfer_validation_loss")]
    transfer = transfer.groupby(
        ["species", "recording_id", "subject", "fraction", "target_seed"],
        as_index=False,
    ).agg(test_f1=("test_f1", "mean"), stable_step=("stable_step", "mean"))
    transfer["model"] = "GRU pretrained"
    scratch = phase4e[phase4e.condition.eq("scratch_matched")].copy()
    scratch = scratch.groupby(
        ["species", "recording_id", "subject", "fraction", "target_seed"],
        as_index=False,
    ).agg(test_f1=("test_f1", "mean"), stable_step=("stable_step", "mean"))
    scratch["model"] = "GRU scratch"
    eegnet = runs[runs.phase.eq("global_eegnet")].copy()
    eegnet = eegnet.groupby(
        ["species", "recording_id", "subject", "fraction", "target_seed"],
        as_index=False,
    ).agg(test_f1=("test_f1", "mean"), stable_step=("stable_step", "mean"))
    eegnet["model"] = "EEGNet (global z-score)"
    pooled = pd.concat([transfer, scratch, eegnet], ignore_index=True)
    recording = pooled.groupby(
        ["model", "species", "recording_id", "subject", "fraction"],
        as_index=False,
    ).mean(numeric_only=True)
    return recording.groupby(
        ["model", "species", "subject", "fraction"], as_index=False
    ).mean(numeric_only=True)


def draw_line(
    axis: Any,
    data: pd.DataFrame,
    *,
    color: str,
    linestyle: str,
    label: str,
    metric: str,
    scale: float = 1.0,
    ribbon: bool = True,
) -> None:
    data = data.sort_values("fraction")
    x = data.fraction.to_numpy(dtype=float) * 100
    mean = data["mean"].to_numpy(dtype=float) * scale
    low = data.ci_low.to_numpy(dtype=float) * scale
    high = data.ci_high.to_numpy(dtype=float) * scale
    if ribbon:
        axis.fill_between(
            x, low, high, color=color, alpha=0.13, linewidth=0, zorder=1
        )
    else:
        axis.errorbar(
            x,
            mean,
            yerr=[mean - low, high - mean],
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=2,
            alpha=0.48,
            zorder=1,
        )
    axis.plot(
        x,
        mean,
        color=color,
        linestyle=linestyle,
        marker="o",
        markersize=5,
        linewidth=2.3,
        label=label,
        zorder=2,
    )


def setup_x(axis: Any) -> None:
    axis.set_xscale("log")
    axis.set_xlim(4.4, 115)
    axis.set_xticks(np.array(FRACTIONS) * 100)
    axis.set_xticklabels(["5", "10", "25", "50", "100"])
    axis.grid(alpha=0.22, which="major")
    axis.minorticks_off()
    axis.set_xlabel("Target training data (%)")


def species_legend() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color="#333333",
            lw=2.2,
            linestyle=LINESTYLES[s],
            label=SPECIES_LABELS[s],
        )
        for s in SPECIES
    ]


def plot_main(delta: pd.DataFrame, path: Path, title: str) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(12.4, 4.55), constrained_layout=True
    )
    specs = [
        ("f1_advantage", "Transfer advantage in test supported-F1 (pp)", 100.0),
        ("steps_saved", "Transfer advantage in stable-convergence steps", 1.0),
    ]
    for axis, (metric, ylabel, scale) in zip(axes, specs, strict=True):
        data = delta[delta.metric.eq(metric)]
        for species in SPECIES:
            draw_line(
                axis,
                data[data.species.eq(species)],
                color=MODEL_COLORS["GRU pretrained"],
                linestyle=LINESTYLES[species],
                label=SPECIES_LABELS[species],
                metric=metric,
                scale=scale,
            )
        axis.axhline(0, color="#333333", linewidth=0.9, zorder=0)
        setup_x(axis)
        axis.set_ylabel(ylabel)
    axes[0].set_title("Final performance")
    axes[1].set_title("Convergence speed")
    axes[1].legend(
        handles=species_legend(), title="Species", frameon=False, loc="best"
    )
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_absolute(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), constrained_layout=True)
    specs = [
        ("test_f1", "Subject-balanced test supported-F1", False),
        ("stable_step", "Stable convergence (optimizer steps)", True),
    ]
    for row, species in enumerate(SPECIES):
        for column, (metric, ylabel, starts_zero) in enumerate(specs):
            axis = axes[row, column]
            data = summary[
                summary.metric.eq(metric) & summary.species.eq(species)
            ]
            for model in MODEL_COLORS:
                subset = data[data.model.eq(model)]
                draw_line(
                    axis,
                    subset,
                    color=MODEL_COLORS[model],
                    linestyle=LINESTYLES[species],
                    label=model,
                    metric=metric,
                    ribbon=False,
                )
            setup_x(axis)
            axis.set_ylabel(ylabel)
            axis.set_title(
                f"{SPECIES_LABELS[species]} — "
                f"{'Final performance' if metric == 'test_f1' else 'Convergence speed'}"
            )
            if starts_zero:
                axis.set_ylim(bottom=0)
    model_handles = [
        Line2D([0], [0], color=color, lw=2.5, label=model)
        for model, color in MODEL_COLORS.items()
    ]
    first = axes[0, 1].legend(
        handles=model_handles, title="Model", frameon=False, loc="upper left"
    )
    axes[0, 1].add_artist(first)
    axes[1, 1].legend(
        handles=species_legend(),
        title="Line style",
        frameon=False,
        loc="lower right",
    )
    fig.suptitle(
        "Absolute performance and convergence across target-data fractions",
        fontsize=14,
    )
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_session_distributions(recording: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.3), constrained_layout=True)
    specs = [
        ("f1_advantage", "Transfer advantage in test supported-F1 (pp)", 100.0),
        ("steps_saved", "Transfer advantage in stable-convergence steps", 1.0),
    ]
    for row, species in enumerate(SPECIES):
        for column, (metric, xlabel, scale) in enumerate(specs):
            axis = axes[row, column]
            data = recording[recording.species.eq(species)]
            series = [
                data[np.isclose(data.fraction, fraction)][metric]
                .dropna()
                .to_numpy()
                * scale
                for fraction in FRACTIONS
            ]
            violin = axis.violinplot(
                series,
                positions=np.array(FRACTIONS) * 100,
                widths=[3.2, 5.6, 9.5, 13.5, 19],
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(MODEL_COLORS["GRU pretrained"])
                body.set_alpha(0.24)
                body.set_edgecolor(MODEL_COLORS["GRU pretrained"])
            violin["cmedians"].set_color(MODEL_COLORS["GRU pretrained"])
            axis.axhline(0, color="#333333", linewidth=0.9)
            setup_x(axis)
            axis.set_ylabel(xlabel)
            axis.set_title(SPECIES_LABELS[species])
    fig.suptitle("Session-level paired transfer effects", fontsize=14)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def top_half(recording: pd.DataFrame) -> pd.DataFrame:
    """Keep ceil(n/2) sessions by scratch 100%-data F1, defined upstream."""
    # `recording` alone has contrasts. The caller attaches the precomputed score.
    selected: list[pd.DataFrame] = []
    for species, data in recording.groupby("species"):
        cutoff_n = int(np.ceil(data.recording_id.nunique() / 2))
        rank = (
            data[["recording_id", "scratch_full_f1"]]
            .drop_duplicates()
            .sort_values("scratch_full_f1", ascending=False)
        )
        selected.append(rank.head(cutoff_n).assign(species=species))
    selected_keys = pd.concat(selected, ignore_index=True)[
        ["species", "recording_id"]
    ]
    return recording.merge(
        selected_keys, on=["species", "recording_id"], validate="many_to_one"
    )


def attach_scratch_full_score(
    recording: pd.DataFrame, runs: pd.DataFrame
) -> pd.DataFrame:
    scratch = runs[
        (runs.phase.eq("phase4e"))
        & (runs.condition.eq("scratch_matched"))
        & np.isclose(runs.fraction, 1.0)
    ]
    score = (
        scratch.groupby(["species", "recording_id"], as_index=False)
        .test_f1.mean()
        .rename(columns={"test_f1": "scratch_full_f1"})
    )
    return recording.merge(
        score, on=["species", "recording_id"], validate="many_to_one"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entity", default=default_entity(), required=default_entity() is None
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--fetch-phase4e-chunk", nargs=2, type=int, metavar=("OFFSET", "LIMIT")
    )
    parser.add_argument(
        "--fetch-eegnet-chunk", nargs=2, type=int, metavar=("OFFSET", "LIMIT")
    )
    parser.add_argument("--render-cached", action="store_true")
    args = parser.parse_args()
    out_csv, out_figures = csv_dir(__file__), figures_dir(__file__)
    if args.fetch_phase4e_chunk:
        offset, limit = args.fetch_phase4e_chunk
        expected = load_phase4e_cells()[offset : offset + limit]
        print(
            f"Fetching Phase 4E chunk {offset}:{offset + len(expected)}...",
            flush=True,
        )
        api = wandb.Api(timeout=120)
        records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(fetch_phase4e_run, api, args.entity, row)
                for row in expected
            ]
            for future in as_completed(futures):
                records.append(future.result())
        path = out_csv / f"{PREFIX}_phase4e_cache_{offset:04d}.csv"
        pd.DataFrame(records).to_csv(path, index=False)
        print(f"Wrote {path} ({len(records)} rows)")
        return
    if args.fetch_eegnet_chunk:
        offset, limit = args.fetch_eegnet_chunk
        api = wandb.Api(timeout=120)
        eegnet = fetch_global_eegnet(api, args.entity, offset, limit)
        path = out_csv / f"{PREFIX}_eegnet_cache_{offset:04d}.csv"
        eegnet.to_csv(path, index=False)
        print(f"Wrote {path} ({len(eegnet)} rows)")
        return
    if args.render_cached:
        phase4e_paths = sorted(out_csv.glob(f"{PREFIX}_phase4e_cache_*.csv"))
        eegnet_paths = sorted(out_csv.glob(f"{PREFIX}_eegnet_cache_*.csv"))
        phase4e = pd.concat(
            [pd.read_csv(path) for path in phase4e_paths], ignore_index=True
        )
        eegnet = pd.concat(
            [pd.read_csv(path) for path in eegnet_paths], ignore_index=True
        )
        if len(phase4e) != 3060 or phase4e.run_id.duplicated().any():
            raise RuntimeError(
                "Cached Phase 4E records do not contain 3,060 unique runs"
            )
        runs = pd.concat([phase4e, eegnet], ignore_index=True)
    else:
        api = wandb.Api(timeout=120)
        expected = load_phase4e_cells()
        print(
            f"Fetching {len(expected)} completed Phase 4E downstream runs...",
            flush=True,
        )
        records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(fetch_phase4e_run, api, args.entity, row)
                for row in expected
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                records.append(future.result())
                if index % 100 == 0:
                    print(f"  fetched {index}/{len(expected)}", flush=True)
        phase4e = pd.DataFrame(records)
        print("Fetching global-z-score EEGNet baseline...", flush=True)
        eegnet = fetch_global_eegnet(api, args.entity)
        runs = pd.concat([phase4e, eegnet], ignore_index=True)
    runs.to_csv(out_csv / f"{PREFIX}_downstream_runs.csv", index=False)
    if phase4e.state.eq("finished").sum() != 3060:
        raise RuntimeError(
            "Phase 4E W&B audit did not resolve 3,060 finished runs"
        )

    recording_effects, subject_effects = phase4e_units(runs)
    delta = pd.concat(
        [
            summarize_subjects(subject_effects, metric, ["species", "fraction"])
            for metric in ("f1_advantage", "steps_saved")
        ],
        ignore_index=True,
    )
    absolute_units = absolute_subject_units(runs)
    absolute = pd.concat(
        [
            summarize_subjects(
                absolute_units, metric, ["model", "species", "fraction"]
            )
            for metric in ("test_f1", "stable_step")
        ],
        ignore_index=True,
    )
    ranked_recordings = attach_scratch_full_score(recording_effects, runs)
    top_recordings = top_half(ranked_recordings)
    top_subjects = top_recordings.groupby(
        ["species", "subject", "fraction"], as_index=False
    ).mean(numeric_only=True)
    top_delta = pd.concat(
        [
            summarize_subjects(top_subjects, metric, ["species", "fraction"])
            for metric in ("f1_advantage", "steps_saved")
        ],
        ignore_index=True,
    )
    for label, table in {
        "paired recording effects": recording_effects,
        "paired subject effects": subject_effects,
        "main delta summary": delta,
        "absolute summary": absolute,
        "top-half recording effects": top_recordings,
        "top-half delta summary": top_delta,
    }.items():
        path = out_csv / f"{PREFIX}_{label.replace(' ', '_')}.csv"
        table.to_csv(path, index=False)
        print(f"Wrote {label}: {path} ({len(table)} rows)")

    plot_main(
        delta,
        out_figures / f"{PREFIX}_main_transfer_advantage.png",
        "Validation-loss-selected transfer advantage versus matched scratch",
    )
    plot_absolute(absolute, out_figures / f"{PREFIX}_absolute_comparison.png")
    plot_session_distributions(
        recording_effects,
        out_figures / f"{PREFIX}_session_paired_distributions.png",
    )
    plot_main(
        top_delta,
        out_figures / f"{PREFIX}_top_half_robustness.png",
        "Top-half by scratch 100%-data F1 sensitivity analysis (post hoc)",
    )
    print("\nMain paired summaries:")
    print(
        delta.to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
