"""Class-weight smoothing sweep: minipigs vs monkeys (intrasession multisubj).

Fetches finished runs from paired species sweeps that vary
``class_weights.smoothing`` with species-optimal HPs frozen. Reports max
validation metrics for ``neurosoft_acoustic_stim_8band``, prints comparison
tables, and saves F1 / multi-metric figures.

Usage:
    uv run python analysis/20260729-LS-class-weight-smoothing.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from analysis._wandb_utils import (
    default_entity,
    figures_dir,
    unwrap_summary_value,
)

PROJECT = "auditory_decoding"
GROUP = "NEUROSOFT_INTRASESSION_MULTISUBJ"
ENTITY = default_entity()

# Paired species sweeps (class-weight smoothing @ optimal HPs).
SWEEP_IDS: dict[str, str] = {
    "minipigs": "w74jfier",
    "monkeys": "nxx4a4pn",
}

# Optional no-CW multisubject baselines (same HPs, mode none / unset).
BASELINE_SWEEP_IDS: dict[str, str] = {
    "minipigs": "47jd29ds",
    "monkeys": "bvcgw95o",
}

TASK = "neurosoft_acoustic_stim_8band"
METRIC_KEYS = {
    "f1": f"val/{TASK}_f1",
    "auroc": f"val/{TASK}_auroc",
    "precision": f"val/{TASK}_precision",
    "recall": f"val/{TASK}_recall",
    "balanced_acc": f"val/{TASK}_balanced_acc",
}
METRICS = list(METRIC_KEYS)
SPECIES_ORDER = ["minipigs", "monkeys"]
SMOOTHING_ORDER = [0.5, 0.75, 1.0]

STEM = Path(__file__).stem
FIGURES_DIR = figures_dir(__file__)
N_WORKERS = 8


def _run_path(run_id: str) -> str:
    if ENTITY:
        return f"{ENTITY}/{PROJECT}/{run_id}"
    return f"{PROJECT}/{run_id}"


def _sweep_path(sweep_id: str) -> str:
    if ENTITY:
        return f"{ENTITY}/{PROJECT}/{sweep_id}"
    return f"{PROJECT}/{sweep_id}"


def _species_from_run(run: Any, fallback: str | None = None) -> str:
    tags = {t.lower() for t in (run.tags or [])}
    if "minipigs" in tags:
        return "minipigs"
    if "monkeys" in tags:
        return "monkeys"
    data = (run.config or {}).get("data") or {}
    ds = str(data.get("dataset_class", "")).lower()
    if "minipig" in ds:
        return "minipigs"
    if "monkey" in ds:
        return "monkeys"
    return fallback or "unknown"


def _extract_meta(config: dict[str, Any]) -> dict[str, Any]:
    hp = config.get("hyperparameters") or {}
    model = config.get("model") or {}
    trainer = config.get("trainer") or {}
    cw = config.get("class_weights") or {}

    smoothing = config.get("class_weights.smoothing")
    if smoothing is None and isinstance(cw, dict):
        smoothing = cw.get("smoothing")

    mode = config.get("class_weights.mode")
    if mode is None and isinstance(cw, dict):
        mode = cw.get("mode")

    fold = config.get("hyperparameters.fold_number")
    if fold is None:
        fold = hp.get("fold_number")

    tokenizer = config.get("model/tokenizer")
    if tokenizer is None:
        tok = model.get("tokenizer")
        if isinstance(tok, dict):
            tokenizer = tok.get("_name_") or tok.get("name") or str(tok)
        else:
            tokenizer = tok

    return {
        "smoothing": smoothing,
        "cw_mode": mode,
        "fold": fold,
        "tokenizer": tokenizer,
        "atn_dropout": config.get("model.atn_dropout", model.get("atn_dropout")),
        "learning_rate": config.get(
            "hyperparameters.learning_rate", hp.get("learning_rate")
        ),
        "weight_decay": config.get(
            "hyperparameters.weight_decay", hp.get("weight_decay")
        ),
        "grad_clip": config.get(
            "trainer.gradient_clip_val", trainer.get("gradient_clip_val")
        ),
        "split_type": config.get("data.split_type")
        or (config.get("data") or {}).get("split_type"),
    }


def _summary_max(run: Any, wandb_key: str) -> float | None:
    val = unwrap_summary_value(run.summary.get(wandb_key), "max")
    return float(val) if isinstance(val, (int, float)) else None


def _history_maxes(run: Any, wandb_keys: list[str]) -> dict[str, float | None]:
    """Batch-fetch history maxima for keys missing summary max."""
    if not wandb_keys:
        return {}
    history = run.history(keys=wandb_keys, samples=10_000, pandas=True)
    out: dict[str, float | None] = {}
    for key in wandb_keys:
        if key not in history.columns or history[key].dropna().empty:
            raw = run.summary.get(key)
            val_min = unwrap_summary_value(raw, "min")
            out[key] = (
                float(val_min) if isinstance(val_min, (int, float)) else None
            )
        else:
            out[key] = float(history[key].max())
    return out


def _fetch_one(
    run_id: str,
    species_hint: str,
    sweep_id: str,
    *,
    is_baseline: bool = False,
) -> dict[str, Any]:
    api = wandb.Api()
    run = api.run(_run_path(run_id))
    config = dict(run.config)
    row: dict[str, Any] = {
        "species": _species_from_run(run, fallback=species_hint),
        "sweep_id": sweep_id,
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "is_baseline": is_baseline,
        **_extract_meta(config),
    }
    missing: list[str] = []
    for short, key in METRIC_KEYS.items():
        val = _summary_max(run, key)
        if val is None:
            missing.append(key)
        else:
            row[short] = val
    hist_max = _history_maxes(run, missing)
    for short, key in METRIC_KEYS.items():
        if short not in row:
            row[short] = hist_max.get(key)
    return row


def fetch_finished_runs(api: wandb.Api | None = None) -> pd.DataFrame:
    if api is None:
        api = wandb.Api()

    stubs: list[tuple[str, str, str, bool]] = []
    for species, sweep_id in SWEEP_IDS.items():
        sweep = api.sweep(_sweep_path(sweep_id))
        for run in sweep.runs:
            if run.state != "finished":
                continue
            stubs.append((run.id, species, sweep_id, False))

    for species, sweep_id in BASELINE_SWEEP_IDS.items():
        sweep = api.sweep(_sweep_path(sweep_id))
        for run in sweep.runs:
            if run.state != "finished":
                continue
            stubs.append((run.id, species, sweep_id, True))

    print(f"Fetching {len(stubs)} finished runs ({N_WORKERS} workers)...")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one, rid, species, sid, is_baseline=is_base): rid
            for rid, species, sid, is_base in stubs
        }
        done = 0
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 10 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["smoothing", "fold", "learning_rate", *METRICS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Baselines: treat as smoothing=NaN / label "none"
    return df.sort_values(
        ["is_baseline", "species", "smoothing", "fold"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def fold_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std across folds for each species × smoothing (CW runs only)."""
    cw = df.loc[~df["is_baseline"]].copy()
    g = (
        cw.groupby(["species", "smoothing"], as_index=False)[METRICS]
        .agg(["mean", "std", "count"])
    )
    # Flatten MultiIndex columns
    g.columns = [
        "species" if c[0] == "species" else (
            "smoothing" if c[0] == "smoothing" else f"{c[0]}_{c[1]}"
        )
        for c in g.columns.to_list()
    ]
    return g


def best_per_species(df: pd.DataFrame) -> pd.DataFrame:
    cw = df.loc[~df["is_baseline"]].copy()
    idx = cw.groupby("species")["f1"].idxmax()
    cols = [
        "species",
        "smoothing",
        "fold",
        *METRICS,
        "run_name",
        "run_id",
        "sweep_id",
    ]
    return cw.loc[idx, cols].reset_index(drop=True)


def mean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Species × smoothing mean F1/etc with fold std, plus baseline row."""
    rows: list[dict[str, Any]] = []
    for species in SPECIES_ORDER:
        base = df.loc[(df["species"] == species) & df["is_baseline"]]
        if not base.empty:
            row: dict[str, Any] = {
                "species": species,
                "smoothing": "none (baseline)",
                "n_folds": int(base["fold"].nunique()),
            }
            for m in METRICS:
                row[f"{m}_mean"] = float(base[m].mean())
                row[f"{m}_std"] = float(base[m].std(ddof=1)) if len(base) > 1 else 0.0
            rows.append(row)

        sub = df.loc[(df["species"] == species) & ~df["is_baseline"]]
        for sm in SMOOTHING_ORDER:
            block = sub.loc[sub["smoothing"] == sm]
            if block.empty:
                continue
            row = {
                "species": species,
                "smoothing": sm,
                "n_folds": int(len(block)),
            }
            for m in METRICS:
                row[f"{m}_mean"] = float(block[m].mean())
                row[f"{m}_std"] = (
                    float(block[m].std(ddof=1)) if len(block) > 1 else 0.0
                )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_f1_by_smoothing(df: pd.DataFrame) -> Path:
    """Bar plot of mean±std F1 vs smoothing, species side-by-side."""
    means = mean_table(df)
    # Drop baseline from primary smoothing figure for a clean x-axis;
    # include a separate baseline reference line per species.
    cw_means = means.loc[means["smoothing"] != "none (baseline)"].copy()
    cw_means["smoothing"] = pd.to_numeric(cw_means["smoothing"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(SMOOTHING_ORDER))
    width = 0.35
    colors = {"minipigs": "#4C78A8", "monkeys": "#F58518"}

    for i, species in enumerate(SPECIES_ORDER):
        sub = cw_means.loc[cw_means["species"] == species].set_index("smoothing")
        ys = [sub.loc[s, "f1_mean"] if s in sub.index else np.nan for s in SMOOTHING_ORDER]
        es = [sub.loc[s, "f1_std"] if s in sub.index else np.nan for s in SMOOTHING_ORDER]
        ax.bar(
            x + (i - 0.5) * width,
            ys,
            width,
            yerr=es,
            label=species,
            color=colors[species],
            capsize=4,
            alpha=0.9,
        )
        base = means.loc[
            (means["species"] == species)
            & (means["smoothing"] == "none (baseline)")
        ]
        if not base.empty:
            ax.axhline(
                float(base.iloc[0]["f1_mean"]),
                color=colors[species],
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SMOOTHING_ORDER])
    ax.set_xlabel("class_weights.smoothing")
    ax.set_ylabel("max val F1 (mean ± std over folds)")
    ax.set_title("Class-weight smoothing vs F1 (intrasession multisubject)")
    ax.legend(title="Species / dashed = no-CW baseline")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_f1_by_smoothing.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_metrics_grid(df: pd.DataFrame) -> Path:
    """Small-multiples of mean±std for all five metrics vs smoothing."""
    means = mean_table(df)
    cw_means = means.loc[means["smoothing"] != "none (baseline)"].copy()
    cw_means["smoothing"] = pd.to_numeric(cw_means["smoothing"])
    colors = {"minipigs": "#4C78A8", "monkeys": "#F58518"}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 3.6), sharex=True)
    x = np.arange(len(SMOOTHING_ORDER))
    width = 0.35

    for ax, metric in zip(axes, METRICS):
        for i, species in enumerate(SPECIES_ORDER):
            sub = cw_means.loc[cw_means["species"] == species].set_index(
                "smoothing"
            )
            ys = [
                sub.loc[s, f"{metric}_mean"] if s in sub.index else np.nan
                for s in SMOOTHING_ORDER
            ]
            es = [
                sub.loc[s, f"{metric}_std"] if s in sub.index else np.nan
                for s in SMOOTHING_ORDER
            ]
            ax.bar(
                x + (i - 0.5) * width,
                ys,
                width,
                yerr=es,
                label=species if metric == METRICS[0] else None,
                color=colors[species],
                capsize=3,
                alpha=0.9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in SMOOTHING_ORDER])
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
        if metric == "f1":
            ax.set_ylabel("max val (mean ± std)")

    axes[0].legend(loc="best")
    fig.suptitle(
        "Effect of class_weights.smoothing on validation metrics",
        y=1.02,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_metrics_by_smoothing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    print(f"Project: {PROJECT}")
    print(f"Group:   {GROUP}")
    print(f"Sweeps:  {SWEEP_IDS}")
    print(f"Baselines: {BASELINE_SWEEP_IDS}")

    df = fetch_finished_runs()
    if df.empty:
        print("No finished runs found.")
        return

    csv_path = FIGURES_DIR / f"{STEM}_runs.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved run table → {csv_path}")

    cw = df.loc[~df["is_baseline"]]
    print(f"\nFinished CW runs: {len(cw)} "
          f"({(cw['species']=='minipigs').sum()} minipigs, "
          f"{(cw['species']=='monkeys').sum()} monkeys)")
    print(f"Baseline runs: {df['is_baseline'].sum()}")

    print("\n=== Best config per species (by max val F1) ===")
    best = best_per_species(df)
    print(best.to_string(index=False))

    print("\n=== Mean ± std across folds (incl. no-CW baseline) ===")
    means = mean_table(df)
    # Compact display
    disp = means.copy()
    for m in METRICS:
        disp[m] = disp.apply(
            lambda r, mm=m: f"{r[f'{mm}_mean']:.4f}±{r[f'{mm}_std']:.4f}",
            axis=1,
        )
    print(
        disp[["species", "smoothing", "n_folds", *METRICS]].to_string(
            index=False
        )
    )

    print("\n=== Full grid (CW runs) ===")
    full_cols = [
        "species",
        "smoothing",
        "fold",
        *METRICS,
        "run_id",
        "sweep_id",
    ]
    print(
        cw.sort_values(["species", "smoothing", "fold"])[full_cols].to_string(
            index=False
        )
    )

    fig1 = plot_f1_by_smoothing(df)
    fig2 = plot_metrics_grid(df)
    print(f"\nFigures:\n  {fig1}\n  {fig2}")


if __name__ == "__main__":
    main()
