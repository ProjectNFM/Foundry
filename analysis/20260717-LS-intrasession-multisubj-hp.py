"""Intrasession multisubject HP search: minipigs vs monkeys.

Fetches finished runs from paired species sweeps in
``NEUROSOFT_INTRASESSION_MULTISUBJ``, extracts varied hyperparameters and
max validation metrics for ``neurosoft_acoustic_stim_8band``, prints
comparison tables, and saves a primary F1 comparison figure.

Usage:
    uv run python analysis/20260717-LS-intrasession-multisubj-hp.py
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

# Paired species sweeps (grid HP search, intrasession multisubject).
SWEEP_IDS: dict[str, str] = {
    "minipigs": "9cr4zl3u",
    "monkeys": "meu5wgw5",
}

TASK = "neurosoft_acoustic_stim_8band"
METRIC_KEYS = {
    "f1": f"val/{TASK}_f1",
    "auroc": f"val/{TASK}_auroc",
    "precision": f"val/{TASK}_precision",
    "recall": f"val/{TASK}_recall",
    "balanced_acc": f"val/{TASK}_balanced_acc",
}

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


def _extract_hps(config: dict[str, Any]) -> dict[str, Any]:
    hp = config.get("hyperparameters") or {}
    model = config.get("model") or {}
    trainer = config.get("trainer") or {}

    tokenizer = config.get("model/tokenizer")
    if tokenizer is None:
        tok = model.get("tokenizer")
        if isinstance(tok, dict):
            tokenizer = tok.get("_name_") or tok.get("name") or str(tok)
        else:
            tokenizer = tok

    return {
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
        "fold": hp.get("fold_number"),
        "split_type": config.get("data.split_type")
        or (config.get("data") or {}).get("split_type"),
        "batch_size": config.get("hyperparameters.batch_size", hp.get("batch_size")),
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
            out[key] = float(val_min) if isinstance(val_min, (int, float)) else None
        else:
            out[key] = float(history[key].max())
    return out


def _fetch_one(run_id: str, species_hint: str, sweep_id: str) -> dict[str, Any]:
    api = wandb.Api()
    run = api.run(_run_path(run_id))
    config = dict(run.config)
    row = {
        "species": _species_from_run(run, fallback=species_hint),
        "sweep_id": sweep_id,
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        **_extract_hps(config),
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

    stubs: list[tuple[str, str, str]] = []
    for species, sweep_id in SWEEP_IDS.items():
        sweep = api.sweep(_sweep_path(sweep_id))
        for run in sweep.runs:
            if run.state != "finished":
                continue
            stubs.append((run.id, species, sweep_id))

    print(f"Fetching {len(stubs)} finished runs ({N_WORKERS} workers)...")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one, rid, species, sid): (rid, species)
            for rid, species, sid in stubs
        }
        done = 0
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 20 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ["learning_rate", "weight_decay", "atn_dropout", "grad_clip", "f1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["species", "f1"], ascending=[True, False]).reset_index(
        drop=True
    )


def best_per_species(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("species")["f1"].idxmax()
    cols = [
        "species",
        "tokenizer",
        "atn_dropout",
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "f1",
        "auroc",
        "precision",
        "recall",
        "balanced_acc",
        "run_name",
        "run_id",
    ]
    return df.loc[idx, cols].sort_values("species").reset_index(drop=True)


def _fmt_table(df: pd.DataFrame) -> str:
    """Pretty-print with scientific LR and 4-decimal metrics."""
    out = df.copy()
    if "learning_rate" in out.columns:
        out["learning_rate"] = out["learning_rate"].map(
            lambda x: f"{x:.2e}" if pd.notna(x) else ""
        )
    float_cols = [
        c
        for c in [
            "atn_dropout",
            "weight_decay",
            "grad_clip",
            "f1",
            "auroc",
            "precision",
            "recall",
            "balanced_acc",
            "mean",
            "std",
            "max",
        ]
        if c in out.columns
    ]
    for c in float_cols:
        out[c] = out[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return out.to_string(index=False)


def print_tables(df: pd.DataFrame) -> None:
    best = best_per_species(df)
    print("\n=== Best configuration per species (by max val F1) ===")
    with pd.option_context("display.max_columns", 20, "display.width", 140):
        print(_fmt_table(best))

    top = (
        df.sort_values(["species", "f1"], ascending=[True, False])
        .groupby("species", group_keys=False)
        .head(10)
    )
    show = top[
        [
            "species",
            "tokenizer",
            "atn_dropout",
            "learning_rate",
            "weight_decay",
            "grad_clip",
            "f1",
            "auroc",
            "precision",
            "recall",
            "balanced_acc",
            "run_id",
        ]
    ]
    print("\n=== Top-10 configurations per species ===")
    with pd.option_context("display.max_columns", 20, "display.width", 140):
        print(_fmt_table(show))

    tok = (
        df.groupby(["species", "tokenizer"])["f1"]
        .agg(["count", "mean", "std", "max"])
        .reset_index()
        .sort_values(["species", "mean"], ascending=[True, False])
    )
    print("\n=== F1 by species × tokenizer ===")
    with pd.option_context("display.width", 120):
        print(_fmt_table(tok))


def plot_f1_by_tokenizer(df: pd.DataFrame) -> Path:
    order = (
        df.groupby("tokenizer")["f1"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    species_list = sorted(df["species"].unique())
    colors = {"minipigs": "#4C72B0", "monkeys": "#DD8452"}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.35
    x = np.arange(len(order))

    for i, species in enumerate(species_list):
        means, stds, ns = [], [], []
        for tok in order:
            vals = df.loc[
                (df["species"] == species) & (df["tokenizer"] == tok), "f1"
            ].dropna()
            means.append(float(vals.mean()) if len(vals) else np.nan)
            stds.append(float(vals.std(ddof=0)) if len(vals) else 0.0)
            ns.append(len(vals))
        offset = (i - (len(species_list) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=species,
            color=colors.get(species, "gray"),
            edgecolor="white",
            error_kw=dict(lw=1.0),
        )
        for bar, mean, n in zip(bars, means, ns):
            if np.isnan(mean):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{mean:.3f}\n(n={n})",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    short = [t.replace("per_channel_", "") for t in order]
    ax.set_xticklabels(short, rotation=15, ha="right")
    ax.set_xlabel("Tokenizer")
    ax.set_ylabel(f"Max val {TASK} F1")
    ax.set_title("Intrasession multisubject HP search: F1 by tokenizer (mean ± std)")
    ax.legend(title="Species")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_f1_by_tokenizer.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_best_bar(df: pd.DataFrame) -> Path:
    best = best_per_species(df)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = {"minipigs": "#4C72B0", "monkeys": "#DD8452"}
    x = range(len(best))
    bars = ax.bar(
        x,
        best["f1"],
        color=[colors.get(s, "gray") for s in best["species"]],
        edgecolor="white",
    )
    for bar, row in zip(bars, best.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{row.f1:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        label = (
            f"{row.tokenizer}\n"
            f"atn={row.atn_dropout}, lr={row.learning_rate:g}\n"
            f"wd={row.weight_decay}, clip={row.grad_clip}"
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color="white",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(best["species"])
    ax.set_ylabel(f"Max val {TASK} F1")
    ax.set_ylim(0, max(best["f1"].max() * 1.2, 0.1))
    ax.set_title("Best HP config per species (max val F1)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_best_f1_per_species.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def main() -> None:
    print(
        f"Resolved: sweeps {SWEEP_IDS}, group={GROUP}, project={PROJECT}, "
        f"entity={ENTITY}"
    )
    df = fetch_finished_runs()
    if df.empty:
        raise SystemExit("No finished runs found.")

    print(
        f"Loaded {len(df)} finished runs | species counts: "
        f"{df['species'].value_counts().to_dict()}"
    )
    print_tables(df)
    plot_best_bar(df)
    plot_f1_by_tokenizer(df)

    csv_path = FIGURES_DIR / f"{STEM}_runs.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
