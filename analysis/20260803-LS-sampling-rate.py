"""Post-CNN target token rate (sampling rate) sweep: minipigs vs monkeys.

Fetches finished runs from paired species sweeps that vary
``model.tokenizer.temporal_embedding.target_token_rate`` (50 / 200 Hz)
with species-optimal HPs frozen, and compares against 100 Hz CW baselines
from the class-weight smoothing sweeps. Primary analysis uses
``weight_decay=0.08`` (species optima).

Usage:
    uv run python analysis/20260803-LS-sampling-rate.py
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

# Paired species sweeps (token rate @ optimal HPs + CW).
SWEEP_IDS: dict[str, str] = {
    "minipigs": "04jorgw5",
    "monkeys": "eh63y1v7",
}

# 100 Hz CW baselines (species-optimal smoothing) from class-weight sweep.
CW_BASELINE_SWEEP_IDS: dict[str, str] = {
    "minipigs": "w74jfier",
    "monkeys": "nxx4a4pn",
}
# Species-optimal CW smoothing used as 100 Hz reference.
CW_BASELINE_SMOOTHING: dict[str, float] = {
    "minipigs": 0.75,
    "monkeys": 1.0,
}

OPTIMAL_WEIGHT_DECAY = 0.08

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
RATE_ORDER = [50, 100, 200]

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
    cw = config.get("class_weights") or {}

    rate = config.get("model.tokenizer.temporal_embedding.target_token_rate")
    if rate is None:
        tok = model.get("tokenizer") if isinstance(model, dict) else None
        te = (tok or {}).get("temporal_embedding") if isinstance(tok, dict) else {}
        if isinstance(te, dict):
            rate = te.get("target_token_rate")

    smoothing = config.get("class_weights.smoothing")
    if smoothing is None and isinstance(cw, dict):
        smoothing = cw.get("smoothing")

    fold = config.get("hyperparameters.fold_number")
    if fold is None:
        fold = hp.get("fold_number")

    wd = config.get("hyperparameters.weight_decay", hp.get("weight_decay"))

    tokenizer = config.get("model/tokenizer")
    if tokenizer is None:
        tok = model.get("tokenizer")
        if isinstance(tok, dict):
            tokenizer = tok.get("_name_") or tok.get("name") or str(tok)
        else:
            tokenizer = tok

    return {
        "target_token_rate": rate,
        "weight_decay": wd,
        "smoothing": smoothing,
        "fold": fold,
        "tokenizer": tokenizer,
        "atn_dropout": config.get("model.atn_dropout", model.get("atn_dropout")),
        "learning_rate": config.get(
            "hyperparameters.learning_rate", hp.get("learning_rate")
        ),
        "split_type": config.get("data.split_type")
        or (config.get("data") or {}).get("split_type"),
    }


def _summary_max(run: Any, wandb_key: str) -> float | None:
    val = unwrap_summary_value(run.summary.get(wandb_key), "max")
    return float(val) if isinstance(val, (int, float)) else None


def _history_maxes(run: Any, wandb_keys: list[str]) -> dict[str, float | None]:
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
    is_baseline: bool,
) -> dict[str, Any] | None:
    api = wandb.Api()
    run = api.run(_run_path(run_id))
    if run.state != "finished":
        return {
            "species": species_hint,
            "sweep_id": sweep_id,
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "is_baseline": is_baseline,
            **_extract_meta(dict(run.config)),
            **{m: None for m in METRICS},
        }

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


def fetch_runs(api: wandb.Api | None = None) -> pd.DataFrame:
    if api is None:
        api = wandb.Api()

    stubs: list[tuple[str, str, str, bool]] = []
    for species, sweep_id in SWEEP_IDS.items():
        for run in api.sweep(_sweep_path(sweep_id)).runs:
            stubs.append((run.id, species, sweep_id, False))

    for species, sweep_id in CW_BASELINE_SWEEP_IDS.items():
        for run in api.sweep(_sweep_path(sweep_id)).runs:
            stubs.append((run.id, species, sweep_id, True))

    print(f"Fetching {len(stubs)} runs ({N_WORKERS} workers)...")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one, rid, sp, sid, is_baseline=is_base): rid
            for rid, sp, sid, is_base in stubs
        }
        done = 0
        for fut in as_completed(futures):
            row = fut.result()
            if row is not None:
                rows.append(row)
            done += 1
            if done % 10 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}")

    df = pd.DataFrame(rows)
    for col in [
        "target_token_rate",
        "weight_decay",
        "smoothing",
        "fold",
        "learning_rate",
        *METRICS,
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def primary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Finished runs at optimal weight_decay; CW baselines at optimal smoothing."""
    sr = df.loc[
        (~df["is_baseline"])
        & (df["state"] == "finished")
        & (df["weight_decay"] == OPTIMAL_WEIGHT_DECAY)
    ].copy()
    sr["source"] = "sampling_rate"

    base_parts: list[pd.DataFrame] = []
    for species, sm in CW_BASELINE_SMOOTHING.items():
        part = df.loc[
            df["is_baseline"]
            & (df["state"] == "finished")
            & (df["species"] == species)
            & (df["smoothing"] == sm)
            & (df["weight_decay"] == OPTIMAL_WEIGHT_DECAY)
        ].copy()
        part["source"] = "baseline_100hz"
        # Ensure rate labeled 100
        part["target_token_rate"] = 100.0
        base_parts.append(part)

    out = pd.concat([sr, *base_parts], ignore_index=True)
    return out.sort_values(
        ["species", "target_token_rate", "fold"]
    ).reset_index(drop=True)


def mean_table(prim: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for species in SPECIES_ORDER:
        for rate in RATE_ORDER:
            block = prim.loc[
                (prim["species"] == species)
                & (prim["target_token_rate"] == rate)
            ]
            if block.empty:
                continue
            row: dict[str, Any] = {
                "species": species,
                "target_token_rate": rate,
                "n_folds": int(len(block)),
                "folds": ",".join(
                    str(int(f)) for f in sorted(block["fold"].dropna().unique())
                ),
            }
            for m in METRICS:
                row[f"{m}_mean"] = float(block[m].mean())
                row[f"{m}_std"] = (
                    float(block[m].std(ddof=1)) if len(block) > 1 else 0.0
                )
            rows.append(row)
    return pd.DataFrame(rows)


def best_max_per_species(prim: pd.DataFrame) -> pd.DataFrame:
    """Best single finished run by F1 among 50/200 (not baseline)."""
    sr = prim.loc[prim["source"] == "sampling_rate"]
    idx = sr.groupby("species")["f1"].idxmax()
    cols = [
        "species",
        "target_token_rate",
        "fold",
        "weight_decay",
        *METRICS,
        "run_name",
        "run_id",
        "sweep_id",
    ]
    return sr.loc[idx, cols].reset_index(drop=True)


def best_mean_rate(means: pd.DataFrame) -> pd.DataFrame:
    """Best target_token_rate by mean F1 including 100 Hz baseline."""
    idx = means.groupby("species")["f1_mean"].idxmax()
    return means.loc[idx].reset_index(drop=True)


def plot_f1_by_rate(means: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(RATE_ORDER))
    width = 0.35
    colors = {"minipigs": "#4C78A8", "monkeys": "#F58518"}

    for i, species in enumerate(SPECIES_ORDER):
        sub = means.loc[means["species"] == species].set_index(
            "target_token_rate"
        )
        ys = [
            sub.loc[r, "f1_mean"] if r in sub.index else np.nan
            for r in RATE_ORDER
        ]
        es = [
            sub.loc[r, "f1_std"] if r in sub.index else np.nan
            for r in RATE_ORDER
        ]
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

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r} Hz" for r in RATE_ORDER])
    ax.set_xlabel("target_token_rate (post-CNN)")
    ax.set_ylabel("max val F1 (mean ± std over folds)")
    ax.set_title(
        "Token rate vs F1 (wd=0.08; 100 Hz = CW optimal baseline)"
    )
    ax.legend(title="Species")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_f1_by_rate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_metrics_grid(means: pd.DataFrame) -> Path:
    colors = {"minipigs": "#4C78A8", "monkeys": "#F58518"}
    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 3.6), sharex=True)
    x = np.arange(len(RATE_ORDER))
    width = 0.35

    for ax, metric in zip(axes, METRICS):
        for i, species in enumerate(SPECIES_ORDER):
            sub = means.loc[means["species"] == species].set_index(
                "target_token_rate"
            )
            ys = [
                sub.loc[r, f"{metric}_mean"] if r in sub.index else np.nan
                for r in RATE_ORDER
            ]
            es = [
                sub.loc[r, f"{metric}_std"] if r in sub.index else np.nan
                for r in RATE_ORDER
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
        ax.set_xticklabels([str(r) for r in RATE_ORDER])
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
        if metric == "f1":
            ax.set_ylabel("max val (mean ± std)")

    axes[0].legend(loc="best")
    fig.suptitle(
        "Effect of target_token_rate on validation metrics (wd=0.08)",
        y=1.02,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_metrics_by_rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    print(f"Project: {PROJECT}")
    print(f"Group:   {GROUP}")
    print(f"Sweeps:  {SWEEP_IDS}")
    print(f"CW 100Hz baselines: {CW_BASELINE_SWEEP_IDS}")
    print(f"Primary weight_decay: {OPTIMAL_WEIGHT_DECAY}")

    df = fetch_runs()
    csv_path = FIGURES_DIR / f"{STEM}_runs.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved all runs → {csv_path}")

    # Crash inventory
    crashed = df.loc[(~df["is_baseline"]) & (df["state"] != "finished")]
    if not crashed.empty:
        print("\n=== Crashed / unfinished SR runs ===")
        print(
            crashed[
                [
                    "species",
                    "target_token_rate",
                    "weight_decay",
                    "fold",
                    "state",
                    "run_id",
                ]
            ]
            .sort_values(
                ["species", "target_token_rate", "weight_decay", "fold"]
            )
            .to_string(index=False)
        )

    prim = primary_df(df)
    print(
        f"\nPrimary finished cells (wd={OPTIMAL_WEIGHT_DECAY}): "
        f"{len(prim)} "
        f"(SR={int((prim['source']=='sampling_rate').sum())}, "
        f"100Hz baseline={int((prim['source']=='baseline_100hz').sum())})"
    )

    print("\n=== Best single-run F1 among 50/200 Hz (wd=0.08) ===")
    best_max = best_max_per_species(prim)
    print(best_max.to_string(index=False))

    means = mean_table(prim)
    print("\n=== Fold mean ± std (incl. 100 Hz CW baseline) ===")
    disp = means.copy()
    for m in METRICS:
        disp[m] = disp.apply(
            lambda r, mm=m: f"{r[f'{mm}_mean']:.4f}±{r[f'{mm}_std']:.4f}",
            axis=1,
        )
    print(
        disp[
            ["species", "target_token_rate", "n_folds", "folds", *METRICS]
        ].to_string(index=False)
    )

    print("\n=== Best rate by fold-mean F1 (incl. 100 Hz) ===")
    print(best_mean_rate(means).to_string(index=False))

    print("\n=== Full primary grid ===")
    cols = [
        "species",
        "target_token_rate",
        "fold",
        "source",
        *METRICS,
        "run_id",
        "sweep_id",
    ]
    print(prim[cols].to_string(index=False))

    # Extra WD cells (secondary)
    extra = df.loc[
        (~df["is_baseline"])
        & (df["state"] == "finished")
        & (df["weight_decay"] != OPTIMAL_WEIGHT_DECAY)
    ]
    if not extra.empty:
        print("\n=== Secondary: non-optimal weight_decay (finished) ===")
        g = (
            extra.groupby(["species", "target_token_rate", "weight_decay"])[
                "f1"
            ]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        print(g.to_string(index=False))

    fig1 = plot_f1_by_rate(means)
    fig2 = plot_metrics_grid(means)
    print(f"\nFigures:\n  {fig1}\n  {fig2}")


if __name__ == "__main__":
    main()
