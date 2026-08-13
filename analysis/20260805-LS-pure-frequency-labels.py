"""Pure-frequency labels vs multi-frequency 8-band grouping.

Compares ``pure_freq`` sweeps (one stim frequency per class label) against
matched class-weight baselines that used the original multi-frequency
band mapping. Primary analysis: ``weight_decay=0.08`` with species-optimal
CW smoothing.

Usage:
    uv run python analysis/20260805-LS-pure-frequency-labels.py
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

# Pure-frequency label sweeps (mode=auto + species-optimal smoothing).
SWEEP_IDS: dict[str, str] = {
    "minipigs": "w8y76p9g",
    "monkeys": "xtnzcpor",
}

# Matched CW baselines (original multi-frequency 8-band mapping).
CW_BASELINE_SWEEP_IDS: dict[str, str] = {
    "minipigs": "w74jfier",
    "monkeys": "nxx4a4pn",
}
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
LABEL_ORDER = ["multi-freq bands", "pure-freq"]

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
        "weight_decay": config.get(
            "hyperparameters.weight_decay", hp.get("weight_decay")
        ),
        "learning_rate": config.get(
            "hyperparameters.learning_rate", hp.get("learning_rate")
        ),
        "tokenizer": tokenizer,
        "atn_dropout": config.get("model.atn_dropout", model.get("atn_dropout")),
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
    label_scheme: str,
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
        "label_scheme": label_scheme,
        **_extract_meta(config),
    }
    if run.state != "finished":
        for m in METRICS:
            row[m] = None
        return row

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

    stubs: list[tuple[str, str, str, str]] = []
    for species, sweep_id in SWEEP_IDS.items():
        for run in api.sweep(_sweep_path(sweep_id)).runs:
            stubs.append((run.id, species, sweep_id, "pure-freq"))
    for species, sweep_id in CW_BASELINE_SWEEP_IDS.items():
        for run in api.sweep(_sweep_path(sweep_id)).runs:
            stubs.append((run.id, species, sweep_id, "multi-freq bands"))

    print(f"Fetching {len(stubs)} runs ({N_WORKERS} workers)...")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(
                _fetch_one, rid, sp, sid, label_scheme=scheme
            ): rid
            for rid, sp, sid, scheme in stubs
        }
        done = 0
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 10 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}")

    df = pd.DataFrame(rows)
    for col in ["fold", "weight_decay", "smoothing", "learning_rate", *METRICS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def primary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Finished runs at wd=0.08; baselines filtered to species-optimal smoothing."""
    parts: list[pd.DataFrame] = []

    pure = df.loc[
        (df["state"] == "finished")
        & (df["label_scheme"] == "pure-freq")
        & (df["weight_decay"] == OPTIMAL_WEIGHT_DECAY)
    ]
    parts.append(pure)

    for species, sm in CW_BASELINE_SMOOTHING.items():
        base = df.loc[
            (df["state"] == "finished")
            & (df["label_scheme"] == "multi-freq bands")
            & (df["species"] == species)
            & (df["smoothing"] == sm)
            & (df["weight_decay"] == OPTIMAL_WEIGHT_DECAY)
        ]
        parts.append(base)

    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(
        ["species", "label_scheme", "fold"]
    ).reset_index(drop=True)


def mean_table(prim: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for species in SPECIES_ORDER:
        for scheme in LABEL_ORDER:
            block = prim.loc[
                (prim["species"] == species) & (prim["label_scheme"] == scheme)
            ]
            if block.empty:
                continue
            row: dict[str, Any] = {
                "species": species,
                "label_scheme": scheme,
                "n_folds": int(len(block)),
            }
            for m in METRICS:
                row[f"{m}_mean"] = float(block[m].mean())
                row[f"{m}_std"] = (
                    float(block[m].std(ddof=1)) if len(block) > 1 else 0.0
                )
            rows.append(row)
    return pd.DataFrame(rows)


def delta_table(means: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for species in SPECIES_ORDER:
        multi = means.loc[
            (means["species"] == species)
            & (means["label_scheme"] == "multi-freq bands")
        ]
        pure = means.loc[
            (means["species"] == species) & (means["label_scheme"] == "pure-freq")
        ]
        if multi.empty or pure.empty:
            continue
        m = multi.iloc[0]
        p = pure.iloc[0]
        row: dict[str, Any] = {"species": species}
        for metric in METRICS:
            row[f"multi_{metric}"] = float(m[f"{metric}_mean"])
            row[f"pure_{metric}"] = float(p[f"{metric}_mean"])
            row[f"delta_{metric}"] = float(p[f"{metric}_mean"]) - float(
                m[f"{metric}_mean"]
            )
            denom = float(m[f"{metric}_mean"])
            row[f"rel_delta_{metric}"] = (
                row[f"delta_{metric}"] / denom if denom else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def best_max_pure(prim: pd.DataFrame) -> pd.DataFrame:
    pure = prim.loc[prim["label_scheme"] == "pure-freq"]
    idx = pure.groupby("species")["f1"].idxmax()
    cols = [
        "species",
        "fold",
        "weight_decay",
        "smoothing",
        *METRICS,
        "run_name",
        "run_id",
        "sweep_id",
    ]
    return pure.loc[idx, cols].reset_index(drop=True)


def plot_f1_by_scheme(means: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(LABEL_ORDER))
    width = 0.35
    colors = {"minipigs": "#4C78A8", "monkeys": "#F58518"}

    for i, species in enumerate(SPECIES_ORDER):
        sub = means.loc[means["species"] == species].set_index("label_scheme")
        ys = [
            sub.loc[s, "f1_mean"] if s in sub.index else np.nan
            for s in LABEL_ORDER
        ]
        es = [
            sub.loc[s, "f1_std"] if s in sub.index else np.nan
            for s in LABEL_ORDER
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
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_ylabel("max val F1 (mean ± std over folds)")
    ax.set_title("Pure-freq vs multi-freq band labels (wd=0.08, CW matched)")
    ax.legend(title="Species")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_f1_by_scheme.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_delta_metrics(deltas: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(METRICS))
    width = 0.35
    colors = {"minipigs": "#4C78A8", "monkeys": "#F58518"}

    for i, species in enumerate(SPECIES_ORDER):
        sub = deltas.loc[deltas["species"] == species]
        if sub.empty:
            continue
        ys = [float(sub.iloc[0][f"delta_{m}"]) for m in METRICS]
        ax.bar(
            x + (i - 0.5) * width,
            ys,
            width,
            label=species,
            color=colors[species],
        )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylabel("pure-freq − multi-freq (fold mean)")
    ax.set_title("Metric change under pure-frequency labels")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_delta_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    print(f"Project: {PROJECT}")
    print(f"Group:   {GROUP}")
    print(f"Pure-freq sweeps: {SWEEP_IDS}")
    print(f"CW multi-freq baselines: {CW_BASELINE_SWEEP_IDS}")
    print(f"CW smoothing: {CW_BASELINE_SMOOTHING}")
    print(f"Primary weight_decay: {OPTIMAL_WEIGHT_DECAY}")

    df = fetch_runs()
    csv_path = FIGURES_DIR / f"{STEM}_runs.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved all runs → {csv_path}")

    prim = primary_df(df)
    print(f"\nPrimary finished cells: {len(prim)}")

    print("\n=== Best pure-freq single-run F1 (wd=0.08) ===")
    print(best_max_pure(prim).to_string(index=False))

    means = mean_table(prim)
    print("\n=== Fold mean ± std ===")
    disp = means.copy()
    for m in METRICS:
        disp[m] = disp.apply(
            lambda r, mm=m: f"{r[f'{mm}_mean']:.4f}±{r[f'{mm}_std']:.4f}",
            axis=1,
        )
    print(
        disp[["species", "label_scheme", "n_folds", *METRICS]].to_string(
            index=False
        )
    )

    deltas = delta_table(means)
    print("\n=== Pure − multi (fold means) ===")
    ddisp = deltas.copy()
    for m in METRICS:
        ddisp[f"Δ{m}"] = ddisp.apply(
            lambda r, mm=m: (
                f"{r[f'delta_{mm}']:+.4f} "
                f"({100 * r[f'rel_delta_{mm}']:+.1f}%)"
            ),
            axis=1,
        )
    print(
        ddisp[["species", "Δf1", "Δauroc", "Δprecision", "Δrecall"]].to_string(
            index=False
        )
    )

    print("\n=== Full primary grid ===")
    cols = [
        "species",
        "label_scheme",
        "fold",
        "smoothing",
        *METRICS,
        "run_id",
        "sweep_id",
    ]
    print(
        prim.sort_values(["species", "label_scheme", "fold"])[cols].to_string(
            index=False
        )
    )

    extra = df.loc[
        (df["state"] == "finished")
        & (df["label_scheme"] == "pure-freq")
        & (df["weight_decay"] != OPTIMAL_WEIGHT_DECAY)
    ]
    if not extra.empty:
        print("\n=== Secondary: pure-freq non-optimal weight_decay ===")
        g = (
            extra.groupby(["species", "weight_decay"])[METRICS]
            .agg(["mean", "std", "count"])
        )
        print(g)

    fig1 = plot_f1_by_scheme(means)
    fig2 = plot_delta_metrics(deltas)
    print(f"\nFigures:\n  {fig1}\n  {fig2}")


if __name__ == "__main__":
    main()
