"""Causal vs block split: minipigs vs monkeys (intrasession multisubj).

Compares ``intrasession-causal`` sweeps against the optimal-HP
``intrasession-block`` multisubject baselines at ``weight_decay=0.08``.
Causal runs are fold 0 only; primary deltas are fold-0 matched.

Usage:
    uv run python analysis/20260805-LS-causal-split.py
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
    csv_dir,
    default_entity,
    figures_dir,
    unwrap_summary_value,
)

PROJECT = "auditory_decoding"
GROUP = "NEUROSOFT_INTRASESSION_MULTISUBJ"
ENTITY = default_entity()

# Causal split sweeps (optimal HPs; fold 0).
SWEEP_IDS: dict[str, str] = {
    "minipigs": "5t68w2o3",
    "monkeys": "83o9h925",
}

# Optimal-HP multisubject block baselines (no CW follow-ups).
BASELINE_SWEEP_IDS: dict[str, str] = {
    "minipigs": "47jd29ds",
    "monkeys": "bvcgw95o",
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
SPLIT_ORDER = ["intrasession-block", "intrasession-causal"]

STEM = Path(__file__).stem
FIGURES_DIR = figures_dir(__file__)
CSV_DIR = csv_dir(__file__)
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
        "split_type": config.get("data.split_type")
        or (config.get("data") or {}).get("split_type"),
        "fold": fold,
        "weight_decay": config.get(
            "hyperparameters.weight_decay", hp.get("weight_decay")
        ),
        "learning_rate": config.get(
            "hyperparameters.learning_rate", hp.get("learning_rate")
        ),
        "tokenizer": tokenizer,
        "atn_dropout": config.get(
            "model.atn_dropout", model.get("atn_dropout")
        ),
        "grad_clip": config.get(
            "trainer.gradient_clip_val",
            (config.get("trainer") or {}).get("gradient_clip_val"),
        ),
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

    stubs: list[tuple[str, str, str, bool]] = []
    for species, sweep_id in SWEEP_IDS.items():
        for run in api.sweep(_sweep_path(sweep_id)).runs:
            stubs.append((run.id, species, sweep_id, False))
    for species, sweep_id in BASELINE_SWEEP_IDS.items():
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
            rows.append(fut.result())
            done += 1
            if done % 5 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}")

    df = pd.DataFrame(rows)
    for col in ["fold", "weight_decay", "learning_rate", *METRICS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def primary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Finished runs at optimal weight_decay."""
    return df.loc[
        (df["state"] == "finished")
        & (df["weight_decay"] == OPTIMAL_WEIGHT_DECAY)
    ].copy()


def fold0_comparison(prim: pd.DataFrame) -> pd.DataFrame:
    """Fold-0 matched causal vs block, with absolute and relative F1 delta."""
    rows: list[dict[str, Any]] = []
    for species in SPECIES_ORDER:
        block = prim.loc[
            (prim["species"] == species)
            & (prim["is_baseline"])
            & (prim["fold"] == 0)
        ]
        causal = prim.loc[
            (prim["species"] == species)
            & (~prim["is_baseline"])
            & (prim["fold"] == 0)
        ]
        if block.empty or causal.empty:
            continue
        b = block.iloc[0]
        c = causal.iloc[0]
        row: dict[str, Any] = {
            "species": species,
            "fold": 0,
            "weight_decay": OPTIMAL_WEIGHT_DECAY,
        }
        for m in METRICS:
            row[f"block_{m}"] = float(b[m])
            row[f"causal_{m}"] = float(c[m])
            row[f"delta_{m}"] = float(c[m]) - float(b[m])
            row[f"rel_delta_{m}"] = (
                (float(c[m]) - float(b[m])) / float(b[m])
                if float(b[m]) != 0
                else np.nan
            )
        row["block_run_id"] = b["run_id"]
        row["causal_run_id"] = c["run_id"]
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_fold_means(prim: pd.DataFrame) -> pd.DataFrame:
    base = prim.loc[prim["is_baseline"]]
    rows: list[dict[str, Any]] = []
    for species in SPECIES_ORDER:
        block = base.loc[base["species"] == species]
        if block.empty:
            continue
        row: dict[str, Any] = {
            "species": species,
            "n_folds": int(len(block)),
        }
        for m in METRICS:
            row[f"{m}_mean"] = float(block[m].mean())
            row[f"{m}_std"] = (
                float(block[m].std(ddof=1)) if len(block) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_f1_fold0(comp: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(SPECIES_ORDER))
    width = 0.35
    colors = {"block": "#4C78A8", "causal": "#E45756"}

    block_ys = [
        float(comp.loc[comp["species"] == sp, "block_f1"].iloc[0])
        if (comp["species"] == sp).any()
        else np.nan
        for sp in SPECIES_ORDER
    ]
    causal_ys = [
        float(comp.loc[comp["species"] == sp, "causal_f1"].iloc[0])
        if (comp["species"] == sp).any()
        else np.nan
        for sp in SPECIES_ORDER
    ]

    ax.bar(
        x - width / 2,
        block_ys,
        width,
        label="block (baseline)",
        color=colors["block"],
    )
    ax.bar(
        x + width / 2, causal_ys, width, label="causal", color=colors["causal"]
    )
    ax.set_xticks(x)
    ax.set_xticklabels(SPECIES_ORDER)
    ax.set_ylabel("max val F1 (fold 0, wd=0.08)")
    ax.set_title("Causal vs block split (optimal-HP baselines)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_f1_fold0.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_delta_metrics(comp: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(METRICS))
    width = 0.35
    colors = {"minipigs": "#4C78A8", "monkeys": "#F58518"}

    for i, species in enumerate(SPECIES_ORDER):
        sub = comp.loc[comp["species"] == species]
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
    ax.set_ylabel("causal − block (fold 0, wd=0.08)")
    ax.set_title("Metric drop under causal split")
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
    print(f"Causal sweeps: {SWEEP_IDS}")
    print(f"Block baselines: {BASELINE_SWEEP_IDS}")
    print(f"Primary weight_decay: {OPTIMAL_WEIGHT_DECAY}")

    df = fetch_runs()
    csv_path = CSV_DIR / f"{STEM}_runs.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved all runs → {csv_path}")

    prim = primary_df(df)
    print(f"\nPrimary finished cells (wd={OPTIMAL_WEIGHT_DECAY}): {len(prim)}")

    print("\n=== Fold-0 matched comparison (causal vs block) ===")
    comp = fold0_comparison(prim)
    disp = comp.copy()
    for m in METRICS:
        disp[f"Δ{m}"] = disp.apply(
            lambda r, mm=m: (
                f"{r[f'delta_{mm}']:+.4f} ({100 * r[f'rel_delta_{mm}']:+.1f}%)"
            ),
            axis=1,
        )
    print(
        disp[
            [
                "species",
                "block_f1",
                "causal_f1",
                "Δf1",
                "Δauroc",
                "Δprecision",
                "Δrecall",
                "block_run_id",
                "causal_run_id",
            ]
        ].to_string(index=False)
    )

    print(
        "\n=== Block baseline fold mean±std (context; causal is fold 0 only) ==="
    )
    means = baseline_fold_means(prim)
    mdisp = means.copy()
    for m in METRICS:
        mdisp[m] = mdisp.apply(
            lambda r, mm=m: f"{r[f'{mm}_mean']:.4f}±{r[f'{mm}_std']:.4f}",
            axis=1,
        )
    print(mdisp[["species", "n_folds", *METRICS]].to_string(index=False))

    print("\n=== Full primary grid (wd=0.08) ===")
    cols = [
        "species",
        "split_type",
        "fold",
        "is_baseline",
        *METRICS,
        "run_id",
        "sweep_id",
    ]
    print(
        prim.sort_values(["species", "is_baseline", "fold"])[cols].to_string(
            index=False
        )
    )

    extra = df.loc[
        (df["state"] == "finished")
        & (~df["is_baseline"])
        & (df["weight_decay"] != OPTIMAL_WEIGHT_DECAY)
    ]
    if not extra.empty:
        print("\n=== Secondary: causal non-optimal weight_decay ===")
        print(
            extra[
                [
                    "species",
                    "split_type",
                    "fold",
                    "weight_decay",
                    *METRICS,
                    "run_id",
                ]
            ].to_string(index=False)
        )

    fig1 = plot_f1_fold0(comp)
    fig2 = plot_delta_metrics(comp)
    print(f"\nFigures:\n  {fig1}\n  {fig2}")


if __name__ == "__main__":
    main()
