"""Focal loss sweep: minipigs vs monkeys (fold 0).

Fetches finished runs from paired focal-loss sweeps, reports best configs
by max val F1, contrasts ``weight_decay=0.08`` vs stronger WD, and compares
to prior opt-HP / class-weight CE baselines.

Usage:
    uv run python analysis/20260807-LS-focal-loss.py
    uv run python analysis/20260807-LS-focal-loss.py --cached
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

SWEEP_IDS: dict[str, str] = {
    "minipigs": "jotbhxmv",
    "monkeys": "jwdf3c4z",
}

# Fold-0 CE baselines (default capacity 256/4/8/8).
OPT_BASELINE_RUN_IDS: dict[str, str] = {
    "minipigs": "skkz2nec",
    "monkeys": "ljqfklu4",
}
CW_BASELINE_RUN_IDS: dict[str, str] = {
    "minipigs": "wj09rzw3",  # smoothing=0.75
    "monkeys": "vv4a5uv7",  # smoothing=1.0
}
CW_SMOOTHING_USED: dict[str, float] = {
    "minipigs": 0.75,
    "monkeys": 1.0,
}
STRONG_WD: dict[str, float] = {
    "minipigs": 0.1,
    "monkeys": 0.3,
}

FOLD = 0
TASK = "neurosoft_acoustic_stim_8band"
METRIC_KEYS = {
    "f1": f"val/{TASK}_f1",
    "auroc": f"val/{TASK}_auroc",
    "precision": f"val/{TASK}_precision",
    "recall": f"val/{TASK}_recall",
    "balanced_acc": f"val/{TASK}_balanced_acc",
}
METRICS = list(METRIC_KEYS)

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
    return {
        "gamma": config.get("task_loss.gamma"),
        "alpha_smoothing": config.get("task_loss.alpha_smoothing"),
        "label_smoothing": config.get("task_loss.label_smoothing"),
        "weight_decay": config.get("hyperparameters.weight_decay"),
        "learning_rate": config.get("hyperparameters.learning_rate"),
        "atn_dropout": config.get("model.atn_dropout"),
        "tokenizer": config.get("model/tokenizer"),
        "fold": config.get("hyperparameters.fold_number"),
        "split_type": config.get("data.split_type"),
        "grad_clip": config.get("trainer.gradient_clip_val"),
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
            out[key] = float(val_min) if isinstance(val_min, (int, float)) else None
        else:
            out[key] = float(history[key].max())
    return out


def _attach_metrics(run: Any, row: dict[str, Any]) -> dict[str, Any]:
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


def _fetch_one(run_id: str, species_hint: str, sweep_id: str) -> dict[str, Any] | None:
    api = wandb.Api()
    run = api.run(_run_path(run_id))
    config = dict(run.config)
    hps = _extract_hps(config)
    if hps.get("fold") not in (FOLD, str(FOLD)):
        return None
    row: dict[str, Any] = {
        "species": _species_from_run(run, fallback=species_hint),
        "sweep_id": sweep_id,
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        **hps,
    }
    return _attach_metrics(run, row)


def fetch_finished_runs(api: wandb.Api | None = None) -> pd.DataFrame:
    if api is None:
        api = wandb.Api()
    stubs: list[tuple[str, str, str]] = []
    for species, sweep_id in SWEEP_IDS.items():
        sweep = api.sweep(_sweep_path(sweep_id))
        for run in sweep.runs:
            if run.state == "finished":
                stubs.append((run.id, species, sweep_id))

    print(f"Fetching {len(stubs)} finished runs ({N_WORKERS} workers)...")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one, rid, species, sid): rid
            for rid, species, sid in stubs
        }
        done = 0
        for fut in as_completed(futures):
            row = fut.result()
            if row is not None:
                rows.append(row)
            done += 1
            if done % 10 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} kept={len(rows)}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in [
        "gamma",
        "alpha_smoothing",
        "label_smoothing",
        "weight_decay",
        "learning_rate",
        *METRICS,
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["species", "f1"], ascending=[True, False]).reset_index(
        drop=True
    )


def fetch_baselines(api: wandb.Api | None = None) -> pd.DataFrame:
    if api is None:
        api = wandb.Api()
    rows: list[dict[str, Any]] = []
    for species, run_id in OPT_BASELINE_RUN_IDS.items():
        run = api.run(_run_path(run_id))
        row = {
            "label": "opt_baseline_no_cw",
            "species": species,
            "run_id": run.id,
            "run_name": run.name,
            "weight_decay": (run.config or {}).get("hyperparameters.weight_decay"),
            "loss": "CE",
        }
        rows.append(_attach_metrics(run, row))
    for species, run_id in CW_BASELINE_RUN_IDS.items():
        run = api.run(_run_path(run_id))
        row = {
            "label": f"cw_baseline_s{CW_SMOOTHING_USED[species]:g}",
            "species": species,
            "run_id": run.id,
            "run_name": run.name,
            "weight_decay": (run.config or {}).get("hyperparameters.weight_decay"),
            "loss": "CE+CW",
        }
        rows.append(_attach_metrics(run, row))
    return pd.DataFrame(rows)


def best_per_species(
    df: pd.DataFrame, *, weight_decay: float | None = None
) -> pd.DataFrame:
    sub = df if weight_decay is None else df.loc[np.isclose(df["weight_decay"], weight_decay)]
    if sub.empty:
        return sub
    idx = sub.groupby("species")["f1"].idxmax()
    cols = [
        "species",
        "gamma",
        "alpha_smoothing",
        "label_smoothing",
        "weight_decay",
        *METRICS,
        "run_name",
        "run_id",
    ]
    return sub.loc[idx, cols].sort_values("species").reset_index(drop=True)


def _fmt_table(df: pd.DataFrame) -> str:
    out = df.copy()
    float_cols = [
        c
        for c in out.columns
        if c
        in {
            "gamma",
            "alpha_smoothing",
            "label_smoothing",
            "weight_decay",
            *METRICS,
            "baseline_f1",
            "best_f1",
            "delta_f1",
            "baseline_auroc",
            "best_auroc",
            "delta_auroc",
        }
        or c.startswith("delta_")
    ]
    for c in float_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return out.to_string(index=False)


def print_tables(df: pd.DataFrame, baselines: pd.DataFrame) -> None:
    overall = best_per_species(df)
    print("\n=== Best focal config per species (overall max val F1) ===")
    print(_fmt_table(overall))

    print("\n=== Best focal config at WD=0.08 ===")
    best_wd08 = best_per_species(df, weight_decay=0.08)
    print(_fmt_table(best_wd08))

    print("\n=== Best focal config at strong WD (0.1 / 0.3) ===")
    strong_rows = []
    for species, wd in STRONG_WD.items():
        b = best_per_species(df.loc[df["species"] == species], weight_decay=wd)
        if not b.empty:
            strong_rows.append(b)
    best_strong = pd.concat(strong_rows, ignore_index=True) if strong_rows else pd.DataFrame()
    print(_fmt_table(best_strong))

    print("\n=== CE baselines (fold 0) ===")
    print(
        _fmt_table(
            baselines[
                ["label", "species", "weight_decay", *METRICS, "run_id"]
            ].sort_values(["species", "label"])
        )
    )

    # Deltas vs baselines for overall best and WD slices
    for title, best in [
        ("overall best focal", overall),
        ("best focal @ WD=0.08", best_wd08),
        ("best focal @ strong WD", best_strong),
    ]:
        rows = []
        for _, brow in baselines.iterrows():
            species = brow["species"]
            match = best.loc[best["species"] == species]
            if match.empty:
                continue
            bbest = match.iloc[0]
            rows.append(
                {
                    "species": species,
                    "focal_slice": title,
                    "baseline": brow["label"],
                    "baseline_f1": brow["f1"],
                    "best_f1": bbest["f1"],
                    "delta_f1": bbest["f1"] - brow["f1"],
                    "baseline_auroc": brow["auroc"],
                    "best_auroc": bbest["auroc"],
                    "delta_auroc": bbest["auroc"] - brow["auroc"],
                    "gamma": bbest["gamma"],
                    "alpha_smoothing": bbest["alpha_smoothing"],
                    "label_smoothing": bbest["label_smoothing"],
                    "weight_decay": bbest["weight_decay"],
                    "best_run": bbest["run_id"],
                    "baseline_run": brow["run_id"],
                }
            )
        print(f"\n=== {title} vs baselines (delta = focal − baseline) ===")
        print(_fmt_table(pd.DataFrame(rows)))

    # WD contrast: best@0.08 vs best@strong within species
    print("\n=== WD contrast: best@0.08 vs best@strong WD ===")
    contrast = []
    for species in sorted(df["species"].unique()):
        a = best_wd08.loc[best_wd08["species"] == species]
        b = best_strong.loc[best_strong["species"] == species]
        if a.empty or b.empty:
            continue
        a0, b0 = a.iloc[0], b.iloc[0]
        contrast.append(
            {
                "species": species,
                "wd08_f1": a0["f1"],
                "strong_f1": b0["f1"],
                "delta_f1_strong_minus_08": b0["f1"] - a0["f1"],
                "wd08_auroc": a0["auroc"],
                "strong_auroc": b0["auroc"],
                "delta_auroc_strong_minus_08": b0["auroc"] - a0["auroc"],
                "wd08_gamma": a0["gamma"],
                "strong_gamma": b0["gamma"],
                "wd08_run": a0["run_id"],
                "strong_run": b0["run_id"],
                "strong_wd": STRONG_WD[species],
            }
        )
    print(_fmt_table(pd.DataFrame(contrast)))

    # Best F1 by gamma within each WD slice (max over alpha/label smoothing)
    print("\n=== Best F1 by species × WD × gamma (max over alpha/label smoothing) ===")
    for species in sorted(df["species"].unique()):
        sp = df.loc[df["species"] == species]
        pivot = (
            sp.groupby(["weight_decay", "gamma"])["f1"]
            .max()
            .unstack("gamma")
            .sort_index()
        )
        print(f"\n{species}:")
        print(pivot.map(lambda x: f"{x:.4f}" if pd.notna(x) else "").to_string())

    print("\n=== Top-5 focal configs per species ===")
    top = (
        df.sort_values(["species", "f1"], ascending=[True, False])
        .groupby("species", group_keys=False)
        .head(5)
    )
    print(
        _fmt_table(
            top[
                [
                    "species",
                    "gamma",
                    "alpha_smoothing",
                    "label_smoothing",
                    "weight_decay",
                    *METRICS,
                    "run_id",
                ]
            ]
        )
    )


def plot_best_vs_baselines(df: pd.DataFrame, baselines: pd.DataFrame) -> Path:
    overall = best_per_species(df)
    best_wd08 = best_per_species(df, weight_decay=0.08)
    strong_rows = []
    for species, wd in STRONG_WD.items():
        b = best_per_species(df.loc[df["species"] == species], weight_decay=wd)
        if not b.empty:
            strong_rows.append(b)
    best_strong = pd.concat(strong_rows, ignore_index=True)

    species_list = sorted(overall["species"].unique())
    series = {
        "opt_CE": ("opt_baseline_no_cw", baselines),
        "CW_CE": ("cw", baselines),
        "focal_WD0.08": None,
        "focal_strongWD": None,
        "focal_best": None,
    }
    colors = {
        "opt_CE": "#9e9e9e",
        "CW_CE": "#2ca02c",
        "focal_WD0.08": "#4C72B0",
        "focal_strongWD": "#DD8452",
        "focal_best": "#8172B3",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    width = 0.15
    x = np.arange(len(species_list))

    def _vals(metric: str, key: str) -> list[float]:
        out = []
        for species in species_list:
            if key == "opt_CE":
                out.append(
                    float(
                        baselines.loc[
                            (baselines["species"] == species)
                            & (baselines["label"] == "opt_baseline_no_cw"),
                            metric,
                        ].iloc[0]
                    )
                )
            elif key == "CW_CE":
                out.append(
                    float(
                        baselines.loc[
                            (baselines["species"] == species)
                            & baselines["label"].astype(str).str.startswith("cw_"),
                            metric,
                        ].iloc[0]
                    )
                )
            elif key == "focal_WD0.08":
                out.append(
                    float(best_wd08.loc[best_wd08["species"] == species, metric].iloc[0])
                )
            elif key == "focal_strongWD":
                out.append(
                    float(
                        best_strong.loc[best_strong["species"] == species, metric].iloc[0]
                    )
                )
            else:
                out.append(
                    float(overall.loc[overall["species"] == species, metric].iloc[0])
                )
        return out

    for ax, metric in zip(axes, ["f1", "auroc"]):
        for i, key in enumerate(series):
            vals = _vals(metric, key)
            offset = (i - (len(series) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=key,
                color=colors[key],
                edgecolor="white",
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(species_list)
        ax.set_ylabel(f"Max val {TASK} {metric}")
        ax.set_title(metric.upper())
        ax.legend(fontsize=7, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Focal loss vs CE baselines (fold 0, best configs)", y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_best_vs_baselines.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_f1_by_gamma(df: pd.DataFrame) -> Path:
    """Best F1 vs gamma for WD=0.08 and strong WD (max over other focal HPs)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    for ax, species in zip(axes, sorted(df["species"].unique())):
        sp = df.loc[df["species"] == species]
        for wd, style, label in [
            (0.08, "-o", "WD=0.08"),
            (STRONG_WD[species], "--s", f"WD={STRONG_WD[species]}"),
        ]:
            sub = sp.loc[np.isclose(sp["weight_decay"], wd)]
            g = (
                sub.groupby("gamma")["f1"]
                .max()
                .sort_index()
            )
            ax.plot(g.index, g.values, style, label=label, linewidth=2, markersize=7)
            for x, y in zip(g.index, g.values):
                ax.text(x, y + 0.002, f"{y:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xlabel("task_loss.gamma")
        ax.set_ylabel(f"Max val {TASK} F1")
        ax.set_title(species)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([0.5, 1.0, 1.5, 2.0])
    fig.suptitle("Best F1 vs focal γ (max over α / label smoothing)", y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_f1_by_gamma.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_heatmap_gamma_alpha(df: pd.DataFrame) -> list[Path]:
    """Fix WD; axes gamma × alpha_smoothing; cell = max F1 over label_smoothing."""
    paths: list[Path] = []
    for species in sorted(df["species"].unique()):
        sp = df.loc[df["species"] == species]
        wds = [0.08, STRONG_WD[species]]
        fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
        for col_i, wd in enumerate(wds):
            sub = sp.loc[np.isclose(sp["weight_decay"], wd)]
            for row_i, metric in enumerate(["f1", "auroc"]):
                ax = axes[row_i][col_i]
                grid = (
                    sub.groupby(["alpha_smoothing", "gamma"])[metric]
                    .max()
                    .unstack("gamma")
                    .sort_index()
                )
                if grid.empty:
                    ax.set_visible(False)
                    continue
                im = ax.imshow(grid.values, aspect="auto", cmap="viridis")
                ax.set_xticks(range(len(grid.columns)))
                ax.set_xticklabels([str(c) for c in grid.columns])
                ax.set_yticks(range(len(grid.index)))
                ax.set_yticklabels([str(i) for i in grid.index])
                ax.set_xlabel("gamma")
                ax.set_ylabel("alpha_smoothing")
                ax.set_title(f"{metric.upper()} | WD={wd}")
                mean_v = float(np.nanmean(grid.values))
                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        val = grid.values[i, j]
                        if np.isnan(val):
                            continue
                        ax.text(
                            j,
                            i,
                            f"{val:.3f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if val >= mean_v else "black",
                        )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"{species}: max {TASK} over label_smoothing "
            f"(axes γ × α-smoothing; fixed WD)",
            y=1.02,
        )
        fig.tight_layout()
        out = FIGURES_DIR / f"{STEM}_heatmap_{species}_gamma_alpha.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")
        paths.append(out)
    return paths


def main() -> None:
    import sys

    use_cached = "--cached" in sys.argv
    print(
        f"Resolved: sweeps {SWEEP_IDS}, group={GROUP}, project={PROJECT}, "
        f"entity={ENTITY}, fold={FOLD}"
    )
    api = wandb.Api()
    csv_path = FIGURES_DIR / f"{STEM}_runs.csv"
    base_csv = FIGURES_DIR / f"{STEM}_baselines.csv"

    if use_cached and csv_path.exists():
        print(f"Loading cached runs from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = fetch_finished_runs(api)
        if df.empty:
            raise SystemExit("No finished runs found.")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    if use_cached and base_csv.exists():
        print(f"Loading cached baselines from {base_csv}")
        baselines = pd.read_csv(base_csv)
    else:
        baselines = fetch_baselines(api)
        baselines.to_csv(base_csv, index=False)
        print(f"Saved: {base_csv}")

    print(
        f"Loaded {len(df)} runs | species counts: "
        f"{df['species'].value_counts().to_dict()}"
    )
    print_tables(df, baselines)
    plot_best_vs_baselines(df, baselines)
    plot_f1_by_gamma(df)
    plot_heatmap_gamma_alpha(df)


if __name__ == "__main__":
    main()
