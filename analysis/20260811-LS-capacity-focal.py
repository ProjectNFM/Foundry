"""Small capacity + focal loss: minipigs vs monkeys (fold 0).

Freezes species-best small-capacity configs from the capacity ablation and
sweeps focal-loss HPs. Compares best combo configs to:
  1. small-capacity CE winners
  2. default-capacity focal winners

Usage:
    uv run python analysis/20260811-LS-capacity-focal.py
    uv run python analysis/20260811-LS-capacity-focal.py --cached
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
    "minipigs": "bvig2bi8",
    "monkeys": "weslebt0",
}

# Small-capacity CE winners (20260805-LS-model-capacity).
CAPACITY_CE_RUN_IDS: dict[str, str] = {
    "minipigs": "ncx1been",  # 32/2/6/6 cef=1/2, wd=0.08, F1=0.3936
    "monkeys": "zrvjtixp",  # 64/4/6/8, wd=0.30, F1=0.5382
}

# Default-capacity focal winners (20260807-LS-focal-loss).
FOCAL_DEFAULT_CAP_RUN_IDS: dict[str, str] = {
    "minipigs": "gebswvlu",  # γ=1.5, α=0.75, ls=0.1, wd=0.08
    "monkeys": "ubdan13a",  # γ=1.0, α=1.0, ls=0.1, wd=0.30
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
    model = config.get("model") or {}
    return {
        "gamma": config.get("task_loss.gamma"),
        "alpha_smoothing": config.get("task_loss.alpha_smoothing"),
        "label_smoothing": config.get("task_loss.label_smoothing"),
        "weight_decay": config.get("hyperparameters.weight_decay"),
        "cw_smoothing": config.get("class_weights.smoothing"),
        "embed_dim": config.get("model.embed_dim", model.get("embed_dim")),
        "depth": config.get("model.depth", model.get("depth")),
        "self_heads": config.get("model.self_heads", model.get("self_heads")),
        "cross_heads": config.get("model.cross_heads", model.get("cross_heads")),
        "channel_emb_fraction": config.get("model.tokenizer.channel_emb_fraction"),
        "tokenizer": config.get("model/tokenizer"),
        "fold": config.get("hyperparameters.fold_number"),
        "learning_rate": config.get("hyperparameters.learning_rate"),
        "atn_dropout": config.get("model.atn_dropout"),
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

    print(f"Fetching {len(stubs)} finished combo runs ({N_WORKERS} workers)...")
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
            if done % 8 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} kept={len(rows)}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in [
        "gamma",
        "alpha_smoothing",
        "label_smoothing",
        "weight_decay",
        "cw_smoothing",
        "embed_dim",
        "depth",
        "self_heads",
        "cross_heads",
        *METRICS,
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["species", "f1"], ascending=[True, False]).reset_index(
        drop=True
    )


def _fetch_reference(
    run_id: str,
    species: str,
    label: str,
    *,
    api: wandb.Api,
) -> dict[str, Any]:
    run = api.run(_run_path(run_id))
    config = dict(run.config)
    model = config.get("model") or {}
    row: dict[str, Any] = {
        "label": label,
        "species": species,
        "run_id": run.id,
        "run_name": run.name,
        "gamma": config.get("task_loss.gamma"),
        "alpha_smoothing": config.get("task_loss.alpha_smoothing"),
        "label_smoothing": config.get("task_loss.label_smoothing"),
        "weight_decay": config.get("hyperparameters.weight_decay"),
        "cw_smoothing": config.get("class_weights.smoothing"),
        "embed_dim": model.get("embed_dim") or config.get("model.embed_dim"),
        "depth": model.get("depth") or config.get("model.depth"),
        "self_heads": model.get("self_heads") or config.get("model.self_heads"),
        "cross_heads": model.get("cross_heads") or config.get("model.cross_heads"),
        "channel_emb_fraction": config.get("model.tokenizer.channel_emb_fraction"),
        "tokenizer": config.get("model/tokenizer"),
    }
    # Capacity CE minipigs: infer cef from channel_emb_dim if needed
    if row["channel_emb_fraction"] is None:
        tok = model.get("tokenizer") if isinstance(model.get("tokenizer"), dict) else {}
        ced = tok.get("channel_emb_dim")
        ed = row["embed_dim"]
        if ced is not None and ed:
            try:
                row["channel_emb_fraction"] = (
                    f"1/{int(round(int(ed) / int(ced)))}"
                )
            except (TypeError, ValueError, ZeroDivisionError):
                pass
    return _attach_metrics(run, row)


def fetch_baselines(api: wandb.Api | None = None) -> pd.DataFrame:
    if api is None:
        api = wandb.Api()
    rows: list[dict[str, Any]] = []
    for species, run_id in CAPACITY_CE_RUN_IDS.items():
        rows.append(
            _fetch_reference(run_id, species, "small_cap_CE", api=api)
        )
    for species, run_id in FOCAL_DEFAULT_CAP_RUN_IDS.items():
        rows.append(
            _fetch_reference(run_id, species, "default_cap_focal", api=api)
        )
    return pd.DataFrame(rows)


def best_per_species(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("species")["f1"].idxmax()
    cols = [
        "species",
        "gamma",
        "alpha_smoothing",
        "label_smoothing",
        "weight_decay",
        "embed_dim",
        "depth",
        "self_heads",
        "cross_heads",
        "channel_emb_fraction",
        *METRICS,
        "run_name",
        "run_id",
    ]
    return df.loc[idx, [c for c in cols if c in df.columns]].sort_values(
        "species"
    ).reset_index(drop=True)


def _fmt_table(df: pd.DataFrame) -> str:
    out = df.copy()
    float_like = set(METRICS) | {
        "gamma",
        "alpha_smoothing",
        "label_smoothing",
        "weight_decay",
        "cw_smoothing",
        "baseline_f1",
        "best_f1",
        "delta_f1",
        "baseline_auroc",
        "best_auroc",
        "delta_auroc",
    }
    for c in out.columns:
        if c in float_like or c.startswith("delta_"):
            out[c] = out[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    for c in ["embed_dim", "depth", "self_heads", "cross_heads"]:
        if c in out.columns:
            out[c] = out[c].map(
                lambda x: f"{int(x)}"
                if pd.notna(x) and float(x) == int(float(x))
                else x
            )
    return out.to_string(index=False)


def print_tables(df: pd.DataFrame, baselines: pd.DataFrame) -> None:
    best = best_per_species(df)
    print("\n=== Best small-cap + focal config per species (max val F1) ===")
    print(_fmt_table(best))

    print("\n=== Reference baselines ===")
    print(
        _fmt_table(
            baselines[
                [
                    c
                    for c in [
                        "label",
                        "species",
                        "embed_dim",
                        "depth",
                        "gamma",
                        "alpha_smoothing",
                        "label_smoothing",
                        "weight_decay",
                        *METRICS,
                        "run_id",
                    ]
                    if c in baselines.columns
                ]
            ].sort_values(["species", "label"])
        )
    )

    rows = []
    for _, brow in baselines.iterrows():
        species = brow["species"]
        bbest = best.loc[best["species"] == species].iloc[0]
        rows.append(
            {
                "species": species,
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
                "best_run": bbest["run_id"],
                "baseline_run": brow["run_id"],
            }
        )
    print("\n=== Best combo vs baselines (delta = combo − baseline) ===")
    print(_fmt_table(pd.DataFrame(rows)))

    print("\n=== Best F1 by species × gamma (max over α / label smoothing) ===")
    for species in sorted(df["species"].unique()):
        g = (
            df.loc[df["species"] == species]
            .groupby("gamma")["f1"]
            .max()
            .sort_index()
        )
        print(f"{species}: " + ", ".join(f"γ={k:g}→{v:.4f}" for k, v in g.items()))

    print("\n=== Full grid (sorted by F1 within species) ===")
    print(
        _fmt_table(
            df[
                [
                    "species",
                    "gamma",
                    "alpha_smoothing",
                    "label_smoothing",
                    *METRICS,
                    "run_id",
                ]
            ]
        )
    )


def plot_best_vs_baselines(df: pd.DataFrame, baselines: pd.DataFrame) -> Path:
    best = best_per_species(df)
    species_list = sorted(best["species"].unique())
    keys = ["small_cap_CE", "default_cap_focal", "small_cap_focal"]
    colors = {
        "small_cap_CE": "#2ca02c",
        "default_cap_focal": "#9e9e9e",
        "small_cap_focal": "#4C72B0",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    width = 0.25
    x = np.arange(len(species_list))

    def _val(species: str, key: str, metric: str) -> float:
        if key == "small_cap_focal":
            return float(best.loc[best["species"] == species, metric].iloc[0])
        return float(
            baselines.loc[
                (baselines["species"] == species) & (baselines["label"] == key),
                metric,
            ].iloc[0]
        )

    for ax, metric in zip(axes, ["f1", "auroc"]):
        for i, key in enumerate(keys):
            vals = [_val(s, key, metric) for s in species_list]
            offset = (i - 1) * width
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
                    fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(species_list)
        ax.set_ylabel(f"Max val {TASK} {metric}")
        ax.set_title(metric.upper())
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Small-cap + focal vs small-cap CE / default-cap focal (fold 0)",
        y=1.02,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_best_vs_baselines.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_f1_by_gamma(df: pd.DataFrame, baselines: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    for ax, species in zip(axes, sorted(df["species"].unique())):
        sp = df.loc[df["species"] == species]
        g = sp.groupby("gamma")["f1"].max().sort_index()
        ax.plot(g.index, g.values, "-o", color="#4C72B0", label="small_cap+focal", lw=2)
        for x, y in zip(g.index, g.values):
            ax.text(x, y + 0.002, f"{y:.3f}", ha="center", va="bottom", fontsize=7)

        for label, color, style in [
            ("small_cap_CE", "#2ca02c", "--"),
            ("default_cap_focal", "#9e9e9e", ":"),
        ]:
            f1 = float(
                baselines.loc[
                    (baselines["species"] == species)
                    & (baselines["label"] == label),
                    "f1",
                ].iloc[0]
            )
            ax.axhline(f1, color=color, linestyle=style, lw=1.5, label=f"{label} ({f1:.3f})")

        ax.set_xlabel("task_loss.gamma")
        ax.set_ylabel(f"Max val {TASK} F1")
        ax.set_title(species)
        ax.legend(fontsize=7)
        ax.set_xticks([0.5, 1.0, 1.5, 2.0])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Best F1 vs γ on small capacity (max over α / ls)", y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_f1_by_gamma.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_heatmap_gamma_alpha(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    for species in sorted(df["species"].unique()):
        sp = df.loc[df["species"] == species]
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
        for ax, metric in zip(axes, ["f1", "auroc"]):
            grid = (
                sp.groupby(["alpha_smoothing", "gamma"])[metric]
                .max()
                .unstack("gamma")
                .sort_index()
            )
            im = ax.imshow(grid.values, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(grid.columns)))
            ax.set_xticklabels([str(c) for c in grid.columns])
            ax.set_yticks(range(len(grid.index)))
            ax.set_yticklabels([str(i) for i in grid.index])
            ax.set_xlabel("gamma")
            ax.set_ylabel("alpha_smoothing")
            ax.set_title(metric.upper())
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
            f"(axes γ × α-smoothing; small capacity frozen)",
            y=1.05,
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
        f"Loaded {len(df)} combo runs | species: "
        f"{df['species'].value_counts().to_dict()}"
    )
    print_tables(df, baselines)
    plot_best_vs_baselines(df, baselines)
    plot_f1_by_gamma(df, baselines)
    plot_heatmap_gamma_alpha(df)


if __name__ == "__main__":
    main()
