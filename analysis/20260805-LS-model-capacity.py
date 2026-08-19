"""Model capacity sweep: best max-val configs vs baselines (fold 0).

Fetches finished fold-0 runs from paired capacity sweeps in
``NEUROSOFT_INTRASESSION_MULTISUBJ``. Focuses on the **best** hyperparameter
combination by max validation F1 (not averages across runs), compares that
best set to prior opt-HP and class-weight baselines, and saves heatmaps over
the capacity grid (fix one parameter; two on the axes).

Sweeps are unfinished; fold > 0 is ignored. Minipigs additionally vary
``channel_emb_fraction`` (concat tokenizer); monkeys use add tokens.

Usage:
    uv run python analysis/20260805-LS-model-capacity.py
    uv run python analysis/20260805-LS-model-capacity.py --cached
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

SWEEP_IDS: dict[str, str] = {
    "minipigs": "ov9f1g0n",
    "monkeys": "104ze4mt",
}

# Fold-0 references from prior multisubject experiments (default capacity:
# embed_dim=256, depth=4, self/cross heads=8). Capacity sweeps already use
# the preferred CW smoothing per species (minipigs 0.75, monkeys 1.0).
OPT_BASELINE_RUN_IDS: dict[str, str] = {
    "minipigs": "skkz2nec",  # 20260727 opt-HP multi-subject, fold 0, no CW
    "monkeys": "ljqfklu4",
}
CW_BASELINE_RUN_IDS: dict[str, str] = {
    "minipigs": "wj09rzw3",  # 20260729 CW smoothing=0.75, fold 0
    "monkeys": "vv4a5uv7",  # 20260729 CW smoothing=1.0, fold 0
}
CW_SMOOTHING_USED: dict[str, float] = {
    "minipigs": 0.75,
    "monkeys": 1.0,
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
    conf = run.config or {}
    for key in ("data/dataset", "data"):
        val = str(conf.get(key, "")).lower()
        if "minipig" in val:
            return "minipigs"
        if "monkey" in val:
            return "monkeys"
    return fallback or "unknown"


def _extract_hps(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "embed_dim": config.get("model.embed_dim"),
        "depth": config.get("model.depth"),
        "self_heads": config.get("model.self_heads"),
        "cross_heads": config.get("model.cross_heads"),
        "channel_emb_fraction": config.get(
            "model.tokenizer.channel_emb_fraction"
        ),
        "tokenizer": config.get("model/tokenizer"),
        "weight_decay": config.get("hyperparameters.weight_decay"),
        "learning_rate": config.get("hyperparameters.learning_rate"),
        "atn_dropout": config.get("model.atn_dropout"),
        "grad_clip": config.get("trainer.gradient_clip_val"),
        "fold": config.get("hyperparameters.fold_number"),
        "split_type": config.get("data.split_type"),
        "batch_size": config.get("hyperparameters.batch_size"),
        "smoothing": config.get("class_weights.smoothing"),
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
    run_id: str, species_hint: str, sweep_id: str
) -> dict[str, Any] | None:
    api = wandb.Api()
    run = api.run(_run_path(run_id))
    config = dict(run.config)
    hps = _extract_hps(config)
    fold = hps.get("fold")
    if fold not in (FOLD, str(FOLD)):
        return None

    row: dict[str, Any] = {
        "species": _species_from_run(run, fallback=species_hint),
        "sweep_id": sweep_id,
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        **hps,
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

    print(
        f"Fetching {len(stubs)} finished stubs; keeping fold={FOLD} only "
        f"({N_WORKERS} workers)..."
    )
    rows: list[dict[str, Any]] = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one, rid, species, sid): (rid, species)
            for rid, species, sid in stubs
        }
        done = 0
        for fut in as_completed(futures):
            row = fut.result()
            done += 1
            if row is None:
                skipped += 1
            else:
                rows.append(row)
            if done % 20 == 0 or done == len(futures):
                print(
                    f"  {done}/{len(futures)} (kept={len(rows)}, skip_fold={skipped})"
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in [
        "embed_dim",
        "depth",
        "self_heads",
        "cross_heads",
        "weight_decay",
        "learning_rate",
        "atn_dropout",
        "grad_clip",
        "smoothing",
        "f1",
        "auroc",
        "precision",
        "recall",
        "balanced_acc",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(
        ["species", "f1"], ascending=[True, False]
    ).reset_index(drop=True)


def best_per_species(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("species")["f1"].idxmax()
    cols = [
        "species",
        "embed_dim",
        "depth",
        "self_heads",
        "cross_heads",
        "channel_emb_fraction",
        "weight_decay",
        "smoothing",
        "tokenizer",
        "f1",
        "auroc",
        "precision",
        "recall",
        "balanced_acc",
        "run_name",
        "run_id",
    ]
    return (
        df.loc[idx, [c for c in cols if c in df.columns]]
        .sort_values("species")
        .reset_index(drop=True)
    )


def _fetch_reference_run(
    run_id: str,
    species: str,
    label: str,
    *,
    api: wandb.Api | None = None,
) -> dict[str, Any]:
    if api is None:
        api = wandb.Api()
    run = api.run(_run_path(run_id))
    config = dict(run.config)
    model = config.get("model") or {}
    tok = (
        model.get("tokenizer")
        if isinstance(model.get("tokenizer"), dict)
        else {}
    )
    channel_emb_dim = tok.get("channel_emb_dim")
    embed_dim = model.get("embed_dim") or config.get("model.embed_dim")
    cef = config.get("model.tokenizer.channel_emb_fraction")
    if cef is None and channel_emb_dim is not None and embed_dim:
        try:
            cef = f"1/{int(round(int(embed_dim) / int(channel_emb_dim)))}"
        except (TypeError, ValueError, ZeroDivisionError):
            cef = None

    row: dict[str, Any] = {
        "label": label,
        "species": species,
        "run_id": run.id,
        "run_name": run.name,
        "embed_dim": embed_dim,
        "depth": model.get("depth") or config.get("model.depth"),
        "self_heads": model.get("self_heads") or config.get("model.self_heads"),
        "cross_heads": model.get("cross_heads")
        or config.get("model.cross_heads"),
        "channel_emb_fraction": cef,
        "tokenizer": config.get("model/tokenizer"),
        "weight_decay": config.get("hyperparameters.weight_decay"),
        "smoothing": config.get("class_weights.smoothing"),
        "fold": config.get("hyperparameters.fold_number"),
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


def fetch_baselines(api: wandb.Api | None = None) -> pd.DataFrame:
    if api is None:
        api = wandb.Api()
    rows: list[dict[str, Any]] = []
    for species, run_id in OPT_BASELINE_RUN_IDS.items():
        rows.append(
            _fetch_reference_run(run_id, species, "opt_baseline_no_cw", api=api)
        )
    for species, run_id in CW_BASELINE_RUN_IDS.items():
        rows.append(
            _fetch_reference_run(
                run_id,
                species,
                f"cw_baseline_s{CW_SMOOTHING_USED[species]:g}",
                api=api,
            )
        )
    return pd.DataFrame(rows)


def _fmt_table(df: pd.DataFrame) -> str:
    out = df.copy()
    metric_float = {
        "weight_decay",
        "smoothing",
        "f1",
        "auroc",
        "precision",
        "recall",
        "balanced_acc",
        "baseline_f1",
        "best_cap_f1",
        "delta_f1",
        "baseline_auroc",
        "best_cap_auroc",
        "delta_auroc",
    }
    for c in out.columns:
        if c in metric_float or (
            c.startswith("delta_")
            and c.endswith(("f1", "auroc", "precision", "recall", "acc"))
        ):
            out[c] = out[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    for c in ["embed_dim", "depth", "self_heads", "cross_heads"]:
        if c in out.columns:
            out[c] = out[c].map(
                lambda x: (
                    f"{int(x)}"
                    if pd.notna(x) and float(x) == int(float(x))
                    else x
                )
            )
    return out.to_string(index=False)


def _pivot_max(
    df: pd.DataFrame,
    index: str,
    columns: str,
    value: str,
    fixed: dict[str, Any] | None = None,
) -> pd.DataFrame:
    sub = df
    if fixed:
        for key, val in fixed.items():
            sub = sub.loc[sub[key] == val]
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby([index, columns], dropna=False)[value]
        .max()
        .unstack(columns)
        .sort_index()
    )


def print_tables(df: pd.DataFrame, baselines: pd.DataFrame) -> None:
    best = best_per_species(df)
    print("\n=== Best model-size configuration per species (max val F1) ===")
    with pd.option_context("display.max_columns", 24, "display.width", 160):
        print(_fmt_table(best))

    print("\n=== Reference baselines (fold 0, default capacity 256/4/8/8) ===")
    base_cols = [
        "label",
        "species",
        "embed_dim",
        "depth",
        "self_heads",
        "cross_heads",
        "channel_emb_fraction",
        "smoothing",
        "f1",
        "auroc",
        "precision",
        "recall",
        "balanced_acc",
        "run_id",
    ]
    with pd.option_context("display.max_columns", 24, "display.width", 160):
        print(
            _fmt_table(
                baselines[
                    [c for c in base_cols if c in baselines.columns]
                ].sort_values(["species", "label"])
            )
        )

    cmp_rows: list[dict[str, Any]] = []
    for _, brow in baselines.iterrows():
        species = brow["species"]
        bbest = best.loc[best["species"] == species].iloc[0]
        cmp_rows.append(
            {
                "species": species,
                "baseline": brow["label"],
                "baseline_f1": brow["f1"],
                "best_cap_f1": bbest["f1"],
                "delta_f1": bbest["f1"] - brow["f1"],
                "baseline_auroc": brow["auroc"],
                "best_cap_auroc": bbest["auroc"],
                "delta_auroc": bbest["auroc"] - brow["auroc"],
                "best_embed_dim": int(bbest["embed_dim"]),
                "best_depth": int(bbest["depth"]),
                "best_self_heads": int(bbest["self_heads"]),
                "best_cross_heads": int(bbest["cross_heads"]),
                "best_run": bbest["run_id"],
                "baseline_run": brow["run_id"],
            }
        )
    print(
        "\n=== Best capacity vs baselines (fold 0; delta = best − baseline) ==="
    )
    with pd.option_context("display.max_columns", 20, "display.width", 160):
        print(_fmt_table(pd.DataFrame(cmp_rows)))

    # Best cell at each (embed_dim, depth) — max over heads/cef/wd — for heatmap context
    print("\n=== Best F1 at each (embed_dim, depth) [max over other HPs] ===")
    for species in sorted(df["species"].unique()):
        pivot = _pivot_max(
            df.loc[df["species"] == species], "depth", "embed_dim", "f1"
        )
        print(f"\n{species}:")
        print(
            pivot.map(lambda x: f"{x:.4f}" if pd.notna(x) else "").to_string()
        )


def _annotate_heatmap(ax: Any, grid: pd.DataFrame) -> None:
    for i, idx in enumerate(grid.index):
        for j, col in enumerate(grid.columns):
            val = grid.iloc[i, j]
            if pd.isna(val):
                continue
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white"
                if val
                >= (
                    np.nanmean(grid.values)
                    if np.isfinite(np.nanmean(grid.values))
                    else 0
                )
                else "black",
            )


def plot_heatmap_embed_dim_vs_depth(df: pd.DataFrame) -> list[Path]:
    """Fix self_heads; axes embed_dim × depth; cell = max metric over remaining HPs."""
    paths: list[Path] = []
    for species in sorted(df["species"].unique()):
        sp = df.loc[df["species"] == species]
        self_vals = sorted(sp["self_heads"].dropna().unique())
        fig, axes = plt.subplots(
            2,
            len(self_vals),
            figsize=(4.2 * len(self_vals), 7.2),
            squeeze=False,
        )
        for col_i, self_h in enumerate(self_vals):
            for row_i, metric in enumerate(["f1", "auroc"]):
                ax = axes[row_i][col_i]
                grid = _pivot_max(
                    sp,
                    index="depth",
                    columns="embed_dim",
                    value=metric,
                    fixed={"self_heads": self_h},
                )
                if grid.empty:
                    ax.set_visible(False)
                    continue
                im = ax.imshow(grid.values, aspect="auto", cmap="viridis")
                ax.set_xticks(range(len(grid.columns)))
                ax.set_xticklabels([str(int(c)) for c in grid.columns])
                ax.set_yticks(range(len(grid.index)))
                ax.set_yticklabels([str(int(i)) for i in grid.index])
                ax.set_xlabel("embed_dim")
                ax.set_ylabel("depth")
                ax.set_title(f"{metric.upper()} | self_heads={int(self_h)}")
                _annotate_heatmap(ax, grid)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"{species}: max {TASK} over remaining HPs "
            f"(axes embed_dim × depth; fixed self_heads)",
            y=1.02,
        )
        fig.tight_layout()
        out = FIGURES_DIR / f"{STEM}_heatmap_{species}_embed_dim_vs_depth.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")
        paths.append(out)
    return paths


def plot_heatmap_heads(df: pd.DataFrame, best: pd.DataFrame) -> list[Path]:
    """Fix embed_dim to species best; axes self_heads × cross_heads."""
    paths: list[Path] = []
    for _, brow in best.iterrows():
        species = brow["species"]
        embed = brow["embed_dim"]
        sp = df.loc[df["species"] == species]
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
        for ax, metric in zip(axes, ["f1", "auroc"]):
            grid = _pivot_max(
                sp,
                index="self_heads",
                columns="cross_heads",
                value=metric,
                fixed={"embed_dim": embed},
            )
            if grid.empty:
                ax.set_visible(False)
                continue
            im = ax.imshow(grid.values, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(grid.columns)))
            ax.set_xticklabels([str(int(c)) for c in grid.columns])
            ax.set_yticks(range(len(grid.index)))
            ax.set_yticklabels([str(int(i)) for i in grid.index])
            ax.set_xlabel("cross_heads")
            ax.set_ylabel("self_heads")
            ax.set_title(f"{metric.upper()} | embed_dim={int(embed)}")
            _annotate_heatmap(ax, grid)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"{species}: max {TASK} over remaining HPs "
            f"(axes self×cross heads; fixed embed_dim={int(embed)})",
            y=1.05,
        )
        fig.tight_layout()
        out = FIGURES_DIR / f"{STEM}_heatmap_{species}_heads.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")
        paths.append(out)
    return paths


def plot_heatmap_minipigs_cef(
    df: pd.DataFrame, best: pd.DataFrame
) -> Path | None:
    """Minipigs only: fix depth to best; axes embed_dim × channel_emb_fraction."""
    brow = best.loc[best["species"] == "minipigs"]
    if brow.empty:
        return None
    depth = brow.iloc[0]["depth"]
    sp = df.loc[df["species"] == "minipigs"].dropna(
        subset=["channel_emb_fraction"]
    )
    if sp.empty:
        return None

    # Order cef as 1/2, 1/3, 1/4
    cef_order = ["1/2", "1/3", "1/4"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric in zip(axes, ["f1", "auroc"]):
        grid = _pivot_max(
            sp,
            index="channel_emb_fraction",
            columns="embed_dim",
            value=metric,
            fixed={"depth": depth},
        )
        if grid.empty:
            ax.set_visible(False)
            continue
        # Reorder rows if present
        ordered = [c for c in cef_order if c in grid.index] + [
            i for i in grid.index if i not in cef_order
        ]
        grid = grid.loc[ordered]
        im = ax.imshow(grid.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels([str(int(c)) for c in grid.columns])
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels([str(i) for i in grid.index])
        ax.set_xlabel("embed_dim")
        ax.set_ylabel("channel_emb_fraction")
        ax.set_title(f"{metric.upper()} | depth={int(depth)}")
        _annotate_heatmap(ax, grid)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"minipigs: max {TASK} over remaining HPs "
        f"(axes embed_dim × cef; fixed depth={int(depth)}; concat only)",
        y=1.05,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_heatmap_minipigs_embed_dim_vs_cef.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_best_vs_baselines(df: pd.DataFrame, baselines: pd.DataFrame) -> Path:
    best = best_per_species(df)
    species_list = sorted(best["species"].unique())
    labels = ["opt_baseline_no_cw", "cw_baseline", "best_capacity"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = {
        "opt_baseline_no_cw": "#9e9e9e",
        "cw_baseline": "#2ca02c",
        "best_capacity": "#4C72B0",
    }
    width = 0.25
    x = np.arange(len(species_list))

    for ax, metric in zip(axes, ["f1", "auroc"]):
        for i, lab in enumerate(labels):
            vals = []
            for species in species_list:
                if lab == "best_capacity":
                    vals.append(
                        float(
                            best.loc[best["species"] == species, metric].iloc[0]
                        )
                    )
                elif lab == "opt_baseline_no_cw":
                    vals.append(
                        float(
                            baselines.loc[
                                (baselines["species"] == species)
                                & (baselines["label"] == "opt_baseline_no_cw"),
                                metric,
                            ].iloc[0]
                        )
                    )
                else:
                    vals.append(
                        float(
                            baselines.loc[
                                (baselines["species"] == species)
                                & baselines["label"]
                                .astype(str)
                                .str.startswith("cw_baseline"),
                                metric,
                            ].iloc[0]
                        )
                    )
            offset = (i - 1) * width
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=lab,
                color=colors[lab],
                edgecolor="white",
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
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
        "Best model-size config vs opt / CW baselines (fold 0)",
        y=1.02,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_best_vs_baselines.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def main() -> None:
    import sys

    use_cached = "--cached" in sys.argv
    print(
        f"Resolved: sweeps {SWEEP_IDS}, group={GROUP}, project={PROJECT}, "
        f"entity={ENTITY}, fold={FOLD}"
    )
    api = wandb.Api()
    csv_path = CSV_DIR / f"{STEM}_runs.csv"
    base_csv = CSV_DIR / f"{STEM}_baselines.csv"

    if use_cached and csv_path.exists():
        print(f"Loading cached runs from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = fetch_finished_runs(api)
        if df.empty:
            raise SystemExit("No finished fold-0 runs found.")
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
        f"Loaded {len(df)} fold-{FOLD} finished runs | species counts: "
        f"{df['species'].value_counts().to_dict()}"
    )
    best = best_per_species(df)
    print_tables(df, baselines)
    plot_best_vs_baselines(df, baselines)
    plot_heatmap_embed_dim_vs_depth(df)
    plot_heatmap_heads(df, best)
    plot_heatmap_minipigs_cef(df, best)


if __name__ == "__main__":
    main()
