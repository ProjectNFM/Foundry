"""Class-weight smoothing sweep report: minipigs vs monkeys.

Reproduces the comparative report for WandB sweeps:
  - w74jfier (minipigs_class-weights)
  - nxx4a4pn (monkeys_class-weights)
in group NEUROSOFT_INTRASESSION_MULTISUBJ / project auditory_decoding.

Usage:
    uv run python analysis/class_weights_smoothing_sweep_report.py
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import wandb

from analysis._wandb_utils import default_entity, figures_dir, unwrap_summary_value

PROJECT = "auditory_decoding"
GROUP = "NEUROSOFT_INTRASESSION_MULTISUBJ"
SWEEP_IDS: dict[str, str] = {
    "minipigs": "w74jfier",
    "monkeys": "nxx4a4pn",
}

TASK = "neurosoft_acoustic_stim_8band"
# Report summary ``.max`` for all classification metrics (loss would use ``.min``).
METRICS = {
    "f1": f"val/{TASK}_f1",
    "auroc": f"val/{TASK}_auroc",
    "precision": f"val/{TASK}_precision",
    "recall": f"val/{TASK}_recall",
    "balanced_acc": f"val/{TASK}_balanced_acc",
}

SLUG = "class_weights_smoothing"


def _entity() -> str | None:
    entity = default_entity()
    if entity:
        return entity
    api = wandb.Api()
    try:
        return api.default_entity
    except Exception:
        viewer = api.viewer
        if isinstance(viewer, dict):
            return viewer.get("entity")
        return getattr(viewer, "entity", None)


def _cfg_get(cfg: dict[str, Any], dotted: str) -> Any:
    """Read a config value; prefer flat WandB keys, then nested dicts."""
    if not isinstance(cfg, dict):
        return None
    if dotted in cfg:
        return cfg[dotted]
    slash = dotted.replace(".", "/")
    if slash in cfg:
        return cfg[slash]
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _as_float(val: Any) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def metric_max(run: Any, wandb_key: str, history_cache: dict[str, float] | None = None) -> float | None:
    """Read ``{wandb_key}.max`` from the run summary (fallback: history max).

    WandB stores maximized metrics as SummarySubDict ``{"max": ...}``. Some runs
    only persist ``min`` for precision/recall/balanced_acc; in that case we take
    the max over logged history so every reported metric is still a max.
    """
    raw = run.summary.get(wandb_key)
    # Prefer explicit .max field on the summary value.
    parsed = _as_float(unwrap_summary_value(raw, "max"))
    if parsed is not None:
        return parsed

    if history_cache is not None and wandb_key in history_cache:
        return history_cache[wandb_key]

    hist = run.history(keys=[wandb_key], samples=10_000, pandas=True)
    if wandb_key not in hist.columns or hist[wandb_key].dropna().empty:
        return None
    return float(hist[wandb_key].max())


def detect_species(run: Any, sweep_hint: str | None = None) -> str:
    for tag in run.tags or []:
        if str(tag).lower() in ("minipigs", "monkeys"):
            return str(tag).lower()

    cfg = run.config or {}
    dataset_class = str(_cfg_get(cfg, "data.dataset_class") or "").lower()
    if "minipigs" in dataset_class:
        return "minipigs"
    if "monkeys" in dataset_class:
        return "monkeys"

    blob = str(cfg).lower()
    if "neurosoft_minipigs" in blob or "minipigs" in blob:
        return "minipigs"
    if "neurosoft_monkeys" in blob or "monkeys" in blob:
        return "monkeys"

    return sweep_hint or "unknown"


def collect_runs(
    api: Any,
    entity: str | None,
    project: str,
    group: str,
    sweep_ids: dict[str, str],
) -> pd.DataFrame:
    path_prefix = f"{entity}/{project}" if entity else project

    # Sweep.runs often ship empty configs; keep ids/species hints then re-fetch.
    sweep_meta: list[tuple[str, str, str]] = []  # species_hint, sweep_id, run_id
    for species_hint, sweep_id in sweep_ids.items():
        sweep = api.sweep(f"{path_prefix}/{sweep_id}")
        for run in sweep.runs:
            sweep_meta.append((species_hint, sweep_id, run.id))

    group_runs = list(api.runs(path_prefix, filters={"group": group}, per_page=500))
    group_ids = {r.id for r in group_runs}

    records: list[dict[str, Any]] = []
    for species_hint, sweep_id, run_id in sweep_meta:
        run = api.run(f"{path_prefix}/{run_id}")
        if run.state != "finished":
            continue
        in_group = run.id in group_ids or getattr(run, "group", None) == group
        if not in_group:
            continue

        cfg = dict(run.config or {})
        species = detect_species(run, species_hint)

        need_history = any(
            _as_float(unwrap_summary_value(run.summary.get(k), "max")) is None
            for k in METRICS.values()
        )
        history_cache: dict[str, float] | None = None
        if need_history:
            keys = list(METRICS.values())
            hist = run.history(keys=keys, samples=10_000, pandas=True)
            history_cache = {
                k: float(hist[k].max())
                for k in keys
                if k in hist.columns and not hist[k].dropna().empty
            }

        row: dict[str, Any] = {
            "run_id": run.id,
            "run_name": run.name,
            "species": species,
            "sweep_id": sweep_id,
            "smoothing": _cfg_get(cfg, "class_weights.smoothing"),
            "fold": _cfg_get(cfg, "hyperparameters.fold_number"),
            "split_type": _cfg_get(cfg, "data.split_type"),
            "learning_rate": _cfg_get(cfg, "hyperparameters.learning_rate"),
            "weight_decay": _cfg_get(cfg, "hyperparameters.weight_decay"),
            "atn_dropout": _cfg_get(cfg, "model.atn_dropout"),
            "batch_size": _cfg_get(cfg, "hyperparameters.batch_size"),
            "tokenizer": cfg.get("model/tokenizer")
            or _cfg_get(cfg, "model.tokenizer._target_"),
        }
        for out_name, wandb_key in METRICS.items():
            row[out_name] = metric_max(run, wandb_key, history_cache)
        records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["smoothing"] = pd.to_numeric(df["smoothing"], errors="coerce")
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["species", "smoothing", "fold"]).reset_index(drop=True)


def fold_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df["smoothing"].isna().all():
        raise ValueError(
            "No runs with valid class_weights.smoothing; check config fetch."
        )
    metric_cols = list(METRICS)
    flat_rows = []
    for (species, smoothing), sub in df.groupby(["species", "smoothing"]):
        row: dict[str, Any] = {
            "species": species,
            "smoothing": smoothing,
            "n_folds": len(sub),
        }
        for m in metric_cols:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std(ddof=1) if len(sub) > 1 else 0.0
        flat_rows.append(row)
    out = pd.DataFrame(flat_rows)
    return out.sort_values(["species", "smoothing"]).reset_index(drop=True)


def best_config_per_species(df: pd.DataFrame) -> pd.DataFrame:
    """Best single run per species by max val F1."""
    idx = df.groupby("species")["f1"].idxmax()
    cols = [
        "species",
        "smoothing",
        "fold",
        "f1",
        "auroc",
        "precision",
        "recall",
        "balanced_acc",
        "run_name",
        "run_id",
        "sweep_id",
    ]
    return df.loc[idx, cols].sort_values("species").reset_index(drop=True)


def best_smoothing_by_mean_f1(fold_df: pd.DataFrame) -> pd.DataFrame:
    idx = fold_df.groupby("species")["f1_mean"].idxmax()
    return fold_df.loc[idx].sort_values("species").reset_index(drop=True)


def print_tables(df: pd.DataFrame, fold_df: pd.DataFrame) -> None:
    print("=" * 100)
    print("CLASS WEIGHTS SMOOTHING SWEEP — minipigs vs monkeys")
    print(f"Project={PROJECT}  Group={GROUP}  Sweeps={SWEEP_IDS}")
    print(f"Finished ∩ group runs: {len(df)}")
    print("=" * 100)

    best_run = best_config_per_species(df)
    print("\n### Best configuration per species (single run, max val F1)")
    print(
        best_run.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        )
    )

    best_mean = best_smoothing_by_mean_f1(fold_df)
    print("\n### Best smoothing per species (mean ± std F1 across folds)")
    show = best_mean[
        [
            "species",
            "smoothing",
            "n_folds",
            "f1_mean",
            "f1_std",
            "auroc_mean",
            "precision_mean",
            "recall_mean",
            "balanced_acc_mean",
        ]
    ]
    print(
        show.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        )
    )

    print("\n### Fold-averaged grid (Species × smoothing)")
    grid_cols = [
        "species",
        "smoothing",
        "n_folds",
        "f1_mean",
        "f1_std",
        "auroc_mean",
        "auroc_std",
        "precision_mean",
        "recall_mean",
        "balanced_acc_mean",
    ]
    print(
        fold_df[grid_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        )
    )

    print("\n### Full run table")
    full_cols = [
        "species",
        "smoothing",
        "fold",
        "f1",
        "auroc",
        "precision",
        "recall",
        "balanced_acc",
        "run_name",
        "run_id",
    ]
    print(
        df[full_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        )
    )


def plot_f1_comparison(fold_df: pd.DataFrame, out_dir: Any) -> list[str]:
    paths: list[str] = []
    species_order = [s for s in ("minipigs", "monkeys") if s in set(fold_df["species"])]
    colors = {"minipigs": "#4C72B0", "monkeys": "#DD8452"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for species in species_order:
        sub = fold_df[fold_df["species"] == species].sort_values("smoothing")
        ax.errorbar(
            sub["smoothing"],
            sub["f1_mean"],
            yerr=sub["f1_std"],
            marker="o",
            capsize=4,
            label=species,
            color=colors.get(species),
            linewidth=2,
        )
    ax.set_xlabel("class_weights.smoothing")
    ax.set_ylabel(f"max val F1 (mean ± std over folds)\n{TASK}")
    ax.set_title("Class-weight smoothing vs 8-band decoding F1")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"{SLUG}_f1_vs_smoothing.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))
    print(f"Saved: {path}")

    # Grouped bar of mean F1
    fig, ax = plt.subplots(figsize=(8, 5))
    smoothings = sorted(fold_df["smoothing"].unique())
    x = range(len(smoothings))
    width = 0.35
    for i, species in enumerate(species_order):
        means, stds = [], []
        for s in smoothings:
            row = fold_df[
                (fold_df["species"] == species) & (fold_df["smoothing"] == s)
            ]
            means.append(float(row["f1_mean"].iloc[0]) if len(row) else float("nan"))
            stds.append(float(row["f1_std"].iloc[0]) if len(row) else 0.0)
        offsets = [xi + (i - 0.5) * width for xi in x]
        ax.bar(
            offsets,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=species,
            color=colors.get(species),
            edgecolor="white",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in smoothings])
    ax.set_xlabel("class_weights.smoothing")
    ax.set_ylabel(f"max val F1 (mean ± std)\n{TASK}")
    ax.set_title("Class-weight smoothing — fold-mean F1 by species")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path2 = out_dir / f"{SLUG}_f1_bars.png"
    fig.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path2))
    print(f"Saved: {path2}")
    return paths


def main() -> None:
    entity = _entity()
    api = wandb.Api(timeout=60)
    df = collect_runs(api, entity, PROJECT, GROUP, SWEEP_IDS)
    if df.empty:
        raise SystemExit("No finished runs found for the resolved sweep ∩ group set.")

    fold_df = fold_summary(df)
    print_tables(df, fold_df)
    out_dir = figures_dir(__file__)
    plot_f1_comparison(fold_df, out_dir)


if __name__ == "__main__":
    main()
