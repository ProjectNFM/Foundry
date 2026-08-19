"""Intrasession optimal-HP baselines across training paradigms.

Applies species-specific optimal hyperparameters from the multisubject HP
search and compares single-session, single-subject, and multi-subject
training for minipigs and monkeys (intrasession-block evaluation).

Usage:
    uv run python analysis/20260727-LS-intrasession-opt-baselines.py
    uv run python analysis/20260727-LS-intrasession-opt-baselines.py --cached
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
ENTITY = default_entity()

# (species, paradigm) → sweep id. HPs fixed at species optima.
SWEEP_IDS: dict[tuple[str, str], str] = {
    ("minipigs", "single-session"): "hiyb4224",
    ("minipigs", "single-subject"): "4k9zt970",
    ("minipigs", "multi-subject"): "47jd29ds",
    ("monkeys", "single-session"): "h5gf9jn1",
    ("monkeys", "single-subject"): "aycfxm9b",
    ("monkeys", "multi-subject"): "bvcgw95o",
}

PARADIGM_ORDER = ["single-session", "single-subject", "multi-subject"]
# Extended labels used in summary tables / figures (reports both singlesess variants).
SUMMARY_PARADIGM_ORDER = [
    "single-session (all)",
    "single-session (excl. outliers)",
    "single-subject",
    "multi-subject",
]
SPECIES_ORDER = ["minipigs", "monkeys"]

# Single-session units excluded when fold-mean F1 is missing or at ceiling.
SINGLESESS_OUTLIER_F1_GE = 0.99

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
    return fallback or "unknown"


def _extract_meta(config: dict[str, Any]) -> dict[str, Any]:
    hp = config.get("hyperparameters") or {}
    model = config.get("model") or {}
    trainer = config.get("trainer") or {}
    data = config.get("data") or {}
    dkw = data.get("dataset_kwargs") or {}

    tokenizer = config.get("model/tokenizer")
    if tokenizer is None:
        tok = model.get("tokenizer")
        if isinstance(tok, dict):
            tokenizer = tok.get("_name_") or tok.get("name") or str(tok)
        else:
            tokenizer = tok

    recording_ids = config.get("data.dataset_kwargs.recording_ids")
    if recording_ids is None:
        recording_ids = dkw.get("recording_ids")
    if isinstance(recording_ids, list):
        recording_id = recording_ids[0] if len(recording_ids) == 1 else None
        n_recordings = len(recording_ids)
    else:
        recording_id = None
        n_recordings = None

    subject = config.get("data.subject", data.get("subject"))
    if (
        subject is None
        and isinstance(recording_id, str)
        and recording_id.startswith("sub-")
    ):
        subject = recording_id.split("_")[0]

    return {
        "tokenizer": tokenizer,
        "atn_dropout": config.get(
            "model.atn_dropout", model.get("atn_dropout")
        ),
        "learning_rate": config.get(
            "hyperparameters.learning_rate", hp.get("learning_rate")
        ),
        "weight_decay": config.get(
            "hyperparameters.weight_decay", hp.get("weight_decay")
        ),
        "grad_clip": config.get(
            "trainer.gradient_clip_val", trainer.get("gradient_clip_val")
        ),
        "fold": hp.get("fold_number", dkw.get("fold")),
        "split_type": config.get("data.split_type") or data.get("split_type"),
        "subject": subject,
        "recording_id": recording_id,
        "n_recordings": n_recordings,
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
    run_id: str, species_hint: str, paradigm: str, sweep_id: str
) -> dict[str, Any]:
    api = wandb.Api()
    run = api.run(_run_path(run_id))
    config = dict(run.config)
    row = {
        "species": _species_from_run(run, fallback=species_hint),
        "paradigm": paradigm,
        "sweep_id": sweep_id,
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "group": run.group,
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

    stubs: list[tuple[str, str, str, str]] = []
    for (species, paradigm), sweep_id in SWEEP_IDS.items():
        sweep = api.sweep(_sweep_path(sweep_id))
        for run in sweep.runs:
            if run.state != "finished":
                continue
            stubs.append((run.id, species, paradigm, sweep_id))

    print(f"Fetching {len(stubs)} finished runs ({N_WORKERS} workers)...")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one, rid, species, paradigm, sid): rid
            for rid, species, paradigm, sid in stubs
        }
        done = 0
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 25 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in METRICS + [
        "learning_rate",
        "weight_decay",
        "atn_dropout",
        "grad_clip",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce").astype("Int64")
    df["paradigm"] = pd.Categorical(
        df["paradigm"], categories=PARADIGM_ORDER, ordered=True
    )
    return df.sort_values(["species", "paradigm", "fold"]).reset_index(
        drop=True
    )


def _unit_key(row: pd.Series) -> str:
    if row["paradigm"] == "single-session":
        return str(row["recording_id"] or row["run_name"])
    if row["paradigm"] == "single-subject":
        return str(row["subject"] or row["run_name"])
    return "all"


def fold_means_by_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics across folds within each (species, paradigm, unit)."""
    work = df.copy()
    work["unit"] = work.apply(_unit_key, axis=1)
    agg = (
        work.groupby(["species", "paradigm", "unit"], observed=True)[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    # flatten columns
    flat = agg[["species", "paradigm", "unit"]].copy()
    for m in METRICS:
        flat[f"{m}_mean"] = agg[(m, "mean")]
        flat[f"{m}_std"] = agg[(m, "std")]
        flat[f"{m}_n"] = agg[(m, "count")]
    return flat


def flag_singlesess_outliers(unit_df: pd.DataFrame) -> pd.DataFrame:
    """Flag single-session units with missing or ceiling fold-mean F1."""
    out = unit_df.copy()
    is_sess = out["paradigm"].astype(str) == "single-session"
    missing = is_sess & out["f1_mean"].isna()
    ceiling = is_sess & (out["f1_mean"] >= SINGLESESS_OUTLIER_F1_GE)
    out["is_outlier"] = missing | ceiling
    out["outlier_reason"] = np.where(
        missing,
        "missing_f1",
        np.where(ceiling, f"f1>={SINGLESESS_OUTLIER_F1_GE}", ""),
    )
    return out


def _summarize_units(
    g: pd.DataFrame, species: str, paradigm_label: str
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "species": species,
        "paradigm": paradigm_label,
        "n_units": int(len(g)),
        "n_folds": int(g["f1_n"].mean()) if len(g) and "f1_n" in g else np.nan,
    }
    for m in METRICS:
        vals = g[f"{m}_mean"].dropna()
        row[f"{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
        row[f"{m}_std"] = float(vals.std(ddof=0)) if len(vals) else np.nan
    return row


def paradigm_summary(df: pd.DataFrame, unit_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize metrics per species × paradigm.

    Single-session is reported twice: all sessions, and excluding outliers
    (missing F1 or fold-mean F1 >= ``SINGLESESS_OUTLIER_F1_GE``).
    Single-subject: mean±std across units of fold-mean metrics.
    Multi-subject: mean±std across folds (one pooled training set).
    """
    flagged = flag_singlesess_outliers(unit_df)
    rows = []
    for species in SPECIES_ORDER:
        sess = flagged[
            (flagged["species"] == species)
            & (flagged["paradigm"].astype(str) == "single-session")
        ]
        rows.append(_summarize_units(sess, species, "single-session (all)"))
        rows.append(
            _summarize_units(
                sess[~sess["is_outlier"]],
                species,
                "single-session (excl. outliers)",
            )
        )

        subj = flagged[
            (flagged["species"] == species)
            & (flagged["paradigm"].astype(str) == "single-subject")
        ]
        rows.append(_summarize_units(subj, species, "single-subject"))

        multi = df[
            (df["species"] == species) & (df["paradigm"] == "multi-subject")
        ]
        row: dict[str, Any] = {
            "species": species,
            "paradigm": "multi-subject",
            "n_units": 1,
            "n_folds": int(len(multi)),
        }
        for m in METRICS:
            vals = multi[m].dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{m}_std"] = float(vals.std(ddof=0)) if len(vals) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    out["paradigm"] = pd.Categorical(
        out["paradigm"], categories=SUMMARY_PARADIGM_ORDER, ordered=True
    )
    return out.sort_values(["species", "paradigm"]).reset_index(drop=True)


def multisub_fold_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["paradigm"] == "multi-subject"].copy()
    cols = [
        "species",
        "fold",
        "f1",
        "auroc",
        "precision",
        "recall",
        "balanced_acc",
        "run_id",
    ]
    return sub[cols].sort_values(["species", "fold"]).reset_index(drop=True)


def print_tables(
    df: pd.DataFrame, unit_df: pd.DataFrame, summary: pd.DataFrame
) -> None:
    flagged = flag_singlesess_outliers(unit_df)
    outliers = flagged[flagged["is_outlier"]]

    print("\n=== Paradigm summary ===")
    print(
        "  (single-session/subject: mean±std across units of fold-means; "
        "multi-subject: mean±std across folds)"
    )
    print(
        f"  Single-session outliers: missing F1 or fold-mean F1 "
        f">= {SINGLESESS_OUTLIER_F1_GE}"
    )
    show = summary.copy()
    for m in METRICS:
        show[m] = [
            f"{mean:.4f}±{std:.4f}"
            for mean, std in zip(show[f"{m}_mean"], show[f"{m}_std"])
        ]
    print(
        show[["species", "paradigm", "n_units", *METRICS]].to_string(
            index=False
        )
    )

    print("\n=== Excluded single-session outliers ===")
    if outliers.empty:
        print("  (none)")
    else:
        print(
            outliers[
                [
                    "species",
                    "unit",
                    "f1_mean",
                    "auroc_mean",
                    "f1_n",
                    "outlier_reason",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    print("\n=== Multi-subject fold-wise (raw max-val metrics) ===")
    print(
        multisub_fold_table(df).to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n=== Single-subject fold-means ===")
    ss = unit_df[unit_df["paradigm"] == "single-subject"].copy()
    ss = ss.sort_values(["species", "f1_mean"], ascending=[True, False])
    ss_out = pd.DataFrame(
        {
            "species": ss["species"].astype(str).values,
            "subject": ss["unit"].astype(str).values,
            "f1": [
                f"{m:.4f}±{(0.0 if pd.isna(s) else float(s)):.4f}"
                for m, s in zip(ss["f1_mean"], ss["f1_std"])
            ],
            "n_folds": ss["f1_n"].astype(int).values,
        }
    )
    print(ss_out.to_string(index=False))

    print("\n=== Single-session: all vs excl. outliers ===")
    sess = flagged[flagged["paradigm"].astype(str) == "single-session"]
    for species in SPECIES_ORDER:
        g_all = sess[sess["species"] == species]["f1_mean"]
        g_filt = sess[(sess["species"] == species) & (~sess["is_outlier"])][
            "f1_mean"
        ]
        n_out = int(
            sess[(sess["species"] == species) & sess["is_outlier"]].shape[0]
        )
        print(
            f"  {species}: all n={int(g_all.notna().sum())} "
            f"(+{int(g_all.isna().sum())} missing) "
            f"mean={g_all.mean():.4f}±{g_all.std(ddof=0):.4f} "
            f"| excl. outliers n={len(g_filt)} (−{n_out}) "
            f"mean={g_filt.mean():.4f}±{g_filt.std(ddof=0):.4f}"
        )


def plot_paradigm_f1(summary: pd.DataFrame) -> Path:
    colors = {"minipigs": "#4C72B0", "monkeys": "#DD8452"}
    labels = SUMMARY_PARADIGM_ORDER
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(labels))
    width = 0.35

    for i, species in enumerate(SPECIES_ORDER):
        sub = summary[summary["species"] == species].set_index("paradigm")
        means = [
            float(sub.loc[p, "f1_mean"]) if p in sub.index else np.nan
            for p in labels
        ]
        stds = [
            float(sub.loc[p, "f1_std"]) if p in sub.index else 0.0
            for p in labels
        ]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=species,
            color=colors[species],
            edgecolor="white",
            error_kw=dict(lw=1.1),
        )
        for j, bar in enumerate(bars):
            if labels[j] == "single-session (excl. outliers)":
                bar.set_hatch("//")
                bar.set_alpha(0.9)
        for bar, mean in zip(bars, means):
            if np.isnan(mean):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "single-session\n(all)",
            "single-session\n(excl. outliers)",
            "single-subject",
            "multi-subject",
        ]
    )
    ax.set_ylabel(f"Max val {TASK} F1")
    ax.set_title("Optimal-HP intrasession baselines by training paradigm")
    ax.legend(title="Species")
    ymax = summary["f1_mean"].max() + summary["f1_std"].max()
    ax.set_ylim(0, max(ymax, 0.1) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_f1_by_paradigm.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_singlesub_f1(unit_df: pd.DataFrame) -> Path:
    ss = unit_df[unit_df["paradigm"] == "single-subject"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colors = {"minipigs": "#4C72B0", "monkeys": "#DD8452"}
    for ax, species in zip(axes, SPECIES_ORDER):
        g = ss[ss["species"] == species].sort_values("unit")
        ax.bar(
            g["unit"],
            g["f1_mean"],
            yerr=g["f1_std"].fillna(0),
            capsize=3,
            color=colors[species],
            edgecolor="white",
        )
        ax.set_title(species)
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Subject")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(f"Max val {TASK} F1 (fold mean±std)")
    fig.suptitle("Single-subject intrasession baselines", y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_f1_singlesubject.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def main() -> None:
    import sys

    use_cached = "--cached" in sys.argv
    print(
        f"Resolved: {len(SWEEP_IDS)} sweeps, project={PROJECT}, entity={ENTITY}"
    )
    for (species, paradigm), sid in SWEEP_IDS.items():
        print(f"  {species:9s} {paradigm:15s} → {sid}")

    csv_runs = CSV_DIR / f"{STEM}_runs.csv"
    if use_cached and csv_runs.exists():
        print(f"Loading cached runs from {csv_runs}")
        df = pd.read_csv(csv_runs)
        df["paradigm"] = pd.Categorical(
            df["paradigm"], categories=PARADIGM_ORDER, ordered=True
        )
    else:
        df = fetch_finished_runs()
    if df.empty:
        raise SystemExit("No finished runs found.")

    print(
        f"Loaded {len(df)} runs | "
        f"{df.groupby(['species', 'paradigm'], observed=True).size().to_dict()}"
    )
    unit_df = fold_means_by_unit(df)
    flagged = flag_singlesess_outliers(unit_df)
    summary = paradigm_summary(df, unit_df)
    print_tables(df, unit_df, summary)
    plot_paradigm_f1(summary)
    plot_singlesub_f1(unit_df)

    csv_units = CSV_DIR / f"{STEM}_units.csv"
    csv_summary = CSV_DIR / f"{STEM}_summary.csv"
    csv_outliers = CSV_DIR / f"{STEM}_singlesess_outliers.csv"
    df.to_csv(csv_runs, index=False)
    flagged.to_csv(csv_units, index=False)
    summary.to_csv(csv_summary, index=False)
    flagged[flagged["is_outlier"]].to_csv(csv_outliers, index=False)
    print(f"Saved: {csv_runs}")
    print(f"Saved: {csv_units}")
    print(f"Saved: {csv_summary}")
    print(f"Saved: {csv_outliers}")


if __name__ == "__main__":
    main()
