"""Session-level EEGNet / GRU vs POYO on NeuroSoft 8-band decoding.

Fetches finished runs from ``NEUROSOFT_INTRASESSION_SINGLESESS`` and
compares EEGNet and GRU **session averages** against three POYO
references from this thread:

1. Opt-HP **single-session** POYO (mean across sessions)
2. Opt-HP **single-subject** POYO (mean across subjects)
3. **Best multi-subject** POYO (reduced-capacity fold-0 winners)

Per-session scatterplots and supplementary bars use ``val_session/``
history-max metrics from (2) and (3), matched to the same recordings
as EEGNet/GRU — not the pooled run summary.

Usage:
    uv run python analysis/20260818-LS-singlesess-eegnet-gru-baselines.py
    uv run python analysis/20260818-LS-singlesess-eegnet-gru-baselines.py --cached
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
GROUP = "NEUROSOFT_INTRASESSION_SINGLESESS"
ENTITY = default_entity()

# Opt-HP POYO singlesess sweeps from 20260727.
POYO_OPT_SWEEPS = {"hiyb4224", "h5gf9jn1"}

# Opt-HP POYO single-subject sweeps from 20260727.
POYO_SINGLESUB_SWEEPS: dict[str, str] = {
    "minipigs": "4k9zt970",
    "monkeys": "aycfxm9b",
}

# Best reduced-capacity POYO (fold 0) from 20260805-LS-model-capacity.
MULTISUBJ_BEST_POYO: dict[str, str] = {
    "minipigs": "ncx1been",
    "monkeys": "zrvjtixp",
}

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

MODEL_ORDER = ["eegnet", "gru", "poyo"]
MODEL_LABELS = {"eegnet": "EEGNet", "gru": "GRU", "poyo": "POYO (session)"}
MODEL_COLORS = {"eegnet": "#E8963E", "gru": "#C44E52", "poyo": "#4C72B0"}
SPECIES_ORDER = ["minipigs", "monkeys"]

CONDITION_ORDER = ["eegnet", "gru", "poyo_sess", "poyo_subj", "poyo_multi"]
CONDITION_LABELS = {
    "eegnet": "EEGNet\n(session)",
    "gru": "GRU\n(session)",
    "poyo_sess": "POYO\nsession",
    "poyo_subj": "POYO\nsubject",
    "poyo_multi": "POYO multi\n(best)",
}
CONDITION_COLORS = {
    "eegnet": "#E8963E",
    "gru": "#C44E52",
    "poyo_sess": "#4C72B0",
    "poyo_subj": "#8172B2",
    "poyo_multi": "#55A868",
}

PER_SESSION_MODELS = ["eegnet", "gru", "poyo", "poyo_subj", "poyo_multi"]
PER_SESSION_LABELS = {
    "eegnet": "EEGNet",
    "gru": "GRU",
    "poyo": "POYO (session)",
    "poyo_subj": "POYO (subject)",
    "poyo_multi": "POYO multi (best)",
}
PER_SESSION_COLORS = {
    "eegnet": CONDITION_COLORS["eegnet"],
    "gru": CONDITION_COLORS["gru"],
    "poyo": CONDITION_COLORS["poyo_sess"],
    "poyo_subj": CONDITION_COLORS["poyo_subj"],
    "poyo_multi": CONDITION_COLORS["poyo_multi"],
}
STEM = Path(__file__).stem
FIGURES_DIR = figures_dir(__file__)
CSV_DIR = csv_dir(__file__)
N_WORKERS = 8
HISTORY_BATCH = 30
VAL_SESSION_PREFIX = "val_session/"


def _run_path(run_id: str, project: str = PROJECT) -> str:
    if ENTITY:
        return f"{ENTITY}/{project}/{run_id}"
    return f"{project}/{run_id}"


def _sweep_path(sweep_id: str) -> str:
    if ENTITY:
        return f"{ENTITY}/{PROJECT}/{sweep_id}"
    return f"{PROJECT}/{sweep_id}"


def _species_from_run(run: Any) -> str:
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
    return "unknown"


def _model_from_run(run: Any, config: dict[str, Any]) -> str:
    tags = {t.lower() for t in (run.tags or [])}
    for cand in ("eegnet", "gru", "poyo_eeg"):
        if cand in tags:
            return "poyo" if cand == "poyo_eeg" else cand
    hydra = (
        config.get("hydra.runtime.choices.model")
        or config.get("model/_name_")
        or ""
    )
    hydra_s = str(hydra).lower()
    if "eegnet" in hydra_s:
        return "eegnet"
    if hydra_s == "gru" or hydra_s.endswith("/gru"):
        return "gru"
    if "poyo" in hydra_s:
        return "poyo"
    name = (run.name or "").lower()
    if name.startswith("eegnet") or "_eegnet_" in name:
        return "eegnet"
    if name.startswith("gru") or "_gru_" in name:
        return "gru"
    if "poyo" in name:
        return "poyo"
    return "unknown"


def _is_poyo_opt(run: Any) -> bool:
    tags = {t.lower() for t in (run.tags or [])}
    if "opt" in tags:
        return True
    sweep_id = run.sweep.id if getattr(run, "sweep", None) else None
    return sweep_id in POYO_OPT_SWEEPS


def _recording_id(config: dict[str, Any]) -> str | None:
    rec = config.get("data.dataset_kwargs.recording_ids")
    if rec is None:
        dkw = (config.get("data") or {}).get("dataset_kwargs") or {}
        rec = dkw.get("recording_ids")
    if isinstance(rec, list):
        return rec[0] if rec else None
    if isinstance(rec, str):
        return rec
    return None


def _subject(config: dict[str, Any], recording_id: str | None) -> str | None:
    val = config.get("data.subject")
    if val is None:
        val = (config.get("data") or {}).get("subject")
    if (
        val is None
        and isinstance(recording_id, str)
        and recording_id.startswith("sub-")
    ):
        val = recording_id.split("_")[0]
    return val


def _fold(config: dict[str, Any]) -> int | None:
    val = config.get("hyperparameters.fold_number")
    if val is None:
        hp = config.get("hyperparameters") or {}
        val = hp.get("fold_number")
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


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
            out[key] = None
        else:
            out[key] = float(history[key].max())
    return out


def _history_maxes_batched(
    run: Any, wandb_keys: list[str], batch_size: int = HISTORY_BATCH
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for i in range(0, len(wandb_keys), batch_size):
        out.update(_history_maxes(run, wandb_keys[i : i + batch_size]))
    return out


def _shorten_session_id(session_id: str) -> str:
    """Keep sub-/ses-/acq- segments (same as SessionMetricsCallback)."""
    parts = str(session_id).split("_")
    keep = [p for p in parts if p.startswith(("sub-", "ses-", "acq-"))]
    return "_".join(keep) if keep else str(session_id)


def _parse_val_session_keys(summary: Any) -> dict[str, dict[str, str]]:
    """Map short session id → {metric: wandb key} for val_session logs."""
    task_prefix = f"{TASK}_"
    parsed: dict[str, dict[str, str]] = {}
    for key in summary.keys():
        key_s = str(key)
        if not key_s.startswith(VAL_SESSION_PREFIX):
            continue
        parts = key_s.split("/")
        if len(parts) != 3:
            continue
        _, sid, rest = parts
        if not rest.startswith(task_prefix):
            continue
        metric = rest[len(task_prefix) :]
        if metric not in METRIC_KEYS:
            continue
        parsed.setdefault(sid, {})[metric] = key_s
    return parsed


def val_session_rows_from_run(
    run: Any,
    *,
    species: str,
    model: str,
    fold: int | None = None,
    subject: str | None = None,
) -> list[dict[str, Any]]:
    parsed = _parse_val_session_keys(run.summary)
    wandb_keys = [k for mmap in parsed.values() for k in mmap.values()]
    maxes = _history_maxes_batched(run, wandb_keys)
    rows: list[dict[str, Any]] = []
    for sid, mmap in parsed.items():
        row: dict[str, Any] = {
            "run_id": run.id,
            "species": species,
            "model": model,
            "session": sid,
            "session_key": sid,
            "subject": subject or sid.split("_")[0],
            "fold": fold,
        }
        for metric in METRICS:
            key = mmap.get(metric)
            val = maxes.get(key) if key else None
            if val is None and key is not None:
                val = _summary_max(run, key)
            row[metric] = val
        rows.append(row)
    return rows


def fetch_multisubj_session_metrics(
    api: wandb.Api | None = None,
) -> pd.DataFrame:
    if api is None:
        api = wandb.Api()
    rows: list[dict[str, Any]] = []
    for species, run_id in MULTISUBJ_BEST_POYO.items():
        run = api.run(_run_path(run_id))
        part = val_session_rows_from_run(
            run, species=species, model="poyo_multi", fold=0
        )
        print(f"  val_session multi {species} ({run_id}): {len(part)} sessions")
        rows.extend(part)
    return pd.DataFrame(rows)


def _fetch_one_val_sessions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    api = wandb.Api()
    run = api.run(_run_path(str(payload["run_id"])))
    fold = payload.get("fold")
    try:
        fold_i = int(fold) if fold is not None and pd.notna(fold) else None
    except (TypeError, ValueError):
        fold_i = None
    return val_session_rows_from_run(
        run,
        species=str(payload["species"]),
        model="poyo_subj",
        fold=fold_i,
        subject=payload.get("subject"),
    )


def fetch_singlesubj_session_metrics(
    subj_runs: pd.DataFrame, api: wandb.Api | None = None
) -> pd.DataFrame:
    del api
    work = subj_runs[subj_runs["state"] == "finished"].copy()
    payloads = work[["run_id", "species", "fold", "subject"]].to_dict("records")
    print(
        f"Fetching val_session metrics for {len(payloads)} single-subject "
        f"POYO runs ({N_WORKERS} workers)..."
    )
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(_fetch_one_val_sessions, p) for p in payloads]
        done = 0
        for fut in as_completed(futures):
            rows.extend(fut.result())
            done += 1
            if done % 10 == 0 or done == len(payloads):
                print(f"  {done}/{len(payloads)}")
    return pd.DataFrame(rows)


def _row_from_run(run: Any) -> dict[str, Any]:
    config = dict(run.config or {})
    model = _model_from_run(run, config)
    poyo_opt = _is_poyo_opt(run) if model == "poyo" else False
    recording_id = _recording_id(config)
    if recording_id is None and run.name:
        # eegnet_neurosoft_8band_<recording>
        prefix = f"{model}_neurosoft_8band_"
        if run.name.startswith(prefix):
            recording_id = run.name[len(prefix) :]
        elif run.name.startswith("poyo_eeg_neurosoft_8band_"):
            recording_id = run.name[len("poyo_eeg_neurosoft_8band_") :]
    tokenizer = config.get("model/tokenizer")
    if tokenizer is None:
        tok = (config.get("model") or {}).get("tokenizer")
        if isinstance(tok, dict):
            tokenizer = tok.get("_name_") or tok.get("name")
        else:
            tokenizer = tok
    row: dict[str, Any] = {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "group": run.group,
        "sweep_id": run.sweep.id if getattr(run, "sweep", None) else None,
        "species": _species_from_run(run),
        "model": model,
        "poyo_opt": poyo_opt,
        "recording_id": recording_id,
        "subject": _subject(config, recording_id),
        "fold": _fold(config),
        "tokenizer": tokenizer,
        "learning_rate": config.get("hyperparameters.learning_rate")
        or (config.get("hyperparameters") or {}).get("learning_rate"),
        "weight_decay": config.get("hyperparameters.weight_decay")
        or (config.get("hyperparameters") or {}).get("weight_decay"),
        "batch_size": config.get("hyperparameters.batch_size")
        or (config.get("hyperparameters") or {}).get("batch_size"),
        "split_type": config.get("data.split_type")
        or (config.get("data") or {}).get("split_type"),
        "tags": ",".join(sorted(t.lower() for t in (run.tags or []))),
    }
    missing: list[str] = []
    for short, key in METRIC_KEYS.items():
        val = _summary_max(run, key)
        if val is None:
            missing.append(key)
        else:
            row[short] = val
    if missing:
        hist_max = _history_maxes(run, missing)
        for short, key in METRIC_KEYS.items():
            if short not in row:
                row[short] = hist_max.get(key)
    return row


def fetch_group_runs(api: wandb.Api | None = None) -> pd.DataFrame:
    if api is None:
        api = wandb.Api()
    path = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT
    print(f"Fetching runs: {path} group={GROUP}")
    runs = api.runs(path, filters={"group": GROUP})
    rows: list[dict[str, Any]] = []
    for i, run in enumerate(runs, start=1):
        rows.append(_row_from_run(run))
        if i % 100 == 0:
            print(f"  scanned {i}")
    print(f"  scanned {len(rows)}")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in METRICS + ["learning_rate", "weight_decay", "batch_size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce").astype("Int64")
    return df


def fetch_multisubj_best(api: wandb.Api | None = None) -> pd.DataFrame:
    if api is None:
        api = wandb.Api()
    rows = []
    for species, run_id in MULTISUBJ_BEST_POYO.items():
        run = api.run(_run_path(run_id))
        row = _row_from_run(run)
        row["species"] = species
        row["model"] = "poyo_multisubj_best"
        row["poyo_opt"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_sweep_runs(
    sweep_ids: dict[str, str], api: wandb.Api | None = None
) -> pd.DataFrame:
    if api is None:
        api = wandb.Api()
    rows: list[dict[str, Any]] = []
    for species, sweep_id in sweep_ids.items():
        sweep = api.sweep(_sweep_path(sweep_id))
        n_fin = 0
        for run in sweep.runs:
            if run.state != "finished":
                continue
            row = _row_from_run(run)
            sp = _species_from_run(run)
            row["species"] = species if sp == "unknown" else sp
            n_fin += 1
            rows.append(row)
        print(f"  sweep {sweep_id} ({species}): {n_fin} finished")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in METRICS + ["learning_rate", "weight_decay", "batch_size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce").astype("Int64")
    return df


def _fill_one(run_id: str) -> dict[str, Any]:
    api = wandb.Api()
    run = api.run(_run_path(run_id))
    keys = [METRIC_KEYS[s] for s in ("precision", "recall", "balanced_acc")]
    hist = _history_maxes(run, keys)
    out: dict[str, Any] = {"run_id": run_id}
    for short in ("precision", "recall", "balanced_acc"):
        out[short] = hist.get(METRIC_KEYS[short])
    return out


def fill_missing_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill precision/recall/balanced_acc from history max (summary stores min)."""
    if df.empty or "precision" not in df.columns:
        return df
    need = df["precision"].isna() & (df["state"] == "finished")
    poyo_opt = (
        df["poyo_opt"].astype(bool)
        if "poyo_opt" in df.columns
        else pd.Series(False, index=df.index)
    )
    is_primary = df["model"].isin(["eegnet", "gru", "poyo_multisubj_best"]) | (
        (df["model"] == "poyo") & poyo_opt
    )
    need = need & is_primary
    ids = df.loc[need, "run_id"].astype(str).tolist()
    if not ids:
        return df
    print(
        f"Filling history-max P/R/bAcc for {len(ids)} runs ({N_WORKERS} workers)..."
    )
    updates: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_fill_one, rid): rid for rid in ids}
        done = 0
        for fut in as_completed(futures):
            row = fut.result()
            updates[row["run_id"]] = row
            done += 1
            if done % 25 == 0 or done == len(ids):
                print(f"  {done}/{len(ids)}")
    out = df.copy()
    for rid, row in updates.items():
        mask = out["run_id"].astype(str) == rid
        for key in ("precision", "recall", "balanced_acc"):
            if row.get(key) is not None:
                out.loc[mask, key] = row[key]
    return out


def primary_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Finished EEGNet, GRU, and opt-HP POYO."""
    finished = df[df["state"] == "finished"].copy()
    keep_eegnet = finished["model"] == "eegnet"
    keep_gru = finished["model"] == "gru"
    keep_poyo = (finished["model"] == "poyo") & finished["poyo_opt"].astype(
        bool
    )
    out = finished[keep_eegnet | keep_gru | keep_poyo].copy()
    out["model"] = pd.Categorical(
        out["model"], categories=MODEL_ORDER, ordered=True
    )
    return out.reset_index(drop=True)


def fold_means_by_session(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["session"] = work["recording_id"].fillna(work["run_name"]).astype(str)
    grouped = work.groupby(
        ["species", "model", "session"], observed=True, dropna=False
    )
    rows: list[dict[str, Any]] = []
    for (species, model, session), g in grouped:
        row: dict[str, Any] = {
            "species": species,
            "model": model,
            "session": session,
            "n_folds": int(g["fold"].nunique(dropna=True)),
        }
        for m in METRICS:
            vals = pd.to_numeric(g[m], errors="coerce").dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{m}_std"] = (
                float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            )
            row[f"{m}_n"] = int(len(vals))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    present = [m for m in MODEL_ORDER if m in set(out["model"].astype(str))]
    if present:
        out["model"] = pd.Categorical(
            out["model"], categories=MODEL_ORDER, ordered=True
        )
    return out.reset_index(drop=True)


def fold_means_from_session_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Fold-mean val_session metrics; ``session`` is already the short id."""
    if df.empty:
        return df.copy()
    work = df.copy()
    work["session"] = work["session"].astype(str)
    grouped = work.groupby(
        ["species", "model", "session"], observed=True, dropna=False
    )
    rows: list[dict[str, Any]] = []
    for (species, model, session), g in grouped:
        row: dict[str, Any] = {
            "species": species,
            "model": str(model),
            "session": session,
            "n_folds": int(g["fold"].nunique(dropna=True)),
        }
        for m in METRICS:
            vals = pd.to_numeric(g[m], errors="coerce").dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{m}_std"] = (
                float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            )
            row[f"{m}_n"] = int(len(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def flag_outliers(session_df: pd.DataFrame) -> pd.DataFrame:
    out = session_df.copy()
    missing = out["f1_mean"].isna()
    ceiling = out["f1_mean"] >= SINGLESESS_OUTLIER_F1_GE
    out["is_outlier"] = missing | ceiling
    out["outlier_reason"] = np.where(
        missing,
        "missing_f1",
        np.where(ceiling, f"f1>={SINGLESESS_OUTLIER_F1_GE}", ""),
    )
    return out


def model_summary(session_df: pd.DataFrame) -> pd.DataFrame:
    flagged = flag_outliers(session_df)
    rows: list[dict[str, Any]] = []
    for species in SPECIES_ORDER:
        for model in MODEL_ORDER:
            g_all = flagged[
                (flagged["species"] == species) & (flagged["model"] == model)
            ]
            g_filt = g_all[~g_all["is_outlier"]]
            for label, g in (
                ("all", g_all),
                ("excl. outliers", g_filt),
            ):
                row: dict[str, Any] = {
                    "species": species,
                    "model": model,
                    "subset": label,
                    "n_sessions": int(len(g)),
                    "n_outliers": int(g_all["is_outlier"].sum())
                    if label == "all"
                    else int(len(g_all) - len(g)),
                    "mean_n_folds": float(g["n_folds"].mean())
                    if len(g)
                    else np.nan,
                }
                for m in METRICS:
                    vals = g[f"{m}_mean"].dropna()
                    row[f"{m}_mean"] = (
                        float(vals.mean()) if len(vals) else np.nan
                    )
                    row[f"{m}_std"] = (
                        float(vals.std(ddof=0)) if len(vals) else np.nan
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def matched_sessions(session_df: pd.DataFrame) -> pd.DataFrame:
    """Sessions with a non-outlier fold-mean for every primary model."""
    flagged = flag_outliers(session_df)
    keep = flagged[~flagged["is_outlier"]]
    rows = []
    for species in SPECIES_ORDER:
        sub = keep[keep["species"] == species]
        counts = sub.groupby("session")["model"].nunique()
        ok = set(counts[counts == len(MODEL_ORDER)].index)
        g = sub[sub["session"].isin(ok)]
        for model in MODEL_ORDER:
            gm = g[g["model"] == model]
            row: dict[str, Any] = {
                "species": species,
                "model": model,
                "n_sessions": int(gm["session"].nunique()),
            }
            for m in METRICS:
                vals = gm[f"{m}_mean"].dropna()
                row[f"{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
                row[f"{m}_std"] = (
                    float(vals.std(ddof=0)) if len(vals) else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def best_session_per_model(session_df: pd.DataFrame) -> pd.DataFrame:
    flagged = flag_outliers(session_df)
    keep = flagged[~flagged["is_outlier"]]
    idx = keep.groupby(["species", "model"], observed=True)["f1_mean"].idxmax()
    cols = [
        "species",
        "model",
        "session",
        "n_folds",
        "f1_mean",
        "auroc_mean",
        "precision_mean",
        "recall_mean",
        "balanced_acc_mean",
    ]
    return (
        keep.loc[idx, cols]
        .sort_values(["species", "model"])
        .reset_index(drop=True)
    )


def fold_means_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["subject"] = work["subject"].fillna(work["run_name"]).astype(str)
    grouped = work.groupby(["species", "subject"], observed=True, dropna=False)
    rows: list[dict[str, Any]] = []
    for (species, subject), g in grouped:
        row: dict[str, Any] = {
            "species": species,
            "subject": subject,
            "n_folds": int(g["fold"].nunique(dropna=True)),
        }
        for m in METRICS:
            vals = pd.to_numeric(g[m], errors="coerce").dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{m}_std"] = (
                float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            )
            row[f"{m}_n"] = int(len(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def subject_summary(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for species in SPECIES_ORDER:
        g = subject_df[subject_df["species"] == species]
        row: dict[str, Any] = {
            "species": species,
            "n_subjects": int(len(g)),
            "mean_n_folds": float(g["n_folds"].mean()) if len(g) else np.nan,
        }
        for m in METRICS:
            vals = g[f"{m}_mean"].dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{m}_std"] = float(vals.std(ddof=0)) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(
    sess_summary: pd.DataFrame,
    subj_summary_df: pd.DataFrame,
    multi: pd.DataFrame,
) -> pd.DataFrame:
    """EEGNet/GRU/POYO session means vs POYO subject mean vs best multi POYO."""
    filt = sess_summary[sess_summary["subset"] == "excl. outliers"]
    rows: list[dict[str, Any]] = []
    sess_map = {"eegnet": "eegnet", "gru": "gru", "poyo": "poyo_sess"}
    for species in SPECIES_ORDER:
        for model, cond in sess_map.items():
            srow = filt[(filt["species"] == species) & (filt["model"] == model)]
            if srow.empty:
                continue
            row: dict[str, Any] = {
                "species": species,
                "condition": cond,
                "n_units": int(srow["n_sessions"].iloc[0]),
                "unit": "session",
            }
            for m in METRICS:
                row[f"{m}_mean"] = float(srow[f"{m}_mean"].iloc[0])
                row[f"{m}_std"] = float(srow[f"{m}_std"].iloc[0])
            rows.append(row)

        sub = subj_summary_df[subj_summary_df["species"] == species]
        if not sub.empty:
            row = {
                "species": species,
                "condition": "poyo_subj",
                "n_units": int(sub["n_subjects"].iloc[0]),
                "unit": "subject",
            }
            for m in METRICS:
                row[f"{m}_mean"] = float(sub[f"{m}_mean"].iloc[0])
                row[f"{m}_std"] = float(sub[f"{m}_std"].iloc[0])
            rows.append(row)

        mrow = multi[multi["species"] == species]
        if not mrow.empty:
            row = {
                "species": species,
                "condition": "poyo_multi",
                "n_units": 1,
                "unit": "pooled (fold 0)",
            }
            for m in METRICS:
                row[f"{m}_mean"] = (
                    float(mrow[m].iloc[0])
                    if pd.notna(mrow[m].iloc[0])
                    else np.nan
                )
                row[f"{m}_std"] = 0.0
            rows.append(row)
    out = pd.DataFrame(rows)
    out["condition"] = pd.Categorical(
        out["condition"], categories=CONDITION_ORDER, ordered=True
    )
    return out.sort_values(["species", "condition"]).reset_index(drop=True)


def _fmt(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "nan"
    if pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f}±{std:.4f}"


def print_inventory(raw: pd.DataFrame, primary: pd.DataFrame) -> None:
    print("\n=== Inventory (all group runs) ===")
    print(
        raw.groupby(["state", "species", "model"], dropna=False)
        .size()
        .unstack("model", fill_value=0)
        .to_string()
    )
    extra = raw[
        (raw["state"] == "finished")
        & (raw["model"] == "poyo")
        & (~raw["poyo_opt"].astype(bool))
    ]
    print(
        f"\nExcluded non-opt POYO (finished): {len(extra)} "
        f"(minipigs={int((extra.species == 'minipigs').sum())}, "
        f"monkeys={int((extra.species == 'monkeys').sum())})"
    )
    print("\n=== Primary finished runs ===")
    print(
        primary.groupby(["species", "model"], observed=True)
        .size()
        .unstack("model", fill_value=0)
        .to_string()
    )
    print("\nFolds in primary:")
    print(
        primary.groupby(["species", "model", "fold"], observed=True)
        .size()
        .unstack("fold", fill_value=0)
        .to_string()
    )


def print_tables(
    summary: pd.DataFrame,
    matched: pd.DataFrame,
    best: pd.DataFrame,
    outliers: pd.DataFrame,
    comparison: pd.DataFrame,
    subj_units: pd.DataFrame,
) -> None:
    print("\n=== Comparison: session EEGNet/GRU vs POYO paradigms ===")
    print(
        "  EEGNet/GRU/POYO session: mean±std across sessions of fold-means "
        "(excl. outliers)"
    )
    print("  POYO subject: mean±std across subjects of fold-means")
    print("  POYO multi (best): capacity-ablation fold-0 winner")
    show_c = comparison.copy()
    for m in METRICS:
        show_c[m] = [
            _fmt(a, b) for a, b in zip(show_c[f"{m}_mean"], show_c[f"{m}_std"])
        ]
    print(
        show_c[["species", "condition", "n_units", "unit", *METRICS]].to_string(
            index=False
        )
    )

    print(
        "\n=== Session-mean metrics (fold-mean per session, then mean±std) ==="
    )
    show = summary.copy()
    for m in METRICS:
        show[m] = [
            _fmt(a, b) for a, b in zip(show[f"{m}_mean"], show[f"{m}_std"])
        ]
    print(
        show[
            [
                "species",
                "model",
                "subset",
                "n_sessions",
                "mean_n_folds",
                *METRICS,
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    print("\n=== POYO single-subject fold-means ===")
    if subj_units.empty:
        print("  (none)")
    else:
        ss = subj_units.sort_values(
            ["species", "f1_mean"], ascending=[True, False]
        )
        print(
            ss[
                [
                    "species",
                    "subject",
                    "n_folds",
                    "f1_mean",
                    "auroc_mean",
                    "precision_mean",
                    "recall_mean",
                    "balanced_acc_mean",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    print("\n=== Matched sessions (all 3 models, excl. outliers) ===")
    show_m = matched.copy()
    for m in METRICS:
        show_m[m] = [
            _fmt(a, b) for a, b in zip(show_m[f"{m}_mean"], show_m[f"{m}_std"])
        ]
    print(
        show_m[["species", "model", "n_sessions", *METRICS]].to_string(
            index=False
        )
    )

    print("\n=== Best session per model (excl. outliers, by fold-mean F1) ===")
    print(best.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(
        f"\n=== Session outliers (missing F1 or fold-mean F1 "
        f">= {SINGLESESS_OUTLIER_F1_GE}) ==="
    )
    if outliers.empty:
        print("  (none)")
    else:
        print(
            outliers[
                ["species", "model", "session", "f1_mean", "outlier_reason"]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    print("\n=== Deltas: session EEGNet/GRU minus each POYO reference ===")
    for species in SPECIES_ORDER:
        sub = comparison[comparison["species"] == species].set_index(
            "condition"
        )
        refs = ["poyo_sess", "poyo_subj", "poyo_multi"]
        print(f"  {species}")
        for base in ("eegnet", "gru"):
            if base not in sub.index:
                continue
            b_f1 = float(sub.loc[base, "f1_mean"])
            b_auc = float(sub.loc[base, "auroc_mean"])
            for ref in refs:
                if ref not in sub.index:
                    continue
                d_f1 = b_f1 - float(sub.loc[ref, "f1_mean"])
                d_auc = b_auc - float(sub.loc[ref, "auroc_mean"])
                print(
                    f"    {base:6s} − {ref:10s}  "
                    f"ΔF1={d_f1:+.4f}  ΔAUROC={d_auc:+.4f}"
                )


def plot_comparison_bars(
    comparison: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    stem_suffix: str,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    x = np.arange(len(CONDITION_ORDER))
    width = 0.7
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    for ax, species in zip(axes, SPECIES_ORDER):
        sub = comparison[comparison["species"] == species].set_index(
            "condition"
        )
        means = [
            float(sub.loc[c, mean_col]) if c in sub.index else np.nan
            for c in CONDITION_ORDER
        ]
        stds = [
            float(sub.loc[c, std_col]) if c in sub.index else 0.0
            for c in CONDITION_ORDER
        ]
        colors = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
        bars = ax.bar(
            x,
            means,
            width,
            yerr=stds,
            capsize=4,
            color=colors,
            edgecolor="white",
            error_kw=dict(lw=1.1),
        )
        for bar, mean, std in zip(bars, means, stds):
            if np.isnan(mean):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.0 if pd.isna(std) else std) + 0.015,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=8
        )
        ax.set_title(species)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_{stem_suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def _short_session(session: str) -> str:
    """sub-07_ses-03_task-AcousStim_acq-RH_desc-raw → 07-s03-RH."""
    parts = session.split("_")
    sub = parts[0].replace("sub-", "") if parts else session
    ses = ""
    acq = ""
    for p in parts:
        if p.startswith("ses-"):
            ses = p.replace("ses-", "s")
        if p.startswith("acq-"):
            acq = p.replace("acq-", "")
    if ses or acq:
        return f"{sub}-{ses}-{acq}".strip("-")
    return session[:18]


def _subject_from_session(session: str) -> str:
    return str(session).split("_")[0]


def attach_poyo_session_metrics(
    session_df: pd.DataFrame,
    subj_sess: pd.DataFrame,
    multi_sess: pd.DataFrame,
) -> pd.DataFrame:
    """Add per-session POYO subject / multi bars for sessions already plotted."""
    keep = session_df.copy()
    keep["model"] = keep["model"].astype(str)
    keep["session_key"] = keep["session"].map(_shorten_session_id)
    keep_keys = set(zip(keep["species"].astype(str), keep["session_key"]))

    extras = []
    for extra, model_name in (
        (subj_sess, "poyo_subj"),
        (multi_sess, "poyo_multi"),
    ):
        if extra is None or extra.empty:
            continue
        part = extra.copy()
        part["model"] = model_name
        part["session_key"] = part["session"].map(_shorten_session_id)
        mask = [
            (str(sp), sk) in keep_keys
            for sp, sk in zip(part["species"], part["session_key"])
        ]
        extras.append(part.loc[mask])
    if not extras:
        return keep
    return pd.concat([keep, *extras], ignore_index=True)


def plot_per_session_bars(
    session_df: pd.DataFrame,
    subj_sess: pd.DataFrame,
    multi_sess: pd.DataFrame,
    metric: str,
    stem_suffix: str,
) -> Path:
    flagged = flag_outliers(session_df)
    keep = flagged[~flagged["is_outlier"]].copy()
    keep = attach_poyo_session_metrics(keep, subj_sess, multi_sess)
    keep["short"] = keep["session"].map(_short_session)
    keep = keep.drop_duplicates(
        subset=["species", "model", "short"], keep="first"
    )
    fig, axes = plt.subplots(2, 1, figsize=(18, 9.0), sharey=False)
    n_models = len(PER_SESSION_MODELS)
    width = 0.15
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * width
    ylabels = {
        "f1": "Fold-mean max val F1",
        "auroc": "Fold-mean max val AUROC",
    }
    titles = {
        "f1": "Per-session F1 (excl. outliers)",
        "auroc": "Per-session AUROC (excl. outliers)",
    }
    col = f"{metric}_mean"
    for ax, species in zip(axes, SPECIES_ORDER):
        sub = keep[keep["species"] == species]
        sessions = sorted(sub["short"].unique())
        x = np.arange(len(sessions))
        for i, model in enumerate(PER_SESSION_MODELS):
            means = []
            stds = []
            g = sub[sub["model"] == model].set_index("short")
            for sess in sessions:
                if sess in g.index:
                    means.append(float(g.loc[sess, col]))
                    std_val = g.loc[sess, f"{metric}_std"]
                    stds.append(0.0 if pd.isna(std_val) else float(std_val))
                else:
                    means.append(np.nan)
                    stds.append(0.0)
            ax.bar(
                x + offsets[i],
                means,
                width,
                yerr=stds,
                capsize=1.2,
                label=PER_SESSION_LABELS[model],
                color=PER_SESSION_COLORS[model],
                edgecolor="white",
                linewidth=0.3,
                error_kw=dict(lw=0.5),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(sessions, rotation=90, fontsize=6.5)
        ax.set_title(species)
        ax.set_ylabel(ylabels[metric])
        ax.legend(
            frameon=False,
            fontsize=7.5,
            loc="upper right",
            ncol=2,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.7, len(sessions) - 0.3)
    fig.suptitle(titles[metric], y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_{stem_suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_vs_poyo_multi(
    session_df: pd.DataFrame,
    multi_sess: pd.DataFrame,
    metric: str,
    stem_suffix: str,
) -> Path:
    """Per-session EEGNet / GRU vs per-session best multi-subject POYO."""
    flagged = flag_outliers(session_df)
    keep = flagged[~flagged["is_outlier"]].copy()
    keep["model"] = keep["model"].astype(str)
    keep["session_key"] = keep["session"].map(_shorten_session_id)
    multi_s = multi_sess.copy()
    multi_s["session_key"] = multi_s["session"].map(_shorten_session_id)
    col = f"{metric}_mean"
    poyo_col = f"poyo_{metric}"
    metric_label = "F1" if metric == "f1" else "AUROC"
    baselines = ["eegnet", "gru"]
    fig, axes = plt.subplots(
        2, 2, figsize=(10.5, 9.2), sharex=True, sharey=True
    )
    for row, baseline in enumerate(baselines):
        for col_i, species in enumerate(SPECIES_ORDER):
            ax = axes[row, col_i]
            left = keep[
                (keep["species"] == species) & (keep["model"] == baseline)
            ][["session_key", col]]
            right = multi_s[multi_s["species"] == species][
                ["session_key", col]
            ].rename(columns={col: poyo_col})
            merged = left.merge(right, on="session_key", how="inner")
            merged = merged.dropna(subset=[col, poyo_col])
            n_above = (
                int((merged[col] > merged[poyo_col]).sum())
                if len(merged)
                else 0
            )
            if len(merged):
                ax.scatter(
                    merged[poyo_col],
                    merged[col],
                    c=MODEL_COLORS[baseline],
                    s=32,
                    alpha=0.8,
                    edgecolors="white",
                    linewidths=0.3,
                    zorder=3,
                )
            lims = [0.0, 1.0]
            ax.plot(lims, lims, color="0.5", ls="--", lw=1, zorder=0)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_title(
                f"{species} · {MODEL_LABELS[baseline]} "
                f"(n={len(merged)}, {n_above} above)"
            )
            ax.set_aspect("equal", adjustable="box")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 1:
                ax.set_xlabel(f"Best multi-subject POYO session {metric_label}")
            if col_i == 0:
                ax.set_ylabel(
                    f"Session {MODEL_LABELS[baseline]} {metric_label}"
                )
    fig.suptitle(
        f"Per-session {metric_label}: EEGNet / GRU vs best multi-subject POYO",
        y=1.01,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_{stem_suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def main() -> None:
    import sys

    use_cached = "--cached" in sys.argv
    print(
        f"Resolved: project={PROJECT}, group={GROUP}, entity={ENTITY}\n"
        f"  POYO opt singlesess sweeps: {sorted(POYO_OPT_SWEEPS)}\n"
        f"  POYO singlesub sweeps: {POYO_SINGLESUB_SWEEPS}\n"
        f"  Best multisubj POYO: {MULTISUBJ_BEST_POYO}"
    )

    csv_runs = CSV_DIR / f"{STEM}_runs.csv"
    csv_multi = CSV_DIR / f"{STEM}_multisubj_best.csv"
    csv_subj = CSV_DIR / f"{STEM}_poyo_singlesubj_runs.csv"
    csv_multi_val = CSV_DIR / f"{STEM}_poyo_multisubj_val_sessions.csv"
    csv_subj_val = CSV_DIR / f"{STEM}_poyo_singlesubj_val_sessions.csv"
    if (
        use_cached
        and csv_runs.exists()
        and csv_multi.exists()
        and csv_subj.exists()
    ):
        print(f"Loading cached runs from {csv_runs}")
        raw = pd.read_csv(csv_runs)
        multi = pd.read_csv(csv_multi)
        subj_runs = pd.read_csv(csv_subj)
    else:
        api = wandb.Api()
        if use_cached and csv_runs.exists() and csv_multi.exists():
            print(f"Loading cached session runs from {csv_runs}")
            raw = pd.read_csv(csv_runs)
            multi = pd.read_csv(csv_multi)
        else:
            raw = fetch_group_runs(api)
            multi = fetch_multisubj_best(api)
        print("Fetching POYO single-subject sweeps...")
        subj_runs = fetch_sweep_runs(POYO_SINGLESUB_SWEEPS, api)

    if raw.empty:
        raise SystemExit("No runs found in group.")

    raw = fill_missing_metrics(raw)
    multi = fill_missing_metrics(multi)
    subj_runs = fill_missing_metrics(subj_runs)
    raw.to_csv(csv_runs, index=False)
    multi.to_csv(csv_multi, index=False)
    subj_runs.to_csv(csv_subj, index=False)

    if use_cached and csv_multi_val.exists() and csv_subj_val.exists():
        print(f"Loading cached val_session tables from {csv_multi_val}")
        multi_val = pd.read_csv(csv_multi_val)
        subj_val = pd.read_csv(csv_subj_val)
    else:
        api = wandb.Api()
        print("Fetching per-session val_session metrics (history max)...")
        multi_val = fetch_multisubj_session_metrics(api)
        subj_val = fetch_singlesubj_session_metrics(subj_runs, api)
        multi_val.to_csv(csv_multi_val, index=False)
        subj_val.to_csv(csv_subj_val, index=False)

    primary = primary_runs(raw)
    print_inventory(raw, primary)
    session_df = fold_means_by_session(primary)
    flagged = flag_outliers(session_df)
    summary = model_summary(session_df)
    matched = matched_sessions(session_df)
    best = best_session_per_model(session_df)
    outliers = flagged[flagged["is_outlier"]]

    subj_units = fold_means_by_subject(subj_runs)
    subj_sum = subject_summary(subj_units)
    comparison = comparison_table(summary, subj_sum, multi)
    multi_sess = fold_means_from_session_rows(multi_val)
    subj_sess = fold_means_from_session_rows(subj_val)

    print_tables(summary, matched, best, outliers, comparison, subj_units)
    print("\n=== val_session coverage (best multi / single-subject POYO) ===")
    for label, df in (
        ("poyo_multi", multi_sess),
        ("poyo_subj", subj_sess),
    ):
        if df.empty:
            print(f"  {label}: (none)")
            continue
        counts = df.groupby("species").size()
        for species, n in counts.items():
            print(f"  {label} {species}: {int(n)} sessions")

    plot_comparison_bars(
        comparison,
        "f1",
        "Max val F1 (mean ± std across units)",
        "EEGNet / GRU session means vs POYO session, subject, and best multi",
        "f1_by_model",
    )
    plot_comparison_bars(
        comparison,
        "auroc",
        "Max val AUROC (mean ± std across units)",
        "EEGNet / GRU session means vs POYO session, subject, and best multi",
        "auroc_by_model",
    )
    plot_per_session_bars(
        session_df, subj_sess, multi_sess, "f1", "supp_f1_per_session"
    )
    plot_per_session_bars(
        session_df, subj_sess, multi_sess, "auroc", "supp_auroc_per_session"
    )
    plot_vs_poyo_multi(session_df, multi_sess, "f1", "f1_vs_poyo_multi")
    plot_vs_poyo_multi(session_df, multi_sess, "auroc", "auroc_vs_poyo_multi")

    csv_sessions = CSV_DIR / f"{STEM}_sessions.csv"
    csv_summary = CSV_DIR / f"{STEM}_summary.csv"
    csv_matched = CSV_DIR / f"{STEM}_matched.csv"
    csv_comparison = CSV_DIR / f"{STEM}_comparison.csv"
    csv_subj_units = CSV_DIR / f"{STEM}_poyo_singlesubj_units.csv"
    csv_multi_sess = CSV_DIR / f"{STEM}_poyo_multisubj_session_means.csv"
    csv_subj_sess = CSV_DIR / f"{STEM}_poyo_singlesubj_session_means.csv"
    flagged.to_csv(csv_sessions, index=False)
    summary.to_csv(csv_summary, index=False)
    matched.to_csv(csv_matched, index=False)
    comparison.to_csv(csv_comparison, index=False)
    subj_units.to_csv(csv_subj_units, index=False)
    multi_sess.to_csv(csv_multi_sess, index=False)
    subj_sess.to_csv(csv_subj_sess, index=False)
    print(f"Saved: {csv_runs}")
    print(f"Saved: {csv_sessions}")
    print(f"Saved: {csv_summary}")
    print(f"Saved: {csv_matched}")
    print(f"Saved: {csv_comparison}")
    print(f"Saved: {csv_subj}")
    print(f"Saved: {csv_subj_units}")
    print(f"Saved: {csv_multi}")
    print(f"Saved: {csv_multi_val}")
    print(f"Saved: {csv_subj_val}")
    print(f"Saved: {csv_multi_sess}")
    print(f"Saved: {csv_subj_sess}")


if __name__ == "__main__":
    main()
