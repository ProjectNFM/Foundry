"""Session-level EEGNet / GRU vs POYO on NeuroSoft 8-band decoding.

Fetches finished runs from ``NEUROSOFT_INTRASESSION_SINGLESESS`` and
compares EEGNet and GRU against three POYO references, **fold 0 only**:

1. Opt-HP **single-session** POYO
2. Opt-HP **single-subject** POYO (``val_session/`` per recording)
3. **Best multi-subject** POYO (``val_session/`` per recording)

Primary bars / tables: unweighted mean±std **across sessions**.
Supplementary: species-level **pooled** metrics for every condition.
Pooled F1 / precision / recall come from summing validation confusion
matrices at each run's max-F1 epoch (hard-label pool). Pooled AUROC is
the true run ``val/`` when one model spans the pool (multi-subject
POYO); otherwise a trial-count-weighted mean of per-run AUROCs (ranking
scores are not stored).

Usage:
    uv run python analysis/20260818-LS-singlesess-eegnet-gru-baselines.py
    uv run python analysis/20260818-LS-singlesess-eegnet-gru-baselines.py --cached
"""

from __future__ import annotations

import json
import time
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
FOLD_KEEP = 0

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
MODEL_LABELS = {
    "eegnet": "EEGNet",
    "gru": "GRU",
    "poyo": "POYO (single-session)",
}
MODEL_COLORS = {"eegnet": "#E8963E", "gru": "#C44E52", "poyo": "#4C72B0"}
SPECIES_ORDER = ["minipigs", "monkeys"]

CONDITION_ORDER = ["eegnet", "gru", "poyo_sess", "poyo_subj", "poyo_multi"]
CONDITION_LABELS = {
    "eegnet": "EEGNet\nsingle-session",
    "gru": "GRU\nsingle-session",
    "poyo_sess": "POYO\nsingle-session",
    "poyo_subj": "POYO\nsingle-subject",
    "poyo_multi": "POYO\nmulti-subject",
}
CONDITION_NAMES = {k: v.replace("\n", " ") for k, v in CONDITION_LABELS.items()}
CONDITION_COLORS = {
    "eegnet": "#E8963E",
    "gru": "#C44E52",
    "poyo_sess": "#4C72B0",
    "poyo_subj": "#8172B2",
    "poyo_multi": "#55A868",
}

PER_SESSION_MODELS = ["eegnet", "gru", "poyo", "poyo_subj", "poyo_multi"]
PER_SESSION_LABELS = {
    "eegnet": "EEGNet (single-session)",
    "gru": "GRU (single-session)",
    "poyo": "POYO (single-session)",
    "poyo_subj": "POYO (single-subject)",
    "poyo_multi": "POYO (multi-subject)",
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
CM_KEY = f"val/{TASK}_confusion_counts"
F1_KEY = METRIC_KEYS["f1"]
AUROC_KEY = METRIC_KEYS["auroc"]


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


def keep_fold(df: pd.DataFrame, fold: int = FOLD_KEEP) -> pd.DataFrame:
    if df.empty or "fold" not in df.columns:
        return df.copy()
    folds = pd.to_numeric(df["fold"], errors="coerce")
    return df.loc[folds == fold].copy()


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
    subj_sess: pd.DataFrame,
    multi_sess: pd.DataFrame,
    session_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fold-0 session mean±std for EEGNet/GRU/POYO single-session, single-subject, and multi-subject."""
    filt = sess_summary[sess_summary["subset"] == "excl. outliers"]
    drop_keys = _outlier_session_keys(session_df)
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

        for cond, source in (
            ("poyo_subj", subj_sess),
            ("poyo_multi", multi_sess),
        ):
            part = _filter_ref_sessions(source, species, drop_keys)
            if part.empty:
                continue
            rows.append(_session_agg_row(part, species, cond))
    out = pd.DataFrame(rows)
    out["condition"] = pd.Categorical(
        out["condition"], categories=CONDITION_ORDER, ordered=True
    )
    return out.sort_values(["species", "condition"]).reset_index(drop=True)


def _session_agg_row(
    part: pd.DataFrame, species: str, cond: str
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "species": species,
        "condition": cond,
        "n_units": int(len(part)),
        "unit": "session",
    }
    for m in METRICS:
        vals = part[f"{m}_mean"].dropna()
        row[f"{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
        row[f"{m}_std"] = float(vals.std(ddof=0)) if len(vals) else np.nan
    return row


def _outlier_session_keys(session_df: pd.DataFrame) -> set[tuple[str, str]]:
    flagged = flag_outliers(session_df)
    out = flagged[flagged["is_outlier"]]
    return {
        (str(sp), _shorten_session_id(sess))
        for sp, sess in zip(out["species"], out["session"])
    }


def _filter_ref_sessions(
    multi_sess: pd.DataFrame,
    species: str,
    drop_keys: set[tuple[str, str]],
) -> pd.DataFrame:
    if multi_sess.empty:
        return multi_sess
    work = multi_sess[multi_sess["species"] == species].copy()
    work["session_key"] = work["session"].map(_shorten_session_id)
    work = flag_outliers(work)
    keep_mask = [
        (str(sp), sk) not in drop_keys
        for sp, sk in zip(work["species"], work["session_key"])
    ]
    work = work.loc[keep_mask]
    return work[~work["is_outlier"]].reset_index(drop=True)


def _fmt(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "nan"
    if pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f}±{std:.4f}"


def _as_cm(val: Any) -> np.ndarray | None:
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    try:
        arr = np.asarray(val, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.size == 0:
        return None
    return arr


def _macro_from_cm(cm: np.ndarray) -> dict[str, float]:
    """Macro P/R/F1 over classes with support, matching WandB session F1."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    present = support > 0
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * prec * rec,
        prec + rec,
        out=np.zeros_like(tp),
        where=(prec + rec) > 0,
    )
    if not present.any():
        return {
            "f1": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "balanced_acc": np.nan,
            "n_trials": 0.0,
        }
    return {
        "f1": float(f1[present].mean()),
        "precision": float(prec[present].mean()),
        "recall": float(rec[present].mean()),
        "balanced_acc": float(rec[present].mean()),
        "n_trials": float(cm.sum()),
    }


def _history_with_retry(run: Any, tries: int = 4) -> pd.DataFrame:
    delay = 2.0
    last_exc: Exception | None = None
    for _ in range(tries):
        try:
            hist = run.history(samples=20_000, pandas=True)
            if hist is None:
                return pd.DataFrame()
            return hist
        except Exception as exc:
            last_exc = exc
            time.sleep(delay)
            delay = min(delay * 2, 20.0)
    if last_exc is not None:
        print(f"  history failed for {run.id}: {last_exc}")
    return pd.DataFrame()


def _cm_at_max_f1_from_history(hist: pd.DataFrame) -> np.ndarray | None:
    if hist.empty or F1_KEY not in hist.columns:
        return None
    last_cm = None
    best_f1 = -np.inf
    best_cm = None
    has_cm = CM_KEY in hist.columns
    for _, row in hist.iterrows():
        if has_cm:
            cm = _as_cm(row.get(CM_KEY))
            if cm is not None:
                last_cm = cm
        f1 = row.get(F1_KEY)
        if f1 is None or (isinstance(f1, float) and np.isnan(f1)):
            continue
        f1_v = float(f1)
        if f1_v > best_f1:
            best_f1 = f1_v
            best_cm = last_cm
    return None if best_cm is None else np.asarray(best_cm, dtype=float)


def _fetch_one_confusion(run_id: str) -> dict[str, Any]:
    api = wandb.Api()
    run = api.run(_run_path(str(run_id)))
    from_summary = False
    hist = _history_with_retry(run)
    cm = _cm_at_max_f1_from_history(hist)
    if cm is None:
        cm = _as_cm(run.summary.get(CM_KEY))
        from_summary = cm is not None
    if cm is None:
        return {
            "run_id": run_id,
            "n_trials": np.nan,
            "cm_json": None,
            "from_summary": False,
        }
    return {
        "run_id": run_id,
        "n_trials": float(cm.sum()),
        "cm_json": json.dumps(cm.astype(int).tolist()),
        "from_summary": from_summary,
    }


def fetch_confusion_at_max_f1(run_ids: list[str]) -> pd.DataFrame:
    ids = [str(r) for r in run_ids if r]
    print(
        f"Fetching max-F1 confusion matrices for {len(ids)} runs "
        f"({N_WORKERS} workers)..."
    )
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_fetch_one_confusion, rid): rid for rid in ids}
        done = 0
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 20 == 0 or done == len(ids):
                print(f"  {done}/{len(ids)}")
    return pd.DataFrame(rows)


def _cm_map(cm_df: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if cm_df.empty:
        return out
    for _, row in cm_df.iterrows():
        raw = row.get("cm_json")
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        cm = _as_cm(json.loads(raw) if isinstance(raw, str) else raw)
        if cm is not None:
            out[str(row["run_id"])] = cm
    return out


def pooled_reference_table(
    primary: pd.DataFrame,
    session_df: pd.DataFrame,
    subj_fold0: pd.DataFrame,
    multi: pd.DataFrame,
    cm_df: pd.DataFrame,
) -> pd.DataFrame:
    """Species-level pooled metrics for every condition.

    F1 / precision / recall / balanced acc: sum confusion matrices at each
    run's max-F1 epoch, then macro-average over classes with support.
    AUROC: true ``val/`` for the single multi-subject model; otherwise the
    trial-count-weighted mean of per-run history-max AUROC.
    """
    cms = _cm_map(cm_df)
    drop_keys = _outlier_session_keys(session_df)
    n_trials_map = (
        {
            str(r["run_id"]): r["n_trials"]
            for _, r in cm_df.iterrows()
            if pd.notna(r.get("n_trials"))
        }
        if not cm_df.empty
        else {}
    )

    def _keep_session(row: pd.Series) -> bool:
        sid = row.get("recording_id") or row.get("session") or ""
        key = (str(row["species"]), _shorten_session_id(str(sid)))
        return key not in drop_keys

    rows: list[dict[str, Any]] = []
    sess_map = {"eegnet": "eegnet", "gru": "gru", "poyo": "poyo_sess"}
    for species in SPECIES_ORDER:
        for model, cond in sess_map.items():
            part = primary[
                (primary["species"] == species) & (primary["model"] == model)
            ]
            part = part[part.apply(_keep_session, axis=1)]
            rows.append(
                _pooled_condition_row(
                    species,
                    cond,
                    part,
                    cms,
                    n_trials_map,
                    unit="single-session models (summed CM)",
                    f1_source="summed_cm",
                    auroc_source="n_weighted",
                )
            )

        subj = subj_fold0[subj_fold0["species"] == species]
        rows.append(
            _pooled_condition_row(
                species,
                "poyo_subj",
                subj,
                cms,
                n_trials_map,
                unit="single-subject models (summed CM)",
                f1_source="summed_cm",
                auroc_source="n_weighted",
            )
        )

        mrow = multi[multi["species"] == species]
        row: dict[str, Any] = {
            "species": species,
            "condition": "poyo_multi",
            "n_units": 1 if not mrow.empty else 0,
            "n_trials": np.nan,
            "unit": "multi-subject (pooled val/)",
            "f1_source": "val",
            "auroc_source": "val",
            "auroc_estimated": False,
        }
        for m in METRICS:
            if mrow.empty or m not in mrow.columns:
                row[f"{m}_mean"] = np.nan
            else:
                val = mrow[m].iloc[0]
                row[f"{m}_mean"] = float(val) if pd.notna(val) else np.nan
            row[f"{m}_std"] = 0.0
        rows.append(row)

    out = pd.DataFrame(rows)
    out["condition"] = pd.Categorical(
        out["condition"], categories=CONDITION_ORDER, ordered=True
    )
    return out.sort_values(["species", "condition"]).reset_index(drop=True)


def _pooled_condition_row(
    species: str,
    cond: str,
    part: pd.DataFrame,
    cms: dict[str, np.ndarray],
    n_trials_map: dict[str, float],
    *,
    unit: str,
    f1_source: str,
    auroc_source: str,
) -> dict[str, Any]:
    stacked: list[np.ndarray] = []
    aurocs: list[float] = []
    weights: list[float] = []
    for _, run in part.iterrows():
        rid = str(run["run_id"])
        cm = cms.get(rid)
        if cm is None:
            continue
        stacked.append(cm)
        n = float(n_trials_map.get(rid, cm.sum()))
        auc = pd.to_numeric(run.get("auroc"), errors="coerce")
        if pd.notna(auc) and n > 0:
            aurocs.append(float(auc))
            weights.append(n)
    row: dict[str, Any] = {
        "species": species,
        "condition": cond,
        "n_units": int(len(stacked)),
        "n_trials": float(sum(s.sum() for s in stacked)) if stacked else np.nan,
        "unit": unit,
        "f1_source": f1_source,
        "auroc_source": auroc_source,
        "auroc_estimated": auroc_source == "n_weighted",
    }
    if stacked:
        mets = _macro_from_cm(np.sum(stacked, axis=0))
        row["f1_mean"] = mets["f1"]
        row["precision_mean"] = mets["precision"]
        row["recall_mean"] = mets["recall"]
        row["balanced_acc_mean"] = mets["balanced_acc"]
        if mets["n_trials"]:
            row["n_trials"] = mets["n_trials"]
    else:
        for m in ("f1", "precision", "recall", "balanced_acc"):
            row[f"{m}_mean"] = np.nan
    if aurocs:
        row["auroc_mean"] = float(np.average(aurocs, weights=weights))
    else:
        row["auroc_mean"] = np.nan
    for m in METRICS:
        row[f"{m}_std"] = 0.0
    return row


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
    pooled: pd.DataFrame | None = None,
) -> None:
    print("\n=== Comparison: fold-0 session mean ± std (excl. outliers) ===")
    print("  EEGNet / GRU / POYO single-session: fold-0 val/ max per recording")
    print(
        "  POYO single-subject / multi-subject: fold-0 val_session/ history-max per recording"
    )
    show_c = comparison.copy()
    show_c["condition"] = show_c["condition"].map(
        lambda c: CONDITION_NAMES.get(str(c), c)
    )
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
                    f"    {base:6s} − {CONDITION_NAMES.get(ref, ref):20s}  "
                    f"ΔF1={d_f1:+.4f}  ΔAUROC={d_auc:+.4f}"
                )

    if pooled is not None and not pooled.empty:
        print("\n=== Supplementary: species-level pooled metrics ===")
        print(
            "  F1 / P / R / bAcc: sum of val confusion matrices at each "
            "run's max-F1 epoch, then macro over classes with support"
        )
        print(
            "  AUROC: true pooled val/ for POYO multi-subject; otherwise "
            "trial-count-weighted mean of per-run max AUROC (scores not stored)"
        )
        cols = [
            "species",
            "condition",
            "n_units",
            "n_trials",
            "unit",
            "f1_source",
            "auroc_source",
            *METRICS,
        ]
        show_p = pooled.copy()
        show_p["condition"] = show_p["condition"].map(
            lambda c: CONDITION_NAMES.get(str(c), c)
        )
        for m in METRICS:
            show_p[m] = [
                _fmt(a, b)
                for a, b in zip(show_p[f"{m}_mean"], show_p[f"{m}_std"])
            ]
        keep = [c for c in cols if c in show_p.columns]
        print(show_p[keep].to_string(index=False))


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
            [CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=7.5
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


POOLED_PLOT_CONDS = list(CONDITION_ORDER)
POOLED_PLOT_LABELS = CONDITION_LABELS


def plot_pooled_bars(
    pooled: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    stem_suffix: str,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    x = np.arange(len(POOLED_PLOT_CONDS))
    width = 0.7
    mean_col = f"{metric}_mean"
    hatch_est = metric == "auroc"
    for ax, species in zip(axes, SPECIES_ORDER):
        sub = pooled[pooled["species"] == species].set_index("condition")
        means = [
            float(sub.loc[c, mean_col]) if c in sub.index else np.nan
            for c in POOLED_PLOT_CONDS
        ]
        colors = [CONDITION_COLORS[c] for c in POOLED_PLOT_CONDS]
        hatches = []
        for c in POOLED_PLOT_CONDS:
            estimated = False
            if (
                hatch_est
                and c in sub.index
                and "auroc_estimated" in sub.columns
            ):
                estimated = bool(sub.loc[c, "auroc_estimated"])
            hatches.append("//" if estimated else None)
        bars = ax.bar(
            x,
            means,
            width,
            color=colors,
            edgecolor="white",
        )
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor("0.25")
        for bar, mean in zip(bars, means):
            if np.isnan(mean):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [POOLED_PLOT_LABELS[c] for c in POOLED_PLOT_CONDS], fontsize=7.5
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
    """Add per-session POYO single-subject / multi-subject bars for sessions already plotted."""
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
        "f1": "Fold-0 max val F1",
        "auroc": "Fold-0 max val AUROC",
    }
    titles = {
        "f1": "Per-session F1, fold 0 (excl. outliers)",
        "auroc": "Per-session AUROC, fold 0 (excl. outliers)",
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
            fontsize=6.5,
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
        f"  fold={FOLD_KEEP} only\n"
        f"  POYO opt singlesess sweeps: {sorted(POYO_OPT_SWEEPS)}\n"
        f"  POYO singlesub sweeps: {POYO_SINGLESUB_SWEEPS}\n"
        f"  Best multisubj POYO: {MULTISUBJ_BEST_POYO}"
    )

    csv_runs = CSV_DIR / f"{STEM}_runs.csv"
    csv_multi = CSV_DIR / f"{STEM}_multisubj_best.csv"
    csv_subj = CSV_DIR / f"{STEM}_poyo_singlesubj_runs.csv"
    csv_multi_val = CSV_DIR / f"{STEM}_poyo_multisubj_val_sessions.csv"
    csv_subj_val = CSV_DIR / f"{STEM}_poyo_singlesubj_val_sessions.csv"
    csv_cm = CSV_DIR / f"{STEM}_confusion_maxf1.csv"
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

    primary = keep_fold(primary_runs(raw))
    print_inventory(raw, primary)
    session_df = fold_means_by_session(primary)
    flagged = flag_outliers(session_df)
    summary = model_summary(session_df)
    matched = matched_sessions(session_df)
    best = best_session_per_model(session_df)
    outliers = flagged[flagged["is_outlier"]]

    subj_runs_f0 = keep_fold(subj_runs)
    subj_val_f0 = keep_fold(subj_val)
    subj_units = fold_means_by_subject(subj_runs_f0)
    multi_sess = fold_means_from_session_rows(multi_val)
    subj_sess = fold_means_from_session_rows(subj_val_f0)
    comparison = comparison_table(summary, subj_sess, multi_sess, session_df)

    cm_ids = (
        primary["run_id"].astype(str).tolist()
        + subj_runs_f0["run_id"].astype(str).tolist()
    )
    if use_cached and csv_cm.exists():
        print(f"Loading cached confusion matrices from {csv_cm}")
        cm_df = pd.read_csv(csv_cm)
        missing_ids = [
            i for i in cm_ids if i not in set(cm_df["run_id"].astype(str))
        ]
        if missing_ids:
            print(f"  fetching {len(missing_ids)} missing CMs...")
            extra = fetch_confusion_at_max_f1(missing_ids)
            cm_df = pd.concat([cm_df, extra], ignore_index=True)
            cm_df.to_csv(csv_cm, index=False)
    else:
        cm_df = fetch_confusion_at_max_f1(cm_ids)
        cm_df.to_csv(csv_cm, index=False)
    n_sum = (
        int(cm_df["from_summary"].fillna(False).astype(bool).sum())
        if (not cm_df.empty and "from_summary" in cm_df.columns)
        else 0
    )
    n_miss = (
        int(cm_df["cm_json"].isna().sum()) if not cm_df.empty else len(cm_ids)
    )
    print(
        f"  confusion matrices: {len(cm_df) - n_miss} ok, {n_miss} missing, {n_sum} last-epoch fallback"
    )
    pooled = pooled_reference_table(
        primary, session_df, subj_runs_f0, multi, cm_df
    )

    print_tables(
        summary, matched, best, outliers, comparison, subj_units, pooled
    )
    print(
        "\n=== val_session coverage (POYO multi-subject / single-subject) ==="
    )
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
        "Fold-0 max val F1 (mean ± std across sessions)",
        "Fold 0 session mean ± std (excl. outliers)",
        "f1_by_model",
    )
    plot_comparison_bars(
        comparison,
        "auroc",
        "Fold-0 max val AUROC (mean ± std across sessions)",
        "Fold 0 session mean ± std (excl. outliers)",
        "auroc_by_model",
    )
    plot_pooled_bars(
        pooled,
        "f1",
        "Pooled macro-F1 (summed val confusion matrices)",
        "Supplementary: species-level pooled F1 (fold 0)",
        "supp_pooled_f1",
    )
    plot_pooled_bars(
        pooled,
        "auroc",
        "Pooled AUROC (true val/ or n-weighted session AUROC)",
        "Supplementary: species-level pooled AUROC (fold 0; hatched = estimate)",
        "supp_pooled_auroc",
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
    csv_pooled = CSV_DIR / f"{STEM}_pooled.csv"
    flagged.to_csv(csv_sessions, index=False)
    summary.to_csv(csv_summary, index=False)
    matched.to_csv(csv_matched, index=False)
    comparison.to_csv(csv_comparison, index=False)
    subj_units.to_csv(csv_subj_units, index=False)
    multi_sess.to_csv(csv_multi_sess, index=False)
    subj_sess.to_csv(csv_subj_sess, index=False)
    pooled.to_csv(csv_pooled, index=False)
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
    print(f"Saved: {csv_pooled}")
    print(f"Saved: {csv_cm}")


if __name__ == "__main__":
    main()
