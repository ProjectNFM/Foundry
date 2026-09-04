"""Multi-subject EEGNet / GRU-CNN HP search on NeuroSoft 8-band (minipigs).

Fetches finished runs from four WandB grid sweeps in
``HYPER-PARAM-SEARCH`` / ``suarez_auditory_decoding``, extracts architecture
hyperparameters, pooled ``val/`` maxima, per-session ``val_session/``
metrics, and parameter counts (session projector vs backbone).

Primary scoreboard for architecture comparison: unweighted mean±std of
per-session history-max F1 / AUROC on each architecture's best-HP run, plus
the true pooled ``val/`` max (one multi-subject model). Parameter count is
reported as a confound: GRU backbones are 1–2 orders of magnitude larger
than EEGNet even at matched ``num_sources``.

Usage:
    uv run python analysis/20260831-LS-eegnet-gru-multisubj-hp.py
    uv run python analysis/20260831-LS-eegnet-gru-multisubj-hp.py --cached
"""

from __future__ import annotations

import argparse
import itertools
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from analysis._wandb_utils import (
    csv_dir,
    figures_dir,
    unwrap_summary_value,
)

ENTITY = "neurosoft-bioelectronics"
PROJECT = "suarez_auditory_decoding"
GROUP = "HYPER-PARAM-SEARCH"

SWEEP_IDS: dict[str, str] = {
    "eegnet": "d52nj3w1",
    "gru_temporal": "ewm1u7vj",
    "gru_spatial": "qbyn137w",
    "gru_spatiotemporal": "lj06lx64",
}
SWEEP_EXPECTED: dict[str, int] = {
    "eegnet": 36,
    "gru_temporal": 54,
    "gru_spatial": 18,
    "gru_spatiotemporal": 108,
}

TASK = "neurosoft_acoustic_stim_8band"
F1_KEY = f"val/{TASK}_f1"
AUROC_KEY = f"val/{TASK}_auroc"
PREC_KEY = f"val/{TASK}_precision"
RECALL_KEY = f"val/{TASK}_recall"
BAL_KEY = f"val/{TASK}_balanced_acc"
CM_KEY = f"val/{TASK}_confusion_counts"
VAL_SESSION_PREFIX = "val_session/"
N_CLASSES = 8
NUM_SAMPLES = 1000
GRU_HIDDEN = 128
GRU_LAYERS = 2
GRU_BIDIRECTIONAL = True
GRU_PROJ_DIM = 128
N_WORKERS = 8
HISTORY_BATCH = 40

ARCH_ORDER = [
    "eegnet",
    "gru_spatial",
    "gru_temporal",
    "gru_spatiotemporal",
]
ARCH_LABELS = {
    "eegnet": "EEGNet",
    "gru_spatial": "Spatial CNN+GRU",
    "gru_temporal": "Temporal CNN+GRU",
    "gru_spatiotemporal": "Spatiotemporal CNN+GRU",
}
ARCH_COLORS = {
    "eegnet": "#E8963E",
    "gru_spatial": "#C44E52",
    "gru_temporal": "#4C72B0",
    "gru_spatiotemporal": "#55A868",
}

STEM = Path(__file__).stem
FIGURES_DIR = figures_dir(__file__)
CSV_DIR = csv_dir(__file__)


def _run_path(run_id: str) -> str:
    return f"{ENTITY}/{PROJECT}/{run_id}"


def _sweep_path(sweep_id: str) -> str:
    return f"{ENTITY}/{PROJECT}/{sweep_id}"


def _as_float(val: Any, unwrap: str = "max") -> float | None:
    out = unwrap_summary_value(val, unwrap)
    if isinstance(out, (int, float)) and np.isfinite(out):
        return float(out)
    return None


def _summary_tracked(run: Any, key: str, unwrap: str = "max") -> float | None:
    val = _as_float(run.summary.get(key), unwrap)
    if val is not None:
        return val
    suffix = "max" if unwrap == "max" else "min"
    return _as_float(run.summary.get(f"{key}.{suffix}"), unwrap)


def _nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _arch_from_run(run: Any, config: dict[str, Any]) -> str:
    tags = {t.lower() for t in (run.tags or [])}
    model = config.get("model") or {}
    conv = _nested(
        config,
        "model.conv",
        default=model.get("conv") if isinstance(model, dict) else None,
    )
    hydra = str(
        config.get("hydra.runtime.choices.model")
        or config.get("model/_name_")
        or ""
    ).lower()
    name = (run.name or "").lower()
    if "eegnet" in tags or "eegnet" in hydra or name.startswith("eegnet"):
        return "eegnet"
    conv_s = str(conv).lower() if conv is not None else ""
    if conv_s in {"spatial", "temporal", "spatiotemporal"}:
        return f"gru_{conv_s}"
    if "spatiotemporal" in name:
        return "gru_spatiotemporal"
    if "temporal" in name:
        return "gru_temporal"
    if "spatial" in name:
        return "gru_spatial"
    return "unknown"


_EEGNET_NAME_RE = re.compile(
    r"F1-(?P<F1>\d+)_k(?P<kernel>\d+)_D(?P<D>\d+)_ns(?P<ns>\d+)"
)
_GRU_NAME_RE = re.compile(
    r"gru_neurosoft_8band_(?P<conv>spatial|temporal|spatiotemporal)"
    r"_proj(?P<proj>True|False)_F(?P<F>\d+)_k(?P<k>\d+)_D(?P<D>\d+)_ns(?P<ns>\d+)"
)


def _hps_from_name(name: str | None, arch: str) -> dict[str, Any]:
    """Hydra run names encode the swept knobs; sweep.runs often omit config."""
    if not name:
        return {}
    if arch == "eegnet":
        m = _EEGNET_NAME_RE.search(name)
        if not m:
            return {}
        return {
            "F1": int(m.group("F1")),
            "kernel_length": int(m.group("kernel")),
            "D": int(m.group("D")),
            "num_sources": int(m.group("ns")),
            "F2": 16,
        }
    m = _GRU_NAME_RE.search(name)
    if not m:
        return {}
    return {
        "conv": m.group("conv"),
        "use_input_proj": m.group("proj") == "True",
        "conv_filters": int(m.group("F")),
        "conv_kernel": int(m.group("k")),
        "conv_depth_multiplier": int(m.group("D")),
        "num_sources": int(m.group("ns")),
        "hidden_size": GRU_HIDDEN,
        "num_layers": GRU_LAYERS,
        "bidirectional": GRU_BIDIRECTIONAL,
    }


def _extract_hps(config: dict[str, Any], arch: str) -> dict[str, Any]:
    hp = config.get("hyperparameters") or {}
    model = config.get("model") or {}
    data = config.get("data") or {}
    num_sources = _nested(
        config, "hyperparameters.num_sources", default=hp.get("num_sources")
    )
    split = _nested(config, "data.split_type", default=data.get("split_type"))
    lr = _nested(
        config, "hyperparameters.learning_rate", default=hp.get("learning_rate")
    )
    wd = _nested(
        config, "hyperparameters.weight_decay", default=hp.get("weight_decay")
    )
    bs = _nested(
        config, "hyperparameters.batch_size", default=hp.get("batch_size")
    )
    row: dict[str, Any] = {
        "num_sources": num_sources,
        "split_type": split,
        "learning_rate": lr,
        "weight_decay": wd,
        "batch_size": bs,
        "fold": hp.get("fold_number") if isinstance(hp, dict) else None,
        "F1": None,
        "D": None,
        "kernel_length": None,
        "F2": None,
        "conv": None,
        "use_input_proj": None,
        "conv_filters": None,
        "conv_kernel": None,
        "conv_depth_multiplier": None,
        "hidden_size": None,
        "num_layers": None,
        "bidirectional": None,
    }
    if arch == "eegnet":
        row["F1"] = _nested(config, "model.F1", default=model.get("F1"))
        row["D"] = _nested(config, "model.D", default=model.get("D"))
        row["kernel_length"] = _nested(
            config, "model.kernel_length", default=model.get("kernel_length")
        )
        row["F2"] = model.get("F2", 16) if isinstance(model, dict) else 16
    else:
        row["conv"] = _nested(config, "model.conv", default=model.get("conv"))
        row["use_input_proj"] = _nested(
            config, "model.use_input_proj", default=model.get("use_input_proj")
        )
        row["conv_filters"] = _nested(
            config, "model.conv_filters", default=model.get("conv_filters")
        )
        row["conv_kernel"] = _nested(
            config, "model.conv_kernel", default=model.get("conv_kernel")
        )
        row["conv_depth_multiplier"] = _nested(
            config,
            "model.conv_depth_multiplier",
            default=model.get("conv_depth_multiplier"),
        )
        row["hidden_size"] = (
            model.get("hidden_size", GRU_HIDDEN)
            if isinstance(model, dict)
            else GRU_HIDDEN
        )
        row["num_layers"] = (
            model.get("num_layers", GRU_LAYERS)
            if isinstance(model, dict)
            else GRU_LAYERS
        )
        row["bidirectional"] = (
            model.get("bidirectional", GRU_BIDIRECTIONAL)
            if isinstance(model, dict)
            else GRU_BIDIRECTIONAL
        )
    return row


def _session_configs(config: dict[str, Any]) -> dict[str, int] | None:
    hp = config.get("hyperparameters") or {}
    if isinstance(hp, dict) and isinstance(hp.get("session_configs"), dict):
        return {str(k): int(v) for k, v in hp["session_configs"].items()}
    cs = (config.get("model") or {}).get("channel_strategy") or {}
    proj = cs.get("projector") if isinstance(cs, dict) else None
    if isinstance(proj, dict) and isinstance(proj.get("session_configs"), dict):
        return {str(k): int(v) for k, v in proj["session_configs"].items()}
    return None


def _projector_params(
    session_configs: dict[str, int], num_sources: int, common_layer: bool = True
) -> int:
    n = 0
    for n_ch in session_configs.values():
        n += int(n_ch) * num_sources + num_sources
    if common_layer:
        n += num_sources * num_sources + num_sources
    return n


def _gru_recurrent_params(
    input_size: int,
    hidden: int = GRU_HIDDEN,
    num_layers: int = GRU_LAYERS,
    bidirectional: bool = GRU_BIDIRECTIONAL,
) -> int:
    dirs = 2 if bidirectional else 1
    total = 0
    for layer in range(num_layers):
        inp = input_size if layer == 0 else hidden * dirs
        per_dir = 3 * (hidden * inp + hidden * hidden + hidden + hidden)
        total += per_dir * dirs
    return total


def _eegnet_time_after_pools(num_samples: int = NUM_SAMPLES) -> int:
    t1 = (num_samples - 4) // 4 + 1
    return (t1 - 8) // 8 + 1


def _eegnet_backbone_params(
    *,
    F1: int,
    D: int,
    F2: int,
    kernel_length: int,
    num_channels: int,
    num_samples: int = NUM_SAMPLES,
    n_classes: int = N_CLASSES,
) -> int:
    n = F1 * kernel_length
    n += 2 * F1
    n += F1 * D * num_channels
    n += 2 * F1 * D
    n += F1 * D * 16
    n += F2 * F1 * D
    n += 2 * F2
    out_dim = F2 * _eegnet_time_after_pools(num_samples)
    n += out_dim * n_classes + n_classes
    return n


def _gru_backbone_params(
    *,
    conv: str,
    use_input_proj: bool,
    conv_filters: int,
    conv_kernel: int,
    conv_depth_multiplier: int,
    num_channels: int,
    hidden: int = GRU_HIDDEN,
    num_layers: int = GRU_LAYERS,
    bidirectional: bool = GRU_BIDIRECTIONAL,
    proj_dim: int = GRU_PROJ_DIM,
    n_classes: int = N_CLASSES,
) -> int:
    F = conv_filters
    D = conv_depth_multiplier
    K = num_channels
    L = conv_kernel
    if conv == "spatial":
        n = F * K + 2 * F
        feat = F
    elif conv == "temporal":
        n = F * L + 2 * F
        feat = F * K
    elif conv == "spatiotemporal":
        n = F * L + 2 * F + F * D * K + 2 * F * D
        feat = F * D
    else:
        raise ValueError(f"unknown conv {conv}")
    if use_input_proj:
        n += feat * proj_dim + proj_dim
        gru_in = proj_dim
    else:
        gru_in = feat
    n += _gru_recurrent_params(gru_in, hidden, num_layers, bidirectional)
    out_dim = hidden * (2 if bidirectional else 1)
    n += out_dim * n_classes + n_classes
    return n


def _as_bool(val: Any) -> bool:
    if isinstance(val, str):
        return val.strip().lower() in {"true", "1", "yes"}
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    return bool(val)


def _count_params(
    row: dict[str, Any], session_configs: dict[str, int] | None
) -> dict[str, int | None]:
    def _int(key: str, default: int | None = None) -> int | None:
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return int(val)

    ns = _int("num_sources")
    projector = None
    if session_configs is not None and ns is not None:
        projector = _projector_params(session_configs, ns, common_layer=True)
    backbone = None
    arch = str(row.get("arch") or "")
    if arch == "eegnet" and ns is not None:
        f1, d, k = _int("F1"), _int("D"), _int("kernel_length")
        if None not in (f1, d, k):
            backbone = _eegnet_backbone_params(
                F1=f1,
                D=d,
                F2=_int("F2", 16) or 16,
                kernel_length=k,
                num_channels=ns,
            )
    elif arch.startswith("gru_") and ns is not None:
        conv = row.get("conv")
        f = _int("conv_filters")
        if conv and f is not None:
            bidir = row.get("bidirectional")
            if bidir is None or (isinstance(bidir, float) and pd.isna(bidir)):
                bidir_v = GRU_BIDIRECTIONAL
            else:
                bidir_v = _as_bool(bidir)
            backbone = _gru_backbone_params(
                conv=str(conv),
                use_input_proj=_as_bool(row.get("use_input_proj")),
                conv_filters=f,
                conv_kernel=_int("conv_kernel", 64) or 64,
                conv_depth_multiplier=_int("conv_depth_multiplier", 2) or 2,
                num_channels=ns,
                hidden=_int("hidden_size", GRU_HIDDEN) or GRU_HIDDEN,
                num_layers=_int("num_layers", GRU_LAYERS) or GRU_LAYERS,
                bidirectional=bidir_v,
            )
    total = None
    if backbone is not None and projector is not None:
        total = backbone + projector
    return {
        "n_params_backbone": backbone,
        "n_params_projector": projector,
        "n_params_total": total,
    }


def _parse_val_session_summary(run: Any) -> dict[str, dict[str, float | None]]:
    task_prefix = f"{TASK}_"
    parsed: dict[str, dict[str, float | None]] = {}
    for key in run.summary.keys():
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
        if metric not in {"f1", "auroc"}:
            continue
        parsed.setdefault(sid, {})[metric] = _as_float(
            run.summary.get(key), "max"
        )
    return parsed


def _session_means_from_summary(
    parsed: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    f1s = [v["f1"] for v in parsed.values() if v.get("f1") is not None]
    aucs = [v["auroc"] for v in parsed.values() if v.get("auroc") is not None]
    return {
        "n_sessions_logged": int(len(parsed)),
        "session_mean_f1_last": float(np.mean(f1s)) if f1s else None,
        "session_std_f1_last": float(np.std(f1s, ddof=0)) if f1s else None,
        "session_mean_auroc_last": float(np.mean(aucs)) if aucs else None,
        "session_std_auroc_last": float(np.std(aucs, ddof=0)) if aucs else None,
    }


def _cm_n_trials(run: Any) -> float | None:
    cm = run.summary.get(CM_KEY)
    if cm is None:
        return None
    try:
        arr = np.asarray(cm, dtype=float)
        return float(arr.sum())
    except (TypeError, ValueError):
        return None


def _row_from_run(run: Any, sweep_id: str) -> dict[str, Any]:
    config = dict(run.config or {})
    arch = _arch_from_run(run, config)
    hps = _extract_hps(config, arch)
    named = _hps_from_name(run.name, arch)
    for key, val in named.items():
        if hps.get(key) is None:
            hps[key] = val
    session_configs = _session_configs(config) or _SESSION_CONFIGS
    parsed = _parse_val_session_summary(run)
    row: dict[str, Any] = {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "sweep_id": sweep_id,
        "arch": arch,
        "created_at": str(getattr(run, "created_at", "")),
        **hps,
        "pooled_f1": _summary_tracked(run, F1_KEY, "max"),
        "pooled_auroc": _summary_tracked(run, AUROC_KEY, "max"),
        "pooled_precision": _summary_tracked(run, PREC_KEY, "max"),
        "pooled_recall": _summary_tracked(run, RECALL_KEY, "max"),
        "pooled_balanced_acc": _summary_tracked(run, BAL_KEY, "max"),
        "n_val_trials": _cm_n_trials(run),
        **_session_means_from_summary(parsed),
        **_count_params({**hps, "arch": arch}, session_configs),
    }
    return row


_SESSION_CONFIGS: dict[str, int] | None = None


def load_session_configs(api: wandb.Api | None = None) -> dict[str, int]:
    global _SESSION_CONFIGS
    if _SESSION_CONFIGS is not None:
        return _SESSION_CONFIGS
    if api is None:
        api = wandb.Api(timeout=120)
    run = api.run(_run_path("ks62xe0k"))
    sc = _session_configs(dict(run.config or {}))
    if not sc:
        raise RuntimeError(
            "Could not load session_configs from WandB run ks62xe0k"
        )
    _SESSION_CONFIGS = sc
    print(
        f"Loaded session_configs: {len(sc)} sessions, {sum(sc.values())} channels"
    )
    return sc


def _fill_hps_and_params(df: pd.DataFrame) -> pd.DataFrame:
    """Parse run names and recompute parameter counts (sweep.runs omit config)."""
    session_configs = load_session_configs()
    rows = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        named = _hps_from_name(
            str(rec.get("run_name") or ""), str(rec.get("arch") or "")
        )
        for key, val in named.items():
            if rec.get(key) is None or (
                isinstance(rec.get(key), float) and pd.isna(rec.get(key))
            ):
                rec[key] = val
        rec.update(_count_params(rec, session_configs))
        rows.append(rec)
    out = pd.DataFrame(rows)
    for col in [
        "num_sources",
        "F1",
        "D",
        "kernel_length",
        "F2",
        "conv_filters",
        "conv_kernel",
        "conv_depth_multiplier",
        "n_params_backbone",
        "n_params_projector",
        "n_params_total",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def fetch_runs(api: wandb.Api | None = None) -> pd.DataFrame:
    if api is None:
        api = wandb.Api(timeout=120)
    rows: list[dict[str, Any]] = []
    for arch, sweep_id in SWEEP_IDS.items():
        sweep = api.sweep(_sweep_path(sweep_id))
        print(
            f"Fetching sweep {arch} ({sweep_id}) state={getattr(sweep, 'state', None)}..."
        )
        n = 0
        for run in sweep.runs:
            rows.append(_row_from_run(run, sweep_id))
            n += 1
            if n % 20 == 0:
                print(f"  {arch}: {n} runs")
        print(f"  {arch}: {n} runs total")
    df = pd.DataFrame(rows)
    for col in [
        "num_sources",
        "F1",
        "D",
        "kernel_length",
        "F2",
        "conv_filters",
        "conv_kernel",
        "conv_depth_multiplier",
        "pooled_f1",
        "pooled_auroc",
        "n_params_backbone",
        "n_params_projector",
        "n_params_total",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(
        ["arch", "pooled_f1"], ascending=[True, False]
    ).reset_index(drop=True)


def _history_maxes(run: Any, keys: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for i in range(0, len(keys), HISTORY_BATCH):
        batch = keys[i : i + HISTORY_BATCH]
        history = run.history(keys=batch, samples=10_000, pandas=True)
        for key in batch:
            if key not in history.columns or history[key].dropna().empty:
                out[key] = _as_float(run.summary.get(key), "max")
            else:
                out[key] = float(history[key].max())
    return out


def _parse_val_session_keys(summary: Any) -> dict[str, dict[str, str]]:
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
        if metric not in {"f1", "auroc"}:
            continue
        parsed.setdefault(sid, {})[metric] = key_s
    return parsed


def val_session_rows_from_run(run: Any, arch: str) -> list[dict[str, Any]]:
    parsed = _parse_val_session_keys(run.summary)
    wandb_keys = [k for mmap in parsed.values() for k in mmap.values()]
    maxes = _history_maxes(run, wandb_keys)
    rows: list[dict[str, Any]] = []
    for sid, mmap in parsed.items():
        row: dict[str, Any] = {
            "run_id": run.id,
            "arch": arch,
            "session": sid,
            "subject": sid.split("_")[0],
        }
        for metric in ("f1", "auroc"):
            key = mmap.get(metric)
            row[metric] = maxes.get(key) if key else None
        rows.append(row)
    return rows


def fetch_winner_sessions(
    winners: pd.DataFrame, api: wandb.Api | None = None
) -> pd.DataFrame:
    del api
    payloads = winners[["run_id", "arch"]].to_dict("records")
    print(
        f"Fetching val_session history-max for {len(payloads)} winner runs..."
    )
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=min(N_WORKERS, max(1, len(payloads)))
    ) as pool:
        futs = []
        for p in payloads:
            futs.append(pool.submit(_fetch_one_winner, p))
        for i, fut in enumerate(as_completed(futs), start=1):
            part = fut.result()
            rows.extend(part)
            print(f"  {i}/{len(payloads)} ({len(part)} sessions)")
    return pd.DataFrame(rows)


def _fetch_one_winner(payload: dict[str, Any]) -> list[dict[str, Any]]:
    api = wandb.Api(timeout=120)
    run = api.run(_run_path(str(payload["run_id"])))
    return val_session_rows_from_run(run, str(payload["arch"]))


def finished(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["state"] == "finished"].copy()


def best_per_arch(df: pd.DataFrame) -> pd.DataFrame:
    work = finished(df)
    work = work.dropna(subset=["pooled_f1"])
    idx = work.groupby("arch")["pooled_f1"].idxmax()
    return (
        work.loc[idx]
        .set_index("arch")
        .reindex(ARCH_ORDER)
        .reset_index()
        .dropna(subset=["run_id"])
        .reset_index(drop=True)
    )


def top_k(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    work = finished(df).dropna(subset=["pooled_f1"])
    return (
        work.sort_values(["arch", "pooled_f1"], ascending=[True, False])
        .groupby("arch", group_keys=False)
        .head(k)
        .reset_index(drop=True)
    )


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch in ARCH_ORDER:
        sub = df[df["arch"] == arch]
        fin = sub[sub["state"] == "finished"]
        rows.append(
            {
                "arch": arch,
                "sweep_id": SWEEP_IDS[arch],
                "expected": SWEEP_EXPECTED[arch],
                "n_runs": int(len(sub)),
                "n_finished": int(len(fin)),
                "n_running": int((sub["state"] == "running").sum()),
                "n_crashed": int(
                    sub["state"].isin(["crashed", "failed", "killed"]).sum()
                ),
                "coverage": len(fin) / SWEEP_EXPECTED[arch]
                if SWEEP_EXPECTED[arch]
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def session_summary(sess: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch in ARCH_ORDER:
        g = sess[sess["arch"] == arch]
        if g.empty:
            continue
        f1 = pd.to_numeric(g["f1"], errors="coerce").dropna()
        auc = pd.to_numeric(g["auroc"], errors="coerce").dropna()
        rows.append(
            {
                "arch": arch,
                "n_sessions": int(g["session"].nunique()),
                "session_mean_f1": float(f1.mean()) if len(f1) else np.nan,
                "session_std_f1": float(f1.std(ddof=0)) if len(f1) else np.nan,
                "session_mean_auroc": float(auc.mean()) if len(auc) else np.nan,
                "session_std_auroc": float(auc.std(ddof=0))
                if len(auc)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _fmt(mean: float, std: float | None = None) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return ""
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return f"{mean:.4f}"
    return f"{mean:.4f}±{std:.4f}"


def _hp_str(row: pd.Series) -> str:
    def _i(key: str) -> str:
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "?"
        return str(int(val))

    if row["arch"] == "eegnet":
        return f"F1={_i('F1')}, k={_i('kernel_length')}, D={_i('D')}, ns={_i('num_sources')}"
    proj = "proj" if _as_bool(row.get("use_input_proj")) else "no-proj"
    parts = [proj, f"F={_i('conv_filters')}", f"ns={_i('num_sources')}"]
    if row["arch"] != "gru_spatial":
        parts.insert(1, f"k={_i('conv_kernel')}")
    if row["arch"] == "gru_spatiotemporal":
        parts.insert(-1, f"D={_i('conv_depth_multiplier')}")
    return ", ".join(parts)


def print_tables(
    df: pd.DataFrame,
    cov: pd.DataFrame,
    best: pd.DataFrame,
    tops: pd.DataFrame,
    sess_sum: pd.DataFrame,
    sess: pd.DataFrame,
) -> None:
    print("\n=== Sweep coverage ===")
    show = cov.copy()
    show["arch"] = show["arch"].map(lambda a: ARCH_LABELS.get(a, a))
    show["coverage"] = show["coverage"].map(lambda x: f"{100 * x:.0f}%")
    print(show.to_string(index=False))

    print("\n=== Best HP per architecture (finished; max pooled val F1) ===")
    cols = [
        "arch",
        "hps",
        "n_params_total",
        "n_params_backbone",
        "n_params_projector",
        "pooled_f1",
        "pooled_auroc",
        "session_mean_f1",
        "session_mean_auroc",
        "run_id",
    ]
    best_show = best.copy()
    best_show["hps"] = best_show.apply(_hp_str, axis=1)
    if not sess_sum.empty:
        best_show = best_show.merge(sess_sum, on="arch", how="left")
    best_show["arch"] = best_show["arch"].map(lambda a: ARCH_LABELS.get(a, a))
    for c in ["n_params_total", "n_params_backbone", "n_params_projector"]:
        best_show[c] = best_show[c].map(
            lambda x: f"{int(x):,}" if pd.notna(x) else ""
        )
    for c in [
        "pooled_f1",
        "pooled_auroc",
        "session_mean_f1",
        "session_mean_auroc",
    ]:
        if c in best_show.columns:
            std_col = {
                "session_mean_f1": "session_std_f1",
                "session_mean_auroc": "session_std_auroc",
            }.get(c)
            if std_col and std_col in best_show.columns:
                best_show[c] = [
                    _fmt(a, b) for a, b in zip(best_show[c], best_show[std_col])
                ]
            else:
                best_show[c] = best_show[c].map(
                    lambda x: f"{x:.4f}" if pd.notna(x) else ""
                )
    print(
        best_show[[c for c in cols if c in best_show.columns]].to_string(
            index=False
        )
    )

    print("\n=== Winner comparison (best HP; session-mean vs pooled) ===")
    if not sess_sum.empty:
        merged = best.merge(sess_sum, on="arch", how="left")
    else:
        merged = best
    show = merged.copy()
    show["label"] = show["arch"].map(lambda a: ARCH_LABELS.get(a, a))
    show["params"] = show["n_params_total"].map(
        lambda x: f"{int(x):,}" if pd.notna(x) else ""
    )
    show["f1_per_mparam"] = show["pooled_f1"] / (show["n_params_total"] / 1e6)
    print(
        show[
            [
                "label",
                "params",
                "pooled_f1",
                "pooled_auroc",
                "session_mean_f1",
                "session_std_f1",
                "session_mean_auroc",
                "session_std_auroc",
                "f1_per_mparam",
                "run_id",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n=== Top-5 per architecture (pooled F1) ===")
    t = tops.copy()
    t["hps"] = t.apply(_hp_str, axis=1)
    t["arch"] = t["arch"].map(lambda a: ARCH_LABELS.get(a, a))
    t["n_params_total"] = t["n_params_total"].map(
        lambda x: f"{int(x):,}" if pd.notna(x) else ""
    )
    print(
        t[
            [
                "arch",
                "hps",
                "n_params_total",
                "pooled_f1",
                "pooled_auroc",
                "session_mean_f1_last",
                "run_id",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n=== Parameter-count summary (finished runs) ===")
    fin = finished(df)
    for arch in ARCH_ORDER:
        g = fin[fin["arch"] == arch]
        if g.empty:
            continue
        print(
            f"  {ARCH_LABELS[arch]:28s}  "
            f"n={len(g):3d}  "
            f"params {g['n_params_total'].min():,.0f}–{g['n_params_total'].max():,.0f}  "
            f"backbone {g['n_params_backbone'].min():,.0f}–{g['n_params_backbone'].max():,.0f}  "
            f"best F1={g['pooled_f1'].max():.4f}"
        )

    if not sess.empty:
        print(
            "\n=== Supplementary: per-session F1 / AUROC (winners, history-max) ==="
        )
        sess = sess.copy()
        sess["short"] = sess["session"].map(_short_session)
        wide_f1 = sess.pivot_table(
            index="short", columns="arch", values="f1", aggfunc="first"
        )
        wide_auc = sess.pivot_table(
            index="short", columns="arch", values="auroc", aggfunc="first"
        )
        wide_f1 = wide_f1.reindex(
            columns=[a for a in ARCH_ORDER if a in wide_f1.columns]
        )
        wide_auc = wide_auc.reindex(
            columns=[a for a in ARCH_ORDER if a in wide_auc.columns]
        )
        print("\nF1:")
        print(
            wide_f1.rename(columns=ARCH_LABELS).to_string(
                float_format=lambda x: f"{x:.3f}"
            )
        )
        print("\nAUROC:")
        print(
            wide_auc.rename(columns=ARCH_LABELS).to_string(
                float_format=lambda x: f"{x:.3f}"
            )
        )


def _short_session(session: str) -> str:
    parts = str(session).split("_")
    sub = parts[0].replace("sub-", "") if parts else session
    ses = ""
    acq = ""
    for p in parts:
        if p.startswith("ses-"):
            ses = p.replace("ses-", "s")
        if p.startswith("acq-"):
            acq = p.replace("acq-", "")
    return "-".join(p for p in (sub, ses, acq) if p)


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_winner_bars(best: pd.DataFrame, sess_sum: pd.DataFrame) -> Path:
    merged = best.merge(sess_sum, on="arch", how="left")
    merged = merged.set_index("arch").reindex(ARCH_ORDER).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=False)
    x = np.arange(len(ARCH_ORDER))
    width = 0.36
    colors = [ARCH_COLORS[a] for a in ARCH_ORDER]
    # F1
    ax = axes[0]
    sess_means = merged["session_mean_f1"].to_numpy(dtype=float)
    sess_stds = merged["session_std_f1"].to_numpy(dtype=float)
    pooled = merged["pooled_f1"].to_numpy(dtype=float)
    ax.bar(
        x - width / 2,
        sess_means,
        width,
        yerr=sess_stds,
        capsize=3,
        label="Session mean ± std",
        color=colors,
        edgecolor="white",
        error_kw=dict(lw=0.8),
    )
    ax.bar(
        x + width / 2,
        pooled,
        width,
        label="Pooled val/",
        color=colors,
        edgecolor="0.25",
        hatch="//",
        alpha=0.85,
    )
    ax.set_ylabel("F1")
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCH_ORDER], fontsize=8)
    ax.set_title("F1 (best HP)")
    ax.legend(frameon=False, fontsize=8)
    _style_ax(ax)
    # AUROC
    ax = axes[1]
    sess_means = merged["session_mean_auroc"].to_numpy(dtype=float)
    sess_stds = merged["session_std_auroc"].to_numpy(dtype=float)
    pooled = merged["pooled_auroc"].to_numpy(dtype=float)
    ax.bar(
        x - width / 2,
        sess_means,
        width,
        yerr=sess_stds,
        capsize=3,
        label="Session mean ± std",
        color=colors,
        edgecolor="white",
        error_kw=dict(lw=0.8),
    )
    ax.bar(
        x + width / 2,
        pooled,
        width,
        label="Pooled val/",
        color=colors,
        edgecolor="0.25",
        hatch="//",
        alpha=0.85,
    )
    ax.set_ylabel("AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCH_ORDER], fontsize=8)
    ax.set_title("AUROC (best HP)")
    ax.legend(frameon=False, fontsize=8)
    _style_ax(ax)
    fig.suptitle(
        "Best-HP architecture comparison (minipigs, multi-subject, causal)",
        y=1.02,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_best_f1_auroc.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_f1_vs_params(df: pd.DataFrame, best: pd.DataFrame, which: str) -> Path:
    col = "n_params_total" if which == "total" else "n_params_backbone"
    title = (
        "Pooled F1 vs total parameters (incl. session projector)"
        if which == "total"
        else "Pooled F1 vs backbone parameters (excl. session projector)"
    )
    fin = finished(df).dropna(subset=["pooled_f1", col])
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    winner_ids = set(best["run_id"])
    for arch in ARCH_ORDER:
        g = fin[fin["arch"] == arch]
        if g.empty:
            continue
        other = g[~g["run_id"].isin(winner_ids)]
        win = g[g["run_id"].isin(winner_ids)]
        ax.scatter(
            other[col],
            other["pooled_f1"],
            s=28,
            alpha=0.55,
            color=ARCH_COLORS[arch],
            label=ARCH_LABELS[arch],
            edgecolors="none",
        )
        if not win.empty:
            ax.scatter(
                win[col],
                win["pooled_f1"],
                s=110,
                marker="*",
                color=ARCH_COLORS[arch],
                edgecolors="0.15",
                linewidths=0.6,
                zorder=5,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log)")
    ax.set_ylabel("Pooled val F1 (max)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    _style_ax(ax)
    fig.tight_layout()
    suffix = "f1_vs_params" if which == "total" else "f1_vs_backbone_params"
    out = FIGURES_DIR / f"{STEM}_{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_per_session(sess: pd.DataFrame, metric: str, stem_suffix: str) -> Path:
    work = sess.copy()
    work["short"] = work["session"].map(_short_session)
    sessions = sorted(work["short"].unique())
    fig, ax = plt.subplots(figsize=(18, 4.8))
    n = len(ARCH_ORDER)
    width = 0.18
    offsets = (np.arange(n) - (n - 1) / 2) * width
    x = np.arange(len(sessions))
    for i, arch in enumerate(ARCH_ORDER):
        g = work[work["arch"] == arch].set_index("short")
        vals = [
            float(g.loc[s, metric])
            if s in g.index and pd.notna(g.loc[s, metric])
            else np.nan
            for s in sessions
        ]
        ax.bar(
            x + offsets[i],
            vals,
            width,
            label=ARCH_LABELS[arch],
            color=ARCH_COLORS[arch],
            edgecolor="white",
            linewidth=0.3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=90, fontsize=6.5)
    ax.set_ylabel(f"History-max val_session {metric.upper()}")
    ax.set_title(
        f"Per-session {metric.upper()} at each architecture's best HP (minipigs)"
    )
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper right")
    _style_ax(ax)
    ax.set_xlim(-0.7, len(sessions) - 0.3)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_{stem_suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_coverage(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    axes = axes.ravel()
    for ax, arch in zip(axes, ARCH_ORDER):
        sub = finished(df)
        sub = sub[sub["arch"] == arch]
        ax.set_title(
            f"{ARCH_LABELS[arch]}  ({len(sub)}/{SWEEP_EXPECTED[arch]} finished)"
        )
        if sub.empty:
            continue
        if arch == "eegnet":
            pivot = (
                sub.assign(score=sub["pooled_f1"])
                .pivot_table(
                    index="kernel_length",
                    columns="F1",
                    values="pooled_f1",
                    aggfunc="max",
                )
                .sort_index()
            )
            _heatmap(ax, pivot, "kernel", "F1")
        elif arch == "gru_spatial":
            sub = sub.copy()
            sub["proj"] = sub["use_input_proj"].map(
                lambda x: "proj" if bool(x) else "no-proj"
            )
            pivot = sub.pivot_table(
                index="proj",
                columns="conv_filters",
                values="pooled_f1",
                aggfunc="max",
            )
            _heatmap(ax, pivot, "proj", "F")
        elif arch == "gru_temporal":
            sub = sub.copy()
            sub["proj"] = sub["use_input_proj"].map(
                lambda x: "proj" if bool(x) else "no-proj"
            )
            pivot = sub.pivot_table(
                index="conv_kernel",
                columns="conv_filters",
                values="pooled_f1",
                aggfunc="max",
            )
            _heatmap(ax, pivot, "kernel", "F")
        else:
            sub = sub.copy()
            pivot = sub.pivot_table(
                index="conv_kernel",
                columns="conv_filters",
                values="pooled_f1",
                aggfunc="max",
            )
            _heatmap(ax, pivot, "kernel", "F")
        _style_ax(ax)
    fig.suptitle("Max pooled F1 over the HP grid (finished runs)", y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / f"{STEM}_grid_coverage.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def _heatmap(
    ax: plt.Axes, pivot: pd.DataFrame, ylabel: str, xlabel: str
) -> None:
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="YlGnBu",
        vmin=np.nanmin(data) if data.size else 0,
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index], fontsize=8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for i, j in itertools.product(range(data.shape[0]), range(data.shape[1])):
        val = data[i, j]
        if np.isnan(val):
            ax.text(
                j, i, "·", ha="center", va="center", color="0.5", fontsize=8
            )
        else:
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)
    fig = ax.figure
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _save_csv(df: pd.DataFrame, name: str) -> Path:
    path = CSV_DIR / f"{STEM}_{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", action="store_true")
    args = parser.parse_args()

    csv_runs = CSV_DIR / f"{STEM}_runs.csv"
    csv_sess = CSV_DIR / f"{STEM}_winner_sessions.csv"

    if args.cached and csv_runs.exists():
        print(f"Loading cached runs from {csv_runs}")
        df = pd.read_csv(csv_runs)
        df = _fill_hps_and_params(df)
        _save_csv(df, "runs")
    else:
        load_session_configs()
        df = fetch_runs()
        df = _fill_hps_and_params(df)
        _save_csv(df, "runs")

    cov = coverage_table(df)
    best = best_per_arch(df)
    tops = top_k(df, k=5)

    if args.cached and csv_sess.exists():
        print(f"Loading cached winner sessions from {csv_sess}")
        sess = pd.read_csv(csv_sess)
    else:
        sess = fetch_winner_sessions(best)
        _save_csv(sess, "winner_sessions")

    sess_sum = session_summary(sess)
    _save_csv(cov, "coverage")
    _save_csv(best, "best")
    _save_csv(sess_sum, "winner_session_summary")

    print_tables(df, cov, best, tops, sess_sum, sess)

    plot_winner_bars(best, sess_sum)
    plot_f1_vs_params(df, best, "total")
    plot_f1_vs_params(df, best, "backbone")
    plot_coverage(df)
    if not sess.empty:
        plot_per_session(sess, "f1", "supp_f1_per_session")
        plot_per_session(sess, "auroc", "supp_auroc_per_session")


if __name__ == "__main__":
    main()
