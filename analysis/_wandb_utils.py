"""Shared W&B fetch and summary utilities for analysis scripts.

Centralises entity/project/run-path construction, summary-value
unwrapping, run fetching, metric-history retrieval, and run-summary
extraction so that individual analysis scripts declare only
experiment-specific run IDs, metrics, and plots.

All public functions accept an optional *api* parameter so tests can
inject a fake ``wandb.Api`` without network calls.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


class RunNotFoundError(Exception):
    """A W&B run could not be located at the given path."""


class MetricNotFoundError(Exception):
    """One or more requested metrics are absent from a run's history."""


def make_run_path(
    run_id: str,
    project: str,
    entity: str | None = None,
) -> str:
    """Build a ``entity/project/run_id`` path, omitting *entity* when ``None``."""
    if entity:
        return f"{entity}/{project}/{run_id}"
    return f"{project}/{run_id}"


def unwrap_summary_value(val: Any, key: str = "min") -> Any:
    """Extract a scalar from a W&B ``SummarySubDict``.

    W&B summary entries for tracked metrics are often dicts such as
    ``{"min": 0.12, "max": 0.45}``.  This helper tries ``val[key]``
    first, then falls back to ``float(val)``, and returns *val*
    unchanged if neither conversion succeeds.
    """
    try:
        return float(val[key])
    except (TypeError, KeyError, IndexError):
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return val


def get_run(
    run_id: str,
    project: str,
    entity: str | None = None,
    *,
    api: Any | None = None,
) -> Any:
    """Return a W&B run object, raising `.RunNotFoundError` on failure."""
    if api is None:
        import wandb

        api = wandb.Api()

    path = make_run_path(run_id, project, entity)
    try:
        return api.run(path)
    except Exception as exc:
        raise RunNotFoundError(
            f"Could not find run '{run_id}' at path '{path}'. "
            "Check WANDB_ENTITY and project name."
        ) from exc


def fetch_metric_history(
    run_id: str,
    metrics: str | list[str],
    project: str,
    entity: str | None = None,
    *,
    x_axis: str = "_step",
    samples: int = 10_000,
    api: Any | None = None,
) -> pd.DataFrame:
    """Fetch metric history for a single run.

    Returns a ``DataFrame`` with *x_axis* and each requested metric as
    columns, rows with NaN metric values already dropped.

    Raises ``MetricNotFoundError`` if any requested metric column is
    absent from the returned history.
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    run = get_run(run_id, project, entity, api=api)
    keys = list(dict.fromkeys([x_axis] + metrics))
    history = run.history(keys=keys, samples=samples, pandas=True)

    missing = [m for m in metrics if m in keys and m not in history.columns]
    if missing:
        raise MetricNotFoundError(
            f"Metrics {missing} not found in run '{run_id}'. "
            f"Available columns: {sorted(history.columns.tolist())}"
        )

    present_keys = [k for k in keys if k in history.columns]
    history = history[present_keys].dropna(
        subset=[m for m in metrics if m in history.columns]
    )
    return history.reset_index(drop=True)


def fetch_run_summary(
    run_id: str,
    project: str,
    summary_keys: dict[str, tuple[str, str]],
    entity: str | None = None,
    *,
    api: Any | None = None,
    include_state: bool = True,
) -> dict[str, Any]:
    """Fetch specific summary values for a run, applying ``unwrap_summary_value``.

    *summary_keys* maps ``output_name &#8594; (wandb_summary_key, unwrap_key)``.
    For example::

        {"best_val_loss": ("val/loss", "min"),
         "best_val_f1":   ("val/f1",   "max")}
    """
    run = get_run(run_id, project, entity, api=api)
    result: dict[str, Any] = {}
    if include_state:
        result["state"] = run.state
    for out_name, (wandb_key, unwrap_key) in summary_keys.items():
        result[out_name] = unwrap_summary_value(
            run.summary.get(wandb_key), unwrap_key
        )
    return result


def default_entity() -> str | None:
    """Return ``WANDB_ENTITY`` from the environment, or ``None``."""
    return os.environ.get("WANDB_ENTITY") or None


def figures_dir(script_path: str | os.PathLike[str]) -> Any:
    """Return ``analysis/figures/`` relative to *script_path*, creating it."""
    from pathlib import Path

    d = Path(script_path).parent / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d
