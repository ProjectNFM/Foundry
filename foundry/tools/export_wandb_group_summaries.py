"""Export W&B grouped runs to a per-session JSON report.

Example:
    uv run python -m foundry.tools.export_wandb_group_summaries \
        --experiment auditory_decoding/eegnet_freqdec_multisess \
        --entity your-team
"""

import argparse
import json
import logging
import numbers
import os
import re
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _experiment_config_path(experiment_name: str) -> Path:
    return (
        _project_root() / "configs" / "experiment" / f"{experiment_name}.yaml"
    )


def _load_experiment_defaults(
    experiment_name: str,
) -> tuple[str | None, str | None]:
    config_path = _experiment_config_path(experiment_name)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    group = OmegaConf.select(cfg, "run.group")
    metric = OmegaConf.select(cfg, "trainer.callbacks.early_stopping.monitor")
    return group, metric


def _default_project_name() -> str:
    logger_cfg_path = _project_root() / "configs" / "logger" / "wandb.yaml"
    if not logger_cfg_path.exists():
        raise FileNotFoundError(
            f"W&B logger config not found: {logger_cfg_path}"
        )

    logger_cfg = OmegaConf.load(logger_cfg_path)
    project = OmegaConf.select(logger_cfg, "project")
    if not isinstance(project, str):
        raise ValueError(
            "configs/logger/wandb.yaml is missing a string project name."
        )
    return project


def _sanitize_for_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", text)


def _default_output_path(group_name: str) -> Path:
    filename = _sanitize_for_filename(group_name) + "_session_summaries.json"
    return _project_root() / "wandb_exports" / filename


def _to_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _to_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _extract_session_id(run: wandb.apis.public.Run) -> str:
    config = run.config or {}
    if isinstance(config, Mapping):
        data_cfg = config.get("data")
        if isinstance(data_cfg, Mapping):
            recording_ids = data_cfg.get("recording_ids")
            if isinstance(recording_ids, list) and recording_ids:
                return str(recording_ids[0])
            if isinstance(recording_ids, str):
                return recording_ids

        for key in ("data.recording_ids.0", "data.recording_ids[0]"):
            if key in config and config[key]:
                return str(config[key])

    if run.name and run.name.startswith("eegnet_neurosoft_"):
        return run.name[len("eegnet_neurosoft_") :]
    return run.name or run.id


def _numeric_metric_value(
    summary: Mapping[str, Any], metric_key: str
) -> float | None:
    value = summary.get(metric_key)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return float(value)
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export grouped W&B run summaries into a per-session JSON file."
        )
    )
    parser.add_argument(
        "--experiment",
        default="auditory_decoding/eegnet_freqdec_multisess",
        help="Experiment config used to infer defaults.",
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity. Defaults to WANDB_ENTITY env var when available.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="W&B project. Defaults to configs/logger/wandb.yaml project.",
    )
    parser.add_argument(
        "--group",
        default=None,
        help="W&B run group. Defaults to run.group from the experiment config.",
    )
    parser.add_argument(
        "--selection-metric",
        default=None,
        help=(
            "Metric key used to pick the best run per session. "
            "Defaults to trainer.callbacks.early_stopping.monitor."
        ),
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=["finished"],
        help=(
            "W&B run states to include (for example: finished failed crashed)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to ./wandb_exports/<group>_session_summaries.json.",
    )
    parser.add_argument(
        "--api-timeout",
        type=int,
        default=60,
        help="W&B API timeout in seconds.",
    )
    return parser


def _resolve_cli_defaults(
    args: argparse.Namespace,
) -> tuple[str, str, str, str, Path]:
    group_from_experiment, metric_from_experiment = _load_experiment_defaults(
        args.experiment
    )

    project = args.project or _default_project_name()
    group = args.group or group_from_experiment
    selection_metric = args.selection_metric or metric_from_experiment

    if not group:
        raise ValueError(
            "Could not resolve W&B group. Pass --group explicitly."
        )
    if not selection_metric:
        raise ValueError(
            "Could not resolve selection metric. Pass --selection-metric explicitly."
        )

    entity = args.entity
    project_path = f"{entity}/{project}" if entity else project
    output_path = args.output or _default_output_path(group)

    return project_path, project, group, selection_metric, output_path


def _fetch_group_runs(
    project_path: str,
    group: str,
    states: list[str],
    api_timeout: int,
) -> list[wandb.apis.public.Run]:
    api = wandb.Api(timeout=api_timeout)
    filters: dict[str, Any] = {"group": group}
    if states:
        filters["state"] = {"$in": states}

    runs = list(api.runs(path=project_path, filters=filters))
    logger.info(
        "Fetched %d runs from %s (group=%s, states=%s).",
        len(runs),
        project_path,
        group,
        ",".join(states),
    )
    return runs


def _build_report(
    *,
    runs: list[wandb.apis.public.Run],
    project_path: str,
    project: str,
    group: str,
    selection_metric: str,
    states: list[str],
) -> dict[str, Any]:
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for run in runs:
        summary_raw = dict(run.summary.items())
        metric_value = _numeric_metric_value(summary_raw, selection_metric)
        session_id = _extract_session_id(run)

        run_entry = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "url": run.url,
            "created_at": run.created_at,
            "selection_metric_value": metric_value,
            "summary": _to_json_value(summary_raw),
        }
        sessions[session_id].append(run_entry)

    session_entries: dict[str, Any] = {}
    for session_id in sorted(sessions):
        runs_for_session = sessions[session_id]
        best_run = max(
            runs_for_session,
            key=lambda entry: (
                float("-inf")
                if entry["selection_metric_value"] is None
                else entry["selection_metric_value"]
            ),
        )

        if best_run["selection_metric_value"] is None:
            best_run = None

        session_entries[session_id] = {
            "run_count": len(runs_for_session),
            "best_run": best_run,
            "runs": runs_for_session,
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wandb": {
            "project_path": project_path,
            "project": project,
            "group": group,
            "states": states,
        },
        "selection_metric": selection_metric,
        "run_count": len(runs),
        "session_count": len(session_entries),
        "sessions": session_entries,
    }


def _write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args()
    (
        project_path,
        project,
        group,
        selection_metric,
        output_path,
    ) = _resolve_cli_defaults(args)

    runs = _fetch_group_runs(
        project_path=project_path,
        group=group,
        states=args.states,
        api_timeout=args.api_timeout,
    )
    report = _build_report(
        runs=runs,
        project_path=project_path,
        project=project,
        group=group,
        selection_metric=selection_metric,
        states=args.states,
    )
    _write_report(report, output_path)

    print(f"Wrote {report['run_count']} runs to {output_path}")


if __name__ == "__main__":
    main()
