"""Optuna hyperparameter search for single-session behavior classification.

Runs a sequential Optuna study per tokenizer, parallelizing trials across all
available GPUs.  Each trial launches ``main.py`` as a subprocess pinned to one
GPU via ``CUDA_VISIBLE_DEVICES``.

Usage:
    uv run python scripts/run_optuna_search.py
    uv run python scripts/run_optuna_search.py --gpus 0,1,2,3,4,5,6,7
    uv run python scripts/run_optuna_search.py --n-trials 8 --tokenizers spatial_session_cwt per_channel_cwt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
from pathlib import Path

import optuna

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optuna_search")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "tokenizer_explore/optuna_behavior_search"
OUTPUT_BASE = Path(os.environ.get("SCRATCH", str(PROJECT_ROOT / "outputs")))

ALL_TOKENIZERS = [
    "per_channel_cwt",
    "per_channel_per_timepoint_linear",
    "per_channel_resample_cnn",
    "spatial_session_cwt",
    "spatial_session_mlp_per_timepoint_identity",
    "spatial_session_mlp_cwt",
    "spatial_session_per_timepoint_linear",
    "spatial_session_resample_cnn",
    "spatial_session_mlp_resample_cnn",
]

SEARCH_SPACE = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 5e-3, "log": True},
    "weight_decay": {"type": "float", "low": 1e-4, "high": 0.3, "log": True},
}


def detect_gpus() -> list[int]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [int(line.strip()) for line in out.strip().splitlines()]
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return [0]


def read_metric_from_checkpoint(run_dir: Path) -> float | None:
    """Read the best AUROC from the ModelCheckpoint saved in last.ckpt."""
    import torch

    ckpt_dir = run_dir / "checkpoints"
    last_ckpt = ckpt_dir / "last.ckpt"
    best_ckpt = ckpt_dir / "best.ckpt"

    ckpt_path = last_ckpt if last_ckpt.exists() else best_ckpt
    if not ckpt_path.exists():
        return None

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for cb_state in ckpt.get("callbacks", {}).values():
            if "best_model_score" in cb_state:
                score = cb_state["best_model_score"]
                if hasattr(score, "item"):
                    return float(score.item())
                return float(score)
    except Exception as exc:
        log.warning("Failed to read metric from %s: %s", ckpt_path, exc)

    return None


def run_trial(
    trial: optuna.Trial,
    tokenizer: str,
    gpu_pool: queue.Queue[int],
    study_name: str,
) -> float:
    lr = trial.suggest_float("learning_rate", **_float_spec("learning_rate"))
    wd = trial.suggest_float("weight_decay", **_float_spec("weight_decay"))

    run_name = f"optuna_{tokenizer}_t{trial.number:03d}"
    run_dir = OUTPUT_BASE / "runs" / study_name / run_name

    gpu_id = gpu_pool.get()
    try:
        log.info(
            "Trial %d [GPU %d] %s — lr=%.2e wd=%.2e",
            trial.number,
            gpu_id,
            tokenizer,
            lr,
            wd,
        )

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "main.py"),
            f"experiment={EXPERIMENT}",
            f"model/tokenizer={tokenizer}",
            f"hyperparameters.learning_rate={lr}",
            f"hyperparameters.weight_decay={wd}",
            f"run.name={run_name}",
            f"run.group={study_name}",
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        log_file = run_dir / "trial.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(result.stdout)

        if result.returncode != 0:
            log.error(
                "Trial %d FAILED (exit %d). Log: %s",
                trial.number,
                result.returncode,
                log_file,
            )
            return float("-inf")

    finally:
        gpu_pool.put(gpu_id)

    metric = read_metric_from_checkpoint(run_dir)
    if metric is None:
        log.warning("Trial %d: no metric found in %s", trial.number, run_dir)
        return float("-inf")

    log.info("Trial %d finished — AUROC=%.4f", trial.number, metric)
    return metric


def _float_spec(key: str) -> dict:
    spec = SEARCH_SPACE[key]
    return {k: v for k, v in spec.items() if k != "type"}


def run_search_for_tokenizer(
    tokenizer: str,
    gpus: list[int],
    n_trials: int,
    db_path: Path,
) -> optuna.Study:
    study_name = f"OPTUNA_BEHAVIOR_{tokenizer}"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=len(gpus)),
    )

    remaining = n_trials - len(study.trials)
    if remaining <= 0:
        log.info(
            "[%s] Already has %d trials — skipping.",
            tokenizer,
            len(study.trials),
        )
        return study

    log.info(
        "=== Starting search: %s (%d trials, %d GPUs) ===",
        tokenizer,
        remaining,
        len(gpus),
    )

    gpu_pool: queue.Queue[int] = queue.Queue()
    for g in gpus:
        gpu_pool.put(g)

    study.optimize(
        lambda trial: run_trial(trial, tokenizer, gpu_pool, study_name),
        n_trials=remaining,
        n_jobs=len(gpus),
        show_progress_bar=True,
    )

    best = study.best_trial
    log.info(
        "=== Best for %s: AUROC=%.4f (trial %d) ===\n  params=%s",
        tokenizer,
        best.value,
        best.number,
        best.params,
    )
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU indices (default: auto-detect all)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=16,
        help="Max number of Optuna trials per tokenizer (default: 16)",
    )
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=ALL_TOKENIZERS,
        help="Tokenizer names to search (default: all 9)",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=str(OUTPUT_BASE / "optuna"),
        help="Directory for Optuna SQLite databases",
    )
    args = parser.parse_args()

    gpus = (
        [int(g) for g in args.gpus.split(",")] if args.gpus else detect_gpus()
    )
    log.info("Using GPUs: %s", gpus)

    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "behavior_search.db"

    summary: dict[str, dict] = {}
    for tokenizer in args.tokenizers:
        study = run_search_for_tokenizer(
            tokenizer, gpus, args.n_trials, db_path
        )
        if study.best_trial:
            summary[tokenizer] = {
                "best_auroc": study.best_value,
                "best_params": study.best_params,
                "best_trial": study.best_trial.number,
                "n_trials": len(study.trials),
            }

    summary_path = db_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Summary written to %s", summary_path)

    log.info("\n=== Final Summary ===")
    for tok, info in sorted(
        summary.items(), key=lambda x: x[1]["best_auroc"], reverse=True
    ):
        log.info(
            "  %-45s AUROC=%.4f  (trial %d)",
            tok,
            info["best_auroc"],
            info["best_trial"],
        )


if __name__ == "__main__":
    main()
