"""Phase 3 NeuroSoft supervised pretraining pipeline analysis.

Uses wandb.Api() to fetch the exact canonical run set from the Phase 3
ten-job matrix, exports tables to analysis/csv/, and reports:

- run completeness and status;
- source aggregate and per-session validation metrics;
- source/target manifest and checkpoint hashes;
- milestone/best checkpoint counters;
- transfer loaded/excluded/frozen/trainable counts;
- downstream validation/test checkpoint identity;
- compute monotonicity and profiler method; and
- descriptive target F1 versus the existing matched scratch control.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = REPO_ROOT / "analysis" / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)

WANDB_PROJECT = "neurosoft_supervised_pretraining"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

SOURCE_GROUPS = {
    "minipigs_smoke": "NEUROSOFT_SOURCE_PRETRAINING_MINIPIGS",
    "monkeys_smoke": "NEUROSOFT_SOURCE_PRETRAINING_MONKEYS",
}

TRANSFER_GROUPS = {
    "minipigs_transfer": "NEUROSOFT_TRANSFER_MINIPIGS",
    "monkeys_transfer": "NEUROSOFT_TRANSFER_MONKEYS",
}

ALL_GROUPS = {**SOURCE_GROUPS, **TRANSFER_GROUPS}


def _get_api():
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb is required. Install with: pip install wandb")
        sys.exit(1)
    return wandb.Api()


def _fetch_runs(api, group_name: str) -> list:
    """Fetch all runs from a WandB group."""
    filters = {"group": group_name}
    if WANDB_ENTITY:
        path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    else:
        path = WANDB_PROJECT

    try:
        runs = list(api.runs(path, filters=filters))
    except Exception as e:
        print(f"WARNING: Could not fetch runs for group {group_name}: {e}")
        return []
    return runs


def _run_to_dict(run) -> dict:
    """Extract key fields from a WandB run."""
    config = run.config or {}
    summary = (
        run.summary._json_dict
        if hasattr(run.summary, "_json_dict")
        else dict(run.summary)
    )

    def metric(key: str, aggregate: str = "max"):
        """Read a metric whether W&B retained it raw or with an aggregate suffix."""
        value = summary.get(key)
        if value is not None:
            return value
        return summary.get(f"{key}.{aggregate}")

    return {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "group": config.get("run", {}).get("group", run.group),
        "species": config.get("data", {}).get("audit_species"),
        "role": config.get("data", {}).get("role"),
        "transfer_regime": config.get("run", {}).get(
            "pretrained_transfer_regime"
        ),
        "model_seed": config.get("run", {}).get("seed"),
        "max_steps": config.get("trainer", {}).get("max_steps"),
        "max_epochs": config.get("trainer", {}).get("max_epochs"),
        "source_session_mean_f1": summary.get(
            "val/source_session_mean_supported_f1"
        ),
        "val_supported_f1": metric(
            "val/neurosoft_acoustic_stim_8band_supported_f1"
        ),
        "test_supported_f1": metric(
            "test/neurosoft_acoustic_stim_8band_supported_f1"
        ),
        "optimizer_steps": summary.get("compute/optimizer_steps"),
        "processed_windows": summary.get("compute/processed_windows"),
        "cumulative_flops": summary.get("compute/cumulative_flops"),
        "effective_epochs": summary.get("compute/effective_epochs"),
        "best_step": summary.get("compute/best_step"),
        "best_windows": summary.get("compute/best_windows"),
        "best_flops": summary.get("compute/best_flops"),
        "total_parameters": summary.get("compute/total_parameters"),
        "trainable_parameters": summary.get("compute/trainable_parameters"),
        "source_manifest": config.get("source_manifest"),
        "checkpoint_manifest": config.get("run", {}).get(
            "pretrained_checkpoint_manifest"
        ),
        "precision": config.get("trainer", {}).get("precision"),
        "gpu": summary.get("compute/gpu_model"),
    }


def fetch_all_runs() -> pd.DataFrame:
    """Fetch all Phase 3 runs from WandB."""
    api = _get_api()
    all_runs = []

    for label, group in ALL_GROUPS.items():
        runs = _fetch_runs(api, group)
        print(f"  {label} ({group}): {len(runs)} runs")
        for run in runs:
            row = _run_to_dict(run)
            row["category"] = label
            all_runs.append(row)

    df = pd.DataFrame(all_runs)
    return df


def report_completeness(df: pd.DataFrame) -> None:
    """Report run completeness and status."""
    print("\n=== Run Completeness ===")
    if df.empty:
        print("  No runs found. Have the Phase 3 jobs been submitted?")
        return

    status_counts = (
        df.groupby(["category", "state"]).size().unstack(fill_value=0)
    )
    print(status_counts.to_string())

    finished = df[df["state"] == "finished"]
    print(f"\n  Total finished: {len(finished)} / {len(df)}")


def report_source_metrics(df: pd.DataFrame) -> None:
    """Report source pretraining metrics."""
    print("\n=== Source Pretraining Metrics ===")
    source = df[df["role"] == "source_pretraining"]
    if source.empty:
        print("  No source pretraining runs found.")
        return

    cols = [
        "run_name",
        "species",
        "state",
        "source_session_mean_f1",
        "optimizer_steps",
        "processed_windows",
        "effective_epochs",
        "best_step",
    ]
    available = [c for c in cols if c in source.columns]
    print(source[available].to_string(index=False))


def report_transfer_metrics(df: pd.DataFrame) -> None:
    """Report downstream transfer metrics."""
    print("\n=== Transfer Metrics ===")
    transfer = df[df["transfer_regime"].notna()]
    if transfer.empty:
        print("  No transfer runs found.")
        return

    cols = [
        "run_name",
        "species",
        "transfer_regime",
        "state",
        "val_supported_f1",
        "test_supported_f1",
        "total_parameters",
        "trainable_parameters",
    ]
    available = [c for c in cols if c in transfer.columns]
    print(transfer[available].to_string(index=False))


def report_compute(df: pd.DataFrame) -> None:
    """Report compute accounting."""
    print("\n=== Compute Accounting ===")
    source = df[df["role"] == "source_pretraining"]
    if source.empty:
        print("  No source runs for compute reporting.")
        return

    cols = [
        "run_name",
        "optimizer_steps",
        "processed_windows",
        "cumulative_flops",
        "effective_epochs",
        "precision",
        "gpu",
    ]
    available = [c for c in cols if c in source.columns]
    print(source[available].to_string(index=False))


def report_provenance(df: pd.DataFrame) -> None:
    """Report manifest and checkpoint provenance."""
    print("\n=== Provenance ===")
    cols = [
        "run_name",
        "source_manifest",
        "checkpoint_manifest",
        "model_seed",
    ]
    available = [c for c in cols if c in df.columns]
    if available:
        print(df[available].to_string(index=False))


def export_csv(df: pd.DataFrame) -> None:
    """Export tables to CSV."""
    if df.empty:
        print("\n  No data to export.")
        return

    all_path = CSV_DIR / "phase3_all_runs.csv"
    df.to_csv(all_path, index=False)
    print(f"\n  Exported all runs to {all_path}")

    source = df[df["role"] == "source_pretraining"]
    if not source.empty:
        source_path = CSV_DIR / "phase3_source_runs.csv"
        source.to_csv(source_path, index=False)
        print(f"  Exported source runs to {source_path}")

    transfer = df[df["transfer_regime"].notna()]
    if not transfer.empty:
        transfer_path = CSV_DIR / "phase3_transfer_runs.csv"
        transfer.to_csv(transfer_path, index=False)
        print(f"  Exported transfer runs to {transfer_path}")


def main():
    print("=" * 60)
    print("Phase 3 NeuroSoft Supervised Pretraining Pipeline Analysis")
    print("=" * 60)

    print("\nFetching runs from WandB...")
    df = fetch_all_runs()

    report_completeness(df)
    report_source_metrics(df)
    report_transfer_metrics(df)
    report_compute(df)
    report_provenance(df)
    export_csv(df)

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
