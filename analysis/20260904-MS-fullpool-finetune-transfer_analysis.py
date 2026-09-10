"""Analyze the Phase 4A full-pool full-finetuning transfer gate.

Fetches the declared Phase-4A source and transfer runs plus matched Phase-2
100%-data scratch controls through ``wandb.Api()``. Four transfer cells are
recovered from first-attempt local W&B summaries: their packed allocation was
preempted after test evaluation, then its restart failed immediately on a GPU
with an uncorrectable ECC error. Recovery is limited to those immutable IDs.

Usage:
    uv run python analysis/20260904-MS-fullpool-finetune-transfer_analysis.py
"""

from __future__ import annotations

import itertools
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from _wandb_utils import csv_dir, default_entity, figures_dir, unwrap_summary_value


PREFIX = "20260904-MS-fullpool-finetune-transfer"
PROJECT = "neurosoft_supervised_pretraining"
TASK = "neurosoft_acoustic_stim_8band"
SEEDS = {42, 43, 44}
ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "launch/checkpoint_sets/phase4a-mila-best.jsonl"
CELL_LISTS = {
    species: ROOT / f"launch/phase4a/phase4a-downstream-{species}.jsonl"
    for species in ("minipigs", "monkeys")
}
LOCAL_RUN_ROOT = Path("/network/scratch/s/sobralm/runs/PHASE4A_FULL_FINETUNE_MINIPIGS")
SOURCE_GROUPS = {
    "minipigs": "NEUROSOFT_SOURCE_PRETRAINING_MINIPIGS",
    "monkeys": "NEUROSOFT_SOURCE_PRETRAINING_MONKEYS",
}
TRANSFER_GROUPS = {
    "minipigs": "PHASE4A_FULL_FINETUNE_MINIPIGS",
    "monkeys": "PHASE4A_FULL_FINETUNE_MONKEYS",
}
SCRATCH_GROUPS = {
    "minipigs": "NORM_GLOBAL_CONV_BIGRU_MINIPIGS_PROD_OFFLINE_16_20260902",
    "monkeys": "NORM_GLOBAL_CONV_BIGRU_MONKEYS_PROD_OFFLINE_16_20260902",
}
EXPECTED_SOURCE = {"minipigs": 21, "monkeys": 15}
EXPECTED_TRANSFER = {"minipigs": 360, "monkeys": 117}

# Four first attempts completed test evaluation before scheduler preemption;
# their restart hit ECC. ``91cb2c2e`` is W&B-finished but its server summary is
# incomplete despite its completed local result bundle. These are the only
# records permitted to use a local summary.
FAILED_RESTART_RECOVERY_IDS = frozenset(
    {"9a3b3b21", "57bb5d62", "1f1386d5", "01c0f867"}
)
RECOVERED_TRANSFER_RUN_IDS = FAILED_RESTART_RECOVERY_IDS | frozenset({"91cb2c2e"})
RUN_COLUMNS = [
    "kind", "species", "run_id", "run_name", "cell_id", "checkpoint_id",
    "state", "analysis_usable", "metric_source", "recovery_summary_path",
    "created_at", "recording", "subject", "target_fraction", "target_seed",
    "source_seed", "source_model_seed", "source_manifest", "checkpoint_manifest",
    "transfer_regime", "test_supported_f1", "best_val_supported_f1",
    "optimizer_steps", "best_step", "best_windows", "best_flops", "best_wall_time_s",
]
REQUIRED_TRANSFER_METRICS = [
    "test_supported_f1", "best_val_supported_f1", "best_step", "best_windows",
    "best_flops",
]
COMPUTE_METRICS = ("best_step", "best_windows", "best_flops")
VALIDATION_METRIC = f"val/{TASK}_supported_f1"


def nested(config: dict[str, Any], *keys: str) -> Any:
    """Read either nested or Hydra-flattened W&B config fields."""
    dotted = ".".join(keys)
    if dotted in config:
        return config[dotted]
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summary_scalar(summary: dict[str, Any], key: str) -> float | None:
    for candidate in (f"{key}.max", key):
        value = summary.get(candidate)
        if value is not None:
            try:
                return float(unwrap_summary_value(value, "max"))
            except (TypeError, ValueError):
                pass
    return None


def recording_id(config: dict[str, Any], name: str) -> str | None:
    ids = nested(config, "data", "dataset_kwargs", "recording_ids")
    if isinstance(ids, list) and ids:
        return str(ids[0])
    value = nested(config, "neurosoft", "recording_id")
    if value:
        return str(value)
    match = re.search(
        r"(sub-\d+_ses-\d+_task-AcousStim_acq-[A-Za-z]+(?:anest)?_desc-raw)",
        name,
    )
    return match.group(1) if match else None


def subject_id(recording: str | None, config: dict[str, Any]) -> str | None:
    value = nested(config, "neurosoft", "subject")
    if value:
        return str(value)
    match = re.search(r"(sub-\d+)", recording or "")
    return match.group(1) if match else None


def training_fraction(config: dict[str, Any], run_name: str) -> float | None:
    value = nested(config, "data", "training_fraction")
    if value is not None:
        return as_float(value)
    match = re.search(r"_f(0?\.\d+|1\.0+|1)_", run_name)
    return as_float(match.group(1)) if match else None


def source_seed(config: dict[str, Any]) -> int | None:
    value = nested(config, "run", "source_selection_seed")
    return int(value) if value is not None else None


def selected_source_runs() -> dict[str, dict[str, Any]]:
    """Index selected source IDs from the committed checkpoint registry."""
    selected: dict[str, dict[str, Any]] = {}
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        manifest = json.loads(Path(record["manifest_path"]).read_text(encoding="utf-8"))
        selected[str(manifest["wandb"]["run_id"])] = record
    return selected


def compiled_cells() -> dict[str, dict[str, Any]]:
    """Index every declared downstream cell by its immutable identity."""
    selected: dict[str, dict[str, Any]] = {}
    for species, path in CELL_LISTS.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["species"] != species:
                raise ValueError(f"{path}: cell species mismatch")
            cell_id = str(record["cell_id"])
            if cell_id in selected:
                raise ValueError(f"Duplicate compiled cell ID: {cell_id}")
            selected[cell_id] = record
    return selected


def is_declared_transfer_run(
    run_id: str, run_name: str, config: dict[str, Any], species: str,
    selected_cells: dict[str, dict[str, Any]],
) -> bool:
    """Require W&B transfer provenance to match one exact compiled cell."""
    cell_id = nested(config, "run", "cell_id")
    cell = selected_cells.get(str(cell_id))
    if cell is None:
        return False
    recording_ids = nested(config, "data", "dataset_kwargs", "recording_ids")
    observed = {
        "run_name": run_name,
        "species": species,
        "target_recording": str(recording_ids[0]) if isinstance(recording_ids, list) and recording_ids else None,
        "target_fraction": nested(config, "data", "training_fraction"),
        "target_finetuning_seed": nested(config, "run", "seed"),
        "checkpoint_id": nested(config, "run", "checkpoint_id"),
        "checkpoint_manifest": nested(config, "run", "pretrained_checkpoint_manifest"),
        "checkpoint_manifest_hash": nested(config, "run", "pretrained_checkpoint_manifest_hash"),
        "checkpoint_sha256": nested(config, "run", "pretrained_checkpoint_sha256"),
        "source_selection_seed": nested(config, "run", "source_selection_seed"),
        "source_model_seed": nested(config, "run", "source_model_seed"),
        "transfer_regime": nested(config, "run", "pretrained_transfer_regime"),
    }
    if observed != {key: cell[key] for key in observed}:
        print(f"Skipping undeclared/mismatched transfer run {run_name} ({run_id})", flush=True)
        return False
    return True


def recovered_local_summary(run_name: str, run_id: str) -> tuple[dict[str, Any], str]:
    """Load exactly one complete pre-preemption local W&B summary."""
    if run_id not in RECOVERED_TRANSFER_RUN_IDS:
        raise ValueError(f"Local recovery is not allowlisted for run {run_id}.")
    candidates: list[tuple[dict[str, Any], Path]] = []
    for path in (LOCAL_RUN_ROOT / run_name / "wandb").glob(f"run-*-{run_id}/files/wandb-summary.json"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read recovery summary {path}") from exc
        values = [
            summary_scalar(summary, f"test/{TASK}_supported_f1"),
            summary_scalar(summary, f"val/{TASK}_supported_f1"),
            summary_scalar(summary, "compute/best_step"),
            summary_scalar(summary, "compute/best_windows"),
            summary_scalar(summary, "compute/best_flops"),
        ]
        output_path = path.with_name("output.log")
        output = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else ""
        if (
            all(value is not None and np.isfinite(value) for value in values)
            and "uncorrectable ECC error" not in output
        ):
            candidates.append((summary, path))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one complete first-attempt summary for {run_id}; found {len(candidates)}.")
    summary, path = candidates[0]
    return summary, str(path)


def fetch_group(
    api: Any, entity: str, group: str, label: str, species: str,
    selected_sources: dict[str, dict[str, Any]], selected_cells: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    print(f"Fetching {label}: {group}", flush=True)
    rows: list[dict[str, Any]] = []
    for run in api.runs(f"{entity}/{PROJECT}", filters={"group": group}, per_page=500, lazy=False):
        run_id = str(run.id)
        selected = selected_sources.get(run_id)
        if label == "source" and selected is None:
            continue
        config = dict(run.config or {})
        name = str(run.name or "")
        if label == "transfer" and not is_declared_transfer_run(run_id, name, config, species, selected_cells):
            continue
        summary = dict(run.summary or {})
        recovery_path: str | None = None
        metric_source = "wandb_api"
        if label == "transfer" and run_id in RECOVERED_TRANSFER_RUN_IDS:
            summary, recovery_path = recovered_local_summary(name, run_id)
            metric_source = (
                "verified_local_first_attempt"
                if run_id in FAILED_RESTART_RECOVERY_IDS
                else "verified_local_completed_run"
            )
        record = recording_id(config, name)
        target_seed = nested(config, "run", "seed")
        rows.append({
            "kind": label, "species": species, "run_id": run_id, "run_name": name,
            "cell_id": nested(config, "run", "cell_id"),
            "checkpoint_id": nested(config, "run", "checkpoint_id"),
            "state": run.state,
            "analysis_usable": label != "transfer" or run.state == "finished" or run_id in RECOVERED_TRANSFER_RUN_IDS,
            "metric_source": metric_source, "recovery_summary_path": recovery_path,
            "created_at": getattr(run, "created_at", None), "recording": record,
            "subject": subject_id(record, config), "target_fraction": training_fraction(config, name),
            "target_seed": int(target_seed) if target_seed is not None else None,
            "source_seed": selected["source_selection_seed"] if selected else source_seed(config),
            "source_model_seed": selected["source_model_seed"] if selected else nested(config, "run", "source_model_seed"),
            "source_manifest": nested(config, "source_manifest"),
            "checkpoint_manifest": nested(config, "run", "pretrained_checkpoint_manifest"),
            "transfer_regime": nested(config, "run", "pretrained_transfer_regime"),
            "test_supported_f1": summary_scalar(summary, f"test/{TASK}_supported_f1"),
            "best_val_supported_f1": summary_scalar(summary, f"val/{TASK}_supported_f1"),
            "optimizer_steps": summary_scalar(summary, "compute/optimizer_steps"),
            "best_step": summary_scalar(summary, "compute/best_step"),
            "best_windows": summary_scalar(summary, "compute/best_windows"),
            "best_flops": summary_scalar(summary, "compute/best_flops"),
            "best_wall_time_s": summary_scalar(summary, "compute/best_wall_time_s"),
        })
    return rows


def completion_audit(
    source: pd.DataFrame, transfer: pd.DataFrame, selected_cells: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Verify matrix completeness and document the exact four recoveries."""
    print("\n=== Completion and provenance audit ===")
    for label, frame, expected in (("source", source, EXPECTED_SOURCE), ("full finetune", transfer, EXPECTED_TRANSFER)):
        counts = frame.groupby(["species", "state"]).size().unstack(fill_value=0)
        print(f"{label} W&B states:\n{counts.to_string()}")
        for species, n_expected in expected.items():
            if len(frame[frame.species == species]) != n_expected:
                raise RuntimeError(f"{species} {label}: expected {n_expected} declared cells.")
    if not source.state.eq("finished").all():
        raise RuntimeError("Not every selected source run finished.")
    if set(transfer.cell_id.dropna().astype(str)) != set(selected_cells):
        raise RuntimeError("Declared transfer cells do not exactly match W&B provenance.")
    recovered = transfer[transfer.metric_source.str.startswith("verified_local_")].copy()
    if set(recovered.run_id) != RECOVERED_TRANSFER_RUN_IDS:
        raise RuntimeError("Unexpected local-result recovery set.")
    failed_restart = recovered[recovered.run_id.isin(FAILED_RESTART_RECOVERY_IDS)]
    if not failed_restart.state.eq("failed").all():
        raise RuntimeError("The four preemption/ECC recoveries must retain failed W&B states.")
    if not recovered.loc[recovered.run_id.eq("91cb2c2e"), "state"].eq("finished").all():
        raise RuntimeError("91cb2c2e should be W&B-finished with only its server summary missing.")
    if not transfer.analysis_usable.all():
        raise RuntimeError("At least one declared transfer cell has no usable result.")
    missing_metrics = transfer[REQUIRED_TRANSFER_METRICS].isna().sum()
    if missing_metrics.any():
        affected = transfer[
            transfer[REQUIRED_TRANSFER_METRICS].isna().any(axis=1)
        ][["run_id", "cell_id", "state", "metric_source"]]
        raise RuntimeError(
            "Transfer runs lack required metrics: "
            f"{missing_metrics[missing_metrics.gt(0)].to_dict()}; "
            f"affected={affected.to_dict(orient='records')}"
        )
    print(f"Scientifically usable transfer results: {int(transfer.analysis_usable.sum())}/{len(transfer)}")
    print(f"Verified local-summary recoveries: {len(recovered)}")
    return recovered[["run_id", "cell_id", "state", "metric_source", "recovery_summary_path"] + REQUIRED_TRANSFER_METRICS]


def canonical_scratch(scratch: pd.DataFrame) -> pd.DataFrame:
    """Retain only finished, unique 100%-data Phase-2 scratch controls."""
    full = scratch[
        scratch.state.eq("finished")
        & scratch.test_supported_f1.notna()
        & np.isclose(scratch.target_fraction.astype(float), 1.0)
        & scratch.target_seed.isin(SEEDS)
    ].copy()
    unit = ["species", "subject", "recording", "target_seed"]
    if full.duplicated(unit, keep=False).any():
        raise RuntimeError("100%-data scratch controls are not unique by session/seed.")
    return full.sort_values(unit).reset_index(drop=True)


def validation_epoch_to_80pct(
    api: Any, entity: str, run_id: str
) -> tuple[float, float, int]:
    """Return first completed validation epoch at 80% of a run's own maximum."""
    run = api.run(f"{entity}/{PROJECT}/{run_id}")
    history = run.history(
        keys=["epoch", VALIDATION_METRIC], samples=10_000, pandas=True
    )
    if "epoch" not in history or VALIDATION_METRIC not in history:
        raise RuntimeError(f"{run_id}: missing validation epoch/history metric.")
    history = history[["epoch", VALIDATION_METRIC]].dropna().sort_values("epoch")
    if history.empty:
        raise RuntimeError(f"{run_id}: empty validation history.")
    max_f1 = float(history[VALIDATION_METRIC].max())
    reached = history[history[VALIDATION_METRIC].ge(0.8 * max_f1)]
    if reached.empty:
        raise RuntimeError(f"{run_id}: never reached its 80% validation threshold.")
    # W&B epoch is zero-based; report completed downstream epochs as one-based.
    return float(reached.iloc[0].epoch + 1), max_f1, len(history)


def add_validation_convergence(
    api: Any, entity: str, runs: pd.DataFrame, label: str
) -> pd.DataFrame:
    """Fetch exact histories and add the per-run 80%-of-own-maximum epoch."""
    cache_path = csv_dir(__file__) / f"{PREFIX}_{label.replace(' ', '_')}_validation_history.csv"
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path)
        except pd.errors.EmptyDataError:
            # A previous bounded execution may have been interrupted while
            # replacing a cache; restart safely rather than trusting it.
            cached = pd.DataFrame(columns=["run_id", "validation_epoch_to_80pct", "validation_max_f1_history", "validation_records"])
        if cached.run_id.duplicated().any():
            raise RuntimeError(f"Duplicate run IDs in validation-history cache {cache_path}.")
        rows = cached.to_dict(orient="records")
    else:
        rows = []
    existing = {str(row["run_id"]) for row in rows}
    pending = [str(run_id) for run_id in runs.run_id if str(run_id) not in existing]
    total = len(runs)
    for index, run_id in enumerate(pending, start=1):
        epoch, max_f1, records = validation_epoch_to_80pct(api, entity, run_id)
        rows.append({
            "run_id": run_id,
            "validation_epoch_to_80pct": epoch,
            "validation_max_f1_history": max_f1,
            "validation_records": records,
        })
        temporary_path = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.tmp"
        )
        pd.DataFrame(rows).to_csv(temporary_path, index=False)
        temporary_path.replace(cache_path)
        if index % 25 == 0 or index == len(pending):
            print(
                f"Fetched validation histories: {label} {total - len(pending) + index}/{total}",
                flush=True,
            )
    history = pd.DataFrame(rows)
    result = runs.merge(history, on="run_id", how="left", validate="one_to_one")
    if result.validation_epoch_to_80pct.isna().any():
        raise RuntimeError(f"{label}: validation-history cache is incomplete.")
    return result


def paired_effects(transfer: pd.DataFrame, scratch: pd.DataFrame) -> pd.DataFrame:
    """Average source seeds, then pair to identical session/target seeds."""
    unit = ["species", "subject", "recording", "target_seed"]
    transfer = transfer[
        transfer.analysis_usable & transfer.transfer_regime.eq("full_finetuning")
        & transfer.recording.notna() & transfer.target_seed.isin(SEEDS) & transfer.source_seed.isin(SEEDS)
    ].copy()
    pretrained = transfer.groupby(unit, as_index=False).agg(
        pretrain_replicates=("source_seed", "nunique"),
        pretrained_test_supported_f1=("test_supported_f1", "mean"),
        pretrained_best_step=("best_step", "mean"),
        pretrained_best_windows=("best_windows", "mean"),
        pretrained_best_flops=("best_flops", "mean"),
        pretrained_best_wall_time_s=("best_wall_time_s", "mean"),
        pretrained_validation_epoch_to_80pct=("validation_epoch_to_80pct", "mean"),
    )
    pretrained = pretrained[pretrained.pretrain_replicates.eq(len(SEEDS))]
    control = scratch[unit + ["test_supported_f1", "best_step", "best_windows", "best_flops", "best_wall_time_s", "validation_epoch_to_80pct"]].rename(columns={
        "test_supported_f1": "scratch_test_supported_f1", "best_step": "scratch_best_step",
        "best_windows": "scratch_best_windows", "best_flops": "scratch_best_flops",
        "best_wall_time_s": "scratch_best_wall_time_s",
        "validation_epoch_to_80pct": "scratch_validation_epoch_to_80pct",
    })
    paired = pretrained.merge(control, on=unit, how="inner", validate="one_to_one")
    paired["test_f1_gain"] = paired.pretrained_test_supported_f1 - paired.scratch_test_supported_f1
    for metric in COMPUTE_METRICS:
        paired[f"{metric}_saved"] = paired[f"scratch_{metric}"] - paired[f"pretrained_{metric}"]
        paired[f"{metric}_saved_pct"] = 100 * paired[f"{metric}_saved"] / paired[f"scratch_{metric}"]
    paired["validation_epoch_to_80pct_saved"] = (
        paired.scratch_validation_epoch_to_80pct
        - paired.pretrained_validation_epoch_to_80pct
    )
    paired["validation_epoch_to_80pct_saved_pct"] = (
        100 * paired.validation_epoch_to_80pct_saved
        / paired.scratch_validation_epoch_to_80pct
    )
    return paired


def reductions(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average target seeds → sessions → subjects for subject-balanced totals."""
    session = paired.groupby(["species", "subject", "recording"], as_index=False).mean(numeric_only=True)
    subject = session.groupby(["species", "subject"], as_index=False).mean(numeric_only=True)
    return session, subject


def sign_flip_p_positive(values: pd.Series) -> float:
    """Exact one-sided sign-flip p-value for positive mean subject effect."""
    data = np.asarray(values, dtype=float)
    observed = float(data.mean())
    null_means = np.array([np.mean(data * signs) for signs in itertools.product((-1, 1), repeat=len(data))])
    return float(np.mean(null_means >= observed - 1e-12))


def bootstrap_interval(values: pd.Series) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260904)
    means = rng.choice(data, size=(20_000, len(data)), replace=True).mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def species_summary(paired: pd.DataFrame, session: pd.DataFrame, subject: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for species, data in subject.groupby("species"):
        f1_low, f1_high = bootstrap_interval(data.test_f1_gain)
        row: dict[str, Any] = {
            "species": species, "n_subjects": len(data),
            "n_sessions": int(session.species.eq(species).sum()),
            "n_session_seed_pairs": int(paired.species.eq(species).sum()),
            "scratch_test_supported_f1": data.scratch_test_supported_f1.mean(),
            "pretrained_test_supported_f1": data.pretrained_test_supported_f1.mean(),
            "test_f1_gain": data.test_f1_gain.mean(),
            "test_f1_gain_sd_subject": data.test_f1_gain.std(ddof=1),
            "test_f1_gain_bootstrap_95_low": f1_low,
            "test_f1_gain_bootstrap_95_high": f1_high,
            "test_f1_gain_signflip_p_one_sided": sign_flip_p_positive(data.test_f1_gain),
        }
        for metric in COMPUTE_METRICS:
            row[f"{metric}_saved"] = data[f"{metric}_saved"].mean()
            row[f"{metric}_saved_pct"] = data[f"{metric}_saved_pct"].mean()
        convergence_low, convergence_high = bootstrap_interval(
            data.validation_epoch_to_80pct_saved
        )
        row["validation_epoch_to_80pct_saved"] = data.validation_epoch_to_80pct_saved.mean()
        row["validation_epoch_to_80pct_saved_sd_subject"] = data.validation_epoch_to_80pct_saved.std(ddof=1)
        row["validation_epoch_to_80pct_saved_bootstrap_95_low"] = convergence_low
        row["validation_epoch_to_80pct_saved_bootstrap_95_high"] = convergence_high
        row["validation_epoch_to_80pct_saved_signflip_p_one_sided"] = sign_flip_p_positive(
            data.validation_epoch_to_80pct_saved
        )
        row["validation_epoch_to_80pct_saved_pct"] = data.validation_epoch_to_80pct_saved_pct.mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("species").reset_index(drop=True)


def plot_subject_f1(subject: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    for axis, species in zip(axes, ("minipigs", "monkeys"), strict=True):
        data = subject[subject.species.eq(species)].sort_values("subject")
        y = np.arange(len(data))
        axis.scatter(data.test_f1_gain, y, color="#0072b2", s=42, zorder=2, label="Subject")
        axis.scatter([data.test_f1_gain.mean()], [len(data)], color="#d55e00", marker="D", s=55, zorder=3, label="Mean")
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set(title=species.title(), yticks=y, yticklabels=data.subject, xlabel="Transfer − scratch test macro-F1")
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_ylabel("Subject (session- and seed-averaged)")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Paired subject-balanced test effect (positive favors transfer)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_session_f1(session: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for axis, species in zip(axes, ("minipigs", "monkeys"), strict=True):
        data = session[session.species.eq(species)]
        axis.scatter(data.scratch_test_supported_f1, data.pretrained_test_supported_f1, color="#009e73", s=35, alpha=0.85)
        axis.plot([0, 1], [0, 1], color="black", linewidth=0.8)
        axis.set(title=species.title(), xlim=(0, 1), ylim=(0, 1), xlabel="Scratch test macro-F1")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Full-pool transfer test macro-F1")
    fig.suptitle("Session means after averaging target and source seeds")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_compute_savings(subject: pd.DataFrame, output: Path) -> None:
    labels = {
        "best_step": "Best-checkpoint steps saved (%)",
        "best_windows": "Best-checkpoint windows saved (%)",
        "best_flops": "Best-checkpoint FLOPs saved (%)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharex=True)
    for axis, metric in zip(axes.flat, COMPUTE_METRICS, strict=True):
        for x, species, color in ((0, "minipigs", "#0072b2"), (1, "monkeys", "#d55e00")):
            data = subject[subject.species.eq(species)][f"{metric}_saved_pct"]
            offsets = np.linspace(-0.09, 0.09, len(data)) if len(data) > 1 else np.array([0.0])
            axis.scatter(np.full(len(data), x) + offsets, data, color=color, s=36, alpha=0.9)
            axis.scatter([x], [data.mean()], color="black", marker="D", s=42, zorder=3)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=labels[metric], xticks=[0, 1], xticklabels=["Minipigs", "Monkeys"])
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Subject means; positive values favor transfer")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_validation_convergence(subject: pd.DataFrame, output: Path) -> None:
    """Compare subject-level epochs to 80% of each run's own validation peak."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for axis, species in zip(axes, ("minipigs", "monkeys"), strict=True):
        data = subject[subject.species.eq(species)]
        axis.scatter(
            data.scratch_validation_epoch_to_80pct,
            data.pretrained_validation_epoch_to_80pct,
            color="#cc79a7", s=45,
        )
        maximum = max(
            data.scratch_validation_epoch_to_80pct.max(),
            data.pretrained_validation_epoch_to_80pct.max(),
        )
        axis.plot([0, maximum], [0, maximum], color="black", linewidth=0.8)
        axis.set(title=species.title(), xlabel="Scratch validation epochs to 80% peak")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Transfer validation epochs to 80% of own peak")
    fig.suptitle("Subject means; points below the diagonal favor faster transfer")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    api = wandb.Api()
    entity = default_entity() or api.default_entity
    if not entity:
        raise RuntimeError("Set WANDB_ENTITY or configure a default W&B entity.")
    selected_sources = selected_source_runs()
    selected_cells = compiled_cells()
    source_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    scratch_rows: list[dict[str, Any]] = []
    for species in SOURCE_GROUPS:
        source_rows += fetch_group(api, entity, SOURCE_GROUPS[species], "source", species, selected_sources, selected_cells)
        transfer_rows += fetch_group(api, entity, TRANSFER_GROUPS[species], "transfer", species, selected_sources, selected_cells)
        scratch_rows += fetch_group(api, entity, SCRATCH_GROUPS[species], "scratch", species, selected_sources, selected_cells)

    source = pd.DataFrame(source_rows, columns=RUN_COLUMNS)
    transfer = pd.DataFrame(transfer_rows, columns=RUN_COLUMNS)
    scratch = pd.DataFrame(scratch_rows, columns=RUN_COLUMNS)
    recovery = completion_audit(source, transfer, selected_cells)
    scratch_full = canonical_scratch(scratch)
    transfer = add_validation_convergence(api, entity, transfer, "transfer")
    scratch_full = add_validation_convergence(api, entity, scratch_full, "scratch 100%")
    paired = paired_effects(transfer, scratch_full)
    session, subject = reductions(paired)
    summary = species_summary(paired, session, subject)

    expected_transfer_units = transfer.groupby(["species", "subject", "recording", "target_seed"]).ngroups
    if len(paired) != len(scratch_full):
        raise RuntimeError("Every canonical scratch run should produce one paired result.")
    missing_controls = expected_transfer_units - len(scratch_full)
    print(f"Canonical 100%-data scratch controls: {len(scratch_full)}/{expected_transfer_units} transfer session/seed units")
    if missing_controls:
        missing = (transfer[["species", "subject", "recording", "target_seed"]].drop_duplicates()
            .merge(scratch_full[["species", "subject", "recording", "target_seed"]], how="left", indicator=True)
            .query("_merge != 'both'"))
        print("Unpaired transfer units (missing scratch control):")
        print(missing[["species", "subject", "recording", "target_seed"]].to_string(index=False))

    csv_root, figure_root = csv_dir(__file__), figures_dir(__file__)
    tables = {
        "source": source, "transfer": transfer, "scratch_raw": scratch,
        "recovered_first_attempts": recovery, "scratch_100pct_canonical": scratch_full,
        "paired": paired, "session_effects": session, "subject_effects": subject,
        "subject_balanced_summary": summary,
    }
    for label, table in tables.items():
        path = csv_root / f"{PREFIX}_{label}.csv"
        table.to_csv(path, index=False)
        print(f"Wrote {path}")

    figures = {
        "subject_balanced_test_f1": (plot_subject_f1, subject),
        "session_test_f1": (plot_session_f1, session),
        "compute_savings": (plot_compute_savings, subject),
        "validation_epoch_to_80pct": (plot_validation_convergence, subject),
    }
    for suffix, (plotter, table) in figures.items():
        path = figure_root / f"{PREFIX}_{suffix}.png"
        plotter(table, path)
        print(f"Wrote {path}")

    print("\n=== Subject-balanced paired results ===")
    columns = [
        "species", "n_subjects", "n_sessions", "n_session_seed_pairs",
        "scratch_test_supported_f1", "pretrained_test_supported_f1", "test_f1_gain",
        "test_f1_gain_bootstrap_95_low", "test_f1_gain_bootstrap_95_high",
        "test_f1_gain_signflip_p_one_sided", "best_step_saved_pct",
        "best_windows_saved_pct", "best_flops_saved_pct",
        "validation_epoch_to_80pct_saved",
        "validation_epoch_to_80pct_saved_bootstrap_95_low",
        "validation_epoch_to_80pct_saved_bootstrap_95_high",
        "validation_epoch_to_80pct_saved_signflip_p_one_sided",
    ]
    print(summary[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
