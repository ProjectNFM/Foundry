"""Analyze Phase 4B early source-checkpoint transfer with W&B.

The primary analysis uses the complete paired population shared by scratch,
all four early milestones, and Phase 4A's validation-selected-best control.
It deliberately excludes the documented missing monkey scratch cell and the
separate 5K early-checkpoint cell whose W&B summary lacks test/compute metrics.

Run with:

    uv run python analysis/20260910-MS-early-checkpoint-transfer_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

from _wandb_utils import csv_dir, default_entity, figures_dir, unwrap_summary_value


ROOT = Path(__file__).resolve().parents[1]
PROJECT = "neurosoft_supervised_pretraining"
TASK = "neurosoft_acoustic_stim_8band"
SEEDS = {42, 43, 44}
PREFIX = "20260910-MS-early-checkpoint-transfer"
SPECIES = ("minipigs", "monkeys")
EARLY_REGISTRY = ROOT / "launch/checkpoint_sets/phase4a-mila-early-checkpoints.jsonl"
EARLY_CELL_LISTS = {
    species: ROOT / "launch/phase4a" / f"phase4a-early-checkpoint-full-ft-lr1p5e3-{species}.jsonl"
    for species in SPECIES
}
BEST_CELL_LISTS = {
    species: ROOT / "launch/phase4a" / f"phase4a-downstream-lr1p5e3-{species}.jsonl"
    for species in SPECIES
}
EARLY_GROUPS = {
    "minipigs": "PHASE4A_EARLY_CHECKPOINT_FULL_FINETUNE_LR1P5E3_MINIPIGS",
    "monkeys": "PHASE4A_EARLY_CHECKPOINT_FULL_FINETUNE_LR1P5E3_MONKEYS",
}
BEST_GROUPS = {
    "minipigs": "PHASE4A_FULL_FINETUNE_LR1P5E3_MINIPIGS",
    "monkeys": "PHASE4A_FULL_FINETUNE_LR1P5E3_MONKEYS",
}
SCRATCH_GROUPS = {
    "minipigs": "NORM_GLOBAL_CONV_BIGRU_MINIPIGS_PROD_OFFLINE_16_20260902",
    "monkeys": "NORM_GLOBAL_CONV_BIGRU_MONKEYS_PROD_OFFLINE_16_20260902",
}
EXPECTED_EARLY = {"minipigs": 1440, "monkeys": 468}
EXPECTED_BEST = {"minipigs": 360, "monkeys": 117}
CONDITION_ORDER = ["scratch", "step500", "step1500", "step5000", "step15000", "best"]
CONDITION_LABELS = {
    "scratch": "Scratch\n(0)", "step500": "500", "step1500": "1.5K",
    "step5000": "5K", "step15000": "15K", "best": "Val.-selected\nbest",
}
UNIT = ["species", "subject", "recording", "target_seed"]
METRICS = ["test_supported_f1", "best_step", "best_windows", "best_flops", "best_wall_time_s"]


def nested(config: dict[str, Any], *keys: str) -> Any:
    dotted = ".".join(keys)
    if dotted in config:
        return config[dotted]
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def scalar(summary: dict[str, Any], key: str) -> float | None:
    for candidate in (f"{key}.max", key):
        try:
            return float(unwrap_summary_value(summary.get(candidate), "max"))
        except (TypeError, ValueError):
            pass
    return None


def metric_values(summary: dict[str, Any]) -> dict[str, float | None]:
    return {
        metric: scalar(
            summary,
            f"test/{TASK}_supported_f1" if metric == "test_supported_f1" else f"compute/{metric}",
        )
        for metric in METRICS
    }


def compiled_cells(paths: dict[str, Path], expected: dict[str, int]) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    for species, path in paths.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["species"] != species or row["cell_id"] in cells:
                raise RuntimeError(f"Invalid compiled identity in {path}")
            cells[str(row["cell_id"])] = row
    counts = {species: sum(row["species"] == species for row in cells.values()) for species in expected}
    if counts != expected:
        raise RuntimeError(f"Compiled-cell matrix count differs from design: {counts}")
    return cells


def milestones() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in EARLY_REGISTRY.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        result[str(row["checkpoint_id"])] = f"step{int(row['condition']['milestone_step'])}"
    if set(result.values()) != {"step500", "step1500", "step5000", "step15000"}:
        raise RuntimeError("Registry does not contain exactly the declared milestones")
    return result


def transfer_rows(api: Any, entity: str, groups: dict[str, str], cells: dict[str, dict[str, Any]], condition_by_checkpoint: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for species, group in groups.items():
        for run in api.runs(f"{entity}/{PROJECT}", filters={"group": group}, per_page=500, lazy=False):
            config, summary = dict(run.config or {}), dict(run.summary or {})
            cell_id = str(nested(config, "run", "cell_id"))
            expected = cells.get(cell_id)
            if expected is None:
                continue
            observed = {
                "species": species,
                "target_recording": nested(config, "data", "dataset_kwargs", "recording_ids"),
                "target_finetuning_seed": nested(config, "run", "seed"),
                "checkpoint_id": nested(config, "run", "checkpoint_id"),
                "checkpoint_manifest_hash": nested(config, "run", "pretrained_checkpoint_manifest_hash"),
                "source_selection_seed": nested(config, "run", "source_selection_seed"),
                "source_model_seed": nested(config, "run", "source_model_seed"),
                "transfer_regime": nested(config, "run", "pretrained_transfer_regime"),
            }
            expected_values = {
                "species": expected["species"], "target_recording": [expected["target_recording"]],
                "target_finetuning_seed": expected["target_finetuning_seed"], "checkpoint_id": expected["checkpoint_id"],
                "checkpoint_manifest_hash": expected["checkpoint_manifest_hash"],
                "source_selection_seed": expected["source_selection_seed"], "source_model_seed": expected["source_model_seed"],
                "transfer_regime": expected["transfer_regime"],
            }
            if observed != expected_values or str(run.id) != expected["wandb_run_id"]:
                raise RuntimeError(f"W&B provenance mismatch: {run.name} ({run.id})")
            checkpoint_id = str(expected["checkpoint_id"])
            rows.append({
                "run_id": str(run.id), "run_name": str(run.name), "state": run.state, "cell_id": cell_id,
                "species": species, "subject": expected["target_subject"], "recording": expected["target_recording"],
                "target_seed": expected["target_finetuning_seed"], "source_seed": expected["source_selection_seed"],
                "checkpoint_id": checkpoint_id, "condition": condition_by_checkpoint[checkpoint_id],
                **metric_values(summary),
            })
    return rows


def scratch_rows(api: Any, entity: str, early_cells: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    targets = {species: {(row["target_subject"], row["target_recording"]) for row in early_cells.values() if row["species"] == species} for species in SPECIES}
    rows: list[dict[str, Any]] = []
    for species, group in SCRATCH_GROUPS.items():
        for run in api.runs(f"{entity}/{PROJECT}", filters={"group": group}, per_page=500, lazy=False):
            config, summary = dict(run.config or {}), dict(run.summary or {})
            recordings = nested(config, "data", "dataset_kwargs", "recording_ids")
            recording = str(recordings[0]) if isinstance(recordings, list) and recordings else None
            subject = next((item[0] for item in targets[species] if item[1] == recording), None)
            seed, fraction = nested(config, "run", "seed"), nested(config, "data", "training_fraction")
            if run.state != "finished" or (subject, recording) not in targets[species] or seed not in SEEDS or fraction is None or float(fraction) != 1.0:
                continue
            rows.append({
                "species": species, "subject": subject, "recording": recording, "target_seed": int(seed),
                "condition": "scratch", "run_id": str(run.id), "run_name": str(run.name),
                **metric_values(summary),
            })
    return rows


def audit_exact_rows(rows: list[dict[str, Any]], cells: dict[str, dict[str, Any]], expected: dict[str, int], label: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    counts = {species: int(frame.species.eq(species).sum()) for species in expected}
    if counts != expected or set(frame.cell_id) != set(cells):
        raise RuntimeError(f"{label} W&B matrix is incomplete or contains undeclared cells: {counts}")
    if not frame.state.eq("finished").all():
        raise RuntimeError(f"{label} has non-finished declared runs")
    return frame


def aggregate_transfer(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = frame.dropna(subset=METRICS).copy()
    grouped = valid.groupby(UNIT + ["condition"], as_index=False).agg(
        source_replicates=("source_seed", "nunique"), **{metric: (metric, "mean") for metric in METRICS}
    )
    return grouped[grouped.source_replicates.eq(3)].copy(), grouped[~grouped.source_replicates.eq(3)].copy()


def common_units(scratch: pd.DataFrame, early: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    sets = [set(map(tuple, scratch[UNIT].drop_duplicates().to_numpy()))]
    sets += [set(map(tuple, early.loc[early.condition.eq(condition), UNIT].drop_duplicates().to_numpy())) for condition in CONDITION_ORDER[1:5]]
    sets += [set(map(tuple, best[UNIT].drop_duplicates().to_numpy()))]
    common = pd.DataFrame(sorted(set.intersection(*sets)), columns=UNIT)
    if common.groupby("species").size().to_dict() != {"minipigs": 120, "monkeys": 37}:
        raise RuntimeError("Common paired sample was not the predeclared 120 minipig / 37 monkey units")
    return common


def subject_effects(units: pd.DataFrame) -> pd.DataFrame:
    sessions = units.groupby(["species", "condition", "subject", "recording"], as_index=False).mean(numeric_only=True)
    return sessions.groupby(["species", "condition", "subject"], as_index=False).mean(numeric_only=True)


def paired_effects(units: pd.DataFrame) -> pd.DataFrame:
    scratch = units[units.condition.eq("scratch")][UNIT + METRICS].rename(columns={metric: f"scratch_{metric}" for metric in METRICS})
    paired = units[~units.condition.eq("scratch")].merge(scratch, on=UNIT, validate="many_to_one")
    paired["test_f1_gain"] = paired.test_supported_f1 - paired.scratch_test_supported_f1
    for metric in METRICS[1:]:
        paired[f"{metric}_saved_pct"] = 100 * (paired[f"scratch_{metric}"] - paired[metric]) / paired[f"scratch_{metric}"]
    sessions = paired.groupby(["species", "condition", "subject", "recording"], as_index=False).mean(numeric_only=True)
    return sessions.groupby(["species", "condition", "subject"], as_index=False).mean(numeric_only=True)


def summarize(subjects: pd.DataFrame, metric: str) -> pd.DataFrame:
    rng, records = np.random.default_rng(20260910), []
    for (species, condition), data in subjects.groupby(["species", "condition"], sort=False):
        values = data[metric].to_numpy(float)
        bootstrap = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
        records.append({"species": species, "condition": condition, "n_subjects": len(values), "mean": values.mean(), "ci_low": np.quantile(bootstrap, .025), "ci_high": np.quantile(bootstrap, .975)})
    return pd.DataFrame(records)


def plot_series(summary: pd.DataFrame, output: Path, ylabel: str, title: str, *, include_scratch: bool, zero_line: bool = False) -> None:
    order = CONDITION_ORDER if include_scratch else CONDITION_ORDER[1:]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for axis, species in zip(axes, SPECIES, strict=True):
        data = summary[summary.species.eq(species)].set_index("condition").reindex(order).reset_index()
        x = np.arange(len(data))
        axis.errorbar(x, data["mean"], yerr=[data["mean"] - data["ci_low"], data["ci_high"] - data["mean"]], marker="o", capsize=3, linewidth=1.8)
        if zero_line:
            axis.axhline(0, color="black", linewidth=.8)
        axis.set(xticks=x, xticklabels=[CONDITION_LABELS[item] for item in data.condition], title=species.title(), ylabel=ylabel)
        axis.grid(axis="y", alpha=.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_subjects(subjects: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for axis, species in zip(axes, SPECIES, strict=True):
        for subject, series in subjects[subjects.species.eq(species)].groupby("subject"):
            series = series.set_index("condition").reindex(CONDITION_ORDER).reset_index()
            axis.plot(np.arange(len(series)), series.test_supported_f1, marker="o", alpha=.7, label=subject)
        axis.set(xticks=np.arange(len(CONDITION_ORDER)), xticklabels=[CONDITION_LABELS[item] for item in CONDITION_ORDER], title=species.title(), ylabel="Test supported macro-F1")
        axis.grid(axis="y", alpha=.25)
        axis.legend(title="Subject", fontsize=8, ncol=2)
    fig.suptitle("Subject-level absolute-performance trajectories")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    api = wandb.Api()
    entity = default_entity() or api.default_entity
    if not entity:
        raise RuntimeError("Set WANDB_ENTITY or configure a default W&B entity.")
    early_cells = compiled_cells(EARLY_CELL_LISTS, EXPECTED_EARLY)
    best_cells = compiled_cells(BEST_CELL_LISTS, EXPECTED_BEST)
    early_raw = audit_exact_rows(transfer_rows(api, entity, EARLY_GROUPS, early_cells, milestones()), early_cells, EXPECTED_EARLY, "Early checkpoint")
    best_raw = audit_exact_rows(transfer_rows(api, entity, BEST_GROUPS, best_cells, {row["checkpoint_id"]: "best" for row in best_cells.values()}), best_cells, EXPECTED_BEST, "Best checkpoint")
    scratch = pd.DataFrame(scratch_rows(api, entity, early_cells))
    if scratch[METRICS].isna().any().any():
        raise RuntimeError("A retained scratch control lacks test F1 or compute accounting")
    early, incomplete_early = aggregate_transfer(early_raw)
    best, incomplete_best = aggregate_transfer(best_raw)
    if not incomplete_best.empty:
        raise RuntimeError("The validation-selected-best control lacks a source replicate")
    common = common_units(scratch, early, best)
    units = pd.concat([scratch, early, best], ignore_index=True).merge(common, on=UNIT, how="inner", validate="many_to_one")
    subjects, paired_subjects = subject_effects(units), paired_effects(units)
    absolute_summary = summarize(subjects, "test_supported_f1")
    gain_summary = summarize(paired_subjects, "test_f1_gain")
    efficiency_summary = summarize(paired_subjects, "best_step_saved_pct")
    out_csv, out_figures = csv_dir(__file__), figures_dir(__file__)
    for name, frame in {"early_transfer": early_raw, "best_transfer": best_raw, "scratch": scratch, "incomplete_early_units": incomplete_early, "common_units": common, "condition_units": units, "subject_conditions": subjects, "subject_paired_effects": paired_subjects, "absolute_f1_summary": absolute_summary, "paired_f1_gain_summary": gain_summary, "best_step_efficiency_summary": efficiency_summary}.items():
        frame.to_csv(out_csv / f"{PREFIX}_{name}.csv", index=False)
    plot_series(absolute_summary, out_figures / f"{PREFIX}_absolute_test_f1_by_pretraining.png", "Test supported macro-F1", "Absolute performance from scratch through source pretraining", include_scratch=True)
    plot_series(gain_summary, out_figures / f"{PREFIX}_paired_test_f1_gain_by_pretraining.png", "Transfer − scratch test macro-F1", "Paired, subject-balanced transfer effect", include_scratch=False, zero_line=True)
    plot_series(efficiency_summary, out_figures / f"{PREFIX}_best_step_efficiency_by_pretraining.png", "Best-step reduction vs scratch (%)", "Paired downstream optimizer-step efficiency", include_scratch=False, zero_line=True)
    plot_subjects(subjects, out_figures / f"{PREFIX}_subject_test_f1_trajectories.png")
    print("=== Common paired sample ===")
    print(common.groupby("species").size().rename("session_seed_units").to_string())
    print("\n=== Excluded incomplete early units ===")
    print(incomplete_early[UNIT + ["condition", "source_replicates"]].to_string(index=False))
    for title, frame in (("Absolute subject-balanced test supported macro-F1 (95% bootstrap CI)", absolute_summary), ("Paired F1 gain versus scratch (95% bootstrap CI)", gain_summary), ("Paired best-step reduction versus scratch (95% bootstrap CI)", efficiency_summary)):
        print(f"\n=== {title} ===")
        print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
