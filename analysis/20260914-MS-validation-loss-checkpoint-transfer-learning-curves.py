"""Audit Phase 4E source-pretraining runs and validation curves.

Fetches the immutable set of W&B runs, verifies completion and the expected
10,000-step validation history, cross-checks best-loss checkpoint manifests,
and writes a run table plus a five-panel validation-loss figure.

Usage:
    uv run python analysis/20260914-MS-validation-loss-checkpoint-transfer-learning-curves.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb

from _wandb_utils import csv_dir, default_entity, figures_dir

PREFIX = "20260914-MS-validation-loss-checkpoint-transfer-learning-curves"
PROJECT = "neurosoft_supervised_pretraining"
MONKEY_GROUP = "PHASE4E_VALIDATION_LOSS_SOURCE_MONKEYS"
MONKEY_RUN_ROOT = Path(
    "/network/scratch/s/sobralm/runs/PHASE4E_VALIDATION_LOSS_SOURCE_MONKEYS"
)
MONKEY_RUN_IDS = {
    "src_mk_sub-01_s42_m42": "f7a0j1cd",
    "src_mk_sub-01_s43_m43": "je5zv0aa",
    "src_mk_sub-01_s44_m44": "utbkpxqk",
    "src_mk_sub-02_s42_m42": "3g4g6g7z",
    "src_mk_sub-02_s43_m43": "dpykloq5",
    "src_mk_sub-02_s44_m44": "gripx11w",
    "src_mk_sub-03_s42_m42": "3r33nzft",
    "src_mk_sub-03_s43_m43": "s3y9wnon",
    "src_mk_sub-03_s44_m44": "rwwy5dt0",
    "src_mk_sub-04_s42_m42": "vf6yrj8s",
    "src_mk_sub-04_s43_m43": "0tmc41aq",
    "src_mk_sub-04_s44_m44": "dlst881m",
    "src_mk_sub-05_s42_m42": "hwdzv9vv",
    "src_mk_sub-05_s43_m43": "6fcpopwq",
    "src_mk_sub-05_s44_m44": "qewpms97",
}
MINIPIG_GROUP = "PHASE4E_VALIDATION_LOSS_SOURCE_MINIPIGS"
MINIPIG_RUN_ROOT = Path(
    "/network/scratch/s/sobralm/runs/PHASE4E_VALIDATION_LOSS_SOURCE_MINIPIGS"
)
MINIPIG_RUN_IDS = {
    "src_mp_sub-01_s42_m42": "wa3w089g",
    "src_mp_sub-01_s43_m43": "4uui36hs",
    "src_mp_sub-01_s44_m44": "e4x6r37e",
    "src_mp_sub-02_s42_m42": "nd01iyg5",
    "src_mp_sub-02_s43_m43": "z9ppfhmk",
    "src_mp_sub-02_s44_m44": "g0osgbwi",
    "src_mp_sub-03_s42_m42": "w1h5pems",
    "src_mp_sub-03_s43_m43": "vlrkyjiz",
    "src_mp_sub-03_s44_m44": "v8z5sq4d",
    "src_mp_sub-04_s42_m42": "81fyz0nq",
    "src_mp_sub-04_s43_m43": "62jrclt6",
    "src_mp_sub-04_s44_m44": "g38pfpfs",
    "src_mp_sub-05_s42_m42": "cpzjme34",
    "src_mp_sub-05_s43_m43": "3n4dhvkx",
    "src_mp_sub-05_s44_m44": "y24kzdp2",
    "src_mp_sub-06_s42_m42": "5cwm43yd",
    "src_mp_sub-06_s43_m43": "halaot6i",
    "src_mp_sub-06_s44_m44": "phztnkq5",
    "src_mp_sub-07_s42_m42": "b05o39bm",
    "src_mp_sub-07_s43_m43": "wwmbxg8d",
    "src_mp_sub-07_s44_m44": "86x8ah0o",
}
SOURCES = {
    "monkeys": (MONKEY_GROUP, MONKEY_RUN_ROOT, MONKEY_RUN_IDS),
    "minipigs": (MINIPIG_GROUP, MINIPIG_RUN_ROOT, MINIPIG_RUN_IDS),
}
SPECIES_LABEL = {"monkeys": "monkey", "minipigs": "minipig"}
EXPECTED_MILESTONES = {100, 300, 1_000, 3_000, 10_000}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def best_manifest(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    paths = sorted((run_dir / "manifests").glob("best-*.json"))
    if len(paths) != 1:
        raise RuntimeError(
            f"{run_dir.name}: expected one best manifest, found {len(paths)}"
        )
    return paths[0], json.loads(paths[0].read_text())


def validate_artifacts(
    run_dir: Path, run_id: str
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    manifest_path, manifest = best_manifest(run_dir)
    manifest_run_id = manifest.get("wandb", {}).get("run_id")
    if manifest_run_id != run_id:
        issues.append(f"manifest run ID {manifest_run_id!r} != W&B {run_id!r}")

    selection = manifest.get("selection", {})
    if selection.get("monitor") != "val/loss":
        issues.append(f"best manifest monitors {selection.get('monitor')!r}")

    trained = manifest.get("trained_on", {})
    best_step = trained.get("optimizer_steps")
    best_value = selection.get("monitor_value")

    checkpoint_name = Path(manifest.get("checkpoint", {}).get("path", "")).name
    checkpoint = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint.is_file():
        issues.append(f"missing best checkpoint {checkpoint_name}")
    elif sha256(checkpoint) != manifest.get("checkpoint", {}).get("sha256"):
        issues.append(f"best checkpoint hash mismatch ({manifest_path.name})")

    milestone_steps: set[int] = set()
    for path in sorted((run_dir / "manifests").glob("milestone-*.json")):
        payload = json.loads(path.read_text())
        step = payload.get("trained_on", {}).get("optimizer_steps")
        if isinstance(step, int):
            milestone_steps.add(step)
        checkpoint_name = Path(
            payload.get("checkpoint", {}).get("path", "")
        ).name
        checkpoint = run_dir / "checkpoints" / checkpoint_name
        if not checkpoint.is_file():
            issues.append(f"missing milestone checkpoint {checkpoint_name}")
        elif sha256(checkpoint) != payload.get("checkpoint", {}).get("sha256"):
            issues.append(f"milestone checkpoint hash mismatch ({path.name})")
    if milestone_steps != EXPECTED_MILESTONES:
        issues.append(f"milestone steps {sorted(milestone_steps)}")

    return {
        "manifest_best_step": best_step,
        "manifest_best_loss": best_value,
    }, issues


def fetch_history(run: Any) -> pd.DataFrame:
    rows = []
    keys = ["trainer/global_step", "val/loss"]
    for item in run.scan_history(keys=keys, page_size=1_000):
        step = item.get("trainer/global_step")
        loss = item.get("val/loss")
        if step is None or loss is None:
            continue
        rows.append({"optimizer_step": int(step), "val_loss": float(loss)})
    if not rows:
        return pd.DataFrame(columns=["optimizer_step", "val_loss"])
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["optimizer_step"], keep="last")
        .sort_values("optimizer_step")
        .reset_index(drop=True)
    )


def audit(
    entity: str | None,
    species: str,
    group: str,
    run_root: Path,
    expected_run_ids: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    api = wandb.Api()
    entity = entity or api.default_entity
    if not entity:
        raise RuntimeError("Could not resolve W&B entity; set WANDB_ENTITY.")
    runs = list(
        api.runs(
            f"{entity}/{PROJECT}",
            filters={"group": group},
            per_page=100,
            lazy=False,
        )
    )
    by_name = {run.name: run for run in runs}
    issues: list[str] = []
    unexpected = sorted(set(by_name) - set(expected_run_ids))
    missing = sorted(set(expected_run_ids) - set(by_name))
    if unexpected:
        issues.append(f"unexpected W&B runs: {unexpected}")
    if missing:
        issues.append(f"missing W&B runs: {missing}")

    summaries: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    for name, expected_id in sorted(expected_run_ids.items()):
        run = by_name.get(name)
        if run is None:
            continue
        run_issues: list[str] = []
        if run.id != expected_id:
            run_issues.append(f"W&B ID {run.id} != expected {expected_id}")
        if run.state != "finished":
            run_issues.append(f"W&B state is {run.state!r}")

        artifact, artifact_issues = validate_artifacts(run_root / name, run.id)
        run_issues.extend(artifact_issues)
        history = fetch_history(run)
        if history.empty:
            run_issues.append("no validation-loss history")
            first_loss = final_loss = curve_best_loss = math.nan
            curve_best_step = max_step = None
            max_gap = None
        else:
            finite = history["val_loss"].map(math.isfinite).all()
            if not finite:
                run_issues.append("non-finite validation loss")
            max_step = int(history["optimizer_step"].max())
            # Lightning applies an integer val_check_interval within each
            # epoch, resetting its batch counter at epoch boundaries. W&B's
            # validation row also records the zero-based pre-increment global
            # step, so the cadence is near 100 rather than the exact global
            # sequence 100, 200, ..., 10000.
            cadence = history["optimizer_step"].diff().dropna()
            max_gap = int(cadence.max()) if not cadence.empty else None
            if max_gap is not None and 10_000 - max_step > max_gap:
                run_issues.append(
                    f"validation coverage ends too early at step {max_step}"
                )
            if len(history) < 90:
                run_issues.append(
                    f"only {len(history)} validation points, expected at least 90"
                )
            if max_gap is not None and max_gap > 200:
                run_issues.append(f"validation cadence gap is {max_gap} steps")
            first_loss = float(history.iloc[0].val_loss)
            final_loss = float(history.iloc[-1].val_loss)
            best_index = history["val_loss"].idxmin()
            curve_best_loss = float(history.loc[best_index, "val_loss"])
            curve_best_step = int(history.loc[best_index, "optimizer_step"])
            if final_loss >= first_loss:
                run_issues.append(
                    "validation loss did not improve from first to final"
                )
            manifest_loss = artifact["manifest_best_loss"]
            manifest_step = artifact["manifest_best_step"]
            if manifest_loss is None or not math.isclose(
                float(manifest_loss),
                curve_best_loss,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                run_issues.append(
                    f"manifest best loss {manifest_loss} != curve minimum {curve_best_loss:.7g}"
                )
            if manifest_step != curve_best_step + 1:
                run_issues.append(
                    f"manifest best step {manifest_step} != logged curve step + 1 "
                    f"({curve_best_step + 1})"
                )

        subject = name.split("_")[2]
        seed = int(name.rsplit("_s", 1)[1].split("_", 1)[0])
        summary = {
            "species": species,
            "run_id": run.id,
            "run_name": name,
            "state": run.state,
            "excluded_subject": subject,
            "seed": seed,
            "validation_points": len(history),
            "last_step": max_step,
            "max_validation_gap": max_gap,
            "first_val_loss": first_loss,
            "best_val_loss": curve_best_loss,
            "best_step": curve_best_step,
            "final_val_loss": final_loss,
            "relative_improvement": (
                (first_loss - curve_best_loss) / first_loss
                if math.isfinite(first_loss) and first_loss != 0
                else math.nan
            ),
            "final_minus_best": final_loss - curve_best_loss,
            "issues": "; ".join(run_issues),
        }
        summaries.append(summary)
        if run_issues:
            issues.append(f"{name}: {summary['issues']}")
        if not history.empty:
            histories.append(
                history.assign(
                    run_id=run.id, run_name=name, subject=subject, seed=seed
                )
            )

    return (
        pd.DataFrame(summaries),
        pd.concat(histories, ignore_index=True),
        issues,
    )


def plot_curves(history: pd.DataFrame, output: Path, species: str) -> None:
    subjects = sorted(history["subject"].unique())
    fig, axes = plt.subplots(
        1, len(subjects), figsize=(3.6 * len(subjects), 4), sharey=True
    )
    colors = {42: "#1f77b4", 43: "#ff7f0e", 44: "#2ca02c"}
    for axis, subject in zip(axes, subjects, strict=True):
        subset = history[history.subject == subject]
        for seed, curve in subset.groupby("seed"):
            axis.plot(
                curve.optimizer_step,
                curve.val_loss,
                label=f"seed {seed}",
                color=colors[int(seed)],
                linewidth=1.4,
                alpha=0.9,
            )
            best = curve.loc[curve.val_loss.idxmin()]
            axis.scatter(
                best.optimizer_step,
                best.val_loss,
                color=colors[int(seed)],
                s=18,
            )
        axis.set_title(f"Excluded {subject}")
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Source validation loss")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"Phase 4E {SPECIES_LABEL[species]} source pretraining: "
        "validation-loss trajectories"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=default_entity())
    parser.add_argument("--species", choices=["all", *SOURCES], default="all")
    args = parser.parse_args()

    selected = list(SOURCES) if args.species == "all" else [args.species]
    summaries: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    issues: list[str] = []
    figure_paths: list[Path] = []
    for species in selected:
        group, run_root, expected_run_ids = SOURCES[species]
        species_summary, species_history, species_issues = audit(
            args.entity, species, group, run_root, expected_run_ids
        )
        summaries.append(species_summary)
        histories.append(species_history.assign(species=species))
        issues.extend(species_issues)
        figure_path = (
            figures_dir(__file__)
            / f"{PREFIX}_{SPECIES_LABEL[species]}_source_val_loss.png"
        )
        plot_curves(species_history, figure_path, species)
        figure_paths.append(figure_path)

    summary = pd.concat(summaries, ignore_index=True)
    history = pd.concat(histories, ignore_index=True)
    artifact_scope = (
        "source"
        if args.species == "all"
        else f"{SPECIES_LABEL[args.species]}_source"
    )
    summary_path = csv_dir(__file__) / f"{PREFIX}_{artifact_scope}_summary.csv"
    history_path = csv_dir(__file__) / f"{PREFIX}_{artifact_scope}_curves.csv"
    summary.to_csv(summary_path, index=False)
    history.to_csv(history_path, index=False)

    display = summary[
        [
            "run_name",
            "run_id",
            "state",
            "validation_points",
            "best_step",
            "best_val_loss",
            "final_val_loss",
            "final_minus_best",
            "issues",
        ]
    ]
    print(
        display.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    expected_count = sum(len(SOURCES[item][2]) for item in selected)
    print(f"\nW&B runs: {len(summary)}/{expected_count}")
    print(
        f"Finished without audit issues: {(summary.issues == '').sum()}/{expected_count}"
    )
    print(f"Summary CSV: {summary_path}")
    print(f"Curves CSV: {history_path}")
    for figure_path in figure_paths:
        print(f"Figure: {figure_path}")
    if issues:
        print("\nAUDIT ISSUES:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
