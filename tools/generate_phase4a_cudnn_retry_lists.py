"""Generate exact Phase 4A retry lists for the cuDNN/V100 failures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ERROR = (
    "RuntimeError: cuDNN version 92000 is not compatible with devices with "
    "SM < 7.5."
)
TEST_METRIC = "test/neurosoft_acoustic_stim_8band_supported_f1"
SPECIES = ("minipigs", "monkeys")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(species: str) -> Path:
    return ROOT / f"launch/phase4a/phase4a-downstream-lr1p5e3-{species}.jsonl"


def run_root(species: str) -> Path:
    return Path(
        "/network/scratch/s/sobralm/runs/"
        f"PHASE4A_FULL_FINETUNE_LR1P5E3_{species.upper()}"
    )


def output_path(species: str) -> Path:
    return (
        ROOT
        / "launch/phase4a/retries"
        / f"corrected-lr1p5e3-cudnn-v100-{species}.jsonl"
    )


def failed_cell_ids(species: str) -> set[str]:
    root = run_root(species)
    failed: set[str] = set()
    for log_path in root.glob("*/wandb/run-*/files/output.log"):
        if ERROR in log_path.read_text(encoding="utf-8", errors="replace"):
            failed.add(log_path.parents[3].name)
    return failed


def validate_failed_cell(species: str, cell_id: str) -> None:
    cell_root = run_root(species) / cell_id
    if (cell_root / "checkpoints/last.ckpt").exists():
        raise RuntimeError(f"Refusing to retry cell with last.ckpt: {cell_id}")
    summaries = sorted(cell_root.glob("wandb/run-*/files/wandb-summary.json"))
    if not summaries:
        raise RuntimeError(f"Cell has no local W&B summary: {cell_id}")
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if TEST_METRIC in summary:
            raise RuntimeError(
                f"Refusing to retry cell with test metric: {cell_id}"
            )


def generate(species: str) -> tuple[int, Path]:
    source = source_path(species)
    source_lines = source.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in source_lines]
    failed = failed_cell_ids(species)
    known = {str(record["cell_id"]) for record in records}
    unknown = failed - known
    if unknown:
        raise RuntimeError(
            f"Failure logs do not match compiled cells: {sorted(unknown)}"
        )
    for cell_id in sorted(failed):
        validate_failed_cell(species, cell_id)

    selected_lines = [
        line
        for line, record in zip(source_lines, records, strict=True)
        if str(record["cell_id"]) in failed
    ]
    if len(selected_lines) != len(failed):
        raise RuntimeError(f"Duplicate or missing compiled cells for {species}")

    output = output_path(species)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(selected_lines) + "\n", encoding="utf-8")
    lock = {
        "schema_version": 1,
        "species": species,
        "reason": "cuDNN 9.2 rejects V100 GPUs with compute capability 7.0",
        "failure_signature": ERROR,
        "source_cell_list": str(source.relative_to(ROOT)),
        "source_cell_list_sha256": sha256(source),
        "retry_cell_list": str(output.relative_to(ROOT)),
        "retry_cell_list_sha256": sha256(output),
        "retry_cell_count": len(selected_lines),
        "selection_guards": {
            "exact_failure_signature_required": True,
            "last_checkpoint_must_be_absent": True,
            "test_metric_must_be_absent": True,
        },
    }
    lock_path = output.with_suffix(".lock.json")
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return len(selected_lines), output


def main() -> None:
    for species in SPECIES:
        count, output = generate(species)
        print(f"{species}: wrote {count} cells to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
