"""Generate exact Phase 4D retry lists from the completed-cell audit."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPECIES = ("minipigs", "monkeys")
PREFIX = "20260911-MS-transfer-lr-warmup-screen"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_list_path(species: str) -> Path:
    return (
        ROOT / "launch/phase4d" / f"phase4d-transfer-lr-warmup-{species}.jsonl"
    )


def retry_path(species: str) -> Path:
    return (
        ROOT
        / "launch/phase4d/retries"
        / f"phase4d-transfer-lr-warmup-retry-{species}.jsonl"
    )


def audit_path() -> Path:
    return ROOT / "analysis/csv" / f"{PREFIX}_runs.csv"


def load_ineligible_ids(species: str) -> tuple[set[str], Counter[str]]:
    rows: set[str] = set()
    states: Counter[str] = Counter()
    with audit_path().open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["species"] != species:
                continue
            if row["analysis_eligible"].lower() != "true":
                rows.add(row["cell_id"])
                states[row["state"]] += 1
    return rows, states


def generate(species: str) -> None:
    source = cell_list_path(species)
    source_lines = source.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in source_lines if line.strip()]
    failed_ids, states = load_ineligible_ids(species)
    known_ids = {str(record["cell_id"]) for record in records}
    if failed_ids - known_ids:
        raise RuntimeError(
            f"Audit has unknown {species} cells: "
            f"{sorted(failed_ids - known_ids)[:5]}"
        )

    selected: list[dict] = []
    for record in records:
        if str(record["cell_id"]) not in failed_ids:
            continue
        record = dict(record)
        tags = list(record.get("wandb_tags", []))
        if "retry_weights_only_false" not in tags:
            tags.append("retry_weights_only_false")
        record["wandb_tags"] = tags
        overrides = list(record.get("overrides", []))
        for index, override in enumerate(overrides):
            if override.startswith("run.tags="):
                overrides[index] = "run.tags=" + json.dumps(
                    tags, separators=(",", ":")
                )
                break
        record["overrides"] = overrides
        selected.append(record)

    if len(selected) != len(failed_ids):
        raise RuntimeError(
            f"Selected {len(selected)} of {len(failed_ids)} ineligible "
            f"{species} cells"
        )

    output = retry_path(species)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            json.dumps(record, separators=(",", ":")) for record in selected
        )
        + "\n",
        encoding="utf-8",
    )
    lock = {
        "schema": "foundry-phase4d-retry-cell-lock",
        "version": 1,
        "species": species,
        "reason": "retry cells without a valid target test metric after fixing trainer.test checkpoint loading",
        "failure_states": dict(sorted(states.items())),
        "source_cell_list": str(source.relative_to(ROOT)),
        "source_cell_list_sha256": sha256(source),
        "audit_csv": str(audit_path().relative_to(ROOT)),
        "audit_csv_sha256": sha256(audit_path()),
        "retry_cell_list": str(output.relative_to(ROOT)),
        "retry_cell_list_sha256": sha256(output),
        "retry_cell_count": len(selected),
        "selection_guard": "analysis_eligible=false",
        "retry_tag": "retry_weights_only_false",
    }
    output.with_suffix(".lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"{species}: {len(selected)} retry cells; states={dict(sorted(states.items()))}"
    )


if __name__ == "__main__":
    for species in SPECIES:
        generate(species)
