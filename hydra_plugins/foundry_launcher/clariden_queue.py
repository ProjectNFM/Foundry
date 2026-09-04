"""Filesystem-backed work queue for Clariden node-pool allocations.

The queue deliberately uses one JSON file per cell.  Claims are directory
renames on one filesystem, so workers never coordinate through an in-memory
queue or rewrite a shared JSON document.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


QUEUE_STATES = (
    "pending",
    "running",
    "succeeded",
    "failed",
    "not_started_due_to_drain",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_cell_id(overrides: Sequence[str]) -> str:
    """Return a stable ID for one ordered Hydra override vector."""
    encoded = json.dumps(
        list(overrides), ensure_ascii=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True))
    os.replace(temporary, path)


class ClaridenFileQueue:
    """Race-safe queue whose state is encoded by per-cell file location."""

    def __init__(self, manifests_dir: str | Path) -> None:
        self.manifests_dir = Path(manifests_dir)
        self.root = self.manifests_dir / "clariden-queue"
        self.attempts_dir = self.manifests_dir / "clariden-attempts"
        self.index_path = self.manifests_dir / "clariden-queue.jsonl"

    @property
    def initialized(self) -> bool:
        return self.index_path.is_file() and all(
            (self.root / state).is_dir() for state in QUEUE_STATES
        )

    def initialize(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Create a new queue.  Existing queue state is never overwritten."""
        if self.root.exists() or self.index_path.exists():
            raise RuntimeError(f"Clariden queue already exists: {self.root}")

        self.root.mkdir(parents=True)
        self.attempts_dir.mkdir()
        for state in QUEUE_STATES:
            (self.root / state).mkdir()

        index_lines: list[str] = []
        seen: set[str] = set()
        for position, source_record in enumerate(records):
            record = dict(source_record)
            cell_id = str(record["cell_id"])
            if cell_id in seen:
                raise ValueError(
                    f"Duplicate Clariden cell override vector: {cell_id}"
                )
            seen.add(cell_id)
            record.update(
                {
                    "position": position,
                    "state": "pending",
                    "attempt": 0,
                    "created_at": record.get("created_at", utc_now()),
                    "updated_at": utc_now(),
                }
            )
            _atomic_write(self._path("pending", cell_id), record)
            index_lines.append(
                json.dumps(
                    {
                        "cell_id": cell_id,
                        "position": position,
                        "overrides": record["overrides"],
                    },
                    sort_keys=True,
                )
            )

        self.index_path.write_text("\n".join(index_lines) + "\n")

    def _path(self, state: str, cell_id: str) -> Path:
        return self.root / state / f"{cell_id}.json"

    def _read(self, path: Path, state: str | None = None) -> dict[str, Any]:
        record = json.loads(path.read_text())
        if state is not None:
            # The directory move is the atomic state transition and is the
            # source of truth if a worker died before refreshing JSON content.
            record["state"] = state
        return record

    def claim(self, worker: Mapping[str, Any]) -> dict[str, Any] | None:
        """Atomically claim the next pending cell, or return ``None``."""
        pending_dir = self.root / "pending"
        for source in sorted(pending_dir.glob("*.json")):
            destination = self.root / "running" / source.name
            try:
                os.replace(source, destination)
            except FileNotFoundError:
                continue

            record = self._read(destination, "running")
            record["attempt"] = int(record.get("attempt", 0)) + 1
            record["started_at"] = utc_now()
            record["updated_at"] = record["started_at"]
            record["worker"] = dict(worker)
            record.update(dict(worker))
            record["exit_status"] = None
            record["failure_classification"] = None
            _atomic_write(destination, record)
            return record
        return None

    def finish(
        self,
        cell_id: str,
        *,
        succeeded: bool,
        exit_status: int,
        failure_classification: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Move a running cell to its terminal state and persist an attempt."""
        source = self._path("running", cell_id)
        state = "succeeded" if succeeded else "failed"
        destination = self._path(state, cell_id)
        record = self._read(source, state)
        record["finished_at"] = utc_now()
        record["updated_at"] = record["finished_at"]
        record["exit_status"] = int(exit_status)
        record["failure_classification"] = failure_classification
        if extra:
            record.update(dict(extra))
        # Persist the complete terminal record before the directory rename that
        # publishes its state. If a worker dies first, resume still sees a
        # recoverable running record; after the rename, readers see complete JSON.
        _atomic_write(source, record)
        os.replace(source, destination)
        self._write_attempt(record)
        return record

    def update_running(
        self, cell_id: str, extra: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Atomically add provenance known after claim but before execution."""
        path = self._path("running", cell_id)
        record = self._read(path, "running")
        record.update(dict(extra))
        record["updated_at"] = utc_now()
        _atomic_write(path, record)
        return record

    def drain_pending(self) -> int:
        """Atomically preserve all currently pending cells for a later resume."""
        moved = 0
        for source in sorted((self.root / "pending").glob("*.json")):
            destination = self.root / "not_started_due_to_drain" / source.name
            try:
                os.replace(source, destination)
            except FileNotFoundError:
                continue
            record = self._read(destination, "not_started_due_to_drain")
            record["updated_at"] = utc_now()
            record["failure_classification"] = "allocation_draining"
            _atomic_write(destination, record)
            moved += 1
        return moved

    def requeue_for_resume(self, *, retry_failed: bool) -> dict[str, int]:
        """Requeue unfinished work without touching successful cells."""
        sources = ["not_started_due_to_drain", "running"]
        if retry_failed:
            sources.append("failed")

        counts = {state: 0 for state in sources}
        for state in sources:
            for source in sorted((self.root / state).glob("*.json")):
                destination = self.root / "pending" / source.name
                try:
                    os.replace(source, destination)
                except FileNotFoundError:
                    continue
                record = self._read(destination, "pending")
                record["updated_at"] = utc_now()
                if state == "running":
                    interrupted = dict(record)
                    interrupted["state"] = "failed"
                    interrupted["finished_at"] = record["updated_at"]
                    interrupted["failure_classification"] = (
                        "previous_allocation_interrupted"
                    )
                    self._write_attempt(interrupted)
                    record["failure_classification"] = (
                        "previous_allocation_interrupted"
                    )
                _atomic_write(destination, record)
                counts[state] += 1
        return counts

    def records(self, state: str | None = None) -> list[dict[str, Any]]:
        states = [state] if state else list(QUEUE_STATES)
        records: list[dict[str, Any]] = []
        for item_state in states:
            records.extend(
                self._read(path, item_state)
                for path in sorted((self.root / item_state).glob("*.json"))
            )
        return sorted(records, key=lambda item: int(item["position"]))

    def counts(self) -> dict[str, int]:
        return {
            state: sum(1 for _ in (self.root / state).glob("*.json"))
            for state in QUEUE_STATES
        }

    def has_records(self, state: str) -> bool:
        """Return whether a state contains work without scanning every state."""
        if state not in QUEUE_STATES:
            raise ValueError(f"Unknown Clariden queue state: {state}")
        return next((self.root / state).glob("*.json"), None) is not None

    def _write_attempt(self, record: Mapping[str, Any]) -> None:
        path = self.attempts_dir / (
            f"{record['cell_id']}-attempt-{int(record['attempt']):03d}.json"
        )
        _atomic_write(path, record)
