"""Immutable source snapshots for queued Hydra jobs.

Creates a Git-archive-based, read-only source bundle at submission time so
that every multirun task executes the exact code that was committed when the
sweep was launched — regardless of later branch switches or edits.
"""

import datetime
import hashlib
import json
import logging
import os
import platform
import stat
import subprocess
import sys
import tarfile
import uuid
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

COMPLETION_MARKER = ".snapshot_complete"
STAGING_MARKER = ".snapshot_staging"

REQUIRED_SOURCE_ENTRIES = ("main.py", "foundry", "hydra_plugins", "configs")


@dataclass(frozen=True)
class LaunchSnapshot:
    """Immutable descriptor for a sealed source bundle."""

    bundle_dir: str
    source_dir: str
    manifest_path: str
    git_sha: str
    git_branch: str
    source_digest: str
    base_config_path: str
    environment_fingerprint: str
    bundle_id: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "LaunchSnapshot":
        return cls(**json.loads(text))

    @classmethod
    def from_file(cls, path: str | Path) -> "LaunchSnapshot":
        return cls.from_json(Path(path).read_text())


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: str | Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (rc={result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def _validate_clean_repo(project_root: Path) -> tuple[str, str]:
    """Validate the repo is clean and return ``(full_sha, branch)``."""
    _git(["rev-parse", "--show-toplevel"], project_root)
    full_sha = _git(["rev-parse", "HEAD"], project_root)

    try:
        _git(["diff", "--quiet"], project_root)
    except RuntimeError:
        dirty = _git(["diff", "--name-only"], project_root)
        raise RuntimeError(
            "Snapshot aborted: working tree has unstaged changes.\n"
            f"Modified files:\n{dirty}\n"
            "Commit or stash these changes before launching."
        )

    try:
        _git(["diff", "--cached", "--quiet"], project_root)
    except RuntimeError:
        staged = _git(["diff", "--cached", "--name-only"], project_root)
        raise RuntimeError(
            "Snapshot aborted: index has staged-but-uncommitted changes.\n"
            f"Staged files:\n{staged}\n"
            "Commit or reset these changes before launching."
        )

    porcelain = _git(
        ["status", "--porcelain", "--untracked-files=all"], project_root
    )
    if porcelain:
        raise RuntimeError(
            "Snapshot aborted: working tree has untracked files.\n"
            f"Untracked files:\n{porcelain}\n"
            "Commit, remove, or add these files to .gitignore before launching."
        )

    try:
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], project_root)
    except RuntimeError:
        branch = "HEAD"

    return full_sha, branch


# ---------------------------------------------------------------------------
# Digest helpers
# ---------------------------------------------------------------------------


def _compute_tree_digest(root: Path) -> str:
    """SHA-256 over sorted ``(relative_path, file_sha256)`` pairs."""
    entries: list[tuple[str, str]] = []
    for fpath in sorted(root.rglob("*")):
        if not fpath.is_file():
            continue
        h = hashlib.sha256(fpath.read_bytes()).hexdigest()
        entries.append((str(fpath.relative_to(root)), h))

    composite = hashlib.sha256()
    for relpath, fhash in entries:
        composite.update(f"{relpath}\0{fhash}\n".encode())
    return composite.hexdigest()


def _file_hash(path: Path) -> str | None:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


# ---------------------------------------------------------------------------
# Environment fingerprint
# ---------------------------------------------------------------------------


def _build_environment_fingerprint(project_root: Path) -> dict[str, Any]:
    fp: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    pyproject_hash = _file_hash(project_root / "pyproject.toml")
    if pyproject_hash:
        fp["pyproject_toml_sha256"] = pyproject_hash
    lock_hash = _file_hash(project_root / "uv.lock")
    if lock_hash:
        fp["uv_lock_sha256"] = lock_hash
    return fp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_snapshot(
    project_root: Path,
    snapshot_root: Path,
    sweep_name: str,
    job_overrides: Sequence[Sequence[str]],
    hydra_cfg: DictConfig,
    *,
    require_clean_git: bool = True,
) -> LaunchSnapshot:
    """Create a sealed source bundle from the current Git commit.

    Returns a ``LaunchSnapshot`` with absolute paths and identifiers.
    """
    project_root = project_root.resolve()

    if require_clean_git:
        full_sha, branch = _validate_clean_repo(project_root)
    else:
        full_sha = _git(["rev-parse", "HEAD"], project_root)
        try:
            branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], project_root)
        except RuntimeError:
            branch = "HEAD"
        log.warning(
            "Snapshot: clean-git validation skipped (require_clean_git=false). "
            "The bundle may not exactly represent the commit."
        )

    short_sha = full_sha[:8]
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in str(sweep_name)
    )[:64]
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    random_suffix = uuid.uuid4().hex[:8]
    bundle_id = f"{ts}_{safe_name}_{short_sha}_{random_suffix}"

    bundle_dir = (snapshot_root / bundle_id).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=False)

    staging_marker = bundle_dir / STAGING_MARKER
    staging_marker.touch()

    source_dir = bundle_dir / "source"
    manifests_dir = bundle_dir / "manifests"
    task_configs_dir = bundle_dir / "task-configs"
    logs_dir = bundle_dir / "logs"
    for d in (source_dir, manifests_dir, task_configs_dir, logs_dir):
        d.mkdir()

    # --- git archive ---
    archive_bytes = subprocess.run(
        ["git", "archive", "--format=tar", full_sha],
        cwd=str(project_root),
        capture_output=True,
        check=True,
    ).stdout

    with tarfile.open(fileobj=BytesIO(archive_bytes)) as tar:
        tar.extractall(path=str(source_dir))

    for entry_name in REQUIRED_SOURCE_ENTRIES:
        entry = source_dir / entry_name
        if not entry.exists():
            raise RuntimeError(
                f"Staged source is missing required entry: {entry_name}"
            )

    source_digest = _compute_tree_digest(source_dir)

    # --- base config ---
    base_config_path = manifests_dir / "resolved-base-config.yaml"
    base_config_path.write_text(OmegaConf.to_yaml(hydra_cfg, resolve=True))

    # --- submitted overrides ---
    overrides_path = manifests_dir / "submitted-overrides.txt"
    lines: list[str] = []
    for i, ov in enumerate(job_overrides):
        lines.append(f"# task {i}")
        lines.extend(ov)
        lines.append("")
    overrides_path.write_text("\n".join(lines))

    # --- source digest file ---
    digest_path = manifests_dir / "source-files.sha256"
    digest_path.write_text(source_digest + "\n")

    # --- environment fingerprint ---
    env_fp = _build_environment_fingerprint(project_root)

    # --- launch manifest ---
    manifest = {
        "bundle_id": bundle_id,
        "git_sha": full_sha,
        "git_branch": branch,
        "source_digest": source_digest,
        "timestamp_utc": ts,
        "command_line": sys.argv,
        "sweep_name": sweep_name,
        "num_tasks": len(job_overrides),
        "environment": env_fp,
        "paths": {
            "bundle_dir": str(bundle_dir),
            "source_dir": str(source_dir),
            "base_config": str(base_config_path),
            "task_configs_dir": str(task_configs_dir),
            "logs_dir": str(logs_dir),
        },
    }
    manifest_path = manifests_dir / "launch.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # --- per-task resolved configs ---
    for i, ov in enumerate(job_overrides):
        tc_path = task_configs_dir / f"task_{i:04d}.json"
        tc_path.write_text(
            json.dumps({"task_index": i, "overrides": list(ov)}, indent=2)
        )

    # --- seal source (read-only) ---
    for fpath in source_dir.rglob("*"):
        if fpath.is_file():
            fpath.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    for dpath in sorted(source_dir.rglob("*"), reverse=True):
        if dpath.is_dir():
            dpath.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
    source_dir.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )

    env_fp_hash = hashlib.sha256(
        json.dumps(env_fp, sort_keys=True).encode()
    ).hexdigest()[:16]

    snapshot = LaunchSnapshot(
        bundle_dir=str(bundle_dir),
        source_dir=str(source_dir),
        manifest_path=str(manifest_path),
        git_sha=full_sha,
        git_branch=branch,
        source_digest=source_digest,
        base_config_path=str(base_config_path),
        environment_fingerprint=env_fp_hash,
        bundle_id=bundle_id,
    )
    descriptor_path = manifests_dir / "snapshot-descriptor.json"
    descriptor_path.write_text(snapshot.to_json())

    # Signal completion only after every worker-required artifact exists.
    staging_marker.unlink()
    (bundle_dir / COMPLETION_MARKER).touch()

    log.info("Snapshot sealed: %s", bundle_dir)
    log.info("  Git SHA    : %s", full_sha)
    log.info("  Branch     : %s", branch)
    log.info("  Source hash: %s", source_digest[:16])
    log.info("  Manifest   : %s", manifest_path)

    return snapshot


def verify_snapshot(snapshot: LaunchSnapshot) -> None:
    """Verify bundle integrity on a worker node. Raises on mismatch."""
    bundle_dir = Path(snapshot.bundle_dir)
    source_dir = Path(snapshot.source_dir)

    completion = bundle_dir / COMPLETION_MARKER
    if not completion.exists():
        raise RuntimeError(
            f"Snapshot bundle is incomplete (missing {COMPLETION_MARKER}): "
            f"{bundle_dir}"
        )

    if not source_dir.is_dir():
        raise RuntimeError(f"Snapshot source directory missing: {source_dir}")

    actual_digest = _compute_tree_digest(source_dir)
    if actual_digest != snapshot.source_digest:
        raise RuntimeError(
            f"Source digest mismatch in {bundle_dir}.\n"
            f"  Expected: {snapshot.source_digest}\n"
            f"  Actual  : {actual_digest}\n"
            "The snapshot source has been tampered with."
        )

    for entry_name in REQUIRED_SOURCE_ENTRIES:
        if not (source_dir / entry_name).exists():
            raise RuntimeError(
                f"Snapshot source missing required entry: {entry_name}"
            )


def write_task_provenance(
    snapshot: LaunchSnapshot,
    task_index: int,
    overrides: Sequence[str],
    output_dir: str | Path,
) -> Path:
    """Write a ``provenance.json`` file into a task's Hydra output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "bundle_id": snapshot.bundle_id,
        "git_sha": snapshot.git_sha,
        "git_branch": snapshot.git_branch,
        "source_digest": snapshot.source_digest,
        "manifest_path": snapshot.manifest_path,
        "source_dir": snapshot.source_dir,
        "environment_fingerprint": snapshot.environment_fingerprint,
        "task_index": task_index,
        "task_overrides": list(overrides),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
    }

    path = output_dir / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2))
    log.info("Task provenance written: %s", path)
    return path


def build_worker_environment(
    snapshot: LaunchSnapshot,
    *,
    environment_file: str | None = None,
) -> dict[str, str]:
    """Return environment variables and shell setup for a worker process.

    The returned dict contains ``PYTHONPATH``, ``FOUNDRY_SNAPSHOT_*`` vars,
    and references needed by the worker to find the snapshot.
    """
    env: dict[str, str] = {
        "FOUNDRY_SNAPSHOT_BUNDLE_DIR": snapshot.bundle_dir,
        "FOUNDRY_SNAPSHOT_SOURCE_DIR": snapshot.source_dir,
        "FOUNDRY_SNAPSHOT_MANIFEST": snapshot.manifest_path,
        "FOUNDRY_SNAPSHOT_GIT_SHA": snapshot.git_sha,
        "FOUNDRY_SNAPSHOT_SOURCE_DIGEST": snapshot.source_digest,
        "FOUNDRY_SNAPSHOT_BUNDLE_ID": snapshot.bundle_id,
    }
    return env


def build_setup_commands(
    snapshot: LaunchSnapshot,
    *,
    environment_file: str | None = None,
    existing_setup: list[str] | None = None,
    verify_on_worker: bool = True,
) -> list[str]:
    """Build shell setup commands for Submitit workers.

    Replaces bare ``source .env`` and ``cd <checkout>`` with snapshot-safe
    equivalents, then prepends ``PYTHONPATH`` and ``cd`` to the snapshot
    source.
    """
    commands: list[str] = []

    if environment_file:
        resolved = os.path.abspath(environment_file)
        commands.extend(
            [
                "set -a",
                f'source "{resolved}" || true',
                "set +a",
            ]
        )

    if existing_setup:
        for cmd in existing_setup:
            stripped = cmd.strip()
            if stripped.startswith("source .env") or stripped == "source .env":
                continue
            if stripped.startswith("source .env "):
                continue
            if stripped.startswith("cd ") and "Foundry" in stripped:
                continue
            commands.append(cmd)

    snapshot_env = build_worker_environment(snapshot)
    for key, val in snapshot_env.items():
        commands.append(f'export {key}="{val}"')
    commands.append(
        f'export FOUNDRY_SNAPSHOT_VERIFY_ON_WORKER="{int(verify_on_worker)}"'
    )

    commands.append(
        f'export PYTHONPATH="{snapshot.source_dir}:${{PYTHONPATH:-}}"'
    )
    commands.append(f'cd "{snapshot.source_dir}"')

    return commands


def get_snapshot_provenance_for_wandb(
    snapshot_bundle_dir: str | None = None,
) -> dict[str, Any]:
    """Read snapshot identity from env vars set by the launcher.

    Returns a dict suitable for adding to WandB config under a
    ``provenance`` key.  Returns empty dict if no snapshot is active.
    """
    bundle_dir = snapshot_bundle_dir or os.environ.get(
        "FOUNDRY_SNAPSHOT_BUNDLE_DIR"
    )
    if not bundle_dir:
        return {}

    provenance: dict[str, Any] = {
        "provenance.bundle_id": os.environ.get(
            "FOUNDRY_SNAPSHOT_BUNDLE_ID", ""
        ),
        "provenance.git_sha": os.environ.get("FOUNDRY_SNAPSHOT_GIT_SHA", ""),
        "provenance.source_digest": os.environ.get(
            "FOUNDRY_SNAPSHOT_SOURCE_DIGEST", ""
        ),
        "provenance.manifest_path": os.environ.get(
            "FOUNDRY_SNAPSHOT_MANIFEST", ""
        ),
        "provenance.source_dir": os.environ.get(
            "FOUNDRY_SNAPSHOT_SOURCE_DIR", ""
        ),
        "provenance.slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "provenance.slurm_array_task_id": os.environ.get(
            "SLURM_ARRAY_TASK_ID", ""
        ),
        "provenance.slurm_restart_count": os.environ.get(
            "SLURM_RESTART_COUNT", "0"
        ),
    }
    return {k: v for k, v in provenance.items() if v}


def verify_import_paths(source_dir: str | Path) -> None:
    """Verify that key modules are imported from the snapshot, not the live
    checkout. Raises if any module resolves outside the snapshot."""
    source_dir = str(Path(source_dir).resolve())

    checks = {
        "foundry": "foundry",
        "hydra_plugins.foundry_launcher": "hydra_plugins.foundry_launcher",
    }
    issues: list[str] = []
    for label, mod_name in checks.items():
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", None)
        if mod_file and not os.path.abspath(mod_file).startswith(source_dir):
            issues.append(
                f"  {label}: loaded from {mod_file} (expected under {source_dir})"
            )

    main_file = os.path.abspath(sys.argv[0]) if sys.argv else None
    if main_file and not main_file.startswith(source_dir):
        issues.append(f"  main.py: {main_file} (expected under {source_dir})")

    if issues:
        raise RuntimeError(
            "Snapshot import verification failed; application modules were "
            "loaded outside the immutable source bundle:\n" + "\n".join(issues)
        )
