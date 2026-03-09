"""Stage brainset data from scratch storage to SLURM_TMPDIR.

Reads hydra experiment configs to determine which h5 recording files are
needed, packages them into tar archives (cached for reuse across jobs), and
unpacks them at the destination so training can read from fast local storage.

Usage as standalone CLI:
    uv run python -m foundry.tools.stage_data \\
        --experiment poyo_ajile \\
        --source-root ../scratch/brainsets/processed \\
        --compressed-root ../scratch/brainsets/compressed \\
        --dest-root $SLURM_TMPDIR

Usage from Python (called automatically by main.py when SLURM_TMPDIR is set):
    from foundry.tools.stage_data import stage_data
    new_root = stage_data(data_cfg, source_root, compressed_root, dest_root)
"""

import argparse
import hashlib
import logging
import os
import subprocess
import time
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch_brain.dataset import NestedDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def collect_filepaths(dataset) -> dict[str, dict[str, Path]]:
    """Recursively collect h5 file paths from a (possibly nested) dataset.

    Returns a dict mapping each brainset dirname to its
    ``{recording_id: Path}`` mapping.  For a flat ``Dataset`` this is a
    single entry; for a ``NestedDataset`` there may be several.
    """
    if isinstance(dataset, NestedDataset):
        result: dict[str, dict[str, Path]] = {}
        for _name, child in dataset.datasets.items():
            child_paths = collect_filepaths(child)
            for dirname, fmap in child_paths.items():
                result.setdefault(dirname, {}).update(fmap)
        return result

    if not hasattr(dataset, "_filepaths"):
        raise RuntimeError(
            "Dataset was not created with keep_files_open=False; "
            "cannot retrieve file paths for staging."
        )

    filepaths: dict[str, Path] = dict(dataset._filepaths)
    if not filepaths:
        return {}

    sample_path = next(iter(filepaths.values()))
    dirname = sample_path.parent.name
    return {dirname: filepaths}


def _archive_ext(compress: bool) -> str:
    return ".tar.zst" if compress else ".tar"


def compute_archive_name(
    dirname: str,
    recording_ids: list[str],
    source_dir: Path,
    compress: bool = False,
) -> str:
    """Return a deterministic archive filename for a set of recordings.

    Uses ``<dirname>_all`` when every h5 in source_dir is included,
    otherwise ``<dirname>_<sha256[:12]>``.  Extension is ``.tar`` or
    ``.tar.zst`` depending on *compress*.
    """
    ext = _archive_ext(compress)
    all_ids = sorted(p.stem for p in source_dir.glob("*.h5"))
    if sorted(recording_ids) == all_ids:
        return f"{dirname}_all{ext}"

    key = dirname + ":" + ",".join(sorted(recording_ids))
    short_hash = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"{dirname}_{short_hash}{ext}"


def _format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _run_with_progress(
    cmd: list[str],
    monitor_path: Path,
    total_bytes: int,
    desc: str,
) -> None:
    """Run a subprocess while showing byte-level progress.

    Polls the size of *monitor_path* (file or directory) until the
    process finishes, driving a tqdm bar against *total_bytes*.
    """
    initial_size = (
        _dir_size(monitor_path)
        if monitor_path.is_dir()
        else monitor_path.stat().st_size
        if monitor_path.exists()
        else 0
    )
    proc = subprocess.Popen(cmd)

    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc=desc) as pbar:
        while proc.poll() is None:
            time.sleep(0.3)
            if monitor_path.exists():
                current = (
                    _dir_size(monitor_path)
                    if monitor_path.is_dir()
                    else monitor_path.stat().st_size
                ) - initial_size
                if current > pbar.n:
                    pbar.update(current - pbar.n)

        if monitor_path.exists():
            current = (
                _dir_size(monitor_path)
                if monitor_path.is_dir()
                else monitor_path.stat().st_size
            ) - initial_size
            if current > pbar.n:
                pbar.update(current - pbar.n)

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def create_archive(
    filepaths: dict[str, Path],
    archive_path: Path,
    compress: bool = False,
) -> None:
    """Create a tar archive from the given h5 files using system tar.

    When *compress* is True, uses zstd (level 1) for a good
    speed/ratio trade-off on already-partially-compressed h5 data.
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(p.stat().st_size for p in filepaths.values())

    source_dir = next(iter(filepaths.values())).parent
    file_names = sorted(p.name for p in filepaths.values())

    tmp_path = archive_path.with_suffix(".tmp")
    tar_cmd = ["tar", "cf", str(tmp_path)]
    if compress:
        tar_cmd += ["-I", "zstd -1 -T0"]
    tar_cmd += ["-C", str(source_dir)] + file_names

    try:
        _run_with_progress(tar_cmd, tmp_path, total_bytes, "Archiving")
        tmp_path.rename(archive_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    archive_size = archive_path.stat().st_size
    ratio = (
        f", {archive_size / total_bytes:.0%} of original" if compress else ""
    )
    logger.info(
        "Created archive %s (%d files, %s%s)",
        archive_path,
        len(filepaths),
        _format_bytes(archive_size),
        ratio,
    )


def stage_data(
    data_cfg: DictConfig,
    source_root: str | Path,
    compressed_root: str | Path,
    dest_root: str | Path,
    compress: bool = False,
) -> str:
    """Stage brainset data to a fast local directory.

    1. Instantiates the datamodule (with ``keep_files_open=False``) to
       discover all required h5 files.
    2. For each brainset dirname, checks whether a matching tar archive
       already exists under *compressed_root*.  Creates one if not.
    3. Copies the archive to *dest_root* and unpacks it.

    When *compress* is True, archives use zstd compression (smaller
    archives, faster copy over slow links, at the cost of CPU time
    during creation and extraction).

    Returns the new ``data.root`` path that should replace the original
    config value so training reads from the staged location.
    """
    source_root = Path(source_root).resolve()
    compressed_root = Path(compressed_root).resolve()
    dest_root = Path(dest_root).resolve()
    dest_processed = dest_root / "brainsets" / "processed"

    # --- 1. Discover required files -----------------------------------
    cfg_copy = OmegaConf.to_container(data_cfg, resolve=True)
    cfg_copy["root"] = str(source_root)
    cfg_as_dictconfig = OmegaConf.create(cfg_copy)

    datamodule = instantiate(cfg_as_dictconfig, tokenizer=None)
    datamodule.dataset_kwargs["keep_files_open"] = False
    datamodule.setup()

    filepaths_by_dirname = collect_filepaths(datamodule.dataset)
    if not filepaths_by_dirname:
        logger.warning("No recording files discovered; nothing to stage.")
        return str(dest_processed)

    # --- 2. Archive & stage each dirname ------------------------------
    for dirname, filepaths in filepaths_by_dirname.items():
        dest_dir = dest_processed / dirname
        source_dir = source_root / dirname

        already_staged = (
            {p.stem for p in dest_dir.glob("*.h5")}
            if dest_dir.exists()
            else set()
        )
        missing = {
            rid: path
            for rid, path in filepaths.items()
            if rid not in already_staged
        }

        if not missing:
            logger.info(
                "All %d recordings already at %s; skipping.",
                len(filepaths),
                dest_dir,
            )
            continue

        logger.info(
            "%d of %d recordings missing at %s",
            len(missing),
            len(filepaths),
            dest_dir,
        )

        missing_ids = sorted(missing.keys())
        archive_name = compute_archive_name(
            dirname, missing_ids, source_dir, compress=compress
        )
        archive_src = compressed_root / archive_name

        if not archive_src.exists():
            logger.info(
                "Archive not found at %s — creating from %d files in %s",
                archive_src,
                len(missing),
                source_dir,
            )
            create_archive(missing, archive_src, compress=compress)
        else:
            logger.info("Reusing existing archive %s", archive_src)

        dest_dir.mkdir(parents=True, exist_ok=True)
        archive_dst = dest_root / archive_name

        archive_size = archive_src.stat().st_size
        logger.info(
            "Copying %s -> %s (%s)",
            archive_src,
            archive_dst,
            _format_bytes(archive_size),
        )
        _run_with_progress(
            ["cp", str(archive_src), str(archive_dst)],
            archive_dst,
            archive_size,
            "Copying",
        )

        logger.info("Unpacking %s into %s", archive_dst, dest_dir)
        _run_with_progress(
            ["tar", "xf", str(archive_dst), "-C", str(dest_dir)],
            dest_dir,
            archive_size,
            "Extracting",
        )

        archive_dst.unlink()
        logger.info("Staged %d new recordings to %s", len(missing), dest_dir)

    return str(dest_processed)


# -- CLI entry point ---------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage brainset data for a SLURM job."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Hydra experiment name (e.g. poyo_ajile).",
    )
    parser.add_argument(
        "--source-root",
        default="../scratch/brainsets/processed",
        help="Root directory of processed brainset data.",
    )
    parser.add_argument(
        "--compressed-root",
        default="../scratch/brainsets/compressed",
        help="Directory for cached tar archives.",
    )
    parser.add_argument(
        "--dest-root",
        default=None,
        help="Destination root (defaults to $SLURM_TMPDIR).",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help=(
            "Use zstd compression. Smaller archives and faster copies "
            "over slow links, at the cost of extra CPU during "
            "archive creation and extraction."
        ),
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Additional Hydra overrides (e.g. data.recording_ids='[a,b]').",
    )
    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = _build_parser().parse_args()

    dest_root = args.dest_root or os.environ.get("SLURM_TMPDIR")
    if not dest_root:
        raise SystemExit(
            "No destination specified. Pass --dest-root or set $SLURM_TMPDIR."
        )

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    overrides = [f"experiment={args.experiment}"] + args.overrides

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    new_root = stage_data(
        data_cfg=cfg.data,
        source_root=args.source_root,
        compressed_root=args.compressed_root,
        dest_root=dest_root,
        compress=args.compress,
    )
    print(new_root)


if __name__ == "__main__":
    main()
