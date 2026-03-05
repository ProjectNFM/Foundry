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
import shutil
import tarfile
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch_brain.dataset import NestedDataset

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


def compute_archive_name(
    dirname: str,
    recording_ids: list[str],
    source_dir: Path,
) -> str:
    """Return a deterministic archive filename for a set of recordings.

    Uses ``<dirname>_all.tar`` when every h5 in source_dir is included,
    otherwise ``<dirname>_<sha256[:12]>.tar``.
    """
    all_ids = sorted(p.stem for p in source_dir.glob("*.h5"))
    if sorted(recording_ids) == all_ids:
        return f"{dirname}_all.tar"

    key = dirname + ":" + ",".join(sorted(recording_ids))
    short_hash = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"{dirname}_{short_hash}.tar"


def create_archive(
    filepaths: dict[str, Path],
    archive_path: Path,
) -> None:
    """Create a tar archive containing the given h5 files."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = archive_path.with_suffix(".tar.tmp")
    try:
        with tarfile.open(tmp_path, "w") as tar:
            for recording_id, fpath in sorted(filepaths.items()):
                tar.add(fpath, arcname=fpath.name)
        tmp_path.rename(archive_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    logger.info("Created archive %s (%d files)", archive_path, len(filepaths))


def stage_data(
    data_cfg: DictConfig,
    source_root: str | Path,
    compressed_root: str | Path,
    dest_root: str | Path,
) -> str:
    """Stage brainset data to a fast local directory.

    1. Instantiates the datamodule (with ``keep_files_open=False``) to
       discover all required h5 files.
    2. For each brainset dirname, checks whether a matching tar archive
       already exists under *compressed_root*.  Creates one if not.
    3. Copies the archive to *dest_root* and unpacks it.

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
        recording_ids = sorted(filepaths.keys())
        source_dir = source_root / dirname

        archive_name = compute_archive_name(dirname, recording_ids, source_dir)
        archive_src = compressed_root / archive_name

        if not archive_src.exists():
            logger.info(
                "Archive not found at %s — creating from %d files in %s",
                archive_src,
                len(filepaths),
                source_dir,
            )
            create_archive(filepaths, archive_src)
        else:
            logger.info("Reusing existing archive %s", archive_src)

        dest_dir = dest_processed / dirname
        if dest_dir.exists() and any(dest_dir.iterdir()):
            logger.info(
                "Destination %s already populated; skipping unpack.",
                dest_dir,
            )
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        archive_dst = dest_root / archive_name
        logger.info(
            "Copying %s -> %s",
            archive_src,
            archive_dst,
        )
        shutil.copy2(archive_src, archive_dst)

        logger.info("Unpacking %s into %s", archive_dst, dest_dir)
        with tarfile.open(archive_dst, "r") as tar:
            tar.extractall(path=dest_dir)

        archive_dst.unlink()
        logger.info("Staged %s (%d recordings)", dirname, len(filepaths))

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
    )
    print(new_root)


if __name__ == "__main__":
    main()
