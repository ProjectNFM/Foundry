"""OmegaConf custom resolvers and Hydra utilities.

Call :func:`register_resolvers` once before ``@hydra.main`` runs to make these
available in every entry point (``main.py``, ``profile_training.py``, etc.).

Use :func:`hydra_main_wrapper` as a decorator on Hydra main functions to
ensure exceptions are always printed and streams are flushed -- critical for
SLURM jobs where buffered output is lost on crash.
"""

import glob as _glob
import os
import sys
import traceback
from functools import wraps
from typing import List, Optional

from omegaconf import OmegaConf


def _find_checkpoints(root_dir: str, job_id: str, method: str) -> str:
    """Locate checkpoint paths under *root_dir* for a SLURM *job_id*.

    *method* is ``"last"`` (latest recurrent checkpoint) or ``"best"``
    (lowest ``train_loss`` in the best/ directory).
    """
    assert method in ("last", "best")

    job_dirs = _glob.glob(f"{root_dir}/{job_id}*/job*")
    if not job_dirs:
        return ""

    paths: list[str] = []
    for job_dir in job_dirs:
        if method == "last":
            candidate = f"{job_dir}/checkpoints/recurrent/last.ckpt"
            if _glob.glob(candidate):
                paths.append(candidate)
        else:
            job_ckpts = _glob.glob(f"{job_dir}/checkpoints/best/*.ckpt")
            if not job_ckpts:
                continue

            def _loss(p: str) -> float:
                try:
                    return float(p.split("train_loss-")[1].split(".ckpt")[0])
                except (IndexError, ValueError):
                    return float("inf")

            paths.append(min(job_ckpts, key=_loss))

    return ", ".join(paths)


def _get_checkpoints_from_folder(folder_path: str) -> List[str]:
    """Return all ``*.ckpt`` files in *folder_path*."""
    return _glob.glob(f"{folder_path}/*.ckpt")


def _get_overrides_from_ckpt(
    ckpt_path: str, keys_to_include: Optional[List[str]] = None
) -> str:
    """Read ``.hydra/overrides.yaml`` next to a checkpoint and return a
    formatted string like ``key1=val1-key2=val2``.
    """
    if not ckpt_path or ckpt_path == "???":
        return "no-ckpt"

    job_dir = ckpt_path.split("/checkpoints/")[0]
    overrides_path = f"{job_dir}/.hydra/overrides.yaml"

    try:
        import yaml

        with open(overrides_path) as f:
            overrides_list = yaml.safe_load(f)
    except (FileNotFoundError, Exception):
        return "no-overrides"

    if not overrides_list:
        return "no-overrides"

    filtered: dict[str, str] = {}
    for override in overrides_list:
        parts = override.split("=", 1)
        if len(parts) == 2:
            key, value = parts
            if keys_to_include is None or key in keys_to_include:
                filtered[key] = value

    if not filtered:
        return "no-matching-keys"

    return "-".join(f"{k.split('.')[-1]}={v}" for k, v in filtered.items())


def _patch_samples_resolver(patch_duration: float, sampling_rate: float) -> int:
    """Compute ``patch_samples`` from patch duration and sampling rate.

    Equivalent to ``foundry.data.utils.compute_patch_samples``, exposed
    as an OmegaConf resolver so it can be used in YAML configs::

        patch_samples: ${patch_samples:${hyperparameters.patch_duration},${hyperparameters.sampling_rate}}
    """
    return max(1, round(float(patch_duration) * float(sampling_rate)))


def _sweep_choices(values: List[str] | tuple[str, ...]) -> str:
    """Hydra-compatible choice string from a list/tuple of string values."""
    if not values:
        raise ValueError("Cannot build sweep choices from an empty sequence")

    return ",".join(
        "'" + str(value).replace("'", "\\'") + "'" for value in values
    )


def _config_list_sweep_choices(config_path: str, key: str) -> str:
    """Hydra-compatible choice string from a list key inside a YAML config."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    values = OmegaConf.select(cfg, key)
    if values is None:
        raise KeyError(f"Key '{key}' not found in config file: {config_path}")

    if OmegaConf.is_config(values):
        values = OmegaConf.to_container(values, resolve=True)

    if not isinstance(values, (list, tuple)):
        raise TypeError(
            f"Expected '{key}' in {config_path} to be a list/tuple, got {type(values)}"
        )

    return _sweep_choices(tuple(str(value) for value in values))


def _pair_int(pair: str, index: int) -> int:
    """Extract integer item at *index* from a underscore-separated pair."""
    parts = [part.strip() for part in str(pair).split("x")]
    print(parts)
    if len(parts) != 2:
        raise ValueError(f"Expected pair in the form 'axb', got '{pair}'.")
    if index not in (0, 1):
        raise ValueError(f"pair index must be 0 or 1, got {index}.")
    return int(parts[index])


def _neuroprobe_ss_dm_pair_choices(subset_tier: str = "lite") -> str:
    """Hydra choice values for valid Neuroprobe SS-DM subject/session pairs."""
    if subset_tier != "lite":
        raise ValueError(
            "Only subset_tier='lite' is supported by this resolver."
        )

    # SS-DM lite valid target pairs from Neuroprobe benchmark defaults.
    pairs = (
        "1x1",
        "1x2",
        "2x0",
        "2x4",
        "3x0",
        "3x1",
        "4x0",
        "4x1",
        "7x0",
        "7x1",
        "10x0",
        "10x1",
    )
    return _sweep_choices(pairs)


def _count_ecog_channels(h5_path: str) -> int:
    """Count ECoG-like channels in an HDF5 recording file."""
    import h5py
    import numpy as np

    supported_modalities = {"eeg", "ecog", "seeg", "ieeg"}

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"HDF5 file not found: {h5_path}  "
            f"(is the data staged / accessible from this node?)"
        )

    with h5py.File(h5_path, "r") as f:
        raw_types = f["channels/type"][()]
        channel_types = np.array(
            [t.decode() if isinstance(t, bytes) else t for t in raw_types],
            dtype="U",
        )
        return int(
            np.isin(
                np.char.lower(channel_types), list(supported_modalities)
            ).sum()
        )


def _get_num_ecog_channels_by_name(data_dir: str, recording_id: str) -> int:
    """Number of ECoG channels for a recording identified by *recording_id*."""
    h5_path = os.path.join(data_dir, f"{recording_id}.h5")
    return _count_ecog_channels(h5_path)


def _get_suffix(s: str) -> str:
    """Last segment of an underscore-separated string, upper-cased."""
    return s.split("_")[-1].upper()


def hydra_main_wrapper(func):
    """Decorator that ensures exceptions are printed and streams flushed.

    Hydra swallows tracebacks in certain failure modes (especially under
    submitit). This wrapper guarantees the traceback reaches stderr and that
    both stdout/stderr are flushed before the process exits.

    See https://github.com/facebookresearch/hydra/issues/2664
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException:
            traceback.print_exc(file=sys.stderr)
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()

    return wrapper


def register_resolvers() -> None:
    """Register all custom OmegaConf resolvers (idempotent)."""
    _resolvers = {
        "find_checkpoints": _find_checkpoints,
        "get_checkpoints_from_folder": _get_checkpoints_from_folder,
        "get_overrides_from_ckpt": _get_overrides_from_ckpt,
        "patch_samples": _patch_samples_resolver,
        "get_suffix": _get_suffix,
        "sweep_choices": _sweep_choices,
        "config_list_sweep_choices": _config_list_sweep_choices,
        "pair_int": _pair_int,
        "neuroprobe_ss_dm_pair_choices": _neuroprobe_ss_dm_pair_choices,
        "get_num_ecog_channels_by_name": _get_num_ecog_channels_by_name,
    }
    for name, fn in _resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)
