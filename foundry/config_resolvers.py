"""OmegaConf custom resolvers and Hydra utilities.

Call :func:`register_resolvers` once before ``@hydra.main`` runs to make these
available in every entry point (``main.py``, ``profile_training.py``, etc.).

Use :func:`hydra_main_wrapper` as a decorator on Hydra main functions to
ensure exceptions are always printed and streams are flushed -- critical for
SLURM jobs where buffered output is lost on crash.
"""

import glob as _glob
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


def _range_resolver(start: int, end: int, step: int) -> List[int]:
    return list(range(int(start), int(end), int(step)))


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
        "range_resolver": _range_resolver,
        "get_suffix": _get_suffix,
    }
    for name, fn in _resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)
