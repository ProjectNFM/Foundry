"""Capture the NeuralBench P3 / Korczowski2014A reference contract.

Produces a structured report of every observable detail of the NeuralBench
task specification — splits, preprocessing, sample shapes, labels, channels,
loss, metrics, trainer config, and seeds — so the adapter can reproduce
the exact same data contract in Foundry.

Usage:
    uv run python scripts/neuralbench_capture_p3_contract.py
"""

from __future__ import annotations

import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch


def _versions() -> dict:
    pkgs = ["neuralbench", "neuralset", "neuralfetch", "neuraltrain"]
    return {p: version(p) for p in pkgs}


def _load_effective_config():
    from exca import ConfDict
    from neuralbench.experiment_config import prepare_task_configs
    from neuralbench.registry import DEFAULTS_DIR, load_yaml_config

    default_config = load_yaml_config(DEFAULTS_DIR / "config.yaml")
    grid = ConfDict(load_yaml_config(DEFAULTS_DIR / "grid.yaml"))

    configs = prepare_task_configs(
        ConfDict(default_config),
        grid,
        device="eeg",
        task_name="p3",
        use_task_grid=False,
        debug=False,
        force=False,
        prepare=False,
        download=False,
        models=[None],
        datasets=["korczowski2014a"],
        quiet=True,
    )
    return configs[0]


def _build_data_and_loaders(cfg):
    from neuralbench.data import Data

    data = Data(**cfg["data"])
    loaders = data.prepare()
    return data, loaders


def _inspect_task_config(cfg) -> dict:
    """Extract all task-level settings from the effective config."""
    data_cfg = cfg.get("data", {})
    return {
        "study_name": data_cfg.get("study", {}).get("source", {}).get("name"),
        "study_path": str(
            data_cfg.get("study", {}).get("source", {}).get("path", "")
        ),
        "split": {
            "method": data_cfg.get("study", {}).get("split", {}).get("name"),
            "split_by": data_cfg.get("study", {})
            .get("split", {})
            .get("split_by"),
            "valid_split_ratio": data_cfg.get("study", {})
            .get("split", {})
            .get("valid_split_ratio"),
            "test_split_ratio": data_cfg.get("study", {})
            .get("split", {})
            .get("test_split_ratio"),
            "valid_random_state": data_cfg.get("study", {})
            .get("split", {})
            .get("valid_random_state"),
            "test_random_state": data_cfg.get("study", {})
            .get("split", {})
            .get("test_random_state"),
        },
        "epoch": {
            "start": data_cfg.get("start"),
            "duration": data_cfg.get("duration"),
            "trigger_event_type": data_cfg.get("trigger_event_type"),
        },
        "baseline": data_cfg.get("neuro.baseline")
        or data_cfg.get("neuro", {}).get("baseline"),
        "target": {
            k: v
            for k, v in (data_cfg.get("target") or {}).items()
            if k != "=replace="
        },
        "compute_class_weights": cfg.get("compute_class_weights"),
        "brain_model_output_size": cfg.get("brain_model_output_size"),
        "loss": cfg.get("loss"),
        "trainer_config": cfg.get("trainer_config"),
        "metrics": str(cfg.get("metrics")),
        "seed": cfg.get("seed"),
    }


def _inspect_neuro_extractor(data) -> dict:
    """Extract preprocessing pipeline details from the Data object."""
    neuro = data.neuro
    info = {}
    for attr in [
        "frequency",
        "picks",
        "l_freq",
        "h_freq",
        "notch_freqs",
        "baseline",
        "channel_order",
    ]:
        val = getattr(neuro, attr, "NOT_FOUND")
        if val != "NOT_FOUND":
            info[attr] = (
                val if not isinstance(val, (list, tuple)) else list(val)
            )
    if hasattr(neuro, "scaler"):
        info["scaler"] = str(neuro.scaler)
    if hasattr(neuro, "clamp"):
        info["clamp"] = neuro.clamp
    if hasattr(neuro, "_channels"):
        info["num_channels"] = len(neuro._channels)
        info["channel_names"] = list(neuro._channels.keys())
    return info


def _inspect_splits(data, loaders) -> dict:
    """Capture split membership, segment counts, and label distributions."""
    split_info = {}
    for split_name, loader in loaders.items():
        ds = loader.dataset
        n_samples = len(ds)

        label_counts = Counter()
        shapes = {}
        dtypes = {}
        sample_keys = set()
        metadata_samples = []

        n_inspect = min(n_samples, 50)
        for i in range(n_inspect):
            sample = ds[i]
            sample_data = sample.data if hasattr(sample, "data") else sample

            if isinstance(sample_data, dict):
                sample_keys.update(sample_data.keys())
                for key, val in sample_data.items():
                    if isinstance(val, (torch.Tensor, np.ndarray)):
                        shapes[key] = list(val.shape)
                        dtypes[key] = str(val.dtype)
                    elif isinstance(val, (int, float)):
                        shapes[key] = "scalar"
                        dtypes[key] = type(val).__name__

                if "target" in sample_data:
                    t = sample_data["target"]
                    if isinstance(t, torch.Tensor):
                        t = t.numpy()
                    if t.ndim >= 1:
                        label_counts[int(np.argmax(t.flatten()))] += 1
                    else:
                        label_counts[int(t)] += 1

            if hasattr(sample, "segments") and sample.segments:
                seg = sample.segments[0]
                seg_dict = {}
                for attr in dir(seg):
                    if not attr.startswith("_"):
                        try:
                            v = getattr(seg, attr)
                            if not callable(v):
                                seg_dict[attr] = str(v)
                        except Exception:
                            pass
                if i < 3:
                    metadata_samples.append(seg_dict)

        split_info[split_name] = {
            "num_samples": n_samples,
            "sample_keys": sorted(sample_keys),
            "shapes": shapes,
            "dtypes": dtypes,
            "label_distribution_first50": dict(label_counts),
            "segment_metadata_examples": metadata_samples,
        }

    return split_info


def _inspect_full_label_distribution(loaders) -> dict:
    """Get complete label distribution per split."""
    dist = {}
    for split_name, loader in loaders.items():
        ds = loader.dataset
        label_counts = Counter()
        for i in range(len(ds)):
            sample = ds[i]
            sample_data = sample.data if hasattr(sample, "data") else sample
            if isinstance(sample_data, dict) and "target" in sample_data:
                t = sample_data["target"]
                if isinstance(t, torch.Tensor):
                    t = t.numpy()
                if t.ndim >= 1:
                    label_counts[int(np.argmax(t.flatten()))] += 1
                else:
                    label_counts[int(t)] += 1
        dist[split_name] = dict(sorted(label_counts.items()))
    return dist


def _inspect_subjects_per_split(data, loaders) -> dict:
    """Get subject IDs per split for identity audit."""
    subjects = {}
    for split_name, loader in loaders.items():
        ds = loader.dataset
        split_subjects = set()
        for i in range(len(ds)):
            sample = ds[i]
            sample_data = sample.data if hasattr(sample, "data") else sample
            if isinstance(sample_data, dict) and "subject_id" in sample_data:
                sid = sample_data["subject_id"]
                if isinstance(sid, torch.Tensor):
                    sid = sid.item()
                split_subjects.add(int(sid))
        subjects[split_name] = sorted(split_subjects)
    return subjects


def _inspect_signal_stats(loaders) -> dict:
    """Capture signal statistics from a small sample for sanity checking."""
    stats = {}
    for split_name, loader in loaders.items():
        ds = loader.dataset
        sample = ds[0]
        sample_data = sample.data if hasattr(sample, "data") else sample
        if isinstance(sample_data, dict) and "neuro" in sample_data:
            sig = sample_data["neuro"]
            if isinstance(sig, torch.Tensor):
                sig = sig.numpy()
            stats[split_name] = {
                "shape": list(sig.shape),
                "dtype": str(sig.dtype),
                "min": float(np.min(sig)),
                "max": float(np.max(sig)),
                "mean": float(np.mean(sig)),
                "std": float(np.std(sig)),
                "has_nan": bool(np.any(np.isnan(sig))),
                "has_inf": bool(np.any(np.isinf(sig))),
            }
    return stats


def _get_trigger_metadata(data, loaders) -> dict:
    """Inspect trigger/segment metadata from the underlying dataset."""
    info = {}
    for split_name, loader in loaders.items():
        ds = loader.dataset
        if hasattr(ds, "triggers") and ds.triggers is not None:
            triggers = ds.triggers
            info[split_name] = {
                "columns": list(triggers.columns),
                "num_triggers": len(triggers),
            }
            if "subject" in triggers.columns:
                info[split_name]["unique_subjects"] = sorted(
                    triggers["subject"].unique().tolist()
                )
            if "code" in triggers.columns:
                info[split_name]["code_distribution"] = dict(
                    triggers["code"].value_counts()
                )
            if "type" in triggers.columns:
                info[split_name]["event_types"] = sorted(
                    triggers["type"].unique().tolist()
                )
            if "split" in triggers.columns:
                info[split_name]["split_values"] = sorted(
                    triggers["split"].unique().tolist()
                )
            first_row = triggers.iloc[0].to_dict()
            info[split_name]["first_trigger_example"] = {
                k: str(v) for k, v in first_row.items()
            }
    return info


def _foundry_brain_invaders_audit() -> dict:
    """Compare NeuralBench subjects to Foundry Brain Invaders inventory."""
    from pathlib import Path

    bi_dir = Path("./data/processed/korczowski_brain_invaders_2014a")
    audit = {"foundry_data_path": str(bi_dir), "foundry_data_exists": False}
    if bi_dir.exists():
        audit["foundry_data_exists"] = True
        h5_files = sorted(
            f.name
            for f in bi_dir.iterdir()
            if f.is_file() and f.suffix == ".h5"
        )
        audit["foundry_recording_ids"] = h5_files
        audit["foundry_num_recordings"] = len(h5_files)
        subjects = sorted(set(r.split("_")[0] for r in h5_files))
        audit["foundry_subjects"] = subjects
        audit["foundry_num_subjects"] = len(subjects)
    else:
        audit["note"] = (
            "Foundry Brain Invaders H5 data not found at expected path. "
            "Identity audit will compare against expected MOABB subject list."
        )
        audit["expected_moabb_subjects"] = [
            f"sub{str(i).zfill(3)}" for i in range(1, 25)
        ]
        audit["expected_num_subjects"] = 24
    return audit


def main():
    print("=" * 72)
    print("NeuralBench P3 / Korczowski2014A — Reference Contract Capture")
    print("=" * 72)

    report = {}

    print("\n[1/8] Package versions...")
    report["versions"] = _versions()
    print(json.dumps(report["versions"], indent=2))

    print("\n[2/8] Loading effective task config...")
    cfg = _load_effective_config()
    report["task_config"] = _inspect_task_config(cfg)
    print(json.dumps(report["task_config"], indent=2, default=str))

    print("\n[3/8] Downloading data if needed...")
    _download_dataset_standalone(cfg)

    print("\n[4/8] Building Data and loaders...")
    data, loaders = _build_data_and_loaders(cfg)

    print("\n[5/8] Inspecting preprocessing pipeline...")
    report["preprocessing"] = _inspect_neuro_extractor(data)
    print(json.dumps(report["preprocessing"], indent=2, default=str))

    print("\n[6/8] Inspecting splits, shapes, and metadata...")
    report["splits"] = _inspect_splits(data, loaders)
    for split_name, info in report["splits"].items():
        print(f"\n  {split_name}: {info['num_samples']} samples")
        print(f"    keys: {info['sample_keys']}")
        print(f"    shapes: {info['shapes']}")
        print(f"    dtypes: {info['dtypes']}")

    print("\n[7/8] Full label distribution and signal stats...")
    report["label_distribution"] = _inspect_full_label_distribution(loaders)
    report["signal_stats"] = _inspect_signal_stats(loaders)
    report["trigger_metadata"] = _get_trigger_metadata(data, loaders)
    report["subjects_per_split"] = _inspect_subjects_per_split(data, loaders)

    for split_name, dist in report["label_distribution"].items():
        print(f"  {split_name} labels: {dist}")

    print("\n[8/8] Foundry Brain Invaders identity audit...")
    report["foundry_audit"] = _foundry_brain_invaders_audit()
    print(json.dumps(report["foundry_audit"], indent=2, default=str))

    out_path = Path("docs/neuralbench-p3-contract.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report written to {out_path}")

    return report


def _download_dataset_standalone(cfg):
    """Download the dataset using neuralbench's download mechanism."""
    from exca import ConfDict
    from neuralbench.experiment_config import _download_dataset

    _download_dataset(ConfDict(cfg))
    print("  Download step complete.")


if __name__ == "__main__":
    report = main()
