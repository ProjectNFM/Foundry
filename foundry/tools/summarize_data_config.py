"""Summarize the contents of a data config YAML for experiment planning.

Instantiates the dataset(s) defined by a Hydra data config and reports
per-brainset and aggregate statistics: recordings, subjects, channels,
sampling rates, total duration, and effective data.

Usage:
    uv run python -m foundry.tools.summarize_data_config \
        configs/data/openneuro/two_dataset_pretrain.yaml

    # With custom data root:
    uv run python -m foundry.tools.summarize_data_config \
        configs/data/openneuro/kochi_only.yaml --root ./data/processed/
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

SIGNAL_KEYS = ("eeg", "ieeg", "emg", "ecog", "meg")


def _find_signal(recording):
    """Find the first neural signal attribute on a recording."""
    for key in SIGNAL_KEYS:
        if recording.has_nested_attribute(key):
            return key, getattr(recording, key)
    for key in recording.keys():
        attr = getattr(recording, key)
        if hasattr(attr, "sampling_rate"):
            return key, attr
    return None, None


def summarize_child_dataset(name, child_ds):
    """Collect per-recording statistics for a single brainset."""
    rids = child_ds.recording_ids
    if len(rids) == 0:
        return {
            "name": name,
            "num_recordings": 0,
            "num_subjects": 0,
            "channels_per_recording": [],
            "sampling_rates": set(),
            "durations_hours": [],
            "signal_type": "unknown",
        }

    subjects = set()
    channels_per_rec = []
    sampling_rates = set()
    durations = []
    signal_type = "unknown"

    for rid in rids:
        rec = child_ds.get_recording(rid)

        subj = str(rec.subject.id[:])
        subjects.add(subj)

        n_ch = len(rec.channels.id[:])
        channels_per_rec.append(n_ch)

        domain = rec.domain
        start, end = float(domain.start[0]), float(domain.end[0])
        durations.append((end - start) / 3600.0)

        sig_key, sig = _find_signal(rec)
        if sig is not None:
            sampling_rates.add(round(float(sig.sampling_rate), 2))
            signal_type = sig_key

    return {
        "name": name,
        "num_recordings": len(rids),
        "num_subjects": len(subjects),
        "channels_per_recording": channels_per_rec,
        "sampling_rates": sampling_rates,
        "durations_hours": durations,
        "signal_type": signal_type,
    }


def print_summary(config_path, stats_list):
    """Print a formatted summary table."""
    print(f"\n{'=' * 72}")
    print(f"  Data Config Summary: {config_path}")
    print(f"{'=' * 72}")

    total_recordings = 0
    total_subjects = 0
    total_duration = 0.0
    total_effective_data = 0.0
    all_sampling_rates = set()

    for stats in stats_list:
        ch = stats["channels_per_recording"]
        dur = stats["durations_hours"]
        ch_arr = np.array(ch) if ch else np.array([0])
        dur_arr = np.array(dur) if dur else np.array([0.0])

        effective_data = sum(c * d for c, d in zip(ch, dur))

        total_recordings += stats["num_recordings"]
        total_subjects += stats["num_subjects"]
        total_duration += sum(dur)
        total_effective_data += effective_data
        all_sampling_rates.update(stats["sampling_rates"])

        print(f"\n  Brainset: {stats['name']}")
        print(f"  {'─' * 50}")
        print(f"    Signal type:     {stats['signal_type']}")
        print(f"    Recordings:      {stats['num_recordings']}")
        print(f"    Subjects:        {stats['num_subjects']}")
        print(
            f"    Channels/rec:    "
            f"min={ch_arr.min()}, max={ch_arr.max()}, "
            f"mean={ch_arr.mean():.1f}, median={np.median(ch_arr):.0f}"
        )
        print(
            f"    Sampling rate:   "
            f"{', '.join(f'{sr:.0f} Hz' for sr in sorted(stats['sampling_rates']))}"
        )
        print(
            f"    Duration (h):    "
            f"total={sum(dur):.1f}, "
            f"mean={dur_arr.mean():.2f}, "
            f"min={dur_arr.min():.2f}, max={dur_arr.max():.2f}"
        )
        print(f"    Effective data:  {effective_data:.1f} ch·h")

    if len(stats_list) > 1:
        print(f"\n  {'─' * 50}")
        print(f"  AGGREGATE ({len(stats_list)} brainsets)")
        print(f"  {'─' * 50}")
        print(f"    Total recordings:      {total_recordings}")
        print(f"    Total subjects:        {total_subjects}")
        print(f"    Total duration:        {total_duration:.1f} h")
        print(f"    Total effective data:  {total_effective_data:.1f} ch·h")
        print(
            f"    Sampling rates:        "
            f"{', '.join(f'{sr:.0f} Hz' for sr in sorted(all_sampling_rates))}"
        )
    elif len(stats_list) == 1:
        pass  # single brainset, no aggregate needed

    print(f"\n{'=' * 72}\n")


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Summarize a data config YAML."
    )
    parser.add_argument(
        "config",
        help="Path to the data config YAML file.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Override the data root directory (default: use config value).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset_class_str = cfg.get("dataset_class", "")

    if "OpenNeuroMultiBrainset" in dataset_class_str:
        from foundry.data.datasets import OpenNeuroMultiBrainset

        kwargs = dict(cfg.get("dataset_kwargs", {}))
        root = args.root or cfg.get("root", "./data/processed/")
        brainsets = kwargs.pop("brainsets", [])
        split_type = kwargs.pop("split_type", "intrasession")

        # Resolve hydra-style ${..split_type} reference and ??? placeholders
        if isinstance(split_type, str) and (
            split_type.startswith("${") or split_type == "???"
        ):
            top_level = cfg.get("split_type", "intrasession")
            split_type = top_level if top_level != "???" else "intrasession"

        recording_ids = kwargs.pop("recording_ids", None)

        ds = OpenNeuroMultiBrainset(
            root=root,
            brainsets=brainsets,
            split_type=split_type,
            recording_ids=recording_ids,
        )

        stats_list = []
        for name, child in ds.datasets.items():
            stats = summarize_child_dataset(name, child)
            stats_list.append(stats)

        print_summary(config_path, stats_list)

    else:
        print(
            f"Unsupported dataset class: {dataset_class_str}. "
            f"Currently only OpenNeuroMultiBrainset configs are supported.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
