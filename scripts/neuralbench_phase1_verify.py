"""Phase 1 data verification and label discovery for MI and Sleep Stage.

Run after NeuralBench data preparation is complete:
    uv run python scripts/neuralbench_phase1_verify.py

This script:
1. Loads both tasks via NeuralBenchDataModule
2. Reports channels, sampling rate, split sizes, and auto-discovered labels
3. Validates that the class_mapping in task configs matches the data
4. Reports timing for data loading
"""

import time
from pathlib import Path

import numpy as np


def verify_task(task_name, dataset_name, interval_name, cache_dir):
    """Verify a NeuralBench task through the adapter."""
    from foundry.data.neuralbench.datamodule import NeuralBenchDataModule

    print(f"\n{'='*60}")
    print(f"  Task: {task_name} / {dataset_name}")
    print(f"{'='*60}")

    t0 = time.time()
    dm = NeuralBenchDataModule(
        task=task_name,
        dataset=dataset_name,
        cache_dir=cache_dir,
        batch_size=64,
        num_workers=0,
        interval_name=interval_name,
        label_attr="targets",
        session_prefix=f"nb/{task_name}",
    )
    dm.setup("fit")
    setup_time = time.time() - t0

    print(f"\n  Setup time: {setup_time:.1f}s")
    print(f"  Channels: {dm._num_channels}")
    print(f"  Channel names: {dm._channel_names[:5]}...")
    print(f"  Sessions: {len(dm._session_ids)}")
    print(f"  Label map (auto-discovered): {dm.label_map}")
    print(f"  Num classes: {len(dm.label_map)}")

    from foundry.tasks.config import TaskConfig

    task_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "tasks"
        / "neuralbench"
        / f"{task_name}.yaml"
    )
    task_config = TaskConfig.from_yaml(task_path)
    source_labels = np.asarray(list(dm.label_map.values()))
    if task_config.class_mapping is not None:
        mapped = task_config.class_mapping.map_to_class_ids(source_labels)
        if np.any(mapped < 0):
            missing = source_labels[mapped < 0].tolist()
            raise ValueError(
                f"Task config {task_path} does not map NeuralBench labels "
                f"{missing}"
            )
        print(f"  Task mapping: {dict(zip(source_labels, mapped.tolist()))}")

    for split, adapter in [
        ("train", dm._train_adapter),
        ("val", dm._val_adapter),
        ("test", dm._test_adapter),
    ]:
        if adapter is not None:
            print(f"  {split}: {len(adapter)} samples")

    # Count class distribution
    print("\n  Class distribution (train):")
    counts = {}
    ds = dm._train_adapter.nb_dataset
    for i in range(min(len(ds), 5000)):
        sample = ds[i]
        data_dict = sample.data if hasattr(sample, "data") else sample
        target = data_dict["target"]
        if hasattr(target, "numpy"):
            target = target.numpy()
        class_idx = int(np.argmax(target.flatten()))
        label = dm.label_map.get(class_idx, f"UNK_{class_idx}")
        counts[label] = counts.get(label, 0) + 1
    for label, count in sorted(counts.items()):
        print(f"    {label}: {count}")

    # Verify a sample through the adapter
    print("\n  Sample verification:")
    adapter = dm._train_adapter
    _, raw_data = adapter._get_sample_data(0)
    sample = adapter[0]
    print(f"    Data type: {type(sample).__name__}")
    print(f"    Has interval '{interval_name}': {hasattr(sample, interval_name)}")
    if hasattr(sample, interval_name):
        interval = getattr(sample, interval_name)
        print(f"    Interval targets: {interval.targets}")
    print(f"    EEG shape: {sample.eeg.signal.shape}")
    print(f"    Sampling rate: {sample.eeg.sampling_rate}")
    print(f"    Duration: {sample.domain.end[0] - sample.domain.start[0]:.2f}s")

    if task_name == "motor_imagery":
        from foundry.models.baselines import EEGNetEncoder

        raw_signal = raw_data["neuro"]
        if hasattr(raw_signal, "numpy"):
            raw_signal = raw_signal.numpy()
        raw_signal = np.asarray(raw_signal)
        if sample.eeg.sampling_rate != 120.0:
            raise ValueError(
                "NeuralBench MI EEGNet comparison requires the pinned 120 Hz "
                f"extractor, got {sample.eeg.sampling_rate} Hz"
            )
        expected_samples = 480
        if raw_signal.shape != (1, dm._num_channels, expected_samples):
            raise ValueError(
                "NeuralBench MI tensor must be (1, C, 480) for a 4-second "
                f"120 Hz EEGNet comparison, got {raw_signal.shape}"
            )
        np.testing.assert_array_equal(sample.eeg.signal, raw_signal[0].T)

        model = EEGNetEncoder(
            task_configs={task_config.name: task_config},
            num_channels=dm._num_channels,
            num_samples=expected_samples,
        )
        tokenized = model.tokenize(sample)
        np.testing.assert_array_equal(
            tokenized["input_values"].obj.numpy(), raw_signal[0].T
        )
        normalized = model._check_input_shape_conv2d(
            tokenized["input_values"].obj.unsqueeze(0)
        )
        np.testing.assert_array_equal(
            normalized.numpy(), raw_signal[np.newaxis, ...]
        )
        print(
            "    MI EEGNet parity: exact values/order preserved "
            f"(1, {dm._num_channels}, {expected_samples})"
        )

    return dm.label_map


def main():
    cache_dir = "/network/scratch/s/sobralm/neuralset-data"

    print("NeuralBench Phase 1 — Data Verification")
    print("=" * 60)

    results = {}

    # Motor Imagery
    try:
        mi_labels = verify_task(
            "motor_imagery", "schalk2004bci2000",
            "motor_imagery_trials", cache_dir
        )
        results["motor_imagery"] = {"status": "OK", "labels": mi_labels}
    except Exception as e:
        print(f"\n  ERROR: {e}")
        results["motor_imagery"] = {"status": "FAILED", "error": str(e)}

    # Sleep Stage
    try:
        sleep_labels = verify_task(
            "sleep_stage", "kemp2000analysis",
            "sleep_stages", cache_dir
        )
        results["sleep_stage"] = {"status": "OK", "labels": sleep_labels}
    except Exception as e:
        print(f"\n  ERROR: {e}")
        results["sleep_stage"] = {"status": "FAILED", "error": str(e)}

    print(f"\n\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for task, info in results.items():
        print(f"  {task}: {info['status']}")
        if info["status"] == "OK":
            print(f"    Labels: {info['labels']}")

    print("\n  Update task configs with discovered labels if needed.")


if __name__ == "__main__":
    main()
