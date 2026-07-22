from pathlib import Path

import pytest


@pytest.fixture
def data_root():
    """Return the root directory for processed datasets."""
    return Path("data/processed")


@pytest.fixture
def embed_dim():
    """Standard embedding dimension for model tests."""
    return 256


@pytest.fixture
def batch_size():
    """Standard batch size for model tests."""
    return 2


def skip_if_missing_dataset(
    dataset_name: str, data_root: Path
) -> pytest.MarkDecorator:
    """
    Create a pytest skip marker if the dataset folder doesn't exist.

    Args:
        dataset_name: Name of the dataset directory to check
        data_root: Root directory containing dataset folders

    Returns:
        pytest.mark.skipif decorator that skips the test if dataset is missing
    """
    dataset_path = data_root / dataset_name
    return pytest.mark.skipif(
        not dataset_path.exists(), reason=f"Dataset not found at {dataset_path}"
    )
