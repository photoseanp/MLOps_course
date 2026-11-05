import pytest
import torch

from src.dataset import SyntheticDataset, create_dataloader


@pytest.mark.unit
def test_dataset_len_and_item() -> None:
    """Unit: dataset length and sample shapes are correct."""
    ds = SyntheticDataset(num_samples=20, in_features=4)
    assert len(ds) == 20
    x, y = ds[0]
    assert x.shape == (4,)
    assert y.item() in (0, 1)


@pytest.mark.unit
def test_dataset_is_reproducible() -> None:
    """Unit: same seed gives the same data."""
    a = SyntheticDataset(num_samples=10, in_features=4, seed=1)
    b = SyntheticDataset(num_samples=10, in_features=4, seed=1)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.y, b.y)


@pytest.mark.unit
def test_dataloader_batching() -> None:
    """Unit: 10 samples with batch_size=4 give batches 4 + 4 + 2."""
    ds = SyntheticDataset(num_samples=10, in_features=4)
    batches = list(create_dataloader(ds, batch_size=4, shuffle=False))
    assert [len(x) for x, _ in batches] == [4, 4, 2]
    first_x, first_y = batches[0]
    assert first_x.shape == (4, 4)
    assert first_y.shape == (4,)
