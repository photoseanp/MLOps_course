import pytest
import torch


@pytest.fixture(autouse=True)
def _fix_seed() -> None:
    """Make every test reproducible."""
    torch.manual_seed(0)


@pytest.fixture
def sample_features() -> int:
    return 4


@pytest.fixture
def sample_classes() -> int:
    return 2


@pytest.fixture
def dummy_batch(
    sample_features: int, sample_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """One batch: X [8, in_features], y [8]."""
    x = torch.randn(8, sample_features)
    y = torch.randint(0, sample_classes, (8,))
    return x, y
