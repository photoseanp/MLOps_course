import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models import LinearBlock


@pytest.mark.unit
def test_linear_block_forward_shape() -> None:
    """Unit: output has shape [batch, out_features]."""
    block = LinearBlock(in_features=4, out_features=8)
    out = block(torch.randn(2, 4))
    assert out.shape == (2, 8)


@pytest.mark.unit
def test_linear_block_relu_non_negative() -> None:
    """Unit: ReLU output is never negative."""
    block = LinearBlock(in_features=4, out_features=8)
    assert (block(torch.randn(16, 4)) >= 0).all()


@pytest.mark.property
@settings(deadline=None, max_examples=50)
@given(
    batch_size=st.integers(min_value=1, max_value=32),
    in_features=st.integers(min_value=1, max_value=16),
    out_features=st.integers(min_value=1, max_value=16),
)
def test_linear_block_shape_invariant(
    batch_size: int, in_features: int, out_features: int
) -> None:
    """Property: shape is preserved for any dimensions; values are finite."""
    block = LinearBlock(in_features=in_features, out_features=out_features)
    out = block(torch.randn(batch_size, in_features))
    assert out.shape == (batch_size, out_features)
    assert torch.isfinite(out).all()
