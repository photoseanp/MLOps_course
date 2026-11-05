import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models import SimpleClassifier


@pytest.mark.unit
def test_classifier_output_dimensions(
    sample_features: int, sample_classes: int
) -> None:
    """Unit: model returns one logit per class."""
    model = SimpleClassifier(in_features=sample_features, num_classes=sample_classes)
    logits = model(torch.randn(4, sample_features))
    assert logits.shape == (4, sample_classes)
    assert torch.isfinite(logits).all()


@pytest.mark.property
@settings(deadline=None, max_examples=50)
@given(batch_size=st.integers(min_value=1, max_value=64))
def test_classifier_batch_invariance(batch_size: int) -> None:
    """Property: any batch size works and gives [batch, num_classes]."""
    model = SimpleClassifier(in_features=4, num_classes=2)
    logits = model(torch.randn(batch_size, 4))
    assert logits.shape == (batch_size, 2)
