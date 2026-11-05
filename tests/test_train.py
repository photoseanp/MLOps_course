import math

import pytest
import torch

from src.dataset import SyntheticDataset, create_dataloader
from src.models import SimpleClassifier
from src.trainer import Trainer


@pytest.mark.smoke
def test_dry_run_fast_dev_run() -> None:
    """Dry run: one batch passes forward / loss / backward / step without errors."""
    loader = create_dataloader(
        SyntheticDataset(num_samples=8), batch_size=2, shuffle=False
    )
    trainer = Trainer(SimpleClassifier(in_features=4, num_classes=2), lr=0.01)

    history = trainer.fit(loader, fast_dev_run=True)

    assert len(history) == 1
    assert math.isfinite(history[0]["loss"])


@pytest.mark.smoke
def test_smoke_training_step_and_gradients() -> None:
    """Smoke: 1 epoch, 8 samples, batch 2.

    1) loss is finite  2) loss decreases  3) every parameter has a gradient.
    """
    loader = create_dataloader(
        SyntheticDataset(num_samples=8), batch_size=2, shuffle=False
    )
    model = SimpleClassifier(in_features=4, num_classes=2)
    trainer = Trainer(model=model, lr=0.05)

    initial_loss, final_loss = trainer.train_epoch_with_checks(loader)

    assert math.isfinite(final_loss)
    assert final_loss < initial_loss
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} got no gradient"


@pytest.mark.overfit
def test_overfit_single_batch(dummy_batch: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Overfit: the model memorises one batch (loss < 0.05)."""
    trainer = Trainer(
        SimpleClassifier(in_features=4, num_classes=2, hidden_dim=32), lr=0.05
    )
    final_loss = trainer.overfit_batch(dummy_batch, steps=300)
    assert final_loss < 0.05
