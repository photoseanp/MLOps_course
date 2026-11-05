import pytest
import torch

from src.dataset import SyntheticDataset, create_dataloader
from src.models import SimpleClassifier
from src.trainer import Trainer


@pytest.mark.e2e
def test_full_training_pipeline_e2e() -> None:
    """E2E: data -> training -> evaluation -> inference."""
    # Arrange
    loader = create_dataloader(SyntheticDataset(num_samples=64), batch_size=8)
    model = SimpleClassifier(in_features=4, num_classes=2)
    trainer = Trainer(model=model, lr=0.02)

    # Act
    history = trainer.fit(loader, epochs=5)
    metrics = trainer.evaluate(loader)

    # Assert: training converges
    assert len(history) == 5
    assert history[-1]["loss"] < history[0]["loss"]
    assert metrics["accuracy"] > 0.8

    # Assert: inference works
    model.eval()
    with torch.no_grad():
        preds = model(torch.randn(1, 4))
    assert preds.shape == (1, 2)
