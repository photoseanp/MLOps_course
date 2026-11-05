"""Training with PyTorch Lightning + mlflow.pytorch.autolog (no manual log_* calls).

Run (with the tracking server started, see README):
    uv run python -m src.train_autolog
"""

import io
import os
import sys

import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
import torch
from torch import nn

from src.dataset import SyntheticDataset, create_dataloader
from src.models import SimpleClassifier

for stream in (sys.stdout, sys.stderr):
    if isinstance(stream, io.TextIOWrapper):
        stream.reconfigure(encoding="utf-8")

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "pytorch_autolog_experiment"


class LightningClassifier(pl.LightningModule):
    """Lightning wrapper around SimpleClassifier."""

    def __init__(
        self,
        in_features: int = 8,
        num_classes: int = 2,
        hidden_dim: int = 32,
        lr: float = 0.01,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()  # autolog picks these up as params
        self.model = SimpleClassifier(in_features, num_classes, hidden_dim=hidden_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        # Lightning metrics are captured by MLflow autolog automatically
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", acc, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def run_autolog_training(lr: float = 0.01, max_epochs: int = 5) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.pytorch.autolog(log_models=True, checkpoint=False)

    pl.seed_everything(42, verbose=False)
    dataset = SyntheticDataset(num_samples=200, in_features=8)
    loader = create_dataloader(dataset, batch_size=16, shuffle=True)
    model = LightningClassifier(in_features=8, num_classes=2, hidden_dim=32, lr=lr)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=False,  # autolog hooks into the trainer itself
        enable_checkpointing=False,
        default_root_dir="output/lightning",
        log_every_n_steps=1,
    )

    with mlflow.start_run(run_name=f"autolog_lr_{lr}"):
        trainer.fit(model, loader)

    print(f"Autologging run for lr={lr} finished.")


if __name__ == "__main__":
    print("--- Autolog run: lr=0.02 ---")
    run_autolog_training(lr=0.02, max_epochs=5)

    print("--- Autolog run: lr=0.002 ---")
    run_autolog_training(lr=0.002, max_epochs=5)
