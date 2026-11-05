"""Training loop, smoke checks and single-batch overfitting."""

import torch
from torch import nn
from torch.utils.data import DataLoader

Batch = tuple[torch.Tensor, torch.Tensor]


class Trainer:
    """Runs forward / backward / optimizer step and simple sanity checks."""

    def __init__(self, model: nn.Module, lr: float = 0.01) -> None:
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def _step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(x), y)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def fit(
        self,
        dataloader: DataLoader[Batch],
        epochs: int = 1,
        fast_dev_run: bool = False,
    ) -> list[dict[str, float]]:
        """Train for `epochs` epochs and return per-epoch history.

        With ``fast_dev_run=True`` only one batch is processed (dry run),
        and a single-element history is returned.
        """
        self.model.train()
        if fast_dev_run:
            x, y = next(iter(dataloader))
            return [{"epoch": 0.0, "loss": self._step(x, y)}]

        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            total_loss = 0.0
            count = 0
            for x, y in dataloader:
                total_loss += self._step(x, y) * len(x)
                count += len(x)
            history.append({"epoch": float(epoch), "loss": total_loss / max(count, 1)})
        return history

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader[Batch]) -> dict[str, float]:
        """Return mean loss and accuracy on the dataloader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        count = 0
        for x, y in dataloader:
            logits = self.model(x)
            total_loss += float(self.criterion(logits, y).item()) * len(x)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            count += len(x)
        count = max(count, 1)
        return {"loss": total_loss / count, "accuracy": correct / count}

    def train_epoch_with_checks(
        self, dataloader: DataLoader[Batch]
    ) -> tuple[float, float]:
        """Smoke check: loss on the first batch before and after one epoch."""
        first_x, first_y = next(iter(dataloader))
        self.model.eval()
        with torch.no_grad():
            initial_loss = float(self.criterion(self.model(first_x), first_y).item())

        self.model.train()
        for x, y in dataloader:
            self._step(x, y)

        self.model.eval()
        with torch.no_grad():
            final_loss = float(self.criterion(self.model(first_x), first_y).item())
        return initial_loss, final_loss

    def overfit_batch(self, batch: Batch, steps: int = 200) -> float:
        """Repeatedly train on one batch; return the final loss."""
        x, y = batch
        self.model.train()
        loss = float("inf")
        for _ in range(steps):
            loss = self._step(x, y)
        return loss
