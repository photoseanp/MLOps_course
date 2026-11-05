"""Baseline training script (toy loop, fixed seed)."""

import math
import random

import numpy as np

SEED = 42
LEARNING_RATE = 5e-2
EPOCHS = 10


def set_seed(seed: int) -> None:
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def cosine_lr_schedule(base_lr: float, epoch: int, total_epochs: int) -> float:
    """Cosine annealing learning rate schedule."""
    return base_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))


def train(lr: float = LEARNING_RATE, epochs: int = EPOCHS) -> dict[str, float]:
    """Run a toy training loop with cosine LR schedule and return final metrics."""
    set_seed(SEED)

    loss = 1.0
    for epoch in range(epochs):
        current_lr = cosine_lr_schedule(lr, epoch, epochs)
        loss *= 1 - current_lr
        print(f"epoch {epoch + 1}/{epochs}  lr={current_lr:.5f}  loss={loss:.4f}")

    return {"loss": loss, "lr": lr, "epochs": epochs}


if __name__ == "__main__":
    metrics = train()
    print(f"\nFinal loss: {metrics['loss']:.4f}")
