"""Baseline training script (toy loop, fixed seed)."""

import random

import numpy as np

SEED = 42
LEARNING_RATE = 1e-3
EPOCHS = 5


def set_seed(seed: int) -> None:
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def train(lr: float = LEARNING_RATE, epochs: int = EPOCHS) -> dict[str, float]:
    """Run a toy training loop and return final metrics."""
    set_seed(SEED)

    loss = 1.0
    for epoch in range(epochs):
        loss *= 1 - lr
        print(f"epoch {epoch + 1}/{epochs}  loss={loss:.4f}")

    return {"loss": loss, "lr": lr, "epochs": epochs}


if __name__ == "__main__":
    metrics = train()
    print(f"\nFinal loss: {metrics['loss']:.4f}")
