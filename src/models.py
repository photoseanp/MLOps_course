"""Neural network building blocks and models."""

import torch
from torch import nn


class LinearBlock(nn.Module):
    """Fully connected block: Linear layer + ReLU activation."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class SimpleClassifier(nn.Module):
    """Classifier: LinearBlock (hidden layer) + linear head producing logits."""

    def __init__(
        self, in_features: int, num_classes: int, hidden_dim: int = 16
    ) -> None:
        super().__init__()
        self.block = LinearBlock(in_features=in_features, out_features=hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(x))
