"""Synthetic dataset and DataLoader utilities."""

import torch
from torch.utils.data import DataLoader, Dataset

Sample = tuple[torch.Tensor, torch.Tensor]


class SyntheticDataset(Dataset[Sample]):
    """Reproducible, linearly separable binary classification dataset."""

    def __init__(
        self,
        num_samples: int = 100,
        in_features: int = 4,
        seed: int = 42,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(num_samples, in_features, generator=generator)
        self.y = (self.x.sum(dim=1) > 0).long()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Sample:
        return self.x[index], self.y[index]


def create_dataloader(
    dataset: Dataset[Sample],
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader[Sample]:
    """Wrap a dataset into a DataLoader with the given batch size."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
