"""
PyTorch Lightning DataModule for synthetic spiral pattern data.

This module generates a 2D spiral pattern dataset for binary classification.
The spiral pattern is visually interesting and demonstrates how MLPs can learn
non-linear decision boundaries.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from typing import Optional


class SpiralDataset(Dataset):
    """
    Custom dataset that generates spiral pattern data.

    Two classes are arranged in interleaved spirals,
    making it a non-linearly separable problem.
    """

    def __init__(self, num_samples: int = 10000, noise: float = 0.2, seed: int = 42):
        """
        Args:
            num_samples: Total number of samples to generate
            noise: Standard deviation of Gaussian noise added to the spirals
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.num_samples = num_samples
        self.noise = noise
        self.seed = seed

        # Generate the data
        self.data, self.labels = self._generate_spiral_data()

    def _generate_spiral_data(self):
        """
        Generate two interleaved spirals.

        Returns:
            data: Tensor of shape (num_samples, 2) with x, y coordinates
            labels: Tensor of shape (num_samples,) with binary labels
        """
        np.random.seed(self.seed)

        n_per_class = self.num_samples // 2

        # Generate spiral for class 0
        theta_0 = np.linspace(0, 4 * np.pi, n_per_class)
        r_0 = np.linspace(0.1, 1, n_per_class)
        x_0 = r_0 * np.cos(theta_0) + np.random.randn(n_per_class) * self.noise
        y_0 = r_0 * np.sin(theta_0) + np.random.randn(n_per_class) * self.noise

        # Generate spiral for class 1 (offset by pi)
        theta_1 = np.linspace(0, 4 * np.pi, n_per_class) + np.pi
        r_1 = np.linspace(0.1, 1, n_per_class)
        x_1 = r_1 * np.cos(theta_1) + np.random.randn(n_per_class) * self.noise
        y_1 = r_1 * np.sin(theta_1) + np.random.randn(n_per_class) * self.noise

        # Combine both classes
        X = np.vstack([np.column_stack([x_0, y_0]),
                       np.column_stack([x_1, y_1])])
        y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

        # Shuffle the data
        indices = np.random.permutation(self.num_samples)
        X = X[indices]
        y = y[indices]

        # Convert to PyTorch tensors
        data = torch.from_numpy(X).float()
        labels = torch.from_numpy(y).long()

        return data, labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SpiralDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for spiral pattern dataset.

    This handles data preparation, splitting, and DataLoader creation
    for training, validation, and testing.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        noise: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
    ):
        """
        Args:
            num_samples: Total number of samples to generate
            noise: Noise level in the spiral pattern
            batch_size: Batch size for DataLoaders
            num_workers: Number of workers for DataLoaders
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_samples = num_samples
        self.noise = noise
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        # Datasets will be populated in setup()
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """
        Create datasets for training, validation, and testing.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        # Generate the full dataset
        full_dataset = SpiralDataset(
            num_samples=self.num_samples,
            noise=self.noise,
            seed=self.seed
        )

        # Calculate split sizes
        train_size = int(self.train_split * self.num_samples)
        val_size = int(self.val_split * self.num_samples)
        test_size = self.num_samples - train_size - val_size

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        """Return DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

    def val_dataloader(self):
        """Return DataLoader for validation."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

    def test_dataloader(self):
        """Return DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )


if __name__ == "__main__":
    """
    Test the DataModule and optionally visualize the spiral pattern.
    Run with: python src/datamodule.py
    """
    import matplotlib.pyplot as plt

    # Create data module
    dm = SpiralDataModule(num_samples=1000, batch_size=32)
    dm.setup()

    # Get some data
    train_loader = dm.train_dataloader()
    batch_x, batch_y = next(iter(train_loader))

    print(f"Batch shape: {batch_x.shape}")
    print(f"Labels shape: {batch_y.shape}")
    print(f"Sample data point: {batch_x[0]}")
    print(f"Sample label: {batch_y[0]}")

    # Visualize the spiral pattern
    full_dataset = SpiralDataset(num_samples=1000)
    X = full_dataset.data.numpy()
    y = full_dataset.labels.numpy()

    plt.figure(figsize=(8, 8))
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', alpha=0.6, s=20)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', alpha=0.6, s=20)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Spiral Pattern Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # Save the plot
    plt.savefig('spiral_pattern.png', dpi=150, bbox_inches='tight')
    print("\nSpiral pattern visualization saved to 'spiral_pattern.png'")

    print("\nDataModule test successful!")
