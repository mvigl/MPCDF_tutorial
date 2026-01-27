"""
PyTorch Lightning MLP model for binary classification.

This module defines a simple Multi-Layer Perceptron (MLP) that can learn
non-linear decision boundaries for the spiral pattern dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Optional
from torchmetrics import Accuracy


class MLP(pl.LightningModule):
    """
    Multi-Layer Perceptron for binary classification.

    This model demonstrates scaling across GPUs using PyTorch Lightning's
    automatic DDP (Distributed Data Parallel) support.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: List[int] = [64, 32, 16],
        output_dim: int = 2,
        learning_rate: float = 1e-3,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Number of input features (2 for spiral data)
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output classes (2 for binary classification)
            learning_rate: Learning rate for optimizer
            dropout: Dropout probability for regularization
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.dropout_prob = dropout

        # Build the network layers
        layers = []
        prev_dim = input_dim

        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=output_dim)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=output_dim)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=output_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, output_dim)
        """
        return self.network(x)

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.

        Args:
            batch: Tuple of (data, labels)
            batch_idx: Index of the batch

        Returns:
            Loss value for this batch
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)

        # Log metrics
        # Lightning automatically handles logging only on rank 0 in DDP
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.

        Args:
            batch: Tuple of (data, labels)
            batch_idx: Index of the batch
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """
        Test step for a single batch.

        Args:
            batch: Tuple of (data, labels)
            batch_idx: Index of the batch
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, y)

        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            Optimizer instance
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch.
        Used to compute epoch-level metrics.
        """
        # Compute and reset metrics
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        """
        Called at the end of each validation epoch.
        Used to compute epoch-level metrics.
        """
        # Compute and reset metrics
        self.val_accuracy.reset()

    def on_test_epoch_end(self):
        """
        Called at the end of testing.
        Used to compute test metrics.
        """
        # Compute and reset metrics
        self.test_accuracy.reset()


if __name__ == "__main__":
    """
    Test the model architecture.
    Run with: python src/model.py
    """
    # Create a sample model
    model = MLP(
        input_dim=2,
        hidden_dims=[64, 32, 16],
        output_dim=2,
        learning_rate=1e-3,
        dropout=0.2
    )

    print("Model Architecture:")
    print(model)
    print("\nNumber of parameters:")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 2)
    y = torch.randint(0, 2, (batch_size,))

    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")

    with torch.no_grad():
        logits = model(x)

    print(f"  Output shape: {logits.shape}")
    print(f"  Output sample: {logits[0]}")

    # Test training step
    batch = (x, y)
    loss = model.training_step(batch, 0)
    print(f"\n  Loss: {loss.item():.4f}")

    print("\nModel test successful!")
