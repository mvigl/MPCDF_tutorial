# Understanding the Code

This guide explains the code structure and how PyTorch Lightning simplifies distributed training.

## Overview

The tutorial consists of three main Python files:
1. [src/datamodule.py](#datamodulepy) - Data generation and loading
2. [src/model.py](#modelpy) - Neural network definition
3. [src/train.py](#trainpy) - Training script

## datamodule.py

### Purpose
Generates synthetic spiral pattern data for binary classification.

### Key Components

#### SpiralDataset
```python
class SpiralDataset(Dataset):
    def __init__(self, num_samples=10000, noise=0.2, seed=42):
        # Generates two interleaved spirals
        # Class 0: one spiral
        # Class 1: another spiral (offset by π)
```

**How it works**:
- Creates points along two spirals using polar coordinates
- Adds Gaussian noise for variation
- Returns (x, y) coordinates and binary labels

#### SpiralDataModule
```python
class SpiralDataModule(pl.LightningDataModule):
    def setup(self, stage):
        # Splits data into train/val/test

    def train_dataloader(self):
        # Returns DataLoader for training
```

**Key features**:
- Handles data splitting (70% train, 15% val, 15% test)
- Creates DataLoaders with proper workers and pinning
- Works automatically with DDP (no changes needed!)

## model.py

### MLP Architecture

```python
class MLP(pl.LightningModule):
    def __init__(self, hidden_dims=[64, 32, 16]):
        # Input: 2D coordinates
        # Hidden: [64, 32, 16] with ReLU and Dropout
        # Output: 2 classes
```

**Architecture**:
```
Input (2) → Linear(64) → ReLU → Dropout
          → Linear(32) → ReLU → Dropout
          → Linear(16) → ReLU → Dropout
          → Linear(2)  → Output
```

### PyTorch Lightning Methods

#### training_step
```python
def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = F.cross_entropy(logits, y)
    self.log('train_loss', loss, sync_dist=True)
    return loss
```

**Key points**:
- `sync_dist=True`: Synchronizes metrics across all GPUs in DDP
- Lightning automatically handles gradient synchronization
- Loss is computed independently on each GPU, then averaged

#### configure_optimizers
```python
def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
```

**In DDP**:
- Each GPU has its own optimizer instance
- Gradients are synchronized before optimizer step
- Learning rate is the same across all GPUs

## train.py

### Main Training Script

This is the entry point that ties everything together.

### Command-Line Arguments

```python
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epochs', type=int, default=50)
# ... and more
```

**Important**: `batch_size` is **per GPU**
- 1 GPU: effective batch size = 32
- 4 GPUs: effective batch size = 128
- 8 GPUs: effective batch size = 256

### PyTorch Lightning Trainer

```python
trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    accelerator="auto",        # Detects GPU/CPU
    devices="auto",            # Uses all GPUs from SLURM
    num_nodes="auto",          # Detects number of nodes
    strategy=DDPStrategy(...), # Distributed training
    logger=comet_logger,       # Experiment tracking
)
```

**How SLURM integration works**:

Lightning reads these environment variables set by SLURM:
- `SLURM_NTASKS`: Total number of processes
- `SLURM_PROCID`: Process rank (0 to N-1)
- `SLURM_LOCALID`: Local rank on this node
- `SLURM_NODELIST`: List of nodes

You don't need to set these manually - `srun` does it!

### DDP Strategy

```python
strategy = DDPStrategy(
    find_unused_parameters=False,  # Optimization: all params are used
    static_graph=True,              # Graph doesn't change
)
```

**What DDP does**:
1. Copies model to each GPU
2. Splits batch across GPUs
3. Each GPU computes forward + backward pass
4. Synchronizes gradients via all-reduce
5. Each GPU updates its model (identically)

### Comet ML Logger

```python
comet_logger = CometLogger(
    api_key=api_key,
    project_name=args.comet_project,
    experiment_name=args.experiment_name,
)
```

**Automatic features**:
- Only rank 0 logs to Comet (avoids duplicates)
- Logs hyperparameters, metrics, system info
- Creates comparison dashboards

## How DDP Works Behind the Scenes

### Single GPU
```
[GPU 0] ← Full batch (32 samples)
```

### Multi-GPU (4 GPUs)
```
Batch (128 samples) split into 4 mini-batches:
[GPU 0] ← 32 samples
[GPU 1] ← 32 samples
[GPU 2] ← 32 samples
[GPU 3] ← 32 samples

Each GPU:
1. Forward pass on its mini-batch
2. Compute loss
3. Backward pass → gradients

All-Reduce:
4. Average gradients across GPUs

Each GPU:
5. Update model with averaged gradients
```

### Multi-Node (2 nodes, 8 GPUs)
```
Node 0:                    Node 1:
[GPU 0] ← 32 samples      [GPU 4] ← 32 samples
[GPU 1] ← 32 samples      [GPU 5] ← 32 samples
[GPU 2] ← 32 samples      [GPU 6] ← 32 samples
[GPU 3] ← 32 samples      [GPU 7] ← 32 samples

Communication:
- Within node: NVLink (fast, ~600 GB/s)
- Between nodes: InfiniBand (slower, 25 GB/s)
```

## Key Takeaways

### 1. No Code Changes Needed for Scaling

The same `train.py` works for:
- Single GPU
- Multi-GPU (1 node)
- Multi-node

Lightning automatically detects the configuration from SLURM!

### 2. Effective Batch Size

Remember: `effective_batch_size = batch_size × num_gpus`

This affects:
- Convergence speed (larger batches may need more epochs)
- Memory usage (per-GPU memory stays the same)
- Learning rate (may need adjustment for very large batches)

### 3. Logging and Synchronization

Use `sync_dist=True` when logging metrics:
```python
self.log('val_loss', loss, sync_dist=True)
```

This ensures metrics are properly averaged across all GPUs.

### 4. Data Loading

DataLoader automatically handles distributed sampling:
- Each GPU gets different data samples
- No overlap between GPUs
- Proper shuffling is maintained

## Code Flow

Here's what happens when you run a training job:

1. **SLURM allocates resources** (GPUs, CPUs, nodes)
2. **`srun` launches processes** (one per GPU)
3. **Each process**:
   - Loads modules
   - Activates venv
   - Runs `train.py`
4. **PyTorch Lightning**:
   - Detects SLURM environment
   - Initializes DDP
   - Sets up communication
5. **Training loop**:
   - Each GPU gets different data
   - Forward/backward pass
   - Gradient synchronization
   - Model update
6. **Logging**:
   - Only rank 0 logs to Comet ML
   - All ranks log to console

## Next Steps

Now that you understand the code:
- Run [single GPU training](03_single_gpu.md)
- Scale to [multi-GPU](04_multi_gpu.md)
- Try [multi-node](05_multi_node.md)

## Further Reading

- [PyTorch Lightning DDP Documentation](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Backend](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
