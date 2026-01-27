"""
Main training script for MLP on spiral pattern data.

This script demonstrates training with PyTorch Lightning on the MPCDF Raven cluster
with automatic DDP scaling across multiple GPUs and nodes.
"""

import os
import argparse
import yaml
import time
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.strategies import DDPStrategy

from datamodule import SpiralDataModule
from model import MLP


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MLP on spiral pattern data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.2,
        help="Noise level in spiral pattern"
    )

    # Model arguments
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[64, 32, 16],
        help="Hidden layer dimensions (space-separated)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )

    # Training arguments
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers"
    )

    # Logging arguments
    parser.add_argument(
        "--comet_api_key",
        type=str,
        default=None,
        help="Comet ML API key (or set COMET_API_KEY env var)"
    )
    parser.add_argument(
        "--comet_project",
        type=str,
        default="mpcdf-raven-tutorial",
        help="Comet ML project name"
    )
    parser.add_argument(
        "--comet_workspace",
        type=str,
        default=None,
        help="Comet ML workspace (optional)"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for this experiment (auto-generated if not provided)"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for outputs (checkpoints, logs)"
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )

    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Load config file if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Update args with config values (CLI args take precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    return args


def setup_comet_logger(args):
    """
    Setup Comet ML logger.

    Args:
        args: Parsed command line arguments

    Returns:
        CometLogger instance or None if Comet is disabled
    """
    # Check if API key is available
    api_key = args.comet_api_key or os.environ.get("COMET_API_KEY")

    if api_key is None:
        print("WARNING: COMET_API_KEY not set. Logging to Comet ML disabled.")
        print("To enable Comet ML logging, set the COMET_API_KEY environment variable")
        print("or pass --comet_api_key argument.")
        return None

    # Generate experiment name if not provided
    if args.experiment_name is None:
        # Use SLURM environment variables (available before DDP initialization)
        num_gpus = int(os.environ.get("SLURM_NTASKS", 1))
        num_nodes = int(os.environ.get("SLURM_NNODES", 1))

        if num_nodes > 1:
            args.experiment_name = f"spiral_mlp_{num_nodes}nodes_{num_gpus}gpus"
        elif num_gpus > 1:
            args.experiment_name = f"spiral_mlp_{num_gpus}gpus_1node"
        else:
            args.experiment_name = "spiral_mlp_1gpu"

    # Create Comet logger
    comet_logger = CometLogger(
        api_key=api_key,
        project=args.comet_project,  # Updated from project_name
        workspace=args.comet_workspace,
        name=args.experiment_name,  # Updated from experiment_name
        offline_directory=args.output_dir,  # Updated from save_dir
    )

    # Log hyperparameters
    comet_logger.log_hyperparams(vars(args))

    return comet_logger


def print_environment_info():
    """Print information about the training environment."""
    print("\n" + "=" * 70)
    print("ENVIRONMENT INFORMATION")
    print("=" * 70)

    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch Lightning version: {pl.__version__}")

    # CUDA information
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # SLURM information
    print("\nSLURM Environment:")
    slurm_vars = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURM_NNODES",
        "SLURM_NTASKS",
        "SLURM_NTASKS_PER_NODE",
        "SLURM_CPUS_PER_TASK",
    ]
    for var in slurm_vars:
        value = os.environ.get(var, "Not set")
        print(f"  {var}: {value}")

    # DDP information
    print("\nDistributed Training:")
    print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 1)}")
    print(f"  RANK: {os.environ.get('RANK', 0)}")
    print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 0)}")
    print(f"  LOCAL_WORLD_SIZE: {os.environ.get('LOCAL_WORLD_SIZE', 1)}")

    print("=" * 70 + "\n")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Set seed for reproducibility
    pl.seed_everything(args.seed)

    # Print environment information (only on rank 0)
    if int(os.environ.get("RANK", 0)) == 0:
        print_environment_info()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Setup Comet ML logger
    logger = setup_comet_logger(args)

    # Create data module
    print("Setting up data module...")
    data_module = SpiralDataModule(
        num_samples=args.num_samples,
        noise=args.noise,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Create model
    print("Creating model...")
    model = MLP(
        input_dim=2,
        hidden_dims=args.hidden_dims,
        output_dim=2,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
    )

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="spiral_mlp-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Setup DDP strategy
    # PyTorch Lightning will automatically detect SLURM environment
    strategy = DDPStrategy(
        find_unused_parameters=False,
        static_graph=True,  # Optimization for fixed architecture
    )

    # Detect number of nodes from SLURM environment
    # Lightning can auto-detect, but we explicitly set it for clarity
    num_nodes = int(os.environ.get("SLURM_NNODES", 1))

    # Create trainer
    print("Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",  # Automatically detect GPU/CPU
        devices="auto",  # Use all available GPUs in SLURM allocation
        num_nodes=num_nodes,  # Explicitly set from SLURM (not "auto")
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,  # For reproducibility
    )

    # Print training configuration
    if int(os.environ.get("RANK", 0)) == 0:
        print("\n" + "=" * 70)
        print("TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Max epochs: {args.max_epochs}")
        print(f"Batch size (per GPU): {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Hidden dimensions: {args.hidden_dims}")
        print(f"Dropout: {args.dropout}")
        print(f"Number of samples: {args.num_samples}")
        print(f"Noise level: {args.noise}")
        print(f"Number of workers: {args.num_workers}")
        print(f"Output directory: {args.output_dir}")

        # Calculate effective batch size
        world_size = trainer.world_size
        effective_batch_size = args.batch_size * world_size
        print(f"\nEffective batch size (batch_size × num_gpus): {effective_batch_size}")
        print("=" * 70 + "\n")

    # Start timing
    start_time = time.time()

    # Train the model
    print("Starting training...")
    trainer.fit(model, data_module)

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    # Print training summary
    if int(os.environ.get("RANK", 0)) == 0:
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Time per epoch: {total_time/args.max_epochs:.2f} seconds")
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
        print("=" * 70 + "\n")

        # Log timing to Comet
        if logger is not None:
            logger.experiment.log_metric("total_training_time", total_time)
            logger.experiment.log_metric("time_per_epoch", total_time / args.max_epochs)

    # Test the model
    print("Running test...")
    trainer.test(model, data_module)

    print("\nDone!")


if __name__ == "__main__":
    main()
