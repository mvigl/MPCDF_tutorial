# MPCDF Raven Cluster Tutorial: PyTorch Lightning with Multi-GPU Training

A comprehensive tutorial for training deep learning models on the MPCDF Raven HPC cluster using PyTorch Lightning with distributed training across single GPU, multi-GPU, and multi-node configurations.

## Overview

This repository demonstrates how to:
- Train a neural network on the Raven cluster using SLURM
- Scale training from **single GPU** to **multi-GPU** to **multi-node** setups
- Use PyTorch Lightning's automatic DDP (Distributed Data Parallel)
- Track experiments with Comet ML
- Benchmark and compare performance across different configurations

### Learning Objectives

After completing this tutorial, you will understand:
- How to set up a Python environment on Raven
- How to submit SLURM jobs for GPU training
- How DDP works and when to use it
- How to scale from 1 GPU to multiple nodes
- How to monitor GPU utilization and training performance

## Quick Start

### 1. Clone and Setup

```bash
# SSH to Raven cluster
ssh <username>@gate.mpcdf.mpg.de
ssh <username>@raven.mpcdf.mpg.de

# Clone this repository
cd /u/<username>/
git clone <repository-url> MPCDF_tutorial
cd MPCDF_tutorial

# Create virtual environment and install dependencies
bash setup_venv.sh
```

### 2. (Optional) Set up Comet ML - super recommended!!!

Get your account (acceemics get pro for free!) [here](https://www.comet.com/site/products/opik/?utm_medium=ppc&utm_campaign=Opik-Tier1PromptTerms-ProductLP&utm_term=playground%20ai%20prompts&utm_source=adwords&hsa_ad=745367345237&hsa_kw=playground%20ai%20prompts&hsa_net=adwords&hsa_tgt=kwd-3500001&hsa_grp=181543648447&hsa_src=g&hsa_ver=3&hsa_cam=22433031556&hsa_mt=b&hsa_acc=3908332976&gad_source=1&gad_campaignid=22433031556&gbraid=0AAAAACv6e9wuOut08yW_V64Oi6oLSUg6u&gclid=Cj0KCQiA4eHLBhCzARIsAJ2NZoKjyirWZicRvtlWl2z88rghsIsid01ZnXzHHyUHYLY6_1-YOhwVwhsaAuO4EALw_wcB) 

```bash
# Set your Comet ML API key for experiment tracking
export COMET_API_KEY='your-api-key-here'

# Add to ~/.bashrc for persistence
echo "export COMET_API_KEY='your-api-key-here'" >> ~/.bashrc
```
Now your experiments (losses, gpu usage etc.) will be automatically logged online :)
### 3. Submit a Training Job

```bash
# Single GPU (baseline)
sbatch slurm_jobs/single_gpu.sh

# Multi-GPU (4 GPUs on one node)
sbatch slurm_jobs/multi_gpu_single_node.sh

# Multi-node (2 nodes, 8 GPUs total)
sbatch slurm_jobs/multi_node_multi_gpu.sh
```

### 4. Monitor Your Job

```bash
# Check job status
squeue -u $USER

# View output (replace <job_id> with your actual job ID)
tail -f outputs/slurm-<job_id>.out

# Cancel a job
scancel <job_id>
```

### 5. Benchmark Results

#### Just look at your comet page online!

## Repository Structure

```
MPCDF_tutorial/
├── README.md                          # This file
├── setup_venv.sh                      # Virtual environment setup script
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore patterns
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── datamodule.py                  # Spiral pattern dataset
│   ├── model.py                       # MLP model definition
│   ├── train.py                       # Main training script
│   └── benchmark.py                   # Performance comparison tool
│
├── slurm_jobs/                        # SLURM job scripts
│   ├── single_gpu.sh                  # 1 GPU training
│   ├── multi_gpu_single_node.sh       # 4 GPUs on 1 node
│   └── multi_node_multi_gpu.sh        # 2 nodes × 4 GPUs
│
├── configs/                           # Configuration files
│   └── default_config.yaml            # Default hyperparameters
│
├── docs/                              # Detailed documentation
│   ├── 01_setup.md                    # Setup instructions
│   ├── 02_understanding_code.md       # Code walkthrough
│   ├── 03_single_gpu.md               # Single GPU tutorial
│   ├── 04_multi_gpu.md                # Multi-GPU tutorial
│   ├── 05_multi_node.md               # Multi-node tutorial
│   ├── 06_benchmarking.md             # Benchmarking guide
│   └── 07_troubleshooting.md          # Common issues
│
└── outputs/                           # Training outputs
    ├── checkpoints/                   # Model checkpoints
    └── slurm-*.out                    # SLURM logs
```

## The Problem: Spiral Pattern Classification

This tutorial uses a **2D spiral pattern dataset** for binary classification:
- Two classes arranged in interleaved spirals
- Small and fast to train (focuses on HPC concepts, not ML complexity)
- Visually intuitive for understanding model performance

### Model Architecture

- **Input**: 2D coordinates (x, y)
- **Hidden layers**: [64, 32, 16] neurons with ReLU activation
- **Output**: 2 classes (binary classification)

## Training Configurations

### 1. Single GPU (Baseline)
- **Hardware**: 1 A100 GPU (40GB), 18 CPU cores
- **Use case**: Development, debugging, small models
- **Submit**: `sbatch slurm_jobs/single_gpu.sh`

### 2. Multi-GPU Single Node (4 GPUs)
- **Hardware**: 4 A100 GPUs, 72 CPU cores (full node)
- **Speedup**: ~3.5-4× compared to single GPU
- **Use case**: Medium to large models, faster experimentation
- **Submit**: `sbatch slurm_jobs/multi_gpu_single_node.sh`

### 3. Multi-Node (2 nodes, 8 GPUs)
- **Hardware**: 2 nodes × 4 A100 GPUs = 8 GPUs total
- **Speedup**: ~6-7× compared to single GPU
- **Use case**: Large models, very large datasets
- **Submit**: `sbatch slurm_jobs/multi_node_multi_gpu.sh`

## Key Features

### PyTorch Lightning Integration
- Automatic DDP setup (no manual distributed code!)
- Works seamlessly with SLURM environment
- Handles multi-node communication automatically
- Unified interface across all configurations

### Experiment Tracking with Comet ML
- Automatic logging of metrics, hyperparameters, and system info
- Compare runs across different GPU configurations
- Visualize training progress in real-time
- Share results with your team

### Comprehensive Documentation
- Step-by-step tutorials for each configuration
- Detailed explanations of DDP and scaling
- Troubleshooting guide for common issues
- Best practices for Raven cluster

## Documentation

For detailed information, see the [docs/](docs/) directory:

1. **[Setup Guide](docs/01_setup.md)** - Environment setup and installation
2. **[Understanding the Code](docs/02_understanding_code.md)** - Code walkthrough
3. **[Single GPU Training](docs/03_single_gpu.md)** - Baseline training
4. **[Multi-GPU Training](docs/04_multi_gpu.md)** - DDP on one node
5. **[Multi-Node Training](docs/05_multi_node.md)** - DDP across nodes
6. **[Benchmarking](docs/06_benchmarking.md)** - Performance comparison
7. **[Troubleshooting](docs/07_troubleshooting.md)** - Common issues and solutions

## Expected Performance

| Configuration | GPUs | Nodes | Speedup | Efficiency |
|--------------|------|-------|---------|------------|
| Single GPU | 1 | 1 | 1.0× | 100% |
| Multi-GPU | 4 | 1 | ~3.5× | ~88% |
| Multi-Node | 8 | 2 | ~6.0× | ~75% |

**Notes:**
- Speedup is relative to single GPU baseline
- Efficiency = (Speedup / Number of GPUs) × 100%
- Near-linear scaling within a node (thanks to NVLink)
- Slight overhead for multi-node (InfiniBand communication)

## Hardware Specifications (Raven Cluster)

### GPU Nodes
- **GPU**: 4× Nvidia A100 (40 GB HBM2) per node
- **CPU**: Intel Xeon IceLake Platinum 8360Y (72 cores per node)
- **RAM**: 512 GB per node
- **GPU Interconnect**: NVLink 3 (within node)
- **Network**: HDR InfiniBand (200 Gbit/s standard, 400 Gbit/s high-bandwidth)

### Resource Allocation
- **1 GPU jobs**: 18 CPUs, 125 GB RAM (shared node)
- **4 GPU jobs**: 72 CPUs, 500 GB RAM (exclusive node)
- **Multi-node jobs**: Up to 80 nodes (320 GPUs) for standard, 16 nodes for high-BW

## Prerequisites

- Access to MPCDF Raven cluster
- Basic knowledge of Python and PyTorch
- Familiarity with command line and SSH
- (Optional) Comet ML account for experiment tracking

## Tips for Success

1. **Start small**: Begin with single GPU to verify your code works
2. **Use gpudev**: Test with the `gpudev` partition (15 min limit) for quick debugging
3. **Monitor resources**: Check GPU utilization to ensure efficient use
4. **Batch size**: Remember that effective batch size = batch_size × num_gpus
5. **Checkpointing**: Save checkpoints regularly for long-running jobs
6. **File locations**: Use `/ptmp` for temporary outputs, `/u` for code and important results

## Common Questions

**Q: Do I need to modify my code for multi-GPU training?**
A: No! PyTorch Lightning handles all DDP setup automatically.

**Q: Why is multi-node speedup less than linear?**
A: Inter-node communication is slower than intra-node (InfiniBand vs NVLink).

**Q: Can I use more than 2 nodes?**
A: Yes! Just change `--nodes=N` in the SLURM script.

**Q: How do I check my job's GPU utilization?**
A: Use `nvidia-smi` in your SLURM script or check the logs.

**Q: What if my job runs out of memory?**
A: Reduce batch size or request more memory with `#SBATCH --mem=<amount>`

## Troubleshooting

If you encounter issues:

1. Check [docs/07_troubleshooting.md](docs/07_troubleshooting.md)
2. Verify environment setup: `source venv/bin/activate && python -c "import torch; print(torch.cuda.is_available())"`
3. Check SLURM logs: `cat outputs/slurm-<job_id>.err`
4. Verify module loading: `module list` should show `gcc/13` and `cuda/12.6`


## Happy Training! ⚽

<p align="center">
  <img src="strunz.gif" alt="Fast training celebration" width="400"/>
</p>

---

