#!/bin/bash -l
#
# SLURM job script for multi-GPU training on a single node (MPCDF Raven cluster)
#
# This script demonstrates training with DDP (Distributed Data Parallel)
# across 4 A100 GPUs on a single node.
#
# Key features:
# - Uses all 4 GPUs on one node
# - PyTorch Lightning automatically handles DDP setup
# - Effective batch size = batch_size × 4 GPUs
# - Expected speedup: ~3.5-4x compared to single GPU
#

#SBATCH --job-name=mlp_multi_gpu           # Job name
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=4                 # Number of tasks per node (1 per GPU)
#SBATCH --cpus-per-task=18                  # CPUs per task (18 × 4 = 72 total)
#SBATCH --gres=gpu:a100:4                   # Request all 4 A100 GPUs
#SBATCH --constraint="gpu"                  # Request GPU nodes
#SBATCH --mem=500000                        # Memory per node in MB (500 GB for full node)
#SBATCH --time=00:10:00                     # Maximum runtime (10 min)
#SBATCH --output=outputs/slurm-%j.out       # Standard output (%j = job ID)
#SBATCH --error=outputs/slurm-%j.err        # Standard error

# =============================================================================
# IMPORTANT NOTES:
# =============================================================================
# - This script will use an EXCLUSIVE node (all 4 GPUs reserved for this job)
# - The effective batch size is (batch_size × num_gpus)
# - PyTorch Lightning automatically detects SLURM DDP environment
# - GPUs are connected via NVLink for fast communication
# - Make sure you have created the virtual environment first:
#   $ bash setup_venv.sh
# - Set your COMET_API_KEY environment variable before submitting (optional):
#   $ export COMET_API_KEY='your-api-key-here'
# =============================================================================

echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node(s): $SLURM_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total CPUs: $((SLURM_NTASKS_PER_NODE * SLURM_CPUS_PER_TASK))"
echo "========================================"
echo ""

# Load required modules
echo "Loading modules..."
module purge
module load gcc/13 cuda/12.6

echo "Loaded modules:"
module list
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Verify Python and package versions
echo "Python environment:"
python --version
echo ""

# Display GPU information
echo "========================================"
echo "GPU Information"
echo "========================================"
nvidia-smi
echo "========================================"
echo ""

# DDP configuration info
echo "========================================"
echo "DDP Configuration"
echo "========================================"
echo "WORLD_SIZE: 4 GPUs on 1 node"
echo "Batch size per GPU: 32"
echo "Effective batch size: 32 × 4 = 128"
echo "Communication: NVLink (fast intra-node)"
echo "========================================"
echo ""

# Record start time
echo "Training started at: $(date)"
START_TIME=$(date +%s)

# Run training script with srun
# srun will launch 4 processes (1 per GPU)
# PyTorch Lightning will automatically:
# - Detect SLURM environment variables
# - Initialize DDP across the 4 GPUs
# - Synchronize gradients after each batch
echo "Starting multi-GPU training..."
srun python src/train.py \
    --max_epochs 50 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --hidden_dims 64 32 16 \
    --dropout 0.2 \
    --num_samples 10000 \
    --num_workers 4 \
    --output_dir outputs

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "Training completed at: $(date)"
echo "Total duration: ${DURATION} seconds ($(echo "scale=2; ${DURATION}/60" | bc) minutes)"
echo "========================================"

# Print GPU utilization one more time
echo ""
echo "Final GPU status:"
nvidia-smi

echo ""
echo "Job completed successfully!"
echo "Expected speedup: ~3.5-4x compared to single GPU"
echo "Check Comet ML dashboard for detailed metrics and visualizations"
