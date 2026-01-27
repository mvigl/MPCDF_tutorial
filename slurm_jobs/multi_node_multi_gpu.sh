#!/bin/bash -l
#
# SLURM job script for multi-node multi-GPU training (MPCDF Raven cluster)
#
# This script demonstrates training with DDP (Distributed Data Parallel)
# across 2 nodes with 4 A100 GPUs each (8 GPUs total).
#
# Key features:
# - Uses 2 nodes, 4 GPUs per node (8 GPUs total)
# - PyTorch Lightning automatically handles multi-node DDP setup
# - Effective batch size = batch_size × 8 GPUs
# - Expected speedup: ~6-7x compared to single GPU
# - Inter-node communication via HDR200 InfiniBand (200 Gbit/s)
#

#SBATCH --job-name=mlp_multi_node          # Job name
#SBATCH --nodes=2                           # Number of nodes
#SBATCH --ntasks-per-node=4                 # Number of tasks per node (1 per GPU)
#SBATCH --cpus-per-task=18                  # CPUs per task (18 × 4 = 72 per node)
#SBATCH --gres=gpu:a100:4                   # Request all 4 A100 GPUs per node
#SBATCH --constraint="gpu"                  # Request GPU nodes
#SBATCH --mem=500000                        # Memory per node in MB (500 GB per node)
#SBATCH --time=00:10:00                     # Maximum runtime (10 min)
#SBATCH --output=outputs/slurm-%j.out       # Standard output (%j = job ID)
#SBATCH --error=outputs/slurm-%j.err        # Standard error

# =============================================================================
# IMPORTANT NOTES:
# =============================================================================
# - This script will use 2 EXCLUSIVE nodes (8 GPUs total)
# - The effective batch size is (batch_size × num_gpus) = batch_size × 8
# - PyTorch Lightning automatically detects SLURM multi-node DDP environment
# - Inter-node communication uses NCCL over InfiniBand
# - Communication within nodes: NVLink (faster)
# - Communication between nodes: InfiniBand HDR200 (200 Gbit/s)
# - For even faster inter-node communication, use --constraint="gpu-bw"
#   to request high-bandwidth nodes (400 Gbit/s)
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
echo "Total CPUs per node: $((SLURM_NTASKS_PER_NODE * SLURM_CPUS_PER_TASK))"
echo "Total GPUs: $((SLURM_NNODES * SLURM_NTASKS_PER_NODE))"
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

# Display GPU information on all nodes
echo "========================================"
echo "GPU Information (all nodes)"
echo "========================================"
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 bash -c '
    echo "=== Node: $(hostname) ==="
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
'
echo "========================================"
echo ""

# DDP configuration info
echo "========================================"
echo "Multi-Node DDP Configuration"
echo "========================================"
echo "Number of nodes: ${SLURM_NNODES}"
echo "GPUs per node: ${SLURM_NTASKS_PER_NODE}"
echo "Total GPUs: $((SLURM_NNODES * SLURM_NTASKS_PER_NODE))"
echo "Batch size per GPU: 32"
echo "Effective batch size: 32 × $((SLURM_NNODES * SLURM_NTASKS_PER_NODE)) = $((32 * SLURM_NNODES * SLURM_NTASKS_PER_NODE))"
echo "Communication within nodes: NVLink"
echo "Communication between nodes: InfiniBand HDR200 (200 Gbit/s)"
echo "========================================"
echo ""

# Record start time
echo "Training started at: $(date)"
START_TIME=$(date +%s)

# Run training script with srun
# srun will launch processes across all nodes and GPUs
# PyTorch Lightning will automatically:
# - Detect SLURM multi-node environment variables
# - Initialize DDP across all 8 GPUs
# - Use NCCL for inter-GPU communication
# - Synchronize gradients after each batch
echo "Starting multi-node multi-GPU training..."
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

# Print GPU utilization on all nodes
echo ""
echo "Final GPU status (all nodes):"
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 bash -c '
    echo "=== Node: $(hostname) ==="
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader
    echo ""
'

echo ""
echo "Job completed successfully!"
echo "Expected speedup: ~6-7x compared to single GPU"
echo "Scaling efficiency: ~75-85% (due to inter-node communication overhead)"
echo "Check Comet ML dashboard for detailed metrics and visualizations"
echo ""
echo "TIP: For even better performance on large-scale multi-node training,"
echo "     use --constraint=\"gpu-bw\" to request high-bandwidth nodes"
