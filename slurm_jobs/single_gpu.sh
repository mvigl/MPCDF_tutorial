#!/bin/bash -l
#
# SLURM job script for single GPU training on MPCDF Raven cluster
#
# This script demonstrates training an MLP on a single A100 GPU.
# It's suitable for:
# - Small to medium datasets
# - Model development and debugging
# - Baseline performance measurements
#

#SBATCH --job-name=mlp_single_gpu          # Job name
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=1                          # Number of tasks (1 task per GPU)
#SBATCH --cpus-per-task=18                  # CPUs per task (18 is standard for 1 GPU on raven)
#SBATCH --gres=gpu:a100:1                   # Request 1 A100 GPU
#SBATCH --constraint="gpu"                  # Request GPU nodes
#SBATCH --mem=125000                        # Memory per node in MB (125 GB for 1 GPU)
#SBATCH --time=00:10:00                     # Maximum runtime (10 min)
#SBATCH --output=outputs/slurm-%j.out       # Standard output (%j = job ID)
#SBATCH --error=outputs/slurm-%j.err        # Standard error

# =============================================================================
# IMPORTANT NOTES:
# =============================================================================
# - This script will run on a shared node (other jobs may use other GPUs)
# - Memory is limited to 125 GB (adjust --mem if you need more)
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
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
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

# Record start time
echo "Training started at: $(date)"
START_TIME=$(date +%s)

# Run training script
# The training script will automatically detect the single GPU
# and run in non-distributed mode
echo "Starting training..."
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
echo "Check Comet ML dashboard for detailed metrics and visualizations"
