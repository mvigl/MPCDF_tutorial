#!/bin/bash -l
#
# Setup script for creating Python virtual environment
# on MPCDF Raven cluster
#
# Usage: bash setup_venv.sh
#
# This script:
# 1. Loads required modules (gcc and CUDA)
# 2. Creates a Python virtual environment
# 3. Installs PyTorch with CUDA support
# 4. Installs all other dependencies from requirements.txt
#

echo "=========================================="
echo "MPCDF Raven Tutorial - Virtual Environment Setup"
echo "=========================================="
echo ""

# Clean module environment
echo "Step 1/5: Loading required modules..."
module purge
module load gcc/13 cuda/12.6

echo "Loaded modules:"
module list
echo ""

# Create virtual environment
echo "Step 2/5: Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "Warning: venv directory already exists. Removing it..."
    rm -rf venv
fi

python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi
echo "Virtual environment created successfully"
echo ""

# Activate virtual environment
echo "Step 3/5: Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Step 4/5: Upgrading pip..."
pip install --upgrade pip
echo ""

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.6)
echo "Step 5/5: Installing PyTorch with CUDA support..."
echo "This may take a few minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if [ $? -ne 0 ]; then
    echo "Error: Failed to install PyTorch"
    exit 1
fi
echo ""

# Install other requirements
echo "Installing other dependencies from requirements.txt..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements"
    exit 1
fi
echo ""

# Verify installation
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python3 << EOF
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"Error importing torch: {e}")
    sys.exit(1)

try:
    import pytorch_lightning as pl
    print(f"PyTorch Lightning version: {pl.__version__}")
except ImportError as e:
    print(f"Error importing pytorch_lightning: {e}")
    sys.exit(1)

try:
    import comet_ml
    print(f"Comet ML version: {comet_ml.__version__}")
except ImportError as e:
    print(f"Error importing comet_ml: {e}")
    sys.exit(1)

print("All core packages installed successfully!")
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Package verification failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Note: The SLURM job scripts will automatically:"
echo "  1. Load the required modules (gcc/13 cuda/12.6)"
echo "  2. Activate the virtual environment"
echo "  3. Run your training script"
echo ""
echo "Next steps:"
echo "  1. Set your Comet ML API key (optional but recommended):"
echo "     export COMET_API_KEY='your-api-key-here'"
echo "     # Or add it to your ~/.bashrc for persistence"
echo ""
echo "  2. Submit a test job:"
echo "     sbatch slurm_jobs/single_gpu.sh"
echo ""
