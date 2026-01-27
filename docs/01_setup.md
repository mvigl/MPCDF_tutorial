# Setup Guide: Getting Started on Raven

This guide walks you through setting up your environment on the MPCDF Raven cluster.

## Table of Contents
1. [Accessing Raven](#accessing-raven)
2. [Cloning the Repository](#cloning-the-repository)
3. [Setting Up the Virtual Environment](#setting-up-the-virtual-environment)
4. [Configuring Comet ML](#configuring-comet-ml)
5. [Verifying the Installation](#verifying-the-installation)
6. [Understanding the Workflow](#understanding-the-workflow)

## 1. Accessing Raven

### Gateway Login

Raven requires two-step login for security:

```bash
# Step 1: Connect to the gateway
ssh <username>@gate.mpcdf.mpg.de

# Step 2: Connect to Raven
ssh <username>@raven.mpcdf.mpg.de
```

You will need:
- Your MPCDF username and password
- One-Time Password (OTP) for the second step

**Note**: SSH keys are not allowed on Raven login nodes.

### File Transfer

To transfer files to/from Raven, use `scp`:

```bash
# From your local machine to Raven
scp myfile.py <username>@raven.mpcdf.mpg.de:/u/<username>/

# From Raven to your local machine
scp <username>@raven.mpcdf.mpg.de:/u/<username>/results.csv ./
```

## 2. Cloning the Repository

Once logged in to Raven, navigate to your home directory and clone the repository:

```bash
# Navigate to your home directory
cd /u/<username>/

# Clone the repository (replace with actual URL)
git clone <repository-url> MPCDF_tutorial

# Enter the directory
cd MPCDF_tutorial
```

**Important**: Your home directory `/u/<username>` has:
- **Quota**: 2.5 TB disk space, 1 million files
- **Backup**: Files are backed up regularly
- **Access**: Available from all nodes (login, compute, interactive)

For large temporary files, use `/ptmp/<username>` instead.

## 3. Setting Up the Virtual Environment

The repository includes an automated setup script that creates a Python virtual environment and installs all required packages.

### Run the Setup Script

```bash
# Make sure you're in the repository directory
cd /u/<username>/MPCDF_tutorial

# Run the setup script
bash setup_venv.sh
```

### What the Script Does

The `setup_venv.sh` script performs the following steps:

1. **Loads required modules**:
   - `gcc/13` (provides Python3 and build tools)
   - `cuda/12.6` (CUDA libraries for GPU support)

2. **Creates a virtual environment**:
   - Uses Python's built-in `venv` module
   - Creates `venv/` directory in the repository

3. **Installs PyTorch with CUDA support**:
   - Downloads PyTorch compiled for CUDA 12.1 (compatible with CUDA 12.6)
   - Includes torchvision and torchaudio

4. **Installs other dependencies**:
   - PyTorch Lightning
   - Comet ML
   - NumPy, scikit-learn, matplotlib, etc.

### Expected Output

You should see output similar to:

```
==========================================
MPCDF Raven Tutorial - Virtual Environment Setup
==========================================

Step 1/5: Loading required modules...
Step 2/5: Creating Python virtual environment...
Step 3/5: Activating virtual environment...
Step 4/5: Upgrading pip...
Step 5/5: Installing PyTorch with CUDA support...

========================================
Verifying installation...
========================================
Python version: 3.x.x
PyTorch version: 2.x.x
CUDA available: True (on compute nodes)
CUDA version: 12.1
PyTorch Lightning version: 2.x.x
Comet ML version: 3.x.x

========================================
Setup complete!
========================================
```

**Note**: On login nodes, you may see "CUDA available: False" because there are no GPUs on login nodes. This is normal and expected.

### Setup Time

The entire setup process takes approximately **10-15 minutes** on the login nodes due to downloading and installing packages.

## 4. Configuring Comet ML

Comet ML is used for experiment tracking and visualization. While optional, it's highly recommended for comparing runs across different configurations.

### Create a Comet ML Account

1. Go to [https://www.comet.com](https://www.comet.com)
2. Sign up for a free account
3. Navigate to your account settings
4. Copy your API key

### Set the API Key

You have two options for providing your Comet ML API key:

#### Option 1: Environment Variable (Recommended)

```bash
# Set the environment variable for the current session
export COMET_API_KEY='your-api-key-here'

# Add it to your ~/.bashrc for persistence
echo "export COMET_API_KEY='your-api-key-here'" >> ~/.bashrc
source ~/.bashrc
```

#### Option 2: Command Line Argument

You can pass the API key directly when running training:

```bash
python src/train.py --comet_api_key 'your-api-key-here' ...
```

### Create a Project

In the Comet ML dashboard:
1. Click "Create New Project"
2. Name it `mpcdf-raven-tutorial` (or update `configs/default_config.yaml`)
3. All your experiments will be logged to this project

### Running Without Comet ML

If you prefer not to use Comet ML, the training script will still work. You'll see a warning message, but training will proceed normally. Metrics will still be logged to the console and SLURM output files.

## 5. Verifying the Installation

### Test the Virtual Environment

Activate the environment and verify packages are installed:

```bash
# Activate the virtual environment
source venv/bin/activate

# Check Python version
python --version

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Verify other packages
python -c "import pytorch_lightning; import comet_ml; import numpy; print('All packages imported successfully!')"
```

### Test Data Generation

You can test the data generation module on a login node:

```bash
source venv/bin/activate
python src/datamodule.py
```

This will:
- Generate a spiral pattern dataset
- Print dataset information
- Save a visualization to `spiral_pattern.png`

**Note**: This runs on CPU and doesn't require a GPU.

### Test the Model (Optional)

```bash
source venv/bin/activate
python src/model.py
```

This will:
- Create a sample MLP model
- Print the architecture
- Run a test forward pass

## 6. Understanding the Workflow

The workflow for using this tutorial is:

### One-Time Setup (Just Completed!)

1. Clone the repository
2. Run `bash setup_venv.sh`
3. Configure Comet ML (optional)

### For Each Training Job

The SLURM scripts automatically handle everything:

```bash
# Just submit the job
sbatch slurm_jobs/single_gpu.sh
```

Behind the scenes, the SLURM script:
1. Loads the same modules (`gcc/13 cuda/12.6`)
2. Activates the virtual environment (`source venv/bin/activate`)
3. Runs the training script

You **don't need to manually activate the environment** or load modules before submitting jobs!

### File System Best Practices

- **Code and Setup**: Store in `/u/<username>/` (your home directory)
  - Backed up
  - 2.5 TB quota
  - Use for: code, virtual environment, important results

- **Temporary Outputs**: Use `/ptmp/<username>/` (scratch space)
  - Not backed up
  - No quota (fair usage)
  - Files deleted after 12 weeks of inactivity
  - Use for: temporary outputs, large datasets, experimental results

- **Long-Term Storage**: Use `/r/<initial>/<userid>/` (archive)
  - Available on login nodes only
  - Automatically migrated to tape
  - Pack files into tar archives (1 GB - 1 TB recommended)
  - Use for: completed experiments, published results

### Example: Using /ptmp

```bash
# Create a directory in /ptmp
mkdir -p /ptmp/$USER/MPCDF_tutorial

# Create a symbolic link from your repository
cd /u/$USER/MPCDF_tutorial
ln -s /ptmp/$USER/MPCDF_tutorial outputs_ptmp

# Modify SLURM scripts to use /ptmp
# In slurm_jobs/*.sh, change:
# --output=outputs/slurm-%j.out
# to:
# --output=outputs_ptmp/slurm-%j.out
```

## Next Steps

Now that your environment is set up, you can:

1. **Understand the Code**: Read [02_understanding_code.md](02_understanding_code.md)
2. **Run Your First Job**: Follow [03_single_gpu.md](03_single_gpu.md)
3. **Scale to Multi-GPU**: Try [04_multi_gpu.md](04_multi_gpu.md)

## Troubleshooting

### Module Not Found Errors

If you see "module: command not found":

```bash
# Re-source your shell profile
source ~/.bashrc
```

### Virtual Environment Not Activating

```bash
# Make sure you're in the repository directory
cd /u/$USER/MPCDF_tutorial

# Check if venv exists
ls -la venv/

# If it doesn't exist, run setup again
bash setup_venv.sh
```

### CUDA Version Mismatch

The setup script installs PyTorch for CUDA 12.1, which is compatible with Raven's CUDA 12.6. If you see warnings about CUDA versions, they can usually be ignored.

### Permission Denied

If you see "Permission denied" when running scripts:

```bash
# Make scripts executable
chmod +x setup_venv.sh
chmod +x slurm_jobs/*.sh
```

## Summary

You've completed the setup! Here's what you've accomplished:

- ✅ Accessed the Raven cluster
- ✅ Cloned the tutorial repository
- ✅ Created a Python virtual environment
- ✅ Installed PyTorch with CUDA support
- ✅ Installed PyTorch Lightning and dependencies
- ✅ (Optional) Configured Comet ML
- ✅ Verified the installation

The workflow going forward is simple:

```bash
# Submit a job (environment setup is automatic!)
sbatch slurm_jobs/single_gpu.sh
```

Ready to train your first model? Proceed to [03_single_gpu.md](03_single_gpu.md)!
