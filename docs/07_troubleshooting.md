# Troubleshooting Guide

Common issues and solutions for training on the Raven cluster.

## Table of Contents

1. [Setup Issues](#setup-issues)
2. [SLURM Job Issues](#slurm-job-issues)
3. [Training Issues](#training-issues)
4. [DDP and Multi-GPU Issues](#ddp-and-multi-gpu-issues)
5. [Comet ML Issues](#comet-ml-issues)
6. [Performance Issues](#performance-issues)
7. [Getting Help](#getting-help)

## Setup Issues

### Virtual Environment Creation Fails

**Symptom**: `setup_venv.sh` fails with module or Python errors

**Solutions**:

```bash
# 1. Clean module environment
module purge

# 2. Load required modules
module load gcc/13 cuda/12.6

# 3. Verify Python is available
which python3

# 4. Remove old venv if exists
rm -rf venv/

# 5. Run setup again
bash setup_venv.sh
```

### Package Installation Fails

**Symptom**: pip install errors during setup

**Solutions**:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install with verbose output to see errors
pip install -r requirements.txt -v

# If specific package fails, install manually
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### ImportError After Setup

**Symptom**: `ImportError: No module named 'torch'` even after setup

**Solutions**:

```bash
# Make sure venv is activated
source venv/bin/activate

# Verify which Python you're using
which python  # Should show path to venv/bin/python

# Check installed packages
pip list | grep torch
```

## SLURM Job Issues

### Job Stays Pending Forever

**Symptom**: Job in `PD` (Pending) state for hours

**Check why**:
```bash
squeue -j <job_id> --start
```

**Common causes**:

1. **Cluster is busy**
   ```bash
   # Check queue
   sinfo -p gpu

   # See how many jobs ahead of you
   squeue -p gpu | grep PD | wc -l
   ```

2. **Requesting unavailable resources**
   ```bash
   # Check available nodes
   sinfo -p gpu --Node --long

   # Reduce requested time
   #SBATCH --time=00:30:00  # instead of 24:00:00
   ```

3. **Hit job limit**
   ```bash
   # Check your running jobs
   squeue -u $USER

   # Check limits
   sacctmgr show qos
   ```

### Job Fails Immediately

**Symptom**: Job goes to `F` (Failed) state right away

**Check error log**:
```bash
cat outputs/slurm-<job_id>.err
```

**Common causes**:

1. **Module not found**
   ```bash
   # In job script, check:
   module purge
   module load gcc/13 cuda/12.6
   ```

2. **Virtual environment not found**
   ```bash
   # Make sure venv exists
   ls -la venv/

   # Path should be relative to job script location
   source venv/bin/activate  # not ../venv/
   ```

3. **Output directory doesn't exist**
   ```bash
   # Create outputs directory
   mkdir -p outputs
   ```

### Permission Denied Errors

**Symptom**: `Permission denied` when running scripts

**Solutions**:

```bash
# Make scripts executable
chmod +x setup_venv.sh
chmod +x slurm_jobs/*.sh

# Check file permissions
ls -la slurm_jobs/
```

## Training Issues

### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. **Reduce batch size**
   ```bash
   # Edit SLURM script or pass argument:
   srun python src/train.py --batch_size 16  # instead of 32
   ```

2. **Check GPU memory**
   ```bash
   nvidia-smi
   # Look for other processes using GPU memory
   ```

3. **Request more GPU memory** (not applicable on Raven - all A100s have 40GB)

### Training is Very Slow

**Symptom**: Much slower than expected training time

**Diagnostics**:

```bash
# Check GPU utilization during training
watch -n 1 nvidia-smi

# Should see ~95-100% GPU utilization
# If low (<50%), likely a data loading bottleneck
```

**Solutions**:

1. **Increase num_workers**
   ```bash
   srun python src/train.py --num_workers 8  # instead of 4
   ```

2. **Check CPU allocation**
   ```bash
   # Make sure you have enough CPUs
   #SBATCH --cpus-per-task=18
   ```

3. **Use persistent workers**
   ```python
   # Already enabled in datamodule.py
   persistent_workers=True if num_workers > 0 else False
   ```

### Training Accuracy is Poor

**Symptom**: Validation accuracy stays low (~50% for binary classification)

**Check**:

1. **Data is loaded correctly**
   ```bash
   python src/datamodule.py  # Test data generation
   ```

2. **Learning rate is reasonable**
   ```bash
   # Try different learning rates
   --learning_rate 0.01   # too high?
   --learning_rate 0.00001 # too low?
   --learning_rate 0.0001  # default (good)
   ```

3. **Model is training**
   ```bash
   # Check logs - loss should decrease
   grep "train_loss" outputs/slurm-<job_id>.out
   ```

## DDP and Multi-GPU Issues

### DDP Hangs at Initialization

**Symptom**: Multi-GPU job hangs, doesn't start training

**Solutions**:

1. **Must use srun**
   ```bash
   # Correct:
   srun python src/train.py

   # Wrong (won't work for multi-GPU):
   python src/train.py
   ```

2. **Check SLURM environment**
   ```bash
   # In output log, verify:
   echo $SLURM_NTASKS      # Should match total tasks
   echo $SLURM_NNODES      # Should match requested nodes
   ```

3. **Firewall/network issues** (contact MPCDF support)

### NCCL Errors

**Symptom**: Errors mentioning NCCL, "Network unreachable", or timeouts

**Solutions**:

1. **Increase NCCL timeout**
   ```bash
   # Add to SLURM script before training:
   export NCCL_TIMEOUT=1800  # 30 minutes
   export NCCL_DEBUG=INFO     # For debugging
   ```

2. **Check InfiniBand**
   ```bash
   # On compute node:
   ibstat  # Should show active ports
   ```

3. **Try different NCCL settings** (if problems persist)
   ```bash
   export NCCL_IB_DISABLE=0  # Use InfiniBand
   export NCCL_SOCKET_IFNAME=ib0  # Use IB interface
   ```

### Different Results Across Configurations

**Symptom**: Single GPU gives different accuracy than multi-GPU

**Expected**: Small differences (<1%) are normal due to:
- Different random initialization
- Numerical precision in averaging
- Different batch ordering

**Solutions if large differences (>5%)**:

1. **Check seed is set**
   ```python
   pl.seed_everything(42)  # Already in train.py
   ```

2. **Verify same hyperparameters**
   ```bash
   grep "learning_rate\|batch_size" outputs/slurm-*.out
   ```

3. **Check effective batch size**
   ```
   Single GPU:  32
   4 GPUs:      128 (may need different learning rate)
   ```

## Comet ML Issues

### API Key Not Found

**Symptom**: "COMET_API_KEY not set" warning

**Solutions**:

```bash
# Option 1: Environment variable
export COMET_API_KEY='your-key-here'

# Option 2: Add to .bashrc
echo "export COMET_API_KEY='your-key'" >> ~/.bashrc
source ~/.bashrc

# Option 3: Pass as argument
python src/train.py --comet_api_key 'your-key'
```

### Experiments Not Logging

**Symptom**: Training runs but nothing appears in Comet dashboard

**Check**:

1. **API key is valid**
   ```bash
   # Test connection
   python -c "import comet_ml; comet_ml.init()"
   ```

2. **Project name is correct**
   ```bash
   # Verify project exists in Comet dashboard
   # Check project name matches
   --comet_project mpcdf-raven-tutorial
   ```

3. **Network access from login nodes**
   ```bash
   # Comet ML requires internet access
   # Note: Compute nodes don't have internet on Raven!
   # Logging happens during/after job on login nodes
   ```

### Duplicate Experiments

**Symptom**: Multiple experiments with same name

**Cause**: Each GPU rank tries to log (in DDP)

**Solution**: Already handled in `train.py`:
```python
# CometLogger only logs from rank 0 automatically
logger = CometLogger(...)
```

## Performance Issues

### Poor Speedup

**Symptom**: Multi-GPU speedup is much lower than expected

**Check**:

1. **Data loading bottleneck**
   ```bash
   # Increase workers
   --num_workers 8

   # Check CPU utilization
   top  # Should see high CPU usage
   ```

2. **Small batch size**
   ```bash
   # Try larger batch per GPU
   --batch_size 64  # instead of 32
   ```

3. **Model too small**
   ```bash
   # Try larger model
   --hidden_dims 256 128 64 32
   ```

4. **Network issues** (for multi-node)
   ```bash
   # Check network speed between nodes
   iperf -c <other-node>
   ```

### Memory Usage Too High

**Symptom**: Job killed for exceeding memory limit

**Solutions**:

```bash
# 1. Reduce batch size
--batch_size 16

# 2. Reduce model size
--hidden_dims 32 16 8

# 3. Request more memory
#SBATCH --mem=250000  # 250 GB instead of 125 GB

# 4. Reduce num_workers
--num_workers 2
```

## Common Error Messages

### "connection refused" or "connection timeout"

**Cause**: Network communication issues in DDP

**Solutions**:
- Check SLURM allocated correct nodes
- Verify InfiniBand is working
- Contact MPCDF support if persistent

### "CUDA error: device-side assert triggered"

**Cause**: Usually a data/label mismatch

**Solutions**:
```bash
# Check data generation
python src/datamodule.py

# Verify number of classes matches model output
# Spiral dataset: 2 classes, model output: 2
```

### "RuntimeError: unable to open shared memory object"

**Cause**: DataLoader shared memory issues

**Solutions**:
```bash
# Reduce num_workers
--num_workers 0  # Disable multiprocessing

# Or use persistent workers
# (already enabled in our datamodule.py)
```

## Debugging Tips

### Enable Verbose Logging

```bash
# In train.py or SLURM script:
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1  # Immediate output

# Lightning trainer:
trainer = pl.Trainer(..., log_every_n_steps=1)
```

### Test on Interactive Node

Before submitting jobs, test on interactive nodes:

```bash
# Request interactive session
srun -p interactive --gres=gpu:1 --pty bash

# Load modules and activate venv
module load gcc/13 cuda/12.6
source venv/bin/activate

# Test training
python src/train.py --max_epochs 2  # Quick test
```

### Check Logs Carefully

```bash
# View full output
cat outputs/slurm-<job_id>.out

# Search for errors
grep -i "error\|fail\|exception" outputs/slurm-<job_id>.err

# Check timing
grep "Total training time" outputs/slurm-<job_id>.out
```

## Getting Help

### MPCDF Support

For cluster-specific issues:
- Email: support@mpcdf.mpg.de
- Documentation: https://docs.mpcdf.mpg.de/

### PyTorch Lightning

For framework issues:
- Documentation: https://lightning.ai/docs/pytorch/
- GitHub: https://github.com/Lightning-AI/pytorch-lightning
- Forum: https://lightning.ai/forums

### Useful Commands

```bash
# Check quota
/usr/lpp/mmfs/bin/mmlsquota raven_u

# Check job efficiency
seff <job_id>

# View detailed job info
scontrol show job <job_id>

# Check node status
sinfo -N -l

# Cancel all your jobs
scancel -u $USER
```

## Quick Checklist

Before asking for help, verify:

- [ ] Modules loaded: `module list` shows gcc/13 and cuda/12.6
- [ ] Virtual env activated: `which python` shows venv path
- [ ] Scripts executable: `ls -la slurm_jobs/*.sh` shows +x
- [ ] Output directory exists: `ls -la outputs/`
- [ ] Using `srun` for multi-GPU jobs
- [ ] Checked both .out and .err log files
- [ ] Tried on interactive node first

## Still Stuck?

If you've tried everything:

1. Save your error logs
2. Note your exact configuration (nodes, GPUs, batch size)
3. Document steps to reproduce
4. Contact MPCDF support or open an issue

Remember: Most issues are simple configuration problems. Double-check the basics before diving deep!

---

**Happy Training!** 🚀
