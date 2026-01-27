# Single GPU Training Tutorial

This guide walks through training on a single A100 GPU - the baseline for all performance comparisons.

## When to Use Single GPU

- **Model development**: Quick iterations and debugging
- **Small datasets**: When data fits comfortably in 40GB GPU memory
- **Baseline measurements**: Establish performance before scaling
- **Learning**: Understand the basics before distributed training

## Prerequisites

- ✅ Completed [setup guide](01_setup.md)
- ✅ Virtual environment created
- ✅ (Optional) Comet ML configured

## Job Configuration

The single GPU job uses these SLURM parameters:

```bash
#SBATCH --nodes=1              # One node
#SBATCH --ntasks=1             # One task (one GPU process)
#SBATCH --cpus-per-task=18     # 18 CPUs (standard for 1 GPU)
#SBATCH --gres=gpu:a100:1      # One A100 GPU
#SBATCH --mem=125000           # 125 GB memory
#SBATCH --time=01:00:00        # 1 hour max runtime
```

**Resource allocation**:
- Shared node (other jobs may use remaining GPUs)
- 18 out of 72 CPU cores
- 125 GB out of 512 GB RAM
- 1 out of 4 GPUs

## Submitting the Job

```bash
# Make sure you're in the repository directory
cd /u/$USER/MPCDF_tutorial

# Submit the job
sbatch slurm_jobs/single_gpu.sh

# Note the job ID from the output
# Example: "Submitted batch job 123456"
```

## Monitoring the Job

### Check Job Status

```bash
# View your jobs in the queue
squeue -u $USER

# Example output:
#   JOBID    PARTITION  NAME           USER   ST   TIME  NODES
#   123456   gpu        mlp_single_g   user   R    2:30  1
```

**Status codes**:
- `PD` (Pending): Waiting for resources
- `R` (Running): Currently executing
- `CG` (Completing): Finishing up
- `CD` (Completed): Finished successfully

### View Live Output

```bash
# Follow the output log (replace 123456 with your job ID)
tail -f outputs/slurm-123456.out

# Press Ctrl+C to stop following
```

### Check Errors

```bash
# View error log
cat outputs/slurm-123456.err
```

**Note**: Some warnings in the error log are normal (e.g., NCCL warnings on single GPU).

## Understanding the Output

### Job Information Section

```
========================================
SLURM Job Information
========================================
Job ID: 123456
Job Name: mlp_single_gpu
Node(s): raven0123
Number of nodes: 1
Number of tasks: 1
CPUs per task: 18
========================================
```

### GPU Information

```
========================================
GPU Information
========================================
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:00:1E.0 Off |                    0 |
| N/A   30C    P0    55W / 400W |      0MiB / 40960MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### Training Progress

```
Epoch 1/50: 100%|██████████| 219/219 [00:08<00:00, 25.21it/s, loss=0.645, v_num=1, val_loss=0.634, val_acc=0.652]
Epoch 2/50: 100%|██████████| 219/219 [00:07<00:00, 28.45it/s, loss=0.582, v_num=1, val_loss=0.571, val_acc=0.712]
...
Epoch 50/50: 100%|██████████| 219/219 [00:07<00:00, 29.13it/s, loss=0.124, v_num=1, val_loss=0.119, val_acc=0.953]
```

**Metrics explained**:
- `loss`: Training loss (should decrease)
- `val_loss`: Validation loss (should decrease)
- `val_acc`: Validation accuracy (should increase)
- `it/s`: Iterations per second (throughput)

### Training Summary

```
========================================
TRAINING COMPLETE
========================================
Total training time: 362.45 seconds (6.04 minutes)
Time per epoch: 7.25 seconds
Best checkpoint: outputs/checkpoints/spiral_mlp-epoch=42-val_loss=0.1187.ckpt
Best validation loss: 0.1187
========================================
```

## Expected Performance

For the default configuration (10,000 samples, 50 epochs):

- **Training time**: ~5-10 minutes
- **Final validation accuracy**: ~95%
- **Final validation loss**: ~0.12
- **Throughput**: ~25-30 iterations/second

## Checking Results

### View Checkpoints

```bash
# List saved checkpoints
ls -lh outputs/checkpoints/

# Example output:
# -rw-r--r-- 1 user group 15K Jan 27 14:30 spiral_mlp-epoch=42-val_loss=0.1187.ckpt
# -rw-r--r-- 1 user group 15K Jan 27 14:30 last.ckpt
```

### View Comet ML Dashboard

If you configured Comet ML:

1. Go to [https://www.comet.com](https://www.comet.com)
2. Navigate to your project (`mpcdf-raven-tutorial`)
3. Find your experiment (named `spiral_mlp_1gpu`)
4. View metrics, charts, and system information

### Extract Timing Information

```bash
# Extract total training time
grep "Total training time" outputs/slurm-123456.out

# Extract validation accuracy
grep "val_acc" outputs/slurm-123456.out | tail -n 1
```

## Common Issues

### Job Stays Pending

**Symptom**: Job stays in `PD` state for a long time

**Causes**:
- Cluster is busy (high demand for GPUs)
- Requesting resources that aren't available

**Solutions**:
```bash
# Check why job is pending
squeue -j 123456 --start

# Try shorter time limit for faster scheduling
#SBATCH --time=00:30:00
```

### Out of Memory

**Symptom**: Error message about CUDA out of memory

**Solution**: Reduce batch size
```bash
# Edit the SLURM script or run directly:
srun python src/train.py --batch_size 16  # instead of 32
```

### Slow Training

**Symptom**: Training is much slower than expected

**Check**:
```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi
```

If GPU utilization is low (<50%), increase `num_workers`:
```bash
srun python src/train.py --num_workers 8
```

## Next Steps

Now that you've completed single GPU training:

1. **Analyze results**: Check Comet ML dashboard or log files
2. **Save timing**: Note the training time for comparison
3. **Scale up**: Try [multi-GPU training](04_multi_gpu.md) to see the speedup!

## Canceling a Job

If you need to stop a running job:

```bash
# Cancel by job ID
scancel 123456

# Cancel all your jobs
scancel -u $USER
```

## Quick Reference

```bash
# Submit job
sbatch slurm_jobs/single_gpu.sh

# Check status
squeue -u $USER

# View output
tail -f outputs/slurm-<job_id>.out

# Cancel job
scancel <job_id>

# After completion: benchmark
python src/benchmark.py --output_dir outputs
```

Ready to scale to multiple GPUs? Continue to [04_multi_gpu.md](04_multi_gpu.md)!
