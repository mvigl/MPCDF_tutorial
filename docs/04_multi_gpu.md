# Multi-GPU Training Tutorial

Learn how to train with 4 GPUs on a single node using DDP (Distributed Data Parallel).

## Why Multi-GPU?

**Benefits**:
- **Faster training**: ~3.5-4× speedup with 4 GPUs
- **Larger effective batch size**: batch_size × 4
- **Same code**: PyTorch Lightning handles all DDP setup

**When to use**:
- Medium to large models
- Datasets that benefit from larger batches
- When you need results faster

## Key Differences from Single GPU

| Aspect | Single GPU | Multi-GPU (4 GPUs) |
|--------|-----------|-------------------|
| GPUs | 1 | 4 |
| CPUs | 18 | 72 (18 per GPU) |
| Memory | 125 GB | 500 GB (full node) |
| Node Type | Shared | Exclusive |
| Batch Size (effective) | 32 | 128 (32 × 4) |
| Expected Time | ~6 min | ~1.7 min |
| Speedup | 1× | ~3.5× |

## Understanding DDP

### How Data is Distributed

```
Input Batch (128 samples total)
         ↓
    Split into 4
         ↓
┌────────┬────────┬────────┬────────┐
│ GPU 0  │ GPU 1  │ GPU 2  │ GPU 3  │
│ 32     │ 32     │ 32     │ 32     │
│ samples│ samples│ samples│ samples│
└────────┴────────┴────────┴────────┘
         ↓
   Forward pass (parallel)
         ↓
   Backward pass (parallel)
         ↓
  Gradient All-Reduce (sync)
         ↓
   Model update (parallel)
```

### Communication

- **NVLink**: High-speed GPU-to-GPU communication (~600 GB/s)
- **All-Reduce**: NCCL library synchronizes gradients
- **Efficient**: Overlaps communication with computation

## Submitting the Job

```bash
cd /u/$USER/MPCDF_tutorial
sbatch slurm_jobs/multi_gpu_single_node.sh
```

## SLURM Configuration

```bash
#SBATCH --nodes=1              # One node
#SBATCH --ntasks-per-node=4    # 4 tasks (one per GPU)
#SBATCH --cpus-per-task=18     # 18 CPUs per task
#SBATCH --gres=gpu:a100:4      # All 4 A100 GPUs
#SBATCH --mem=500000           # 500 GB (full node)
```

**Important**: This reserves an entire node exclusively for your job.

## Monitoring

### Check DDP Initialization

Look for these lines in the output:

```
DDP Configuration
======================================
WORLD_SIZE: 4 GPUs on 1 node
Batch size per GPU: 32
Effective batch size: 32 × 4 = 128
Communication: NVLink (fast intra-node)
```

### Training Progress

```
Epoch 1/50: 100%|██████████| 55/55 [00:02<00:00, 21.34it/s, ...
```

**Note**: Fewer iterations per epoch (55 vs 219) because effective batch size is 4× larger.

### GPU Utilization

All 4 GPUs should show high utilization:

```
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA A100-SXM...  On   |          ... |    99%      Default |
|   1  NVIDIA A100-SXM...  On   |          ... |    99%      Default |
|   2  NVIDIA A100-SXM...  On   |          ... |    99%      Default |
|   3  NVIDIA A100-SXM...  On   |          ... |    99%      Default |
+-----------------------------------------------------------------------------+
```

## Expected Performance

For default configuration (10,000 samples, 50 epochs):

- **Training time**: ~1.5-3 minutes
- **Speedup**: ~3.5-4× vs single GPU
- **Efficiency**: ~88% (near-linear scaling)
- **Final accuracy**: ~95% (same as single GPU)

## Effective Batch Size

**Critical concept**: Each GPU processes `batch_size` samples, but they're trained **in parallel**.

```
Single GPU:  batch_size = 32  →  effective = 32
4 GPUs:      batch_size = 32  →  effective = 128
```

**Effects**:
- **Faster epochs**: Process more data per iteration
- **Possibly fewer epochs needed**: Larger batches can improve generalization
- **May need learning rate adjustment**: For very large batches

## Comparing to Single GPU

After both jobs complete:

```bash
source venv/bin/activate
python src/benchmark.py --output_dir outputs
```

**Expected output**:
```
SPEEDUP ANALYSIS
======================================
Baseline (Single GPU): 362.45 seconds

Multi-GPU (4 GPUs, 1 node):
  Average time: 95.23 seconds
  Speedup: 3.81x
  Parallel efficiency: 95.2%
```

## Troubleshooting

### DDP Initialization Hangs

**Symptom**: Training doesn't start, hangs at initialization

**Solution**: Make sure you're using `srun`:
```bash
# Correct:
srun python src/train.py

# Wrong:
python src/train.py  # Won't work for multi-GPU!
```

### Different Results than Single GPU

**Expected**: Results may differ slightly due to:
- Different random initialization across GPUs
- Different batch ordering
- Numerical precision in gradient averaging

**Fix**: Set seed for reproducibility (already done in our code)

### Low GPU Utilization

**Symptom**: GPUs showing <70% utilization

**Possible causes**:
- `num_workers` too low (data loading bottleneck)
- Batch size too small (computation is too fast)

**Solution**:
```bash
srun python src/train.py --num_workers 8 --batch_size 64
```

## Advanced: Viewing Logs from Each GPU

Each GPU process writes to the same log file, but you can identify them by rank:

```bash
# Search for rank-specific output
grep "RANK: 0" outputs/slurm-<job_id>.out  # Master process
grep "RANK: 1" outputs/slurm-<job_id>.out  # Worker 1
# etc.
```

## Key Takeaways

1. **No code changes needed**: Same `train.py` as single GPU
2. **Linear scaling**: 4 GPUs ≈ 4× faster (thanks to NVLink)
3. **Effective batch size**: Remember it's multiplied by num_gpus
4. **Efficient**: ~95% parallel efficiency within a node

## Next Steps

- Compare results with single GPU baseline
- Try [multi-node training](05_multi_node.md) for even larger scale
- Experiment with different batch sizes

Continue to [05_multi_node.md](05_multi_node.md) to learn about scaling across nodes!
