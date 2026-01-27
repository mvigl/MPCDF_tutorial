# Multi-Node Multi-GPU Training Tutorial

Scale your training across multiple nodes with 8 GPUs total (2 nodes × 4 GPUs).

## When to Use Multi-Node

**Good use cases**:
- Very large models that benefit from massive parallelism
- Huge datasets requiring high throughput
- When single-node training is still too slow

**Consider the tradeoffs**:
- Communication overhead between nodes
- Slightly lower parallel efficiency (~75% vs ~95%)
- Still significant speedup (~6-7× vs single GPU)

## Multi-Node vs Multi-GPU

| Aspect | Multi-GPU (1 node) | Multi-Node (2 nodes) |
|--------|-------------------|---------------------|
| Total GPUs | 4 | 8 |
| Nodes | 1 | 2 |
| Communication | NVLink (~600 GB/s) | InfiniBand (25 GB/s) |
| Speedup | ~3.5-4× | ~6-7× |
| Efficiency | ~95% | ~75-85% |
| Effective Batch | 128 | 256 |

## How Multi-Node DDP Works

```
Node 0                      Node 1
┌────────────────┐         ┌────────────────┐
│ GPU 0 (Rank 0) │ ←───┐   │ GPU 4 (Rank 4) │
│ GPU 1 (Rank 1) │     │   │ GPU 5 (Rank 5) │
│ GPU 2 (Rank 2) │     │   │ GPU 6 (Rank 6) │
│ GPU 3 (Rank 3) │     │   │ GPU 7 (Rank 7) │
└────────────────┘     │   └────────────────┘
       ↑               │          ↑
       │    InfiniBand Network   │
       └──────────────────────────┘

Communication:
  Within node: NVLink (fast)
  Between nodes: InfiniBand HDR200 (slower but still fast)
```

## SLURM Configuration

```bash
#SBATCH --nodes=2              # Two nodes
#SBATCH --ntasks-per-node=4    # 4 tasks per node
#SBATCH --cpus-per-task=18     # 18 CPUs per task
#SBATCH --gres=gpu:a100:4      # 4 GPUs per node
#SBATCH --mem=500000           # 500 GB per node
```

**Total resources**:
- 8 GPUs (2 × 4)
- 144 CPUs (2 × 72)
- 1 TB RAM (2 × 500 GB)

## Submitting the Job

```bash
cd /u/$USER/MPCDF_tutorial
sbatch slurm_jobs/multi_node_multi_gpu.sh
```

## Monitoring Multi-Node Jobs

### Check Node Allocation

```bash
squeue -j <job_id>
```

Look for `NODELIST`:
```
JOBID  PARTITION  NAME          USER  ST  TIME  NODES  NODELIST
123456 gpu        mlp_multi_no  user  R   1:23  2      raven[0123,0124]
```

You're using nodes `raven0123` and `raven0124`.

### GPU Info from All Nodes

The script runs `nvidia-smi` on all nodes:

```
=== Node: raven0123 ===
0, NVIDIA A100-SXM4-40GB, 40960 MiB
1, NVIDIA A100-SXM4-40GB, 40960 MiB
2, NVIDIA A100-SXM4-40GB, 40960 MiB
3, NVIDIA A100-SXM4-40GB, 40960 MiB

=== Node: raven0124 ===
0, NVIDIA A100-SXM4-40GB, 40960 MiB
1, NVIDIA A100-SXM4-40GB, 40960 MiB
2, NVIDIA A100-SXM4-40GB, 40960 MiB
3, NVIDIA A100-SXM4-40GB, 40960 MiB
```

### Training Progress

```
Epoch 1/50: 100%|██████████| 28/28 [00:01<00:00, 15.23it/s, ...
```

**Note**: Even fewer iterations (28 vs 55 vs 219) due to 8× larger effective batch size.

## Expected Performance

For default configuration:

- **Training time**: ~1-2 minutes
- **Speedup**: ~6-7× vs single GPU
- **Efficiency**: ~75-85% (communication overhead)
- **Throughput**: Process 256 samples per iteration

## Understanding the Slowdown

Why not 8× speedup with 8 GPUs?

**Communication overhead**:
1. **Intra-node** (GPUs 0-3 on same node): Fast via NVLink
2. **Inter-node** (e.g., GPU 0 ↔ GPU 4): Slower via InfiniBand
3. **Gradient synchronization**: Must wait for slowest communication path

**Still very good**: 75-85% efficiency is excellent for multi-node training!

## Advanced: High-Bandwidth Nodes

For even better multi-node performance, request high-bandwidth nodes:

```bash
#SBATCH --constraint="gpu-bw"  # 400 Gbit/s instead of 200 Gbit/s
```

**Tradeoff**: Fewer high-BW nodes available (only 32 vs 192 standard)

## Benchmarking All Configurations

After running all three configurations:

```bash
python src/benchmark.py --output_dir outputs
```

**Expected results**:
```
Configuration                  Time (s)   Speedup   Efficiency
================================================================
Single GPU                     362.45     1.00×     100%
Multi-GPU (4 GPUs, 1 node)     95.23      3.81×     95%
Multi-Node (8 GPUs, 2 nodes)   57.14      6.34×     79%
```

## Troubleshooting

### Nodes Can't Communicate

**Symptom**: Job hangs or errors about "connection refused"

**Check**: SLURM environment variables
```bash
echo $SLURM_NODELIST  # Should list both nodes
echo $SLURM_NNODES    # Should be 2
```

### Unbalanced GPU Usage

**Symptom**: Some GPUs at 100%, others at 50%

**Cause**: Data loading bottleneck or slow node

**Solution**: Increase `num_workers` or check node health

### NCCL Errors

**Symptom**: Errors mentioning NCCL timeout or network

**Solutions**:
1. Check that nodes are on same network
2. Increase NCCL timeout (if needed):
   ```bash
   export NCCL_TIMEOUT=1800  # 30 minutes
   ```
3. Verify InfiniBand is working (contact MPCDF if not)

## Best Practices

### 1. Start Small, Scale Up

```
Single GPU → Multi-GPU → Multi-Node
```

Verify each step works before scaling further.

### 2. Adjust Batch Size

For very large multi-node setups, you may want to increase per-GPU batch size:

```bash
# Instead of 256 total (32 × 8)
srun python src/train.py --batch_size 64  # 512 total (64 × 8)
```

### 3. Learning Rate Scaling

For very large batches, consider linear scaling rule:

```
new_lr = base_lr × (new_batch_size / base_batch_size)
```

Example:
```bash
# Base: batch_size=32, lr=0.001
# With 8 GPUs: batch_size=32, effective=256
# New lr = 0.001 × (256 / 32) = 0.008
srun python src/train.py --learning_rate 0.008
```

## Scaling Beyond 2 Nodes

Want to use more nodes?

```bash
#SBATCH --nodes=4  # 16 GPUs total
#SBATCH --nodes=8  # 32 GPUs total
```

**Considerations**:
- Communication overhead increases
- Diminishing returns beyond certain point
- May need learning rate adjustment
- Check cluster limits (`sacctmgr show qos`)

## Key Takeaways

1. **Significant speedup**: ~6-7× with 8 GPUs
2. **Communication matters**: InfiniBand is slower than NVLink
3. **Still efficient**: 75-85% is good for multi-node
4. **No code changes**: PyTorch Lightning handles everything
5. **Effective batch size**: 256 samples per iteration (32 × 8)

## Next Steps

- Run all three configurations for comparison
- Use [benchmarking tools](06_benchmarking.md) to analyze performance
- Check [troubleshooting guide](07_troubleshooting.md) if issues arise

Continue to [06_benchmarking.md](06_benchmarking.md) to learn how to analyze and compare results!
