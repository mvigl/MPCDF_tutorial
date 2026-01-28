# Benchmarking and Performance Analysis

Learn how to analyze and compare training performance across different GPU configurations.

## Comet ML — The Easiest Way to Benchmark

The most effective way to compare runs across GPU configurations is through [Comet ML](https://www.comet.com). Our training script automatically logs everything you need — losses, accuracy, system metrics, hyperparameters — so you can focus on running experiments rather than parsing log files.

### What Gets Logged Automatically

When you run `train.py` with a valid `COMET_API_KEY`, every experiment logs:

- **Training metrics**: `train_loss` per step and epoch
- **Validation metrics**: `val_loss`, `val_acc` at each validation step
- **Hyperparameters**: batch size, learning rate, hidden dims, number of GPUs/nodes, etc.
- **System metrics**: GPU utilization, GPU memory usage, CPU usage
- **Environment**: Python version, PyTorch version, SLURM job ID, hostname

Only rank 0 logs to Comet in multi-GPU runs, so you never get duplicate entries.

### Comparing GPU Configurations

After running single-GPU, multi-GPU, and multi-node jobs, open your Comet project dashboard to compare them side by side:

1. Go to [comet.com](https://www.comet.com) and open your project (`mpcdf-raven-tutorial`)
2. Select the experiments you want to compare (e.g., `spiral_mlp_1gpu`, `spiral_mlp_4gpu`, `spiral_mlp_8gpu`)
3. Use the **Compare** view to see:
   - **Training curves overlay**: Do all configurations converge to the same loss?
   - **Wall-clock time**: How much faster did multi-GPU finish?
   - **System metrics**: Were all GPUs fully utilized?
   - **Hyperparameters diff**: Confirm only GPU count changed between runs

### Useful Comet Features for Benchmarking

- **Panels**: Create custom panels to plot training time vs. GPU count, or efficiency metrics
- **Grouped experiments**: Tag runs by configuration (e.g., `1gpu`, `4gpu`, `8gpu`) and filter by tag
- **Export to CSV**: Download raw metric data for custom analysis or plotting
- **Sharing**: Generate a shareable link to your comparison dashboard for collaborators
- **GPU utilization charts**: Verify that GPUs are not sitting idle — if utilization is low, your data loading or batch size may be the bottleneck

### Setting Up Comet (If You Haven't Already)

Academics get Comet Pro for free. See the [README setup instructions](../README.md#2-optional-set-up-comet-ml---super-recommended) or:

```bash
export COMET_API_KEY='your-api-key-here'

# Make it persistent
echo "export COMET_API_KEY='your-api-key-here'" >> ~/.bashrc
```

No code changes needed — the training script detects the key automatically.

## Manual Benchmarking from SLURM Logs

If you are not using Comet ML, you can extract performance data directly from SLURM output files.

## Understanding Benchmark Output

### Summary Table

```
BENCHMARK RESULTS SUMMARY
====================================================================================================
Job ID     Configuration                  GPUs   Nodes  Time (s)   Time/Epoch (s) Val Loss  Val Acc
----------------------------------------------------------------------------------------------------
123456     Single GPU                     1      1      362.45     7.25           0.1187    0.9530
123457     4 GPUs (1 node)                4      1      95.23      1.90           0.1192    0.9525
123458     8 GPUs (2 nodes)               8      2      57.14      1.14           0.1190    0.9527
====================================================================================================
```

**Columns explained**:
- **Job ID**: SLURM job identifier
- **Configuration**: GPU/node setup
- **GPUs**: Total number of GPUs used
- **Nodes**: Number of nodes
- **Time (s)**: Total training time in seconds
- **Time/Epoch (s)**: Average time per epoch
- **Val Loss**: Best validation loss achieved
- **Val Acc**: Best validation accuracy achieved

### Speedup Analysis

```
SPEEDUP ANALYSIS
================================================================================
Baseline (Single GPU): 362.45 seconds

Single GPU:
  Average time: 362.45 seconds
  Speedup: 1.00x
  Parallel efficiency: 100.0%

Multi-GPU (4 GPUs, 1 node):
  Average time: 95.23 seconds
  Speedup: 3.81x
  Parallel efficiency: 95.2%

Multi-Node (8 GPUs, 2 nodes):
  Average time: 57.14 seconds
  Speedup: 6.34x
  Parallel efficiency: 79.3%
================================================================================
```

**Metrics explained**:
- **Speedup**: `baseline_time / current_time`
- **Parallel efficiency**: `(speedup / num_gpus) × 100%`

## Key Metrics

### 1. Speedup

Measures how much faster training is compared to baseline:

```
Speedup = Time(baseline) / Time(current)
```

**Ideal**: Linear scaling (2 GPUs → 2× speedup, 4 GPUs → 4× speedup)

**Reality**: Slightly sub-linear due to communication overhead

### 2. Parallel Efficiency

Measures how efficiently GPUs are utilized:

```
Efficiency = (Speedup / Number of GPUs) × 100%
```

**Interpretation**:
- **100%**: Perfect linear scaling (rare)
- **90-100%**: Excellent (intra-node with NVLink)
- **75-90%**: Good (multi-node with InfiniBand)
- **<75%**: Communication overhead or bottlenecks

### 3. Throughput

Samples processed per second:

```
Throughput = (Total samples × Epochs) / Training time
```

Higher is better!

## Expected Results

### Ideal vs Realistic

| Configuration | Ideal Speedup | Realistic Speedup | Realistic Efficiency |
|--------------|---------------|-------------------|---------------------|
| 1 GPU | 1.0× | 1.0× | 100% |
| 4 GPUs (1 node) | 4.0× | 3.5-3.8× | 88-95% |
| 8 GPUs (2 nodes) | 8.0× | 6.0-6.5× | 75-81% |

### Why Not Perfect Scaling?

**Communication overhead**:
- **Gradient synchronization**: All-reduce operation takes time
- **Data distribution**: Splitting and gathering data
- **Network latency**: Especially between nodes

**Diminishing returns**:
- Small model relative to GPU power
- Communication becomes bottleneck
- More GPUs → more synchronization

## Creating Visualizations

### Speedup Chart (Manual)

Using the benchmark data, create a speedup plot:

```python
import matplotlib.pyplot as plt

gpus = [1, 4, 8]
speedup = [1.0, 3.81, 6.34]
ideal = [1.0, 4.0, 8.0]

plt.figure(figsize=(10, 6))
plt.plot(gpus, speedup, 'o-', label='Actual Speedup', linewidth=2)
plt.plot(gpus, ideal, '--', label='Ideal (Linear)', linewidth=2)
plt.xlabel('Number of GPUs')
plt.ylabel('Speedup')
plt.title('Training Speedup vs Number of GPUs')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('speedup_chart.png', dpi=150, bbox_inches='tight')
```

### Efficiency Chart

```python
efficiency = [100, 95.2, 79.3]  # From benchmark results

plt.figure(figsize=(10, 6))
plt.bar(gpus, efficiency, color=['green', 'blue', 'orange'])
plt.xlabel('Number of GPUs')
plt.ylabel('Parallel Efficiency (%)')
plt.title('Parallel Efficiency by Configuration')
plt.ylim(0, 100)
plt.axhline(y=75, color='r', linestyle='--', label='75% threshold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('efficiency_chart.png', dpi=150, bbox_inches='tight')
```

## Comparing Across Different Settings

### Effect of Batch Size

Run experiments with different batch sizes:

```bash
# Small batch
sbatch slurm_jobs/single_gpu.sh  # Uses batch_size=32

# Edit the script or pass argument:
# --batch_size 64
```

### Effect of Model Size

Try different hidden dimensions:

```bash
# Small model (default)
--hidden_dims 64 32 16

# Large model
--hidden_dims 256 128 64 32

# Compare training time and speedup
```

## Extracting Specific Metrics

### From SLURM Logs

```bash
# Extract all timing information
grep "Total training time" outputs/slurm-*.out

# Extract validation metrics
grep "Best validation loss" outputs/slurm-*.out

# Extract GPU utilization (if logged)
grep "GPU-Util" outputs/slurm-*.out
```

### From Comet ML

See the [Comet ML section above](#comet-ml--the-easiest-way-to-benchmark) for full details. In short:
1. Open your project dashboard and select experiments to compare
2. Use the built-in comparison view for training curves and system metrics
3. Export metrics to CSV for custom analysis
4. Share a dashboard link with your team

## Common Patterns

### Good Scaling

**Indicators**:
- Efficiency > 85% for single node
- Efficiency > 75% for multi-node
- Validation metrics similar across configurations

**Example**:
```
1 GPU:   100% efficiency, 362s, val_loss=0.1187
4 GPUs:  95% efficiency, 95s, val_loss=0.1192  ← Good!
8 GPUs:  79% efficiency, 57s, val_loss=0.1190  ← Good!
```

### Poor Scaling

**Indicators**:
- Efficiency < 70%
- Large variance in validation metrics
- Disproportionate slowdown

**Possible causes**:
- Batch size too small
- Data loading bottleneck (increase `num_workers`)
- Network issues
- Model too small for multi-GPU

## Reporting Results

### For Documentation

Include these metrics:
- Training time for each configuration
- Speedup factors
- Parallel efficiency
- Final validation accuracy
- Hardware configuration

### Example Summary

```
Training Results on MPCDF Raven
=================================
Dataset: 10,000 spiral pattern samples
Model: MLP [64, 32, 16]
Epochs: 50
Base batch size: 32 per GPU

Configuration       Time      Speedup   Efficiency
--------------------------------------------------
1 GPU (baseline)    6.0 min   1.0×      100%
4 GPUs (1 node)     1.6 min   3.8×      95%
8 GPUs (2 nodes)    1.0 min   6.3×      79%

All configurations achieved ~95% validation accuracy.
```

## Advanced: Profiling

For detailed performance analysis, use PyTorch Profiler:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    trainer.fit(model, datamodule)

prof.export_chrome_trace("trace.json")
```

View trace in Chrome: `chrome://tracing`

## Key Takeaways

1. **Check efficiency**: >75% is good for multi-node
3. **Expect sub-linear scaling**: Communication overhead is normal
4. **Compare metrics**: Ensure accuracy is consistent
5. **Document results**: Help others understand scaling behavior

## Next Steps

- Create visualizations of your results
- Try different batch sizes or model architectures
- Share findings with your team
- Consult [troubleshooting guide](07_troubleshooting.md) if results are unexpected

Continue to [07_troubleshooting.md](07_troubleshooting.md) for common issues and solutions!
