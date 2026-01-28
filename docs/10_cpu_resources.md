# CPU Resources on Raven

An overview of the CPU compute nodes on the MPCDF Raven cluster, including hardware specs, job types, and example SLURM configurations.

## Hardware Overview

Raven has **1592 CPU-only compute nodes** in addition to the 192 GPU nodes.

| Component | Specification |
|-----------|---------------|
| Processor | Intel Xeon IceLake Platinum 8360Y |
| Base frequency | 2.4 GHz |
| Cores per node | 72 physical (144 logical with hyperthreading) |
| NUMA domains | 2 (36 physical cores each) |
| Peak performance (FP64) | 5530 GFlop/s per node |

### Memory Configurations

| RAM per Node | Number of Nodes |
|-------------|-----------------|
| 256 GB | 1524 |
| 512 GB | 64 |
| 2048 GB (2 TB) | 4 |

Most nodes have 256 GB. If you need more, request it explicitly with `--mem`.

### Interconnect

- **CPU nodes**: HDR100 InfiniBand (100 Gbit/s)
- **GPU nodes**: HDR200 InfiniBand (200 Gbit/s), with a subset at 400 Gbit/s

The network uses a pruned fat-tree topology with four non-blocking islands.

## CPU Job Types

### Shared (< 1 node)

Multiple jobs share a single node. You must specify the number of CPUs and memory your job needs.

```bash
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=24000       # 24 GB total
#SBATCH --time=12:00:00
```

Shared CPU jobs can use up to 36 cores (72 in HT mode) and 120 GB per job.

### Exclusive (full nodes)

Your job gets one or more entire nodes.

| RAM Requested | Max Nodes |
|--------------|-----------|
| up to 240 GB (default) | 1–360 |
| up to 500 GB | 1–64 |
| up to 2048 GB | 1–4 |

```bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=72
#SBATCH --time=12:00:00
```

To request high-memory nodes, specify `--mem`:

```bash
#SBATCH --mem=500000     # 500 GB node
#SBATCH --mem=2048000    # 2 TB node
```

Maximum wall time for all CPU jobs: **24 hours**.

## Hyperthreading

Each physical core supports 2 hyperthreads, giving 144 logical CPUs per node. Hyperthreading can improve performance by up to ~20% for some workloads.

To enable hyperthreading, make sure the product `ntasks-per-node × cpus-per-task` goes up to 144 instead of 72:

```bash
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-core=2    # Enable hyperthreading
#SBATCH --cpus-per-task=36     # 36 × 2 HT = 72 logical per task, 4 tasks × 36 = 144 logical total
```

**Note**: With 144 tasks per node, each process gets half the memory compared to non-HT mode. Request more memory with `--mem` if needed.

## Common Job Patterns

### Pure MPI

One MPI rank per core, no threading:

```bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=72
#SBATCH --time=12:00:00

module purge
module load intel/21.4.0 impi/2021.4

srun ./myprog
```

### Hybrid MPI + OpenMP

A few MPI ranks per node, each using multiple threads:

```bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18    # 4 × 18 = 72 cores per node
#SBATCH --time=12:00:00

module purge
module load intel/21.4.0 impi/2021.4

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores

srun ./myprog
```

### Single-core / Sequential

For serial programs, Python scripts, Matlab, etc.:

```bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000        # 2 GB
#SBATCH --time=01:00:00

srun python3 ./myscript.py
```

### Multithreaded (Python, Julia, Matlab, OpenMP)

Single process using multiple cores via threading:

```bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18    # or up to 72 for a full node
#SBATCH --mem=64000
#SBATCH --time=02:00:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python3 ./myscript.py
```

### Python Multiprocessing

Python's multiprocessing spawns its own worker processes internally, so request one SLURM task but many CPUs:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72    # Full node
#SBATCH --time=02:00:00

export OMP_NUM_THREADS=1    # Avoid overloading with nested threading

srun python3 ./myscript.py $SLURM_CPUS_PER_TASK
```

## Interactive Testing

For quick testing and debugging without submitting a batch job:

```bash
srun -n 4 -p interactive --time=01:00:00 --mem=16000 ./myprog
```

- Maximum: 8 cores, 32 GB memory, 2 hours
- Runs on interactive nodes `raven-i.mpcdf.mpg.de`

## Recommended Software Stack

```bash
# Current recommendation
module load intel/21.4.0 impi/2021.4

# For GPU-aware MPI (on GPU nodes)
module load gcc/13 cuda/12.6 openmpi_gpu/5.0
```

No modules are loaded by default on Raven. You must explicitly load compiler and MPI modules both for compilation and in your job scripts.

## Key Constraints

| Limit | Value |
|-------|-------|
| Max wall time | 24 hours |
| Max running jobs per user | 8 (default) |
| Max submitted jobs per user | 300 (default) |
| Max `ntasks-per-node` | 72 (144 in HT mode) |
| `ntasks-per-node × cpus-per-task` | Must not exceed 144 |
| Internet access from compute nodes | Not available (download data from login nodes) |

## Login and Interactive Nodes

| Hostname | Nodes | Purpose | CPU Limit | RAM |
|----------|-------|---------|-----------|-----|
| `raven.mpcdf.mpg.de` | raven01i–02i | Login, editing, compiling, submitting | 2 cores | 512 GB |
| `raven-i.mpcdf.mpg.de` | raven03i–06i | Interactive development and testing | 6 cores | 256 GB |

Running parallel programs on login nodes is **not allowed**. Use `srun -p interactive` or submit batch jobs instead.
