# Quotas and Partitions on Raven

A reference for checking resource limits, partition availability, and storage quotas on the MPCDF Raven cluster.

## GPU Resource Allocation

Raven has **192 GPU-accelerated nodes**, each with 4 Nvidia A100 GPUs (40 GB HBM2), 72 CPU cores, and 512 GB RAM. SLURM automatically assigns your job to a shared or exclusive node based on how many GPUs you request.

### Shared vs Exclusive GPU Jobs

When you request **fewer than 4 GPUs**, your job shares a node with other jobs. When you request **all 4 GPUs**, you get the entire node exclusively.

| GPUs Requested | CPUs Available | Memory Available | Node Allocation | SLURM Flags |
|---------------|---------------|-----------------|-----------------|-------------|
| 1 | 18 (36 in HT mode) | 125 GB | Shared | `--gres=gpu:a100:1 --cpus-per-task=18 --mem=125000` |
| 2 | 36 (72 in HT mode) | 250 GB | Shared | `--gres=gpu:a100:2 --cpus-per-task=36 --mem=250000` |
| 3 | 54 (108 in HT mode) | 375 GB | Shared | `--gres=gpu:a100:3 --cpus-per-task=54 --mem=375000` |
| 4 | 72 (144 in HT mode) | 500 GB | Exclusive | `--gres=gpu:a100:4 --cpus-per-task=72 --mem=500000` |

**Important**: For shared jobs (1–3 GPUs), you **must** specify `--mem` so SLURM can fit multiple jobs on one node. Resources scale proportionally — each GPU comes with 18 CPU cores and ~125 GB of RAM.

### GPU Network Constraints

All GPU jobs require a constraint to select the type of node. A SLURM job filter automatically routes your job to the right partition, so you only need to specify the constraint:

| Constraint | InfiniBand Bandwidth | Available Nodes | Max Nodes per Job |
|------------|---------------------|-----------------|-------------------|
| `gpu` | 200 Gbit/s or 400 Gbit/s | 192 | 80 |
| `gpu-bw` | 400 Gbit/s only | 32 | 16 |
| `no-gpu-bw` | 200 Gbit/s only | 160 | 80 |

```bash
#SBATCH --constraint="gpu"          # Any GPU node (most flexible, recommended default)
#SBATCH --constraint="gpu-bw"       # High-bandwidth nodes only (better for multi-node)
#SBATCH --constraint="no-gpu-bw"    # Standard-bandwidth nodes only
```

- **Single-node jobs**: Use `--constraint="gpu"` — bandwidth between nodes does not matter.
- **Multi-node jobs**: Consider `--constraint="gpu-bw"` for faster inter-node gradient synchronization, but note the smaller pool of available nodes (32 vs 192).

### The `gpudev` Partition

For quick testing and debugging, use the `gpudev` partition instead of a constraint:

```bash
#SBATCH --partition=gpudev
#SBATCH --time=00:15:00    # Max 15 minutes
```

- Only 1 node available, shared between users
- 1–4 GPUs can be requested
- Maximum wall time: 15 minutes
- Jobs typically start faster than on the production GPU partition

### Summary of GPU Job Limits

| Job Type | GPUs/Node | Max Nodes | Max Wall Time |
|----------|-----------|-----------|---------------|
| Shared GPU | 1–3 | < 1 (shared) | 24 h |
| Exclusive GPU | 4 | 1–80 | 24 h |
| Exclusive GPU (high-BW) | 4 | 1–16 | 24 h |
| GPU dev | 1–4 | 1 (shared) | 15 min |

Default job run limit: **8 concurrent running jobs**. Default submit limit: **300 jobs** (running + pending).

## Checking Partition and Node Status

### Available nodes and their state

```bash
# Overview of all GPU nodes
sinfo -p gpu

# Detailed per-node info (state, CPUs, memory, features)
sinfo -p gpu --Node --long
```

**Node states**:
- `idle` — Available
- `alloc` — Fully allocated
- `mix` — Partially allocated (some GPUs free)
- `drain` — Taken offline for maintenance

### Current queue

```bash
# Jobs running or pending on the GPU partition
squeue -p gpu

# Count pending jobs ahead of you
squeue -p gpu | grep PD | wc -l

# Your own jobs
squeue -u $USER
```

## Checking Job Limits (QOS)

Quality of Service (QOS) settings define per-user limits on job count, CPUs, GPUs, and wall time.

```bash
# Show all QOS policies
sacctmgr show qos format=Name,MaxWall,MaxTRESPerUser,MaxJobsPU,MaxSubmitPU

# Check which QOS your account uses
sacctmgr show assoc where user=$USER format=Account,User,QOS
```

Key columns:
- **MaxWall** — Maximum wall-clock time per job
- **MaxTRESPerUser** — Maximum trackable resources (e.g., GPUs) per user
- **MaxJobsPU** — Maximum running jobs per user
- **MaxSubmitPU** — Maximum submitted (running + pending) jobs per user

## Checking Storage Quotas

### Home directory (`/u`)

```bash
/usr/lpp/mmfs/bin/mmlsquota raven_u
```

- **Quota**: 2.5 TB, 1 million files
- **Backed up**: Yes
- **Use for**: Code, virtual environments, important results

### Scratch space (`/ptmp`)

```bash
/usr/lpp/mmfs/bin/mmlsquota raven_ptmp
```

- **Size**: 12 PB total
- **Quota**: No hard limit (fair usage policy)
- **Backed up**: No
- **Auto-deleted**: Files inactive for 12 weeks
- **Use for**: Batch job I/O, temporary outputs, large datasets, checkpoints during training

### Archive (`/r`)

- Available on login nodes only (not from compute nodes)
- Automatically migrated to tape
- Pack files into tar archives (1 GB – 1 TB recommended); avoid archiving many small files
- **Use for**: Completed experiments, published results

### `/tmp` — Do Not Use

Do not use `/tmp` or `$TMPDIR` for scratch data. Use `/ptmp` instead, which is accessible from all nodes. If your application requires node-local storage, use `JOB_TMPDIR` or `JOB_SHMTMPDIR` (set per job by SLURM).

## Checking a Specific Job's Resources

```bash
# Detailed info about a running or recently completed job
scontrol show job <job_id>

# Efficiency report (CPU/GPU utilization, memory usage)
seff <job_id>
```

`seff` is particularly useful after a job completes — it shows whether you over- or under-requested resources.

## Quick Reference

```bash
# Partition status
sinfo -p gpu

# Your jobs
squeue -u $USER

# QOS limits
sacctmgr show qos format=Name,MaxWall,MaxTRESPerUser,MaxJobsPU

# Home quota
/usr/lpp/mmfs/bin/mmlsquota raven_u

# Scratch quota
/usr/lpp/mmfs/bin/mmlsquota raven_ptmp

# Job efficiency (after completion)
seff <job_id>
```
