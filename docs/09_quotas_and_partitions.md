# Quotas and Partitions on Raven

A reference for checking resource limits, partition availability, and storage quotas on the MPCDF Raven cluster.

## GPU Partitions

Raven GPU nodes are accessed via constraints rather than named partitions. The relevant SLURM constraint values are:

| Constraint | Description | Nodes | Time Limit | Use Case |
|------------|-------------|-------|------------|----------|
| `gpu` | Standard GPU nodes | 192 | Up to 24 h | Production training |
| `gpu-bw` | High-bandwidth InfiniBand (400 Gbit/s) | 32 | Up to 24 h | Multi-node jobs needing fast inter-node communication |
| `gpudev` | Development / debugging | Shared | 15 min | Quick tests and debugging |

Use the constraint in your SLURM script:

```bash
#SBATCH --constraint="gpu"       # Standard GPU nodes
#SBATCH --constraint="gpu-bw"    # High-bandwidth nodes
```

For quick debugging with the development partition:

```bash
#SBATCH --partition=gpudev
#SBATCH --time=00:15:00          # Max 15 minutes
```

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

- **Quota**: 2.5 TB
- **Backed up**: Yes
- **Use for**: Code, virtual environments, important results

### Scratch space (`/ptmp`)

```bash
/usr/lpp/mmfs/bin/mmlsquota raven_ptmp
```

- **Quota**: No hard limit (fair usage policy)
- **Backed up**: No
- **Auto-deleted**: Files inactive for 12 weeks
- **Use for**: Temporary outputs, large datasets, checkpoints during training

### Archive (`/r`)

- Available on login nodes only
- Automatically migrated to tape
- Pack files into tar archives (1 GB – 1 TB recommended)
- **Use for**: Completed experiments, published results

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
