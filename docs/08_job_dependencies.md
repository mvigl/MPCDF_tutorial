# SLURM Job Dependencies

SLURM lets you chain jobs so that one starts only after another finishes. This is useful when a later job depends on the output of an earlier one (e.g., training then evaluation, or sequential experiments).

## Basic Syntax

```bash
sbatch --dependency=<condition>:<job_id> <script.sh>
```

## Example: Submit Two Single-GPU Jobs in Sequence

```bash
# Submit the first job and capture its job ID
FIRST_JOB=$(sbatch --parsable slurm_jobs/single_gpu.sh)
echo "First job submitted: $FIRST_JOB"

# Submit the second job, starting only after the first succeeds
sbatch --dependency=afterok:$FIRST_JOB slurm_jobs/single_gpu.sh
```

## Dependency Conditions

| Condition | Meaning |
|-----------|---------|
| `afterok:<job_id>` | Start only if the first job **succeeded** (exit code 0) |
| `afternotok:<job_id>` | Start only if the first job **failed** (non-zero exit code) |
| `afterany:<job_id>` | Start after the first job finishes, **regardless of success or failure** |
| `after:<job_id>` | Start after the first job has **begun** (not necessarily finished) |

## When to Use Each

- **`afterok`**: The second job needs correct output from the first (e.g., train → evaluate checkpoint).
- **`afternotok`**: Retry or cleanup logic when a job fails.
- **`afterany`**: The second job should always run, e.g., a post-processing or logging step.
- **`after`**: Rare; useful when two jobs can partially overlap.

## Checking Dependency Status

```bash
squeue -u $USER -o "%.10i %.20j %.10T %.12r"
```

A pending dependent job will show reason `Dependency` until its condition is met.
