# Slurm

When `execution.executor` is `slurm`, `txtures run` submits its computation
as a Slurm job array and returns once the job is queued. `tx.compute` and
`tx.run` in Python submit the same kind of job and wait for it to end (see
[From Python](#from-python)). Choose Slurm in your site with
`txtures site set execution.executor=slurm`; `txtures run --local` then
computes one request here instead.

## Settings

Set at least the partition, and the account if the cluster requires one,
in your site:

```bash
txtures site set slurm.partition=gpu slurm.account=mylab
```

To use another partition or account yourself without changing a shared
site, use `txtures env set slurm.partition=NAME` (see
[Departing from your site](settings.md#departing-from-your-site)).

| Key | Default | Meaning |
| --- | --- | --- |
| `slurm.partition` | unset; required | the partition to submit to |
| `slurm.account` | unset | the account to charge |
| `slurm.time_hours` | 24 | the time limit of each task |
| `slurm.memory_gb` | 32 | the memory of each task |
| `slurm.cpus` | 4 | the CPUs of each task |
| `slurm.gpus` | 1 | GPUs per task, requested as `--gres=gpu:N`; 0 for none |
| `slurm.workers` | 1 | the number of array tasks |
| `slurm.options` | `[]` | further `sbatch` options, such as `["--exclude=node1"]` |

## Running

```console
$ txtures run stimuli --workers 4
submitted job 81234 (4 tasks) to write .../outputs/stimuli/main-3f9a0c1d2e4b
logs: .../outputs/.txtures-jobs/stimuli/main-3f9a0c1d2e4b/logs; check progress with `txtures status`
```

`--workers N` overrides `slurm.workers` for one run, and `--local` computes
in the calling process instead of submitting. Planning happens before
submission, so invalid inputs and definitions are reported at once.

If the table is already current, nothing is submitted. If an earlier job for
the same table is still pending or running, `txtures run` reports it and
submits nothing.

Tasks share the work. Each task computes its own share of the missing
documents first, then takes over any left, so the job finishes even if a
task fails. The first task to finish writes the table, and the rest find it
current. A task that dies stops holding its documents within 15 minutes, and
another task or a later run computes them.

Each task checks that the input, definitions, and resolved model revisions
are unchanged since submission, and fails without computing if they are not.
Run `txtures run` again to submit the new request.

Tasks never submit jobs of their own. Each task sets `TXTURES_TASK`, which
every process it starts inherits. Where that variable is set, `txtures run`
and the Python API compute in the same process and say so.

## From Python

`tx.compute` and `tx.run` submit a job when anything is missing, then block
until it ends. Interrupting the call, for example with Ctrl-C, cancels a
job it submitted. `workers=N` and `local=True` match `--workers` and `--local`; see
[Python API](python.md#slurm).

For `tx.compute`, the job directory is
`OUTPUTS/.txtures-jobs/tx.compute/WORKFLOW-DIGEST12/`. It also holds
`texts.json`, the text and configs that the tasks read. These tasks store
results in the cache but write no table. The waiting call builds the table
itself.

## Following a job

`txtures status` lists each recorded job with its task states and log
directory, under `OUTPUTS/.txtures-jobs/`. Task states come from
`sacct`, or from `squeue` where accounting is unavailable. `txtures gc`
removes the records and logs of jobs with no active tasks.

The job runs the same installation that submitted it, so the installation,
and the txtures home must be on storage that compute
nodes can reach.
