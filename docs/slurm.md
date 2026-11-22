# Slurm

On a cluster, set a partition in your site, and txtures submits work that
runs language models as a Slurm job array:

```bash
txtures site set slurm.partition=gpu
```

`txtures run` then returns once the job is queued. `tx.compute` and `tx.run`
in Python submit the same kind of job and wait for it to end (see
[From Python](#from-python)). `txtures run --local` and `local=True` compute
one request here instead.

## Choosing local or Slurm

`execution.executor` decides where requests are computed:

| Value | Requests are computed |
| --- | --- |
| `auto` (default) | on Slurm when they have Slurm work missing and Slurm is set up; otherwise here |
| `slurm` | always on Slurm |
| `local` | always here |

With `auto`:

- Slurm work means missing results of a module whose workload is marked
  `slurm`, such as `lm_scores` and `parse`, which run neural models, however
  few documents there are, or a missing resource marked the same way (see
  [Resource builds](#resource-builds)). A request whose missing work is all
  light, such as surprisal from stored scores, is computed here, and so is
  a request whose results are all stored.
- Slurm is set up when `slurm.partition` is set and the `sbatch` command is
  available. Where neither is present, as on a laptop, requests are computed
  here. Where only one is, Slurm work is computed here with a note.

Because the choice depends on the machine, one site can serve a cluster and
the machines without Slurm that share the home. `--nworkers` and `nworkers=`
apply when a request is submitted and are ignored otherwise.

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
| `slurm.cpu_partition` | unset | the partition for jobs that need no GPU; unset uses `slurm.partition` |
| `slurm.account` | unset | the account to charge |
| `slurm.time_hours` | 24 | the time limit of each task |
| `slurm.memory_gb` | 32 | the memory of each task |
| `slurm.cpus` | 4 | the CPUs of each task |
| `slurm.gpus` | 1 | GPUs per task, requested as `--gres=gpu:N`; 0 for none |
| `slurm.workers` | 1 | the number of array tasks |
| `slurm.options` | `[]` | further `sbatch` options, such as `["--exclude=node1"]` |

## Running

```console
$ txtures run stimuli fillers --nworkers 4
submitted job 81233 to build resource hf_model
submitted job 81234 (4 GPU tasks) for the GPU work
submitted job 81235 (1 task) to write 2 tables (stimuli, fillers) after it
logs: .../outputs/.txtures-jobs/.batches/main-3f9a0c1d2e4b/logs; check progress with `txtures status`
```

`--nworkers N` sets the number of array tasks that share the work for one
run (default `slurm.workers`), and `--local` computes in the calling process
instead of submitting. Planning happens before submission, so invalid
inputs and definitions are reported at once.

Several inputs given to one `txtures run` share one set of jobs: one GPU job
array works through the GPU work of all of them, and one job writes all the
tables. Inputs whose tables are already current are left out. If nothing is
missing, nothing is submitted. If an earlier job for the same tables is
still pending or running, `txtures run` reports it and submits nothing.

### GPU jobs

GPU hours are paid for, so GPU work runs in a job of its own that never sits
idle:

1. Resources the work needs, such as a language model's tokenizer and
   weights, are built first by jobs without GPUs (see
   [Resource builds](#resource-builds)), so no GPU waits for a download.
2. A GPU job array of `--nworkers` tasks computes only the results of
   modules whose job needs a GPU (`lm_scores` and `parse`), for every
   input. Each task computes its own share of the missing documents, then
   any document no other task holds, and ends as soon as nothing is left for
   it, releasing its GPU. It never waits for another task.
3. A job without GPUs, on `slurm.cpu_partition` when that is set, starts
   when the GPU job has ended. It computes everything else, such as
   surprisal from the stored scores, and writes the tables.

A module's job needs a GPU when its workload asks for one, or `slurm.gpus`
is above 0 and the workload does not say; `--ngpus` decides for one run
(see [Job sizes](#job-sizes)). With no GPU work missing, one job without
GPUs does everything.

If a GPU task dies, the documents it held are not computed, and the table
job writes only the tables whose GPU results are complete, reporting the
others. It does not run GPU work on CPUs. Run `txtures run` again to submit
the missing GPU work.

Each task checks that the input, definitions, and resolved model revisions
are unchanged since submission, and skips a request that changed. Run
`txtures run` again to submit the new request.

Tasks never submit jobs of their own. Each task sets `TXTURES_TASK`, which
every process it starts inherits. Where that variable is set, `txtures run`
and the Python API compute in the same process and say so.

## Job sizes

Each module and resource declares a workload: whether it runs on Slurm,
and optionally the time (`time_hours`), memory (`memory_gb`), cores
(`ncores`), and GPUs (`ngpus`) its job needs. A job takes the largest value
that any of its Slurm workloads gives for each, and the `slurm` setting
(`slurm.time_hours`, `slurm.memory_gb`, `slurm.cpus`, `slurm.gpus`) for the
rest. A job that needs no GPU goes to `slurm.cpu_partition` when that is
set. The numbers have no effect on work computed here.

One call can set the size of the jobs it submits, overriding the workloads
and the settings: `--time-hours`, `--memory-gb`, `--ncores`, and `--ngpus`
on `txtures run`, `txtures resources build`, and `txtures tools build`, or
`slurm={"ngpus": 0}` and the like in `tx.run` and `tx.compute`. Time,
memory, and cores apply to every job, build jobs included. `--ngpus` sets
the GPUs of the GPU job, and with `--ngpus 0` there is none: `txtures run
stimuli --ngpus 0` scores a small model on CPU nodes. Build jobs and the job
that writes tables never get GPUs from it. These options do nothing for
work computed here.

## Resource builds

Some resources, such as an n-gram model trained on a large corpus, take
hours and more memory than a request's tasks have. A missing resource whose
workload is `slurm` is built by a job of its own, sized by that workload.
The request's job array waits for every build to succeed. If a build
fails, Slurm cancels the array.

```console
$ txtures run stimuli
submitted job 81230 to build resource kenlm_model
submitted job 81231 (1 task) to write .../outputs/stimuli/main-3f9a0c1d2e4b after the builds
```

A build job records itself in
`OUTPUTS/.txtures-jobs/.resources/KIND-IDENTITY12/`, and `txtures status`
lists it as `resources/KIND-IDENTITY12`. A request that needs a resource
whose build job is still active waits for that job instead of starting
another. Resources are shared through the `resources` directory, so a lab
builds each one once.

`txtures resources build NAME` builds the resources that a request needs
without computing anything else: resources with a `slurm` workload as build
jobs where Slurm is set up, and the others here. With `--local`, it builds
every resource in the calling process. Without Slurm, `txtures run` builds
such a resource in the calling process and says so in a note.

## From Python

`tx.compute` and `tx.run` submit a job when anything is missing, then block
until it ends. Interrupting the call, for example with Ctrl-C, cancels a
job it submitted. `nworkers=N` and `local=True` match `--nworkers` and `--local`; see
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
