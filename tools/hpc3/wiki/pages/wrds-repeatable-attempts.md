---
title: WRDS experiments need durable attempts and verified results
tags: [wrds, reproducibility, operations]
hubs: [operations]
related: ["[[one-ledger-per-repository]]", "[[wrds-filesystem-catalog]]"]
source_paths:
  - "src/hpc3/contracts/wrds_run.py"
  - "src/hpc3/core/wrds_jobs.py"
  - "src/hpc3/core/wrds_results.py"
  - "src/hpc3/core/wrds_scheduler.py"
  - "src/hpc3/wrds/worker.py"
source_git_blobs:
  "src/hpc3/contracts/wrds_run.py": "00660ff78017fe0dd51c344ba8f3f0b51ae0b23f"
  "src/hpc3/core/wrds_jobs.py": "e79b573ab5a1f145599018c38c95fcde511d9d7b"
  "src/hpc3/core/wrds_results.py": "5331107e253a397046ce7d173da3cf0401e87bb9"
  "src/hpc3/core/wrds_scheduler.py": "494b5c5b6f402d8b10a4cab03a331d69039a397d"
  "src/hpc3/wrds/worker.py": "a87cc9d46e07f46c4a0d010f4a23566f2d1532b4"
provenance:
  - "Board task d43697b3: operator requested the existing HPC system for repeatable WRDS experiments"
  - "2026-10-27 America/Los_Angeles: WRDS jobs 40530016 and 40530018, same synthetic input/code/runtime/worker/seed, collected through hpc3.cli.wrds"
  - "Ziggurat /home/corvis/research-data/wrds/experiments/system-probe-20261027/repeatability-proof.json pins both completion records and common RunRecord files; metrics SHA256 60dc457a811582434c73d554c0df58942856247aac8874abcab1942e86d52756; observed sum 45195"
  - "Job 40530005 deliberately exited 17; collect refused WRDS_NOT_COMPLETE and created no output directory; job 40530008 was cancelled through the CLI and absent from qstat; neither attempt had completion.json"
  - "Earlier job 40529992 failed at its worker checksum because Grid Engine's shell wrapper altered escaping; job 40529996 completed after -shell n"
fact_checked: 2026-10-27
confidence: high
---

# WRDS experiments need durable attempts and verified results

WRDS execution lives in the existing HPC package, with a Grid Engine adapter
and the shared research `RunRecord`. The run identity includes code and input
digests, Python binary/version and distribution versions, seed, arguments,
resources and outputs. Local checkout paths are removed from that identity.
An explicit attempt label distinguishes repeats of the same experiment.[^identity]

Submission exclusively creates the attempt directory, verifies uploaded
bytes and asks the scheduler to validate the resource request. It submits
held, persists the job ID remotely and in the local ledger, then releases.
An interrupted release is recovered with `resume`; reusing a directory never
silently creates another job. The staged source and metadata are made
read-only to prevent accidental edits, not to isolate hostile payloads.[^submit]

The worker hashes declared inputs before and after execution and rejects
runtime drift, nonzero exits, missing outputs and nonfinite metrics. It writes
its completion record only after those checks pass. Collection additionally
requires successful scheduler accounting, matching run/attempt/job/worker
identity and verified transferred output hashes. Queue absence alone is
`accounting-pending`, never completion.[^worker][^collect][^status]

The synthetic fixture ran twice with the same worker, code, five-number
input, runtime and seed. Jobs **40530016** and **40530018** produced identical
metrics and identical shared run-record bytes; the sum was **45195**. The
deliberate failure and cancellation produced no completion record. This
establishes the tested execution path, not determinism for arbitrary research
programs or permission to export particular licensed datasets.[^live]

Runtime pins cover the interpreter binary and installed distribution
versions; they do not freeze all package contents into a container. Payloads
must use the supplied seed, declare their data dependencies and write finite
metrics. Current limits are individual Python CPU jobs, up to eight slots,
48 GiB total and seven days; scheduler admission remains authoritative.
UCI's Slurm sweep, chain and container commands are not WRDS commands.[^identity][^worker]

The first checksum wrapper also exposed a scheduler-specific failure: passing
a shell command through Grid Engine's default extra shell altered escaping.
The adapter now requests `-shell n` and explicitly invokes one shell with
quoted arguments. The failed job was recorded as failed rather than being
accepted because it had left the queue.[^scheduler][^live]

[^identity]: `src/hpc3/contracts/wrds_run.py:230` `decode_wrds_run` and `:345` `run_identity`.
[^submit]: `src/hpc3/core/wrds_jobs.py:245` `submit`, `:220` `resume` and `:170` `record_event`.
[^worker]: `src/hpc3/wrds/worker.py:85` `execute_run`.
[^collect]: `src/hpc3/core/wrds_results.py:24` `verified_completion` and `:77` `collect`.
[^status]: `src/hpc3/core/wrds_scheduler.py:126` `status`.
[^scheduler]: `src/hpc3/core/wrds_scheduler.py:13` `job_argv`.
[^live]: Actual jobs and hash-pinned local receipts listed in `provenance`.
