---
title: Small-file boots belong on the node's own scratch, never on BeeGFS
tags: [cluster-facts, storage, performance]
hubs: [cluster-facts]
related: ["[[partitions-and-billing]]", "[[preemption-and-campaigns]]", "[[facts-are-code]]"]
source_paths:
  - "src/hpc3/core/sbatch.py"
  - "src/hpc3/core/array_sbatch.py"
  - "src/hpc3/core/preflight.py"
  - "src/hpc3/core/remote.py"
  - "src/hpc3/core/image_sbatch.py"
  - "src/hpc3/core/image_build.py"
source_git_blobs:
  "src/hpc3/core/sbatch.py": "0a247eba2978d1fc73886bf5153003ed127fc5e7"
  "src/hpc3/core/array_sbatch.py": "ed32f686a73e2ea8cc48d556d306a9d3d3fdb40a"
  "src/hpc3/core/preflight.py": "940f3f7520a93221b6718b3310e0eaa651f96465"
  "src/hpc3/core/remote.py": "80dd48cea51b34dbd3746ae7c304c36c6ce51de1"
  "src/hpc3/core/image_sbatch.py": "40f0336fc86894be4569b1c2c8f55e028961f85a"
  "src/hpc3/core/image_build.py": "55143e4d98a086254d42979a475d735ed6943666"
provenance:
  - "probe job 55675199 on hpc3-l18-04, 2026-09-01"
  - "rusted engine log champion-s2707 (ab48-v7), 2026-09-01"
  - "clients/RustedWarfareBot src/rw_bot/harness/campaign.py@member_command (outside this workspaceRoot)"
  - "login-i17, 2026-09-30 23:40Z: /proc/<pid>/task/*/wchan of a hung apptainer exec showed __IBVSocket_waitForRecvCompletionEvent (D) in syscall 262, newfstatat; board task 465689f5"
  - "login-i15, 2026-10-28T06:07Z: PID 2581254 stat of /pub/wagnera3/envs/cleargbm/bin/python waited in D state, IBVSocket_waitForRecvCompletionEvent, syscall 332 (statx); board task eadd19d5"
  - "login-i15, 2026-10-28: PID 2810355 writing the synthetic benchmark entrypoint waited in D state, FhgfsOps_revalidateIntent; dfsquotas reported 342.16 GiB of 1.00 TiB and 564.45k of 8.00M chunk files"
  - "HPC3 job 58566045, hpc3-20-20, COMPLETED 0:0 in 85 seconds, 2026-10-28: paired image build with node-local Apptainer cache and temporary directory"
fact_checked: 2026-10-28
confidence: high
---

# Small-file boots belong on the node's own scratch, never on BeeGFS

Slurm provisions **`$TMPDIR=/tmp/<user>/<jobid>`** on every HPC3 compute
node: per-job, on local disk, removed with the job. Measured on
`hpc3-l18-04` (probe job 55675199): a 256 MiB write lands at **1.9 GB/s**.
The RCIC docs do not state this anywhere findable — it was established by
probe, which is why it is written down here.

## The failure class this closes

`/pub` is BeeGFS — a parallel filesystem built for large streaming I/O and
poor at concurrent small-file metadata storms. A workload that BOOTS by
reading thousands of small files (a game engine loading `.ini` assets, an
interpreter walking a venv, anything with a file-watcher) crawls when many
jobs boot at once against it, and it crawls in a way that looks like
anything but a filesystem problem: the rusted project lost **ten members
across four batches** to what read as random engine crashes, every one
completing on an uncontended retry. The engine's own log told the truth —
asset lines seconds apart, a failed watch attempt per file, and the
process halted by its own 60-second world-liveness guard, not by any
crash.

Two properties made the class hard to see:

- **Retry succeeds.** An uncontended boot is fast, so every resubmission
  completes and the failure reads as "transient". It is not transient; it
  is deterministic under concurrency.
- **The guard names its symptom, not the cause.** "The live world is
  null after 60s" is a true statement about a slow boot, and nothing in it
  says *filesystem*.

## The rule

Per-job disposable data — clones, working copies, extracted archives,
anything the job creates and the result does not need — goes in
`$TMPDIR`. Reference `$TMPDIR` **unexpanded in the submitted command** so
the batch script's bash expands it on the node after Slurm has provisioned
it; the generated scripts run `set -u`, so a node without it fails loudly
rather than quietly landing on shared disk. What the shared filesystem is
for: the one-time streaming copy IN (BeeGFS is good at that), and the
artifacts a run files OUT.

The rusted project's `member_command` is the worked example: it emits
`--clones $TMPDIR/rw-clones`, the shared-filesystem clone helper was
deleted rather than deprecated, and a test pins that no clone path may
start with the cluster root. Cleanup came free — the node removes
`$TMPDIR` with the job.

## The login node stalls on /pub too, and nothing on the hub can bound it

`hpc3-preflight` probes an image job's environment on the LOGIN node: one
`apptainer exec` of the `.sif` on `/pub`, running `test -d <env>/bin`
(`check_env_path`, `src/hpc3/core/preflight.py`). On 2026-09-30, running
`tools/hpc3`'s real-host execution case from the hub (board task 465689f5),
that probe answered in about 3.3 s through the whole jump route in 15 of 15
direct runs, and in 0.4 to 1.5 s run on `login-i17` itself, yet it hung for
minutes in 4 of 11 execution and preflight runs between 23:03Z and 23:40Z.

Inspected live on `login-i17` during a hang, the stuck `apptainer exec` had
no child and no image file open. Ten threads were in `futex_wait`, and one
was in uninterruptible sleep in `__IBVSocket_waitForRecvCompletionEvent`
inside `newfstatat`: the BeeGFS client waiting on InfiniBand for a metadata
answer about a `/pub` path. The ssh session kept answering keepalives
throughout, so the `ConnectTimeout` and `ServerAlive*` bounds that
`SSH_OPTIONS` (`src/hpc3/core/remote.py`) puts on every remote call did not
fire, and they cannot: they bound the route, not a command the cluster is
still running. Killing the hub's ssh left the remote process running. One
from 23:34Z was still in that state at 23:41Z, re-parented to init, because
a process in uninterruptible sleep takes no signal until its I/O returns.

What this means for a reader: a preflight or execution run that sits on the
environment probe for minutes can have this cause. Inspect the remote wait
state before attributing a new hang to storage. A remote `timeout` cannot
complete termination while a process remains in uninterruptible I/O, for
the same reason the orphan outlived its session.

## The October 28 recurrence and image-build mitigation

On `login-i15`, `/pub` resolved to `/dfs6b/pub`. A `stat` of the existing
ClearGBM environment's interpreter exceeded a ten-second probe bound; a
subsequent probe was observed in `IBVSocket_waitForRecvCompletionEvent`
inside `statx`. A later staging writer waited in `FhgfsOps_revalidateIntent`.
SSH and home-directory operations remained responsive, and both byte and
chunk-file quotas were below their limits. These observations locate the delay in
BeeGFS metadata operations; they do not identify the failing storage server
or establish a cluster-wide outage.[^october]

The shared image builder now requires the compute node's `$TMPDIR` and
places `APPTAINER_CACHEDIR` and `APPTAINER_TMPDIR` beneath it. Missing or empty
scratch fails with `HPC3_NODE_SCRATCH_MISSING`; final images and logs remain
in the declared image directory. This reduces shared-filesystem work during
builds, but cannot repair login-node metadata stalls when staging files or
opening an image.[^builder]

Job **58566045** built the paired evaluation image on `hpc3-20-20` in
**85 seconds**, including image self-checks, unprivileged verification and
`pip check`, with exit `0:0`. This is evidence that the build completed with
node-local temporary storage, not a controlled performance comparison or
proof that BeeGFS recovered everywhere.[^build]

[^october]: Live probes on `login-i15`, 2026-10-28, PIDs 2581254 and 2810355 and `dfsquotas` readings recorded in `provenance`; board task eadd19d5.
[^builder]: `src/hpc3/core/image_sbatch.py:73` `_preamble`, including the required scratch check and quoted cache/temp exports; `src/hpc3/core/image_build.py` failure guidance.
[^build]: `sacct` for job 58566045 and `/pub/wagnera3/images/cleargbm-paired-2a50e93ca/build-58566045.out` and `.err`, 2026-10-28; image SHA256 `f5d16ad3732cb1ba196b2c56aa56fa3465dcc183c75fc73924a5aecf455fb16b`.
