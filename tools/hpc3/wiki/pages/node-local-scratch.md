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
source_git_blobs:
  "src/hpc3/core/sbatch.py": "0a247eba2978d1fc73886bf5153003ed127fc5e7"
  "src/hpc3/core/array_sbatch.py": "ed32f686a73e2ea8cc48d556d306a9d3d3fdb40a"
  "src/hpc3/core/preflight.py": "87e0e78c26f696fd02d60245a82928ed8a45d699"
  "src/hpc3/core/remote.py": "10c5e4e3a13c6f0004fd05cdb01d2ebbc251d8ea"
provenance:
  - "probe job 55675199 on hpc3-l18-04, 2026-09-01"
  - "rusted engine log champion-s2707 (ab48-v7), 2026-09-01"
  - "clients/RustedWarfareBot src/rw_bot/harness/campaign.py@member_command (outside this workspaceRoot)"
  - "login-i17, 2026-09-30 23:40Z: /proc/<pid>/task/*/wchan of a hung apptainer exec showed __IBVSocket_waitForRecvCompletionEvent (D) in syscall 262, newfstatat; board task 465689f5"
fact_checked: 2026-09-30
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
environment probe for minutes is this, and it is a storage stall on the
cluster, not a network or ssh fault on this side. A remote `timeout` would
not end it either, for the same reason the orphan outlived its session.
