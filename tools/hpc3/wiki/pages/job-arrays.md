---
title: A sweep is one sbatch call, and the script is the member table
tags: [submission, arrays, identity]
hubs: [submission]
related: ["[[sweeps-and-artifacts]]", "[[preemption-and-campaigns]]", "[[job-identity-on-cluster]]", "[[node-local-scratch]]", "[[triage-conditions]]"]
source_paths:
  - "src/hpc3/core/array_sbatch.py"
  - "src/hpc3/core/array_submit.py"
  - "src/hpc3/contracts/array.py"
source_git_blobs:
  "src/hpc3/core/array_sbatch.py": "8ebc37d14ba8fb7d1dcd8b629225fddb57e1cd53"
  "src/hpc3/core/array_submit.py": "97a1d6240ff6a7032bbba3a86710c3220d624061"
  "src/hpc3/contracts/array.py": "cb1ed261958a7a3b541d972e0d64a69da681692e"
provenance:
  - "probe job 55678543 (free, --array=0-3%2), 2026-09-01"
fact_checked: 2026-09-06
confidence: high
---

# A sweep is one sbatch call, and the script is the member table

Submitting member by member cost three SSH round trips each -- upload,
preflight, sbatch, ~13 seconds -- against a cluster that scheduled everything
instantly: rusted's 96-member waves spent ~18 minutes purely submitting
(2026-09-01). A sweep now goes up as one job array: one upload, one
``--test-only``, one ``sbatch --array``.

## The two shape rules

**The script always carries the full member table**, one ``case`` arm per
document position dispatched on ``$SLURM_ARRAY_TASK_ID``, whatever subset is
submitted. **The ``--array`` selection is the submitter's argument, never a
script directive.** Together these keep "the script on disk is the record of
what ran" true while letting a campaign resubmit its sparse gap --
``--array=3,17-19`` -- against a byte-identical script, so the
task-to-member mapping cannot drift between convergence passes. Uniformity
needs no runtime check: the renderer takes the sweep document itself, whose
members share the template by construction.

## Identity, measured before coded

Probe job 55678543 (``--array=0-3%2``, throttled so both states existed at
once) fixed the facts every parser relies on:

* RUNNING and terminal tasks appear everywhere as ``55678543_0`` --
  individually, in ``squeue`` and ``sacct -X`` alike.
* PENDING tasks aggregate into ONE row -- ``55678543_[2-3%2]`` -- in
  ``squeue`` **and in ``sacct -X``**. The sacct half was the surprise.
* ``sacct -j 55678543_2`` returns **nothing** while task 2 is inside the
  pending aggregate. Absent from accounting means "not finished", never
  "safe to resubmit" -- which is exactly how the campaign and the watch
  already read absence.
* ``sbatch --test-only`` on an array answers with the same single verdict
  line a plain job gets; ``sbatch`` announces ``Submitted batch job N``
  unchanged; logs land per task via ``%A_%a``.
* A cancelled array reports an aggregate **only if it was WHOLLY pending** --
  ``55765275_[0-5]|CANCELLED by 2422328`` (submitted 2026-09-04T16:18, never
  started a task, cancelled 2026-09-06). A mid-drain cancellation of a
  partly-started array reports per task instead: ``rusted.evolve4`` was
  scancelled the same night with tasks COMPLETED, RUNNING and PENDING at once
  and left 507 individual rows and no aggregate. Account-wide there were
  exactly two aggregate rows since 2026-09-01, both from arrays that never
  started. **The reproduction condition is "never started", not "cancelled
  while some tasks were queued"** -- worth stating because the second is the
  intuitive guess and it does not reproduce.

``contracts/array.py`` owns the expansion, and the in-flight artifact check
and triage's unclaimed-job check both expand before matching -- unexpanded,
every pending member reads as not-live, and the double-submission race the
package exists to refuse would be waved through.

**Three of triage's readers did not, until 2026-09-06.** `unaccounted_jobs`,
`live_entries` and `closures_for` matched raw ids, so all six tasks of a
healthy queued array were reported as jobs accounting had never heard of, and
a terminal aggregate (`55765275_[0-5]|CANCELLED by 2422328`) produced a
closure keyed on an expression no ledger entry carries. They expand now.

**Expanding is only half of it, and the half that changes nothing alone.** A
reader can only expand a row it was given, and `sacct -j 55678543_2` does not
return one. The query has to be built from the array BASE id -- `sacct -j
55678543` -- which answers with the aggregate while pending and with every
per-task row once they have run. `base_job_ids` is that reduction, and it
also collapses a 60-task array into a single asked id. See
[[triage-conditions]] and [[command-length-limits]].

## What the ledger and audit now say

The ledger records every member under its task id (``<base>_<index>``)
against its own qualified name, before the ids are returned -- per-member
identity durably lives there, because the array's ``squeue`` rows share one
name. The per-member ``job_submitted`` audit events are gone: they described
member-by-member submission acts that no longer happen, and telemetry of
acts that did not occur is a false trail. The one ``sweep_submitted`` event
carries every task id plus the billing factor the whole array went out
under.

Failure atomicity got stronger, not weaker: the old loop could die on member
four leaving three live; the array submits whole or refuses whole, and every
refusal -- artifact race, environment, scheduler -- lands before ``sbatch``.
