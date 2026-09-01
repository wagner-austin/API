---
title: The budget is per project, because nothing else says stop
tags: [operations, budget, submission]
related: [[submission-rules]], [[partitions-and-billing]], [[triage-conditions]]
sources: [contracts/budget.py, core/budget.py, README.md@4dc63f17]
fact_checked: 2026-09-01
confidence: high
---

# The budget is per project, because nothing else says stop

The QOS bounds what runs *at once*. Nothing bounds the total, and on the free
partitions nothing bills — a 24-GPU three-day sweep is 1,728 GPU-hours,
inside every limit the cluster enforces, and not a reasonable share of a
shared machine. `BUDGET_PROJECTION_EXCEEDED` fires before submission,
`BUDGET_CONSUMPTION_EXCEEDED` from `hpc3-watch` while running.

## Why the budget lives on the project

It lived on the workspace until 2026-08-28, which sounds tidier and was not:
a cap is the one thing that genuinely differs between bodies of work, so the
only way to say so was to fork the whole document — and three forks were
committed, declaring 0.5, 12.0 and 1.0 GPU-hours over the same ledger.
`hpc3-watch` then enforced whichever one you happened to pass against
whatever job you named. `charge_account` moved with the caps for a sharper
reason: accounts are per-PI, and a job charged to the wrong one spends
another lab's allocation.

There are no `--host` / `--root` / `--budget` / `--ledger` flags for the same
family of reason. When they existed, nothing tied `hpc3-triage --ledger` to
the ledger `hpc3-submit` had written — and pointing them at different paths
gives you either a clean board while jobs run unwatched, or every job
reported as `unaccounted` while nothing is wrong. Both readings are wrong and
neither looks wrong.

## What the budget cannot see

`TIME_LIMIT_EXCEEDS_PARTITION` bounds a single attempt, not a total: a
requeue restarts that clock, so nothing caps cumulative wall time across
requeues except the GPU-hour budget — and the budget projects from
*requested* minutes, so a requeued job can exceed its own projection. Watch
it with `hpc3-watch`. The `oversized` triage finding is the other half of the
same lesson: a budget computed from an unmeasured request ratifies the
request ([[triage-conditions]]).
