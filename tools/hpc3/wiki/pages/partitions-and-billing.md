---
title: Billing follows the job's QOS, and free is not a setting
tags: [cluster-facts, billing, partitions]
hubs: [cluster-facts]
related: ["[[facts-are-code]]", "[[submission-rules]]", "[[budget-model]]"]
source_paths:
  - "src/hpc3/clusters/hpc3.py"
source_git_blobs:
  "src/hpc3/clusters/hpc3.py": "7bb86a2753a78657365f4bd69c3a0dedd0f1e82c"
provenance:
  - "sshare RawUsage measurement 2026-08-23 (cjmayer_lab)"
fact_checked: 2026-09-09
confidence: high
---

# Billing follows the job's QOS, and free is not a setting

| partition | GPUs | bills | preempt mode | max hours | per-user ceiling |
| --- | --- | --- | --- | --- | --- |
| `free-gpu` | V100, A30, A100 | no | `CANCEL` | 72 | 24 GPUs |
| `free-gpu32` | L40S, RTX6000 | no | `CANCEL` | 72 | 4 GPUs |
| `free` | — | no | `CANCEL` | 72 | 3500 cores |
| `gpu` | V100, A30, A100 | **yes** | `OFF` | 336 | 40 GPUs |
| `gpu32` | L40S, RTX6000 | **yes** | `OFF` | 336 | 12 GPUs |
| `standard` | — | **yes** | `OFF` | 336 | 2500 cores |

The mode column replaced a yes/no `preemptible` column on 2026-09-09, and the
difference is the one this wiki's own SCHEMA asks for: "free-gpu is
preemptible" is a claim, and `PreemptMode=CANCEL` read from `scontrol show
partition` is evidence. All six values above were measured that way.

It matters beyond bookkeeping. `CANCEL` means Slurm destroys the job and does
not resubmit, so `--requeue` buys nothing there — and the submission guard
demanded that flag anyway until the contract could read the mode rather than a
boolean projection of it ([[preemption-and-campaigns]]).

**This package submits to the free three and refuses the other three**, so the
last two columns are what you get: 72 hours per attempt, and preemption
([[preemption-and-campaigns]]).

## The QOS lesson, measured

This table said `free-gpu32` billed at `UsageFactor 1.0` for a day, and it
was wrong. The 1.0 belongs to `free-gpu32-part`, which is the **partition**
QOS and governs limits. Jobs there run under `low`, at `0.000000`, because
every free partition declares `AllowQos=low,guest`. Reading a factor off the
partition QOS is how that error was made.

Measured, not reasoned — an 8-core, 1-GPU, 2-minute RTX6000 job on
`free-gpu32` moved `sshare` `RawUsage` by **exactly zero**, on a meter that
read 33,654,891 for another user in the same account at the same moment. So
the rule is `AllowQos`: `low,guest` (both `0.0`) is free, `normal,high`
(`1.0` and `2.0`) bills. **L40S and RTX6000 are free**, which the old wrong
fact had been hiding. The billing three are marked from the `AllowQos` rule
rather than from a measurement, deliberately: the safe direction to be wrong
is to refuse something that would have been free, never to spend on something
recorded as free.

There is also a QOS literally *named* `free-gpu` carrying
`UsageFactor 1.000000`, which is not what the `free-gpu` partition uses. Same
lesson twice: a name containing "free" and a QOS containing a factor are two
different questions, and only a job's own accounting answers either.

## Free only, and not as a setting

There is no `accept_billing`. A billing partition is refused outright with
`PARTITION_BILLS`, which names the free ones. A consent flag would be a limit
a run could switch off — the same shape as declaring `max_gpus_per_user: 999`
to raise a ceiling. Both disable the check instead of changing the fact.

Not carried, each for a measured reason: `standard-hbm` needs an explicit
`--qos` this package does not emit; `gpu-hugemem` returns
`Invalid account or account/partition combination`; `highmem`, `hugemem`,
`maxmem` and `admin` appear in `sinfo -a` but not in
`scontrol show partition`.
