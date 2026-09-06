---
title: Two command-length ceilings, and the token count that met both
tags: [operations, remote, incidents]
hubs: [operations]
related: ["[[triage-conditions]]", "[[ledger-closures]]", "[[preemption-and-campaigns]]", "[[job-arrays]]", "[[submission-rules]]"]
source_paths:
  - "src/hpc3/core/remote.py"
  - "src/hpc3/core/status.py"
  - "src/hpc3/core/squeue.py"
  - "src/hpc3/core/logs.py"
  - "src/hpc3/core/campaign.py"
  - "src/hpc3/core/cancel.py"
  - "src/hpc3/cli/triage.py"
source_git_blobs:
  "src/hpc3/core/remote.py": "dc66efd7dadd6bae900bd37d6f61d268960797ca"
  "src/hpc3/core/status.py": "c89b8061f970b1677c1eff9c76010912c2b80828"
  "src/hpc3/core/squeue.py": "62002937c50a30b2a5032d082e3813d3a989b711"
  "src/hpc3/core/logs.py": "436b22a97e08bf32b046ea8e3b154796189ab614"
  "src/hpc3/core/campaign.py": "16eb99599f73899e856393e48ee7f3fe7addc71e"
  "src/hpc3/core/cancel.py": "7587c162b2b98e55431303b5eb21723926a6a623"
  "src/hpc3/cli/triage.py": "46fd93061aff3ed3ebd76e0bc9407657d3db5859"
provenance:
  - "campaign existence probe truncated mid-loop, vhsearch2-r0, 2026-09-02"
  - "hpc3-triage WinError 206 with runs/ledger.jsonl at 6645 rows, austinpc, 2026-09-05"
  - "bash 'unexpected EOF while looking for matching' from a ~29 KB age probe, hpc3 login-i15, 2026-09-05"
  - "hpc3-triage 2026-09-05 run 1: 6706 recorded, 6417 open, 12 findings, 6345 newly closed; run 2: 72 open, 0 newly closed"
  - "fix committed fa8f87f9 (repo ~/PROJECTS/API)"
fact_checked: 2026-09-06
confidence: high
---

# Two command-length ceilings, and the token count that met both

Every query here names its subjects on a command line — `sacct -j a,b,c`
(`src/hpc3/core/status.py`), `squeue -h -j a,b,c` (`src/hpc3/core/squeue.py`),
`scancel a b c` (`src/hpc3/core/cancel.py`), a `stat` probe per log
(`src/hpc3/core/logs.py`). **A command line has a length**, and on this
submitter that limit fires before Slurm, before ssh, and before the cluster is
reached at all.

## Two ceilings, met three days apart, wearing different disguises

**~8 KB, at the shell.** A 136-member campaign packed every artifact into one
existence probe, ~10 KB, past cmd.exe's 8191-character argument limit. The
command arrived at the remote bash **truncated** and died on `unexpected end
of file` (vhsearch2-r0, 2026-09-02; `src/hpc3/core/campaign.py`,
`existence_commands`).

**32767, at `CreateProcess`.** `hpc3-triage` (`src/hpc3/cli/triage.py`) asked
accounting about every open ledger entry — one id per row, and since
2026-09-06 one per array BASE, which collapses a 60-task array into a single
id ([[job-arrays]]). At 6645 rows of `runs/ledger.jsonl` the argv reached
~70 KB and Python raised

```
FileNotFoundError: [WinError 206] The filename or extension is too long
```

That error **names no command**, so it reads as a missing executable rather
than as an argument that outgrew its call. Triage was unrunnable for as long
as the ledger had been long, and nothing said so — every other `hpc3-*`
command worked, because none of them sizes its query by the ledger.

**A third ceiling exists between them and belongs to neither.** After the
first two were fixed, `age_commands` in `src/hpc3/core/logs.py` still built
one ~29 KB `stat` probe. `CreateProcess` accepted it and ssh sent it; it
arrived at bash truncated mid-quote — `bash: -c: line 1: unexpected EOF while
looking for matching '` (hpc3 login-i15, 2026-09-05). So 32767 is not the
number to design against.

## A count is a guess about token length

The 2026-09-02 fix was `EXISTENCE_CHUNK = 60` — sixty paths per command,
chosen because sixty of *that project's* paths fit. It worked, and it was
local to `src/hpc3/core/campaign.py`.

The identical guess, made separately for job ids in
`src/hpc3/core/status.py`, is what let triage build 70 KB three days later.
Two ad-hoc chunkers for one concern is a fork, and the third would have been
written the same way.

So the split is one mechanism, by **measured width**, in
`src/hpc3/core/remote.py`:

- `token_batches(tokens, overhead=…, separator=…)` — the caller counts the
  characters its command spends on everything that is not the list, because
  only the caller knows its own command. Quoted tokens are batched, because
  the quoted token is what is sent.
- `run_remote_batched(host, commands)` — runs the batches of one split query
  and returns their combined output.
- `MAX_COMMAND_CHARS = 4000` — deliberately far under the lower ceiling. A
  query worth splitting at all is worth splitting well inside every limit it
  can meet.

`EXISTENCE_CHUNK` is gone; `existence_commands` batches through the same
call. `sacct_commands` (`src/hpc3/core/status.py`), `squeue_commands`
(`src/hpc3/core/squeue.py`), `age_commands` (`src/hpc3/core/logs.py`) and the
`scancel` line in `cancel()` (`src/hpc3/core/cancel.py`) are all plural now
— committed as `fa8f87f9` (repo `~/PROJECTS/API`).

## Two things that are wrong in ways a test will not show you

**Outputs recombine line by line, not by concatenating stdout**
(`run_remote_batched`, `src/hpc3/core/remote.py`). A batch whose output lacks
a trailing newline fuses its last row onto the next batch's first. That
presents as one malformed row — a parse defect — and it eats both rows rather
than one.

**Every age batch carries its own clock probe and is parsed separately.** An
age is `now - mtime`, and this measurement's whole premise is that both come
from the same instant (see [[triage-conditions]], `silent`). One clock shared
across batches issued seconds apart is a different instant for every batch
after the first. This is why `log_ages` merges parsed dictionaries instead of
using `run_remote_batched` like every other caller.

## What the fix let happen

The first real run after `fa8f87f9` (austinpc → hpc3 via sedona, 2026-09-05):
6706 recorded, 6417 open, 12 findings, **6345 newly closed**. A second run
reported 72 open and 0 newly closed — the closure record ([[ledger-closures]])
had simply been unable to advance while the command that writes it could not
be built.

**Found by running it, not by testing it.** The third ceiling appeared on the
fixed command's first live invocation, with `make check` already green at
1460 passed and 100% statements + branches. A green suite proved the batching
was correct; it could not prove the batch size was.
