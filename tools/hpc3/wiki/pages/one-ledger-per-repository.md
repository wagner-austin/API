---
title: One ledger per repository, whichever worktree submits
tags: [operations, ledger, campaigns, worktrees]
hubs: [operations]
related: ["[[ledger-closures]]", "[[preemption-and-campaigns]]", "[[job-identity-on-cluster]]", "[[job-arrays]]"]
source_paths:
  - "src/hpc3/core/checkout.py"
  - "src/hpc3/cli/_config.py"
  - "src/hpc3/contracts/workspace.py"
  - "src/hpc3/core/inflight.py"
  - "README.md"
source_git_blobs:
  "src/hpc3/core/checkout.py": "e903c5d1c622d088d73dbc1bc018b41dd6077d6c"
  "src/hpc3/cli/_config.py": "5287aecbeca09b166e8932f5ea38b227fbd7e720"
  "src/hpc3/contracts/workspace.py": "ba27d634de77b9ca57d6ccec1dd7815bc95da0fc"
  "src/hpc3/core/inflight.py": "96492a481dad7ead1916b61056585103e87443c5"
  "README.md": "477a86c18ec7d9e5ce711eb378ec1f6567c36115"
provenance:
  - "board task 747c391e (MCPs board): ledger line counts per checkout and the duplicate array 57783755_[0-12], measured 2026-10-05T10:35Z"
fact_checked: 2026-10-16
confidence: high
---

# One ledger per repository, whichever worktree submits

The workspace names its ledger relative to its own directory, `runs/`, and
`runs/*` is gitignored. Until 2026-10-16 that made the ledger **per
checkout**: each `git worktree` of the API repository kept its own file.
Measured on 2026-10-05: the main checkout's ledger held 16030 lines and no
v56 entry, `API-research-w3` held the 20 rows for arrays
57783112/57783131/57783373, and `API-research-w2` held none.

A worker resuming the v56 cartridge-QA campaign in w2 ran `hpc3-campaign`.
The in-flight check (`claimed_artifacts` in `src/hpc3/core/inflight.py`)
reaches a live job's artifact only through that job's ledger row, because
`squeue` has never heard of an artifact. With no rows, it reported
`0 in flight` while 12 members were running or pending, and submitted the
full duplicate array 57783755_[0-12]. It was cancelled before any member
started; the w3 rows were merged into w2's and the main checkout's ledgers by
hand. `hpc3-trace` had the same blind spot.

## The ledger is the main checkout's

Every command loads its workspace through `src/hpc3/cli/_config.py`, which
resolves a relative `ledger` against the document's directory **as it sits
in the repository's main checkout** (`shared_directory` in
`src/hpc3/core/checkout.py`). Each project's `repo` still resolves against
the document's own directory: the code a run names is the worktree's, and
only the record of what is on the cluster is shared.

The main checkout is found from git's own layout, read from the files rather
than by running git:

- a `.git` **directory** marks a main checkout, which is its own answer;
- a `.git` **file** carries `gitdir: <path>`; when that git directory holds
  a `commondir` file it is a linked worktree, and the main checkout is the
  parent of the common directory;
- a `.git` file whose git directory has no `commondir` is a submodule, which
  shares nothing and is its own answer;
- a worktree of a **bare** repository has no main working tree, and is
  refused with `LEDGER_CHECKOUT_UNREADABLE`, as is a `.git` file that is not
  a pointer.

`tests/test_checkout.py` builds each of those with real git, and
`tests/test_campaign_shared_ledger.py` replays the 2026-10-05 campaign from a
linked worktree: the members another checkout recorded now read as in flight,
and what the worktree submits lands in the main checkout's ledger.

## A second machine still has a second ledger

Sharing through the main checkout reaches every worktree on one machine and
no further. So both submitting paths, `submit` and `submit_array`, also ask
the cluster: before any upload, `check_no_unrecorded_live_job` refuses with
`LIVE_JOB_UNRECORDED` when a live job carries the name the submission would
use (the qualified name of a single job, the sweep's name for an array, or a
selected member's) and the ledger has no row for one of its tasks. It refuses
the whole submission rather than one member, because an unrecorded array's
task index names a member only under the member table of the document that
submitted it. The message names each job, its state and the ledger that was
read; the remedy is to run from where it was submitted, copy its rows across,
or cancel it.
