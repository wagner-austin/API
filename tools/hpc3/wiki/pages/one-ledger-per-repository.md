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
  - "src/hpc3/core/submit.py"
  - "src/hpc3/core/array_submit.py"
  - "tests/test_checkout.py"
  - "tests/test_campaign_shared_ledger.py"
  - "README.md"
source_git_blobs:
  "src/hpc3/core/checkout.py": "e903c5d1c622d088d73dbc1bc018b41dd6077d6c"
  "src/hpc3/cli/_config.py": "5287aecbeca09b166e8932f5ea38b227fbd7e720"
  "src/hpc3/contracts/workspace.py": "ba27d634de77b9ca57d6ccec1dd7815bc95da0fc"
  "src/hpc3/core/inflight.py": "96492a481dad7ead1916b61056585103e87443c5"
  "src/hpc3/core/submit.py": "d2e159c723e4c762a56d8c8ed3c1faf4b36c3e27"
  "src/hpc3/core/array_submit.py": "5391cb2c3b8505f23ed25fc0b91fdff332c06729"
  "tests/test_checkout.py": "0e0aecf8997ffab178f84741756ef91c48c42495"
  "tests/test_campaign_shared_ledger.py": "7780e8bc30969ca1ceff2b84b3ef6406bc349ee5"
  "README.md": "c3962b94b036071c6e52e96babff021a11f1b5d8"
provenance:
  - "board task 747c391e (MCPs board): ledger line counts per checkout and the duplicate array 57783755_[0-12], measured 2026-10-05T10:35Z"
fact_checked: 2026-10-16
confidence: high
---

# One ledger per repository, whichever worktree submits

The workspace names its ledger relative to its own directory, `runs/`, and
`runs/*` is gitignored, so until 2026-10-16 the ledger was **per checkout**:
each `git worktree` of the API repository kept its own file. Measured on
2026-10-05: the main checkout's ledger held 16030 lines and no v56 entry,
`API-research-w3` held the 20 rows for arrays 57783112/57783131/57783373,
and `API-research-w2` held none.[^incident]

A worker resuming the v56 cartridge-QA campaign in w2 ran `hpc3-campaign`.
The in-flight check reaches a live job's artifact only through that job's
ledger row, because `squeue` has never heard of an artifact, so it reported
`0 in flight` while 12 members were running or pending and submitted the
full duplicate array 57783755_[0-12]. It was cancelled before any member
started, and the w3 rows were merged into w2's and the main checkout's
ledgers by hand.[^incident][^claimed]

## The ledger is the main checkout's

Every command loads its workspace through one loader, which resolves a
relative `ledger` against the document's directory **as it sits in the
repository's main checkout**. Each project's `repo` still resolves against
the document's own directory: the code a run names is the worktree's, and
only the record of what is on the cluster is shared.[^loader]

The main checkout is found from git's own layout, read from the files rather
than by running git. A `.git` directory marks a main checkout, its own
answer. A `.git` file carries `gitdir: <path>`, and when that git directory
holds a `commondir` file it is a linked worktree whose main checkout is the
common directory's parent. A `.git` file whose git directory has no
`commondir` is a submodule, which shares nothing and is its own answer. A
worktree of a bare repository has no main working tree and is refused with
`LEDGER_CHECKOUT_UNREADABLE`, as is a `.git` file that is not a
pointer.[^checkout]

Each of those layouts is built with real git in the suite, and the
2026-10-05 campaign is replayed from a linked worktree: the members another
checkout recorded now read as in flight, nothing is submitted, and what the
worktree does submit lands in the main checkout's ledger.[^tests]

## A second machine still has a second ledger

Sharing through the main checkout reaches every worktree on one machine and
no further. So both submitting paths also ask the cluster before any upload,
and refuse with `LIVE_JOB_UNRECORDED` when a live job carries the name the
submission would use (a single job's qualified name, an array's sweep name,
or a selected member's) and the ledger has no row for one of its tasks. The
whole submission is refused rather than one member, because an unrecorded
array's task index names a member only under the member table of the
document that submitted it. The message names each job, its state and the
ledger that was read.[^unrecorded]

[^incident]: Board task 747c391e's spec, MEASURED 2026-10-05T10:35Z (see `provenance`); README.md, the workspace table's `ledger` row.
[^claimed]: `src/hpc3/core/inflight.py` section `claimed_artifacts`, which maps a live job to its artifact only through the job's ledger row.
[^loader]: `src/hpc3/cli/_config.py` sections `load_workspace` and `load_workspace_connection`; `src/hpc3/contracts/workspace.py` section `decode_workspace`, whose `ledger_dir` and `config_dir` are the two directories.
[^checkout]: `src/hpc3/core/checkout.py` sections `shared_directory` and `_linked_main_root`.
[^tests]: `tests/test_checkout.py` and `tests/test_campaign_shared_ledger.py`.
[^unrecorded]: `src/hpc3/core/inflight.py` section `check_no_unrecorded_live_job`, called from `src/hpc3/core/submit.py` section `submit` and `src/hpc3/core/array_submit.py` section `submit_array`.
