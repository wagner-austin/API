# Frozen stock collector deployment

Board task: 6b9fb4df-5b17-4fd9-ba95-5c8f695032c2.
Installed and verified on 2026-10-14 UTC. No orders or deposits are involved.

## Installed runtime and implementation identity

The standalone runtime is:

`C:/Users/Test/PROJECTS/files/stock-panel-20261014/equity-runtime-20261014`

It contains copied source, scripts, manifests, locks and documentation for
the six local dependency packages. All **802 copied files** were individually
SHA256-verified against the source worktree; `source-receipts.json` records
their relative paths and hashes. A fresh virtual environment was installed
with `poetry sync --only main`, including the real compiled Rust dependency.
Both `covenant_ml` and `platform_core` resolve inside this runtime. The task
uses its own Python executable directly, not the editable research worktree.

The runtime and checked worktree produce the same implementation digest:

`0086bf80d55d5f08c01dba7a0397db7ca69cfe3f611004ceeaba489c7608ebbd`

`scripts.equity_outcomes` now requires `--implementation-sha256`. A mismatch
raises `EQUITY-021` before reading the study or constructing a network client.
The digest covers package implementation, shared core, all CLI scripts,
pyproject and lock bytes. The verified digest is also recorded in the result.
This catches changes to those bytes; it is not a signature or protection
against someone deliberately replacing both code and the declared digest.

## Real scheduled execution

A temporary task ran the production command through Windows Task Scheduler
under the `Test` S4U account with limited privileges. It started at
`2026-10-14T06:38:45Z` and exited **0**; the actual observation was recorded at
`2026-10-14T06:38:46.552849+00:00`.

That run used the complete original frozen snapshot, the runtime's copied
calendar and the required implementation digest. It produced
`scheduled-diagnostic.json` and its RunRecord with 91 verified evidence paths.
The future window correctly remained pending, with no HTTP price requests,
no scores and zero orders. A separate runtime process replayed the archive
to `scheduled-diagnostic-replay.json`; both report and RunRecord bytes match.
Diagnostic report SHA256:

`ad1c49aa298ab1de39286e8154729155a1b6cd0952b41d5dc101538c7195eed4`

The diagnostic task's XML and execution metadata were saved before removing
that temporary task. Its absence was then verified. It cannot accidentally
run again and consume the future observation's output path.

## Future collection

Task **`API-EquityPaper-Outcome-20261113`** is registered and reads `Ready`.
Its verified trigger is **2026-11-13T00:05:00Z**, which is **November 12 at
4:05 p.m. Pacific**. This follows the November 11 paper exit and the collector's
conservative completed-bar receipt threshold. The future action differs
from the successful diagnostic only in its archive and output destinations:

- `final-capture-20261113/`
- `final-outcomes-20261113.json` and its RunRecord

These reside under `C:/Users/Test/PROJECTS/files/stock-panel-20261014`.
The action's executable, arguments, working directory and next-run UTC time
were read back and checked after registration. `scheduled-outcome-task.xml`
and `deployment-receipt.json` preserve that installed state.

The task runs without an interactive login, requests wake from sleep, allows
battery operation and starts after a missed trigger when Windows becomes
available. It has a 30-minute execution limit and ignores overlapping
instances. There is no startup trigger that would write a premature pending
result to the final destination, and no retry after an acquisition failure.

The computer must eventually be available, with network access and the source
evidence intact. The original snapshot references its original evidence paths,
including the frozen protocol in the research worktree; those must be retained.
A missing file, changed hash, failed response or incomplete company history
propagates as a nonzero task result. Partial response archives do not become
a complete portfolio result. Task registration does not guarantee future
provider availability or prove that the future collection has already run.

## Checks and interpretation

Full `make check` passed **3,758 tests**, strict typing and all guards, with
**100% of 17,661 statements and 3,588 branches** in 119/300 seconds. The
implementation-mismatch test also runs a real CLI subprocess, verifies its
exact error, and proves no archive, result or HTTP request was created.
The existing real HTTP success, 404 and offline replay tests remain intact.

This deployment removes the need for a manual prompt to collect the declared
stock outcome. It does not establish profitability, turn one portfolio window
into independent samples, authorize money, or run an AI session indefinitely.
