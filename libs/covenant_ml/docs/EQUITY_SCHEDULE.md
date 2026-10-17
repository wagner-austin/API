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

## Moved to Linux on 2026-10-17

austinpc, the Windows host above, is reimaged to Linux (MCPs board task
`e2ffed0d`), so the capture now runs on diphtheria. The frozen snapshot records
each of its 85 sources by an absolute Windows path, and verification re-hashes
them there. So it runs from a new runtime that reads moved evidence through a
relocation declaration (`markets/evidence_location.py`; see
[MARKET_RESEARCH.md](MARKET_RESEARCH.md)).

- **Evidence:** `stock-panel-20261014/` was copied byte for byte to
  `/home/corvis/PROJECTS/files/stock-panel-20261014/`. The snapshot's one source
  outside it, `EQUITY_PILOT_PROTOCOL.md` in the research worktree (SHA-256
  `1c3204c2b4120d9e573a180ee0b8b8946e84292067f80252db2c3a27c2f99530`), was copied
  to `/home/corvis/PROJECTS/files/relocated-evidence/API-gpt6-markets-1012/libs/covenant_ml/docs/`.
  `~/.config/covenant_ml/evidence-relocations.json` on diphtheria declares both
  moves.
- **Runtime:** `equity-runtime-linux-20261017/`, the six local packages exported
  from API commit `57fcc042fb45117db9659cebcfb7206982b82e25` with their tests
  left out, and a fresh `poetry sync --only main` on Python 3.11. Its implementation
  digest is `3d41f687a56bd3cf840649143bd14beaa117e06f618baa73aba07f8754a6f80f`. It
  differs from `0086bf80…` above because the verifier now locates moved evidence.
  The calendar passed is still the original runtime's copied
  `equity_pilot_calendar_20261014.json`.
- **Diagnostic:** the production command, run 2026-10-17 23:28 UTC with
  diagnostic destinations (`linux-diagnostic-20261017.json`), exited 0. It
  verified all 89 sources through the declaration and left the future window
  pending, with no returns, no scores and zero orders.
- **Schedule:** the systemd user timer `covenant-equity-outcome-20261113.timer` on
  diphtheria fires once at **2026-11-13 00:05 UTC**. With `Persistent=true` it
  starts at the next boot if diphtheria was down then. It has no boot trigger
  that could write a premature result. The action writes the same destinations
  as above under the Linux root: `final-capture-20261113/` and
  `final-outcomes-20261113.json`. The Windows task `API-EquityPaper-Outcome-20261113`
  was unregistered the same day, so the capture cannot run twice. Remove the
  Linux schedule with
  `systemctl --user disable --now covenant-equity-outcome-20261113.timer`.

## Checks and interpretation

Full `make check` passed **3,758 tests**, strict typing and all guards, with
**100% of 17,661 statements and 3,588 branches** in 119/300 seconds. The
implementation-mismatch test also runs a real CLI subprocess, verifies its
exact error, and proves no archive, result or HTTP request was created.
The existing real HTTP success, 404 and offline replay tests remain intact.

This deployment removes the need for a manual prompt to collect the declared
stock outcome. It does not establish profitability, turn one portfolio window
into independent samples, authorize money, or run an AI session indefinitely.
