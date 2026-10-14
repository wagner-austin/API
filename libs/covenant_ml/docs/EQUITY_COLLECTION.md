# Stock outcome acquisition and offline replay

Board task: 133829ef-ba06-4265-a714-1e4756ec9b96.

`scripts.equity_outcomes` now joins the verified frozen stock snapshot to
actual public daily price acquisition, exact-calendar evaluation and a
source-bound report. It places no orders. The command requires the complete
original snapshot and RunRecord, not the shortened documentation projection.

`load_equity_study` checks the snapshot and every source hash, recomputes its
ordinal ranking, verifies selected membership and equal portfolio/benchmark
weights, and requires computation before the decision. The separate calendar
pins the original snapshot SHA256 and every expected session. Its endpoints
and length must match the frozen holding period. The calendar is declared
independently of prices; missing observed dates cannot redefine it.

`equity_pilot_calendar_20261014.json` lists the 21 ordinary weekdays from
October 14 through November 11, matching the previously frozen protocol.
This acquisition implementation does not change the decision, entry, exit,
members, weights, costs or signal values.

## Acquisition contract

Capture requires a new directory. Before UTC midnight two days after the
exit date, it seals an explicit pending observation without price requests.
For this pilot, acquisition becomes eligible on **2026-11-13T00:00:00Z**.
This conservative receipt threshold gives every final US session bar more
than the native decoder's required 24 hours. It changes collection timing,
not the November 11 paper exit.

Once eligible, capture requests every cohort member from the same declared
date interval. Each native response is hashed and records its actual receipt
time. The observation start, all receipts and finish must not regress.
All identities, USD currencies, completed bars and exact dates must validate
before a complete result is sealed. A failed or missing unselected benchmark
member fails the entire result. Successful partial responses stay on disk;
there is no retry, replacement source, silent resume or completed report.

Replay verifies the study identity and sealed observation, rechecks every
required archived response and recomputes the complete result without HTTP.
Changed evidence or arithmetic fails. The returned report includes every
constituent return and both declared cost cases. The implementation digest,
original evidence paths, calendar and receipt archives are included in the
outer report and RunRecord. `advance_to_funded_trading` stays false even for
a positive paper outcome.

## Actual execution

Original invocation from `libs/covenant_ml`, before the deployment digest
requirement was added. Current commands must also pass the verified
`--implementation-sha256` described in `EQUITY_SCHEDULE.md`:

```powershell
poetry run python -m scripts.equity_outcomes --snapshot C:/Users/Test/PROJECTS/files/stock-panel-20261014/pilot-first.json --calendar docs/equity_pilot_calendar_20261014.json --archive C:/Users/Test/PROJECTS/files/stock-panel-20261014/cli-pending-20261014 --out C:/Users/Test/PROJECTS/files/stock-panel-20261014/cli-pending-first.json --mode capture
```

This command already ran; capture refuses to overwrite it. Its actual start
and finish were `2026-10-14T06:29:27.121298+00:00`. All 18 cohort members and
six selections validated. The result correctly has `complete: false`, empty
scores and returns, and zero orders. It verified 91 evidence paths. A second
process replayed the same archive to `cli-pending-replay.json`; report and
RunRecord bytes matched independently. Report SHA256:

`3d3918f27da59357ebf5e82bf8d25bf8acf937f575d50e67e84ee77ea80691e1`

For offline reproduction, change `--mode` to `replay`, retain the archive,
and choose a new `--out` filename. Existing output or sidecar paths fail
before acquisition. This command by itself does not register a schedule.

## Verification and limits

Full `make check` passed **3,757 tests**, strict mypy over source, tests and
scripts, all guards, and **100% of 17,656 statements and 3,586 branches**,
in 124 seconds against the 300-second budget. The new workflow adds 38 tests.

Its mature integration tests serve the actual archived AAPL and MSFT daily
price responses through a real local HTTP server, run the production client
and CLI, verify source-bound results, and replay them in another process.
The two-stock ranking and historical decision times in those tests are
controlled plumbing inputs, not a historical prediction or strategy result.
Tests also prove actual HTTP 404 propagation, receipt-clock failure, altered
seals, missing benchmark members, acquisition deadline boundaries and late
arithmetic initialization failure. No mocked HTTP transport replaces these
paths. The actual prospective run above does not pretend to have future
prices. Successful engineering checks establish workflow behavior, not alpha.
