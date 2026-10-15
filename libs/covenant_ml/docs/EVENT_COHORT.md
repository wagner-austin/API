# Event inventory and next evaluation gate

Board task e126ab59-190b-45b5-81fa-d0de8c390795.

## Acquisition contract

`scripts.event_cohort` enumerates both native market tiers for each declared
series. The current tier receives a widened close-time query with no status
filter; the historical tier is paginated in full because its documented
interface has no close-time filter. Both undergo the same local half-open
close-time selection. There is no execution, price, volume or winning-side
filter. Empty families, zero-volume contracts, pending results and scalar
payouts remain explicit. Native documents are preserved rather than reduced
to a weather-specific schema.

Repeated identities within a tier, mismatched family/event identities and
conflicting cross-tier snapshots fail. Identical cross-tier overlap counts
once. The sole explicitly reconciled mutable field is `open_interest_fp`:
both native values are retained as a difference record, validated as
nonnegative fixed-point strings, and never interpreted as historical features.
The current-tier row remains the representative native snapshot. Every other
field must agree exactly. An HTTP or schema failure produces no completed report. Capture,
explicit checkpoint continuation and offline replay reuse ResponseArchive;
all inputs, implementation bytes and the result are bound to the shared
RunRecord. COHORT-001 through COHORT-005 distinguish plan, identity,
conflicting snapshot, temporal and future-window failures.

Endpoint-complete means every row returned by every page was processed.
It does not mean delisted markets absent from both endpoints were recovered,
or that current metadata was available in the past. Inventory close times
organize outcomes only. In particular, early-close sports timestamps cannot
define a retrospective entry relative to the final whistle.

## Repeatable commands

Run from `libs/covenant_ml`, using a fixed plan with `series`, `start_ts` and
`end_ts`. The end is exclusive and must precede acquisition.

```powershell
.venv/Scripts/python.exe -m scripts.event_cohort --plan <plan.json> --archive <raw> --out <report.json> --mode capture
.venv/Scripts/python.exe -m scripts.event_cohort --plan <plan.json> --archive <raw> --out <replay.json> --mode replay
```

`--mode resume` explicitly consumes already archived responses and acquires
only missing requests after an interrupted capture. It is never an automatic
retry. Use a new output filename; reports and sidecars cannot be overwritten.

## Future evaluation protocol: admission requirements

The next predictive comparison is market probability versus a simple model
on exactly the same event opportunities. The July 1–October 14 crypto and
football inventory is exploratory development data, including already
inspected tape dates. It is not a holdout, and no result from it may be
advertised as prospective predictive skill.

For a crypto pilot, a candidate decision clock is a fixed offset after the
scheduled interval opens, recorded before outcomes. For sports, use a
preannounced kickoff schedule with its receipt time, never eventual close
time. Neither clock becomes an active study rule until its source availability
and actual capture path pass deployment checks. Adjacent crypto intervals,
both assets in the same interval, and both teams in one game share dependence;
contract count is not independent sample size.

Before the first scored future observation, seal all of the following together:

- Exact eligible families, dates and opportunities, with no-volume and
  missing-input exclusion counts retained; no replacement of missing dates.
- Input receipt cutoffs, model feature definitions, earlier-only tuning and
  calibration, fitted coefficients, and the hashes of their training inputs.
- An unchanged market probability, a constant base-rate benchmark and one
  logistic challenger on identical rows. Add a complex challenger only as a
  separately declared comparison. Market midpoint is a scoring benchmark,
  never an executable entry price.
- One primary paired probability-score improvement and a prespecified effect
  size. Estimate day/block dependence on development data and set the full
  sample before inspecting future outcomes. Count all repeated comparisons;
  do not stop on a favorable interim result.
- Separate trading utility using contemporaneous ask for a yes purchase or
  one minus bid for no, captured sizes, current series/event fees and explicit
  rounding, receipt latency and a fixed sizing rule. Missing execution evidence
  prevents a deployable-profit claim even if probability scores improve.
- Tests that change future outcomes and features while preserving earlier
  predictions, plus real capture/replay, clock-boundary and omitted-event
  failures. Hash-verified inputs alone do not establish absence of leakage.

This is an admission protocol, not a sealed fitted forecasting study. The
future study has not started merely because these requirements are written.
No model is fitted by the inventory command, no capital is required, and
there is no production order interface.

## October 15 acquisition and verification

The fixed plan selected July 1 inclusive through October 15 exclusive, 2026
UTC (106 days), before full acquisition. The four families were chosen after
the prior tape and quote exploration; one recent native record per family
was also inspected. This selection is disclosed, not described as blind.

| Family | Contracts | Event IDs | Zero volume | Scalar payout |
|---|---:|---:|---:|---:|
| KXBTC15M | 10,048 | 10,048 | 6 | 0 |
| KXETH15M | 10,048 | 10,048 | 6 | 0 |
| KXNCAAFGAME | 1,630 | 815 | 50 | 54 |
| KXNFLGAME | 256 | 128 | 0 | 6 |

There are 21,982 contracts, 21,039 event IDs, 62 zero-volume contracts and
60 scalar outcomes. No selected contract has an empty declared result.
Event IDs are not a count of independent observations. The football data
include preseason and scalar outcomes; binary-only filtering would change
the sample and must remain explicit in any later evaluation.

Each crypto series has 10,048 of 10,176 nominal quarter-hour close slots,
with 128 absent slots. Every observed crypto market spans exactly 900
seconds and lies on the quarter-hour grid. Missing slots are recorded in
`verification.json`; their cause is not inferred. There is no forward-fill
or fabricated contract. Sports inventory is complete only relative to the
two market endpoints, not an independently verified league schedule.

The first capture stopped on unequal open interest. Existing immutable
responses were inspected: 567 overlapping records differed only in that
field. After adding explicit reconciliation and tests, `--mode resume`
completed acquisition. Final differences: Bitcoin 567, Ether 567, NFL 28,
college football zero. All other overlapping fields agree. No failed
attempt emitted a completed result, and no response was overwritten.

Evidence root:
`C:/Users/Test/PROJECTS/files/market-edge-20261012/event-cohort-20261015`.
The 68 native envelopes were received between 20:38:13.926724 and
20:40:45.095367 UTC on October 15. The report binds 71 source paths including
its sidecar and plan. The full independent recount uses stdlib datetime
selection and Decimal volumes over every native page, checks the exact
contract identity sets, and agrees with the production totals.

`results.json` and `replay.json` are byte-identical, SHA-256
`b1d92dbebaef11f2ab95dc3ba75281eab1c7ab5f36aff2de2c4638e83b6c7cd8`.
The report is 66,834,856 bytes. `verification.json` contains the independent
counts, missing intervals, receipt bounds and both report digests. The raw
responses remain required for full offline replay; a summary alone is not
a portable copy of the archive. Prior quotes, predictor availability and
execution evidence remain to be attached before scoring a forecasting study.

## Executable verification of the published datasets

`tests/markets/test_native_cohort_reproduction.py` runs during `make check`.
Its committed `fixtures/kalshi-native-20261015.zip` is the unchanged 637-file
portable archive, SHA-256
`f6eeba269313689cc4bf5ea5cb7f8e0147f9ff30f43b286da6f69ab841dbe622`.
It needs no workstation evidence paths, credentials or network requests.

The test verifies the bundle and original report digests, extracts the archive
into a fresh directory, and invokes the real production collectors in replay
mode. Every reconstructed native field must equal the original report; only
the separately recorded persistence metadata is excluded from this comparison.
It recomputes cohort totals and crypto calendar omissions with stdlib datetime
and Decimal, then reconstructs both quote panels, checks their missing slots
and quote ages, and compares every overlapping quote record. Finally it corrupts
and deletes an actual native response, requiring both failures to propagate.

This replaces text-only audit checks that merely found a claimed count or a
`byte_identical_replay` flag. The test never reads that flag to decide success.
It verifies these historical measurements, not independent predictive skill.
