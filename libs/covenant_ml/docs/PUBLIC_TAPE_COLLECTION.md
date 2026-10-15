# Kalshi public execution capture

`scripts.public_tape` collects the public trade feed without a strategy filter
or account credentials. It places no orders. A plan specifies inclusive
`start_ts` and exclusive `end_ts`, in UTC Unix seconds, at most one day apart.
The window must have ended before the command starts. A fixed recent hour can
contain many executions; use bounded windows and retain each original plan.

## Reproduce

From `libs/covenant_ml`, with the package environment installed:

```powershell
python -m scripts.public_tape --plan plan.json --archive raw --out results.json --mode capture
python -m scripts.public_tape --plan plan.json --archive raw --out replay.json --mode replay
```

The production public origin is the default. `--base-url` selects an explicit
origin, including the real loopback HTTP origin in integration tests. A plan
such as `{"start_ts":1792008000,"end_ts":1792011600}` selects October 14, 2026,
20:00 through 21:00 UTC. Replay requires the same URL and every captured page.
It performs no HTTP requests. Compare trade rows, metadata and coverage counts;
the report also records its actual implementation and arithmetic environment.

## Acquisition contract

The existing `ResponseArchive`, `MarketCollector.pages`, collector factory,
native trade decoder and RunRecord writer remain the shared implementation.
Native trade reconciliation is lifted from the maker collector and compares
every native field, including book side and submillisecond timestamps.
Duplicates within an endpoint or conflicting evidence across tiers fail.

The captured historical cutoff routes trades by trade time. Both market
metadata tiers are queried for every selected ticker in batches bounded by
6000 encoded query characters. This avoids both oversized URLs and thousands
of unnecessarily small requests when a window spans many contracts.
Missing, unexpected, duplicate or conflicting metadata fails explicitly.
Declared binary payouts must agree with results; unresolved and scalar
contracts remain present. Cursor pagination runs to termination, with cursor
cycles rejected. There is no automatic retry or silent partial result.

Requests include a one-second lower margin and the exact end instant. The
collector then enforces the half-open interval and counts excluded boundary
rows. This preserves trades in the final fractional second and handles an
exclusive API lower bound. Rows beyond the requested envelope fail validation.
Native time strings retain precision; report order is not a queue ordering.

Archives preserve exact decoded response text, request URLs, receipt times and
SHA256 hashes, not compressed HTTP wire bytes. Reports bind the plan, archive
files and implementation digest to a RunRecord. Existing output files cannot
be overwritten. After an interrupted capture, an operator may explicitly use
`--mode resume` with the same plan and archive: verified completed requests are
read from disk and missing requests are acquired. The report is emitted only
after the whole collection and metadata join succeed.

## What these observations establish

Trade count, contract count and event count are separate measurements. Event
IDs are not a claim of independence: contracts can share an outcome, and
different events or combinations can share underlying legs and market shocks.
Public executions omit untraded contracts and unexecuted opportunities.
Pagination completeness is relative to what the API exposed during capture;
it does not certify the absence of late publications or later corrections.

Metadata is retrieved after the trade window. Its results, quotes, rules and
status are capture-time observations, never historical model inputs. Native
book side and outcome side are retained separately. The dataset does not
establish trader identity, fees paid, an accessible historical quote and size,
our queue position, achievable fills or profitability. No hypothetical return
is computed. These omissions must be resolved before a trading evaluation.

## Validation

Pure native validators check malformed trade fields and invalid payouts.
Real socket tests exercise pagination, both trade tiers, cross-tier overlap,
exact subsecond boundaries, metadata batching, shared event IDs, unresolved
and scalar results, missing metadata and HTTP 503 failures.
The CLI is exercised through capture, explicit resume and child-process offline
replay; output refusal and changed-source verification are tested. The shared
archive tests additionally cover integrity failures and cursor cycles.
The package's existing `make check` enforces strict typing over source, tests
and scripts, guards, and 100 percent statement and branch coverage.

The first independent fleet check passed all 4281 tests and coverage but took
301 seconds against its 300-second budget. Its log is retained as
`files/public-tape-fleet-failure-20261015.log`. Native-field validation was
then separated from HTTP collection: unit tests exercise the same production
validators without paying network pacing delays for every malformed field.
Real transport and failure tests remain in place; neither the check budget
nor production pacing was relaxed.

Board task: 9617e2fe-dbde-43f8-af39-9f049cfef8a2.
