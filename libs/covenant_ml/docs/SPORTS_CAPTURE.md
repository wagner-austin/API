# Repeated cross-venue sports evidence

`scripts.sports_capture` captures a declared set of full-game NHL winner contracts
from Kalshi and Polymarket US, then reproduces the report offline from the same
hashed response archive. It uses the existing HTTP client, dependency hooks,
book record, fixed-point validators, implementation fingerprint and RunRecord
writer. It cannot submit orders.

## Frozen selection

The selection file is an object with a nonempty `pairs` array. Each pair has:

| Field | Meaning |
| --- | --- |
| `kalshi_ticker` | Full KXNHLGAME contract ticker for the selected team |
| `us_slug` | Exact aec-nhl full-game winner market slug |
| `kalshi_team` | Native Kalshi hockey team identity |
| `us_team` | Native integer teamId on the US market's long side |
| `rules_sha256` | SHA-256 of UTF-8 `dump_json_str(sports_rules(kalshi, us))` |

The caller declares that these are the same team in the same game, after reading
both venues' metadata. The collector verifies the pinned identities and side
orientation; it does not infer cross-venue team equivalence from abbreviations.
`sports_rules` extracts Kalshi's primary and secondary rules and the US market's
description. Freeze these before collection. Any subsequent text change fails
the run instead of silently replacing the selection.

Example acquisition, from this package directory:

```powershell
poetry run python -m scripts.sports_capture --selection selection.json --archive sports-run --out capture.json --kalshi-url https://external-api.kalshi.com/trade-api/v2 --us-url https://gateway.polymarket.us --window-start 2026-10-14T00:00:00Z --window-end 2026-10-14T02:00:00Z --mode capture --rounds 3 --interval-seconds 2 --max-skew-ms 3000
```

For offline replay, use the exact arguments with `--mode replay` and a new
`--out` path. The time window refers to original receipt times, not replay time.
Choose fresh selection/window/archive/output paths for a new acquisition.
Existing request envelopes and report outputs are never overwritten. There is
no automatic retry or capture-resume mode. Partial failed-run evidence stays
available for diagnosis; a failed request never becomes a missing-liquidity row.

Each round gets independent archive directories and re-reads both contract
metadata and both books. Receipts must be nondecreasing across all pairs and
rounds and stay inside the declared window. Missing book sides and excessive
book receipt separation are explicit exclusions with no directional estimates.
Closed or mismatched native US books fail. Receipt separation is not an assertion
of simultaneous exchange state or network latency.

## Interpretation

Prices and gross costs are integer ten-thousandths of a dollar. Visible size
is in centi-contracts. US sizes are decoded exactly from four-decimal strings;
sub-centi-contract sizes are refused rather than rounded. Every level is checked
before choosing the best price, including duplicate prices and crossed books.

For each direction the report models buying YES on one venue and the opposing
position on the other. It records the gross pair price and the ordinary-completion
bound `10000 - pair_price`, plus top-level capacity. Size checks for 1, 5 and
20 contracts show gross cost, sufficient displayed depth and whether that cost
fits $20. These are not fee-inclusive budget or execution approvals.

The native US fee coefficient is retained as evidence. `fees_verified` and
`settlement_equivalence_verified` are explicitly false: this workflow does not
implement complete fee schedules, cancellation equivalence, queue position,
atomic execution, realized returns or a predictive model. In particular,
discretionary fair-price settlements can break a nominal opposing-position
hedge. Positive gross rows would be candidates for further verification, not
permission to trade. Gross-nonpositive rows do not require a fee model to reject
as immediate taker arbitrage under ordinary completion.

## Production smoke measurement

On October 13 Pacific time, the workflow captured Pittsburgh-Chicago and
Minnesota-Winnipeg in three rounds: 24 actual public HTTP responses, six paired
snapshots and twelve directional comparisons. Receipt separation was 358-373 ms.
No direction had a positive gross bound. The report replayed byte for byte;
both files have SHA-256
`b90435ebe212efd10ebc56c8a583d877a431d7a4c14a76609b79e559f7949767`.
Evidence is under
`C:/Users/Test/PROJECTS/files/market-edge-20261013-cross-market/`:
`sports-selection.json`, `sports-run-1/`, `sports-live-1.json`,
`sports-replay-1.json`, and their RunRecords.

A second run captured the same pairs in 12 rounds: 96 responses, 24 paired
snapshots and 48 directional comparisons. All passed the 3,000 ms receipt-skew
limit; observed separation was 355-435 ms. Gross bounds ranged from -$0.03 to
$0.00 per pair, so no direction was positive before fees. Its capture and replay
reports (`sports-live-2.json` and `sports-replay-2.json`, archive `sports-run-2/`)
share SHA-256
`77c04c7d676229e13b7d9569db5a860b2047595d41eeb4f54cf052e4d9f00ef5`.

This short smoke measurement validates acquisition and replay, not a durable
absence of opportunity. It does not establish which venue forecasts better.
The [cross-market investigation](CROSS_MARKET_INVESTIGATION.md) describes that
separate prospective question and the current evidence limits.

Tests exercise real HTTP capture, real server failure, exact offline replay,
subprocess success and corruption failure, changed rules, incorrect identities,
missing depth, malformed levels, clock/window checks and repeated acquisition.
The package's existing `make check` covers source, scripts and tests; there is
no parallel harness or network dependency in the automated suite.
