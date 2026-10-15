# Kalshi public tape acquisition, October 15, 2026

## Frozen window and measured coverage

The acquisition pilot selected every publicly exposed Kalshi execution from
2026-10-14T20:00:00Z inclusive through 21:00:00Z exclusive, without selecting
by strategy, price, result, category or block status. It submitted zero orders
and used no account credentials. This is one UTC date and one hour of activity,
not an independent performance holdout.

The completed report contains **895,294 trades across 96,478 contracts and
71,127 event IDs**. Of those contracts, 84,826 had a declared result when their
metadata was captured and 11,652 did not. No block trades occurred in the
selected records. Exact summed quantity is 185,986,966.03 contracts; that is
execution quantity, not cash turnover or independent observations.

There were 896 trade pages with one terminal cursor. The deliberately enlarged
query envelope returned 895,501 unique executions. Local half-open filtering
excluded 207 boundary-margin records. Selected timestamps run from
20:00:00.010863Z to 20:59:59.988419Z. An independent native-record recount using
datetime comparisons and Decimal quantities matched the production counts.

## Dependence and availability

90,826 contract records expose nonempty multivariate selected-leg arrays,
which name 2,766 distinct underlying event IDs. Another 5,652 records have no
such array or a null field. These are observed identity counts, not independent
sample sizes. Different combination contracts can reuse the same outcomes;
even distinct underlying events can share time, asset or sports exposures.

Archive receipts range from 2026-10-15T11:19:53.159156Z through
11:39:49.295111Z across the complete archive, including the cutoff and trades.
The results and quote fields are information available at those later receipt
times, not historical model features. The snapshot does not recover historical
order-book depth, fees paid, trader identity, queue priority or achievable fills.
Untraded contracts are outside this execution-selected cohort. API pagination
completion does not prove that late publications or later corrections cannot
exist. No strategy P&L or predictive skill was measured.

## Evidence and reproduction

Artifact root: `C:/Users/Test/PROJECTS/files/market-edge-20261012/public-tape-20261015`.
The frozen `plan.json`, complete `raw-v2/` archive, `results.json`, associated
RunRecord, `replay.json`, `independent-trade-audit.json` and `verification.json`
are retained. `CAPTURE_NOTES.md` explains the interrupted diagnostic `raw/`
capture and additional metadata envelopes retained during explicit continuation.
The diagnostic capture is not the accepted dataset.

The complete archive has 2,766 checksum-verified response envelopes. The
source verification covers 2,769 paths including the plan, report and sidecar.
Offline replay reproduces every report byte except the explicitly recorded
implementation digest, which changed when pure validation was extracted from
HTTP collection. All trades, metadata, counts, fingerprints and source hashes
are identical. Both report payloads match their own RunRecords.

| Artifact | SHA256 |
| --- | --- |
| plan.json | 20dd49a479c9f785027bf16720d46fc15a416388b3177d1a97657727e14649c5 |
| results.json | 6ed0eb4d006b81b490da75edaa20913e3987820fd735e298ff0b19f07dade3bc |
| replay.json | 92f0ca6cf70a4a37cabf2f1fd78ae1be3b7181a774f1decb660e59a3950e4879 |
| independent-trade-audit.json | e6be95fe27b2c40de15b331d92ff2007035dd4a56a73641c2608ee03b20f8fbe |
| Reports excluding the implementation-digest line | 7762f8b9a8c5ec4d6640f431be509ce46dbc1ceec33dca4d1e46d53f50828cde |

Replay instructions and failure contracts are in `PUBLIC_TAPE_COLLECTION.md`.
The full local package check passed 4,281 tests, strict typing and guards, with
100 percent statement and branch coverage in 121 seconds. The initial fleet
run passed all tests and coverage but exceeded its 300-second budget by one
second; that failed run is retained. Pure validation now has direct unit tests
instead of repeating paced HTTP acquisition for each malformed field. Real
transport, transport failure and offline CLI tests remain intact.

## Next evidence requirement

Broader calendar coverage and predeclared contract eligibility are needed
before testing predictive performance. A later evaluation must group shared
underlying outcomes, retain untraded opportunities, use features demonstrably
available before entry, and compare market probabilities and simple forecasts
under identical costs and execution rules. The sample requirement should be
set around a detectable improvement and its clustered uncertainty, not a target
number of trade rows. This pilot supplies acquisition evidence for that work.

Board task: 9617e2fe-dbde-43f8-af39-9f049cfef8a2.
