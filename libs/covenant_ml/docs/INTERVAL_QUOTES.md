# Crypto interval quote attachment

This stage measures price availability for every bitcoin and ether contract
in a fixed slice of the complete event cohort. It does not fit a model,
estimate executable profit or place an order.

## Fixed acquisition rule

The initial plan covers close times from October 7, 2026 00:00 UTC inclusive
through October 15 00:00 UTC exclusive. Both `KXBTC15M` and `KXETH15M` are
mandatory, including empty families. No contract is selected by outcome,
traded volume, spread or presence of a quote. These dates overlap earlier
exploratory tape inspection and are not an untouched forecasting holdout.

Every selected contract must have an opening aligned to the fifteen-minute
grid and a duration of exactly fifteen minutes. The hypothetical decision
is five minutes after its captured opening time. Minute candles must end
strictly after opening and at least one second before that decision. The
latest qualifying candle is retained; its age may not exceed 120 seconds
for the `two_sided` classification. These choices precede quote acquisition
and are not optimized against profit or accuracy.

No price is carried forward from an older candle to repair a missing side.
The statuses are `two_sided`, `locked`, `crossed`, `missing_side`, `stale`,
`no_prior_candle` and `omitted_by_endpoint`. A stale last candle takes
precedence over its price-side classification. A current response that
omits a requested contract remains a separate row with the omitted status.
Historical transport errors propagate; an error is never an empty panel.

## Native requests and capacity

The archived `/historical/cutoff` routes by each contract's settlement
timestamp. Equality belongs to the current tier. Historical requests use
`/historical/markets/{ticker}/candlesticks`, with its native `close` prices.
Current requests use `/markets/candlesticks` and `close_dollars` prices.
Both use one-minute intervals without synthetic continuity candles.

Current contracts are grouped by their opening into three-hour buckets.
At most 24 unique contracts fit one bucket across the two families.
The collector checks both the 100-contract cap and a conservative upper
bound below 10,000 minute candles before issuing a batch. Each market's
own decision time filters its returned candles; a later contract's query
end cannot admit future prices for an earlier contract. Duplicate response
identities, unexpected contracts and duplicate or non-minute timestamps
are explicit errors. There is no endpoint guessing or retry fallback.

## Reproduction

From `libs/covenant_ml`, using the locked environment:

```powershell
.venv/Scripts/python.exe -m scripts.interval_quotes `
  --cohort <event-cohort>/results.json `
  --plan <interval-quotes>/plan.json `
  --archive <interval-quotes>/raw `
  --mode capture --out <interval-quotes>/results.json
```

Use `--mode replay` and a fresh output name for an offline repeat. An
explicit `resume` reuses immutable captured requests after an interrupted
acquisition; it does not change existing envelopes. Report and sidecar
overwrites are refused. The output binds the cohort, all its original
inputs, the quote plan, quote response envelopes, exact implementation
digest and environment fingerprint through the shared RunRecord.

## Validation and inference limits

Tests run real HTTP capture, disk replay and a child-process command.
They change future candles, final outcomes and final volume and demand
identical quote attachments. Tests also exercise historical cutoff equality,
wrong identities, missing families, duplicate intervals, stale and null
quotes, missing historical endpoints, batch bounds and tampered provenance.
Full package guards, strict typing and coverage remain required.

These are retrospective exchange candle records. Their end timestamps are
not receipts proving when our process could have acted. Captured opening
metadata is not an archived historical listing announcement. Bid/ask closes
have no guaranteed contemporaneous depth, fill, fee or latency evidence.
Adjacent intervals and the two assets share market conditions. More rows
therefore do not automatically mean more independent observations. A later
forecasting comparison must declare its own time split, training-only
tuning, calibration, dependence structure, costs and frozen future sample.

## Measured eight-day acquisition, October 15

The fixed plan was recorded at 21:08:23 UTC before the first new response.
The capture ran from 21:09:12.086549 to 21:09:37.722911 UTC, preserving
66 envelopes: the cutoff and 65 current batch responses containing 22,890
native candles. The largest theoretical response bound was 4,104 candles.

| Family | Contracts | Two-sided prior quotes | Selected quote age |
|---|---:|---:|---:|
| Bitcoin fifteen-minute | 760 | 760 | 60 seconds each |
| Ether fifteen-minute | 760 | 760 | 60 seconds each |

There are 768 nominal close slots in the window per asset. Eight slots
from October 8 07:15 through 09:00 UTC are absent for both assets; no
contracts or prices are fabricated for them. Each asset has 96 contracts
per close date except October 8, which has 88. Thus the panel represents
760 shared intervals across eight close dates, with serial dependence
still unmeasured. No selected contract has zero final volume, but volume
was never a selection criterion.

Bitcoin spreads were $0.001 for 20 contracts, $0.002 for two, and $0.01
for 738. Ether spreads ranged from $0.001 to $0.03, with $0.01 for 668
contracts and $0.02 for 47. These are quote differences, not net returns.

Evidence root: `files/market-edge-20261012/interval-quotes-20261015`.
The report and its offline replay are byte-identical, SHA-256
`200044456a7b9b8b4a1ce52e8af4eeeb83657eed04dafc24d8dcb7b60a5eb358`.
The source verifier checked 140 paths. A separate calculation using
stdlib datetime and Decimal against every native envelope reproduced
all identities, selected prices, eligible counts and quote ages exactly;
its `verification.json` SHA-256 is
`fa53d763b455476c02c7f41e6f942f5f3f95f1fb290d7b81cdb0be49f0aaa0d1`.
The full local check passed 4,353 tests with 100% statement and branch
coverage in 133 seconds. No fitted model or profitability test is part
of this acquisition result.
