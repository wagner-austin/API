# Inflation nowcasts: positive sample profit, insufficient evidence

Task `d673be48-bc8b-4b85-a086-280b870eec21`, October 14, 2026 UTC.
The [declared protocol](INFLATION_EXPERIMENT_PROTOCOL.md) produced 12 hypothetical
one-contract trades across 14 eligible release dates. Total profit was **$0.96**
after the assumed taker fees and one-cent adverse price movement, or **$0.78**
with doubled fees. No orders were submitted.

**This candidate does not advance.** The probability model scored worse than
the market, quote coverage was sparse, the sample-size gates failed, and the
95% mean-profit interval includes losses. A positive historical total does not
establish a repeatable edge. This result neither supports real-money funding
nor proves that inflation markets cannot be traded profitably.

## Complete accounting

| Stage | Count |
|---|---:|
| Unique contracts in the two captured inventories | 713 |
| Outside January 2024–August 2026 target range | 314 |
| Missing October 2025 actual release | 8 |
| September 2025 release and exchange close date mismatch | 8 |
| Selected contracts / release dates before quote filtering | 383 / 30 |
| Missing exact-cutoff candles | 319 |
| Noninterior or missing quote sides | 17 |
| Spreads exceeding five cents | 6 |
| Jointly eligible contracts / release dates | 41 / 14 |
| Selected hypothetical trades / abstentions | 12 / 2 |
| Winning / losing hypothetical trades | 4 / 8 |

The 383 requests exhaust the declared selected set, including the current-data
partition. They do not prove complete historical exchange coverage. Inference
uses fourteen release dates, not forty-one independent threshold contracts.
Eligible release dates run from July 11, 2024 through September 11, 2026.

## Forecasts and hypothetical payoffs

| Measure | Result |
|---|---:|
| Model Brier, averaged within release then across releases | 0.16840902983069067 |
| Same-subset market Brier, identical weighting | 0.1605654761904762 |
| Total hypothetical purchase cost including assumed fees | $3.04 |
| Total assumed fees | $0.18 |
| Primary hypothetical net profit | $0.96 |
| Doubled-fee hypothetical net profit | $0.78 |
| Mean profit per eligible release, including abstentions | $0.06857142857142857 |
| 95% Student-t interval for mean release profit | −$0.14352247741123436 to $0.2806653345540915 |
| Two-sided zero-mean p-value | 0.4971934254658871 |

The gates require 24 release dates, 20 trades, improved Brier, a positive lower
profit bound and positive doubled-fee profit. Only the last gate passed.
The $3.04 cost is across hypothetical purchases made on different dates; it is
not a demonstrated minimum funding requirement or an achievable return rate.

The largest winner was the November 2025 release contract, earning $0.91.
Removing that one winner as a descriptive concentration check leaves only
$0.05 primary profit. This is not a new fitted strategy or a revised result.
The small sample and exposure to individual releases dominate the conclusion.

## Evidence and acquisition correction

Evidence directory: `C:/Users/Test/PROJECTS/files/inflation-experiment-20261014`.
The 383 immutable candle envelopes preserve request URLs, receipt timestamps,
response bodies and hashes. Original forecast and inventory inputs remain in
`C:/Users/Test/PROJECTS/files/inflation-feasibility-20261014`.

`result.json` and `replay.json` were generated in separate offline processes and
match byte for byte, SHA-256:
`46c2cd79aebed4e9648ca653d324d53216596e068cd5f79e8fbbe38459d3cfdf`.
Each has a RunRecord and binds 387 input files: 383 candles, the forecast file,
both native inventories and the protocol. Final protocol SHA-256:
`62190df371c629a296216ba6d49c0b267e7f3ec684a0b9316515345e2de280e9`.

Before scoring, acquisition stopped on a historical-endpoint 404 for an
ordinary-only August 2026 contract. The recorded protocol correction selects
the endpoint from captured historical inventory membership. It never catches
an error and tries another source. The original failure is recorded in
`acquisition_failure.json`; completed successful captures were preserved and
verified on explicit resumption. The shared exact-cutoff validator was lifted
into `decision_quotes`; sports callers now use that implementation directly.

Historical forecast receipt times, advance release calendars, order-book depth
and fills remain unverified. Fees are an explicit assumption, not reconstructed
historical charges. The interval assumes independent months; serial dependence
may make it too narrow. Missing exact-midnight candles materially restrict the
population evaluated. No thresholds or calibration windows were changed after
the result was seen.

## Validation

The complete `make check` passed in 97 seconds: 3,608 tests, strict typing across
602 files, guards, and 100% coverage of 17,191 statements and 3,440 branches.
The real HTTP integration test covers both native quote partitions, exact
payoffs, offline replay, overwrite refusal and corrupt-archive failure.
Existing sports acquisition and comparison tests pass after the shared-code lift.

## Reproduce

From `libs/covenant_ml`, select a new output path:

```powershell
poetry run python -m scripts.inflation_experiment --history C:/Users/Test/PROJECTS/files/inflation-feasibility-20261014/nowcast_year.json --historical-inventory C:/Users/Test/PROJECTS/files/inflation-feasibility-20261014/kalshi-historical.json --current-inventory C:/Users/Test/PROJECTS/files/inflation-feasibility-20261014/kalshi-markets.json --archive C:/Users/Test/PROJECTS/files/inflation-experiment-20261014/candles --protocol docs/INFLATION_EXPERIMENT_PROTOCOL.md --kalshi-url https://api.elections.kalshi.com/trade-api/v2 --mode replay --out C:/Users/Test/PROJECTS/files/inflation-experiment-20261014/reproduced.json
```

Further useful evidence is prospectively received forecasts and books at a
declared decision time, not a larger cash balance. A broader quote window or
different decision time would be a new experiment requiring a new protocol;
the current sparse sample cannot silently be replaced after seeing its result.
