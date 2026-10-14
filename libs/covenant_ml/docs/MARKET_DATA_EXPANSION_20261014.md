# Market data expansion inventory, October 14, 2026

Board task 46fc5d55-c6f6-41db-b8ab-0474f4ca009c. This inventory separates
measured data coverage, strategy results and proposed acquisition. It is not
a claim that every possible market has been investigated.

| Domain | Evidence already measured | Next useful expansion / unresolved input |
|---|---|---|
| Weather | 1,002 saved eligible forecast rows; 359 later rows on 20 dates; fixed and nested boosted comparisons both fail | More seasons and contemporaneously received forecasts; frozen 60-date campaign already collecting |
| Sports | NHL 233 training / 128 later games; rating blend loses $3.16; second-venue study has 94 eligible later games and zero trades | Other seasons/leagues, native team identity, timestamped lineups/goalies/injuries, exact-cutoff quotes and depth |
| Inflation | 41 eligible contracts across 14 releases, 12 hypothetical trades, +$0.96 but worse probability scores and uncertain profit | Other release families, vintage predictors and denser pre-release quotes; count releases rather than thresholds |
| Equities | Forty disclosure responses for twenty current issuers; eighteen complete inputs; frozen prospective cohort | Historically valid security membership, delistings and corporate actions before broad retrospective claims |
| Crypto spot | New BTC-USD and ETH-USD daily capture: 1,017 validated bars each, January 1, 2024 through October 13, 2026 | Chronological return tests, next-bar execution, spread/fee sensitivity; two surviving pairs do not represent all crypto |
| Prediction-market consistency | Prior 3,489 overlapping same/cross-venue comparisons found no positive gross bound | Repeated synchronized books and settlement-equivalent legs across more categories |
| Options/futures/FX | No validated local historical strategy in this inventory | Contract chains, expiry/roll treatment, margins, point-in-time bid/ask and instrument-specific costs |
| Politics/entertainment/esports | Market-series inventory / platform capability only, not an evaluated strategy | Event resolution identities, point-in-time information and sufficient distinct events; no profit claim |

Earlier source records: [cross-market investigation](CROSS_MARKET_INVESTIGATION.md),
[sports ratings](SPORTS_RATINGS_RESULT.md), [cross-venue sports](SPORTS_CROSS_VENUE_RESULT.md),
[inflation](INFLATION_EXPERIMENT_RESULT.md), [issuer coverage](EQUITY_PANEL_COVERAGE.md),
[equity pilot](EQUITY_PILOT_PROTOCOL.md), [forecast selection](FORECAST_SELECTION_RESULTS.md).

## New crypto coverage measurement

Before acquisition, `files/crypto-history-20261014/plan.json` fixed BTC-USD and
ETH-USD, daily granularity and [2024-01-01, 2026-10-14). Ten public Coinbase
Exchange candle requests used at most 250 days each, no authentication and no
orders. Original decoded response text and per-request URL, start/receipt time,
status and SHA-256 are retained. No response was replaced after acquisition.

Both products contain every one of the 1,017 expected UTC daily timestamps,
without duplicates after filtering each request to its declared half-open
window. Each page also returned its end-boundary candle: five outside-window
rows per product were excluded explicitly. All in-window records passed six-
field shape, midnight alignment, positive ordered OHLC and nonnegative-volume
checks; every response hash matched its receipt. `coverage.json` records these
counts. No strategy was fitted or scored on this new data in this measurement.

The [official Coinbase candle specification](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles)
describes bucket-start timestamps, OHLCV fields, a maximum of 300 candles per
request and potentially incomplete history. Daily bars provide price history,
not historical executable order books. Predictors must stop at completed
bars, and a later execution convention must be declared before scoring.

The immediate new nonweather development input is this crypto archive. The
existing sports and inflation sources remain usable, with their missing-quote
counts preserved. Increasing trial count on their already inspected outcomes
does not create independent evidence. Frozen prospective equity and weather
campaigns continue unchanged; this inventory does not extend their hypotheses.
