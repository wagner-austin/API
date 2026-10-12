# Market execution research

`scripts.market_research` tests a frozen daily-weather maker hypothesis against
public Kalshi evidence. It uses the existing `covenant_ml.evaluation` estimators
and the shared `platform_core.run_record` provenance contract. It has no order
placement interface. Forecast accuracy in covenant-radar does not establish
market profitability; this experiment measures hypothetical execution payoffs.

## Run and reproduce

From `libs/covenant_ml`, after `poetry sync --with dev`:

```powershell
poetry run python -m scripts.market_research --plan plan.json --protocol protocol.json --archive raw --mode capture --out results.json --workers 3
poetry run python -m scripts.market_research --plan plan.json --protocol protocol.json --archive raw --mode replay --out reproduced.json --workers 3
```

`capture` refuses existing request files. `resume` explicitly continues a capture
from verified checkpoints, acquiring only absent requests. `replay` requires
every response on disk and issues no network requests. Corrupt checkpoints,
HTTP failures, unexpected schemas, and conflicting identities propagate errors.
There is no retry or substitution of empty data. All workers share request
pacing; an interrupted capture can be continued with `--mode resume` after its
underlying cause is resolved. Never overwrite an old experiment report.

The required plan fields are `name`, `series`, `timezone`, ISO `start` and `end`
event dates, `rule`, `friction`, and `protocol_digest` (SHA-256 of the predeclared
research protocol). The rule requires integer `min_ask`, `max_ask`, `improve`,
`through`, `contracts`, `max_quote_age_ms`, `latency_ms`, and `maker_rate_bps`.
Money uses 10,000 units per dollar; quantity uses 100 units per contract.
The replication evaluator requires the primary `through` to be 200, with
100 and 0 reported as predetermined sensitivities, never selected as winners.

Example rule: ask band 200–1600, improvement 100, primary through 200, one
contract, maximum candle age 3,600,000 ms, latency 1,000 ms, maker coefficient
zero, and friction 100. Placement is 16:00 in the specified IANA timezone on
the day before the event date. Source fee metadata must identify `quadratic`
with multiplier one. That check establishes the current schedule, not its
historical invariance.

The CLI verifies the supplied protocol file against its declared hash before
requesting data. The final report pins both input files, the raw response files,
and a digest over the Python implementation, shared core, CLI and lock file.
Its numeric fingerprint includes the installed NumPy and SciPy versions.

## Evidence and arithmetic

The collector pages both settled-market inventories. Identical cross-tier
records are reconciled; inconsistent records stop the run. Candles are routed
by settlement time, trades by their independent creation-time cutoff. A trade
window crossing that cutoff requests both tiers. Identical cross-tier trades
count once; conflicts and within-endpoint pagination duplicates are refused.
The two current candle endpoints have distinct native schemas: historical
`close` and live `close_dollars`, both dollar strings. Endpoint identity selects
the schema; missing fields do not trigger an alternate parser.

Every response retains its canonical request URL, acquisition timestamp, text,
and content hash. Trade identity, fractional size, aggressor outcome, and block
status survive normalization. Trades are fetched only for orders admitted by
the frozen quote rule. Changing the quote rule therefore requires a new
collection/replay that obtains its required evidence, not reuse of an old
normalized panel as if it were universal history.

A resting sell-YES order must improve the ask while remaining strictly above
the bid. A fill proxy requires enough subsequent non-block YES-taker volume at
or above the offer plus the declared threshold, strictly after decision plus
latency and strictly before close. Volume below the threshold cannot help fill
the order. Each order either reaches its full requested size or is unfilled in
this model; partial executions are not simulated. Integer accounting computes
settlement payoff and upward whole-cent quadratic fees without floating-point
currency rounding. Accounting uses the required `settlement_value_dollars`
field, including fractional `scalar` settlements; it never maps an unknown
result to NO. Binary YES/NO labels must agree with their declared cash payout.

The report includes every exclusion and order, pooled/city/monthly results,
one-cent friction sensitivity, and drawdown aggregated by event date. Its order-level
95% interval clusters by event date across all cities. Its calendar-day P&L
interval includes zero-trade days and uses seven-lag Newey–West dependence
adjustment. These reuse the library estimators; neither is a bootstrap.
Insufficient samples produce explicit null intervals.

## What a result means

A positive lower bound for the frozen primary rule identifies a simulation
candidate. It does not establish executable profit. A later print cannot prove
counterfactual queue position, unchanged competing orders, available size at
our arrival, or an actual fill. Candle-end age is not the age of the underlying
book update. The model omits partial execution, dynamic capital constraints,
cancellation, competing maker response, and mark-to-market drawdown. Selling
YES carries asymmetric losses even when most contracts resolve NO.

`event_day` is the date named by the weather event, not the timestamp when
the exchange finalized settlement or released collateral. Drawdown is grouped
by that date; it does not reconstruct actual intraday account equity.

Current fee metadata does not verify historical fees. Source omissions remain
possible even when pagination is complete. A public-paper replication is not
a pristine holdout if its dates overlap the author's sample. Sensitivity and
per-city results are descriptive, not licenses to select the best result.
Prospective quote/order-book capture and a frozen paper-execution study are
required before treating a candidate as an attainable edge.

## Measured replication: 2026-10-12

The frozen Boston, Philadelphia, and D.C. panel covers 2026-01-01 through
2026-10-10: 4,788 contracts, with one contract per hypothetical filled order.
The offer improves an eligible 2–16 cent ask by one cent at 16:00 Eastern on
the preceding day. The primary rule requires a subsequent qualifying trade
at least two cents above the offer. Current zero maker fees are assumed.

| Fill proxy | Fills | Total P&L | Mean per fill | Event-date clustered 95% interval | P&L with extra 1 cent per fill |
|---|---:|---:|---:|---:|---:|
| Two cents through, primary | 652 | $4.31 | 0.66 cents | −1.23 to 2.55 cents | −$2.21 |
| One cent through, sensitivity | 797 | $10.68 | 1.34 cents | −0.20 to 2.88 cents | $2.71 |
| Touch, sensitivity | 911 | $15.75 | 1.73 cents | 0.37 to 3.08 cents | $6.64 |

**The frozen primary candidate gate failed.** Its mean is positive but its
interval includes losses, and one cent of additional friction erases the
profit. The daily Newey–West interval also includes zero. The touch result
does not replace the primary result: it relies on a weaker execution proxy.
These findings do not establish an attainable edge or disprove every possible
strategy. No actual orders were placed, and scaling the one-contract results
would require new execution and capital evidence.

The primary descriptive city totals are Boston $3.80, Philadelphia $1.42,
and D.C. −$0.91; every city's primary interval includes zero. Maximum
event-date aggregated drawdown is $4.26. The accounting retains 18 scalar
D.C. settlements rather than dropping them or treating them as binary NO.
Protocol addenda preserve the acquisition corrections made before the first
complete maker-return analysis. The external paper overlaps this period,
so this is a historical replication, not an independent prospective holdout.

Machine-local evidence is under
`C:/Users/Test/PROJECTS/files/market-edge-20261012/`: `plan.json`,
`integrated-plan.json`, the three protocol/accounting addenda,
`integrated-raw/`, `verified-results.json`, `reproduced-results.json`, and
their `.runrecord.json` sidecars. Raw responses are deliberately outside the
repository. The verified report SHA-256 is
`1c917140f943a1ab43040fd42d3f8b692b779788675e837b62820acff254a69c`.
The report includes every order, source hash, exclusion, monthly result,
implementation digest, and numeric environment. One-worker and three-worker
offline runs reproduce the same report bytes.

The next evidence needed is prospective quote and order-book capture under
a frozen policy, followed by an execution study that measures fill rates,
adverse selection, fees, and attainable size. This implementation supplies
the historical evidence and replay layer; it is not that execution study.

## Follow-up input audit and prospective pilot

On 2026-10-12 the existing weather paths were traced before attempting a
forecast-versus-market comparison. The service's
`domains/weather/features.py` transforms an observed temperature. Its
`scripts/build_weather_corpus.py` uses day t's completed daily TMAX and
three lagged anomalies to predict t+1's anomaly. Seasonal state is fitted
on 1950–1989; benchmark rows cover summers in 1990–2024 and split by
station. This is a legitimate separate benchmark, but neither its group
split nor its synthetic midnight event timestamps establishes availability
of final day-t TMAX at a 16:00 day-t trading decision. The summer-only
corpus also does not validate October predictions.

The next source probe succeeded: three explicit GFS runs initialized at
2026-10-12 00:00 UTC were actually received around 15:40 UTC, each with
72 hourly temperatures. Three open-market inventories returned 12 contracts
each and empty pagination cursors. All six responses are preserved through
the existing hashed ResponseArchive in the machine-local evidence directory's
`forecast-input-probe/`, beside its frozen `protocol.json`.

The October 13 contracts name **The Weather Company** as settlement source.
GHCN daily observations must therefore not be substituted as exact market
labels. The pilot records all three next-day point forecasts before outcomes:
Boston 61.70 F, Philadelphia 79.16 F, D.C. 82.76 F. These are maxima of
24 hourly GFS values over the Eastern calendar day, a provisional aggregation
whose alignment to the exchange's observation-day convention is still unverified.
They are not calibrated bracket probabilities or trading recommendations.
Outcomes are pending; no continuous collector or scheduled job was launched.

The next model comparison must measure whether forecasts add information to
market probabilities on future dates, with receipt timestamps and calibration
restricted to prior outcomes. This pilot tests input alignment first and
cannot establish forecast superiority with three observations. Open-Meteo's
[single-run documentation](https://open-meteo.com/en/docs/single-runs-api)
explicitly distinguishes model initialization from public availability;
retrospective initialization timestamps alone cannot prove tradability.

## Standards and regression checks

This extends the package's existing Makefile, pyproject, and shared guard
bootstrap. `make lint` covers `src`, `tests`, and `scripts`; `make check` runs
strict mypy, repository guards, and pytest/xdist with 100% source-and-script
statement and branch coverage. There is no parallel testing harness.

The integration tests use the real HTTP adapter against a local socket server,
real archive files, real codecs, replay, estimators, CLI, and RunRecord writer.
They cover complete capture/replay equivalence, corrupt evidence, schema and
identity failures, cutoff-crossing reconciliation, fee refusal, latency,
fractional volume, block trades, quote lookahead, and accounting identities.

API contracts: [historical tiers](https://docs.kalshi.com/getting_started/historical_data),
[live candles](https://docs.kalshi.com/api-reference/market/get-market-candlesticks),
[historical candles](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks),
[public trades](https://docs.kalshi.com/api-reference/market/get-trades).
