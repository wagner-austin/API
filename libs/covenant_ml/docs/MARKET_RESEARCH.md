# Market execution research

`scripts.market_research` tests a frozen daily-weather maker hypothesis against
public Kalshi evidence. It uses the existing `covenant_ml.evaluation` estimators
and the shared `platform_core.run_record` provenance contract. It has no order
placement interface. Forecast accuracy in covenant-radar does not establish
market profitability; this experiment measures hypothetical execution payoffs.

[City weather calibration](CITY_WEATHER_CALIBRATION.md) records the separate
October 13 exploratory bias/uncertainty experiment. It improved weather-only
scores slightly but worsened the market-plus-weather model; neither forecast
cycle supports promoting the change. The original prospective campaign stays frozen.

## Run and reproduce

The [cross-market investigation](CROSS_MARKET_INVESTIGATION.md) compares sports,
economic events, stocks/options and crypto using October 13 public evidence.
It records the negative snapshot screens and the next non-weather experiment.

The [frozen sports strategy result](SPORTS_STRATEGY_RESULT.md) evaluates a
specific chronological probability calibration rule on 233 training and 128
later games. Training selected the unchanged market probability; the cost-aware
rule abstained on all later games and failed its advancement criteria.
The subsequent [team-rating result](SPORTS_RATINGS_RESULT.md) adds information
from earlier settled games. Its fitted blend lost a hypothetical $3.16 on nine
later trades and also failed advancement; these reused dates are exploratory.

The later studies and active observations are written up separately:

- [Covenant warning evidence](COVENANT_WARNING_EVIDENCE.md): strict dated-input
  selection, future labels, censoring and chronological borrower exclusion;
  proposed cross-validation and review-budget metrics, with native-data gaps.
- [Evidence requirements audit](EVIDENCE_REQUIREMENTS_AUDIT_20261014.md): live
  prospective status, sample planning and execution gaps, Radar target leakage,
  provenance portability limits and the pending WRDS request.
- [Evidence recheck](MARKET_RECHECK_20261014.md): independent check passed;
  protocol checkout byte drift repaired and historical results reproduced again.
- [Fixed boosting comparison](FORECAST_BOOSTING_PROTOCOL.md): declared model,
  chronological folds and evaluation limits frozen before the development run.
- [Completed boosting result](FORECAST_BOOSTING_RESULTS.md): the fixed challenger
  had worse predictive loss in both folds and lost $6.94 after assumed taker fees.
- [Earlier-only capacity selection](FORECAST_SELECTION_RESULTS.md): 24 candidates
  and two selected refits still fail; all inner fits and native models are retained.
- [Cross-market data expansion](MARKET_DATA_EXPANSION_20261014.md): current evidence
  across domains and a new 1,017-day BTC/ETH archive, with explicit input gaps.
- [Validity audit](MARKET_VALIDITY_AUDIT.md): reproduced shared splitter and
  feature-selection leakage defects, their repairs, model inventory and limits
  on historical holdout claims.
- [2025 Philadelphia temporal transfer](MAKER_TEMPORAL_TRANSFER_RESULT.md):
  the frozen high-variation/falling-price group lost $5.61 before additional
  execution friction and failed its declared checks.
- [Causal maker regime diagnostic](MAKER_REGIME_RESULTS.md): stronger future-data
  mutation tests, exact preservation of original backtest results, and pre-entry
  probability-movement groups; no group establishes advancement.
- [Second-venue sports result](SPORTS_CROSS_VENUE_RESULT.md): the frozen
  disagreement strategy produced no trades after costs and did not advance.
- [Historical inflation result](INFLATION_EXPERIMENT_RESULT.md): the declared
  experiment failed its advancement criteria; hypothetical profit is not an edge.
- [Frozen September inflation observation](INFLATION_PROSPECTIVE_2026_09.md):
  the original probabilities and book receipts remain separate from the historical
  experiment. Outcome scoring requires the official release and final settlements.
- [Official release capture](BLS_RELEASE_COLLECTION.md): successful public
  acquisition and byte-identical replay, including source hashes and full checks.
  The [scheduled September observation](BLS_RELEASE_SCHEDULE.md) records its
  isolated runtime, actual scheduler diagnostic and verified future trigger.
- [Stock pilot snapshot](EQUITY_PILOT_SNAPSHOT.md) and
  [scheduled stock outcome collection](EQUITY_SCHEDULE.md): the declared portfolio
  and independently installed future collector; no mature return is claimed.

These reports preserve rejected candidates as well as active measurements.
None establishes a reliable profitable edge or authorizes funded trading.

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

## Forecast versus market comparison: 2026-10-12

`scripts.forecast_research` joins explicit GFS initialization runs to the same
archived native market contracts and prior quotes. The plan requires `name`,
`start`, `train_end`, `end`, `excluded_days`, `forecast_hour` and
`protocol_digest`. The hour is an explicit prior-day GFS cycle (0, 6 or 12 UTC);
there is no automatic substitution. The protocol file must match its SHA-256.
Missing target hours, conflicting metadata, a missing required city, and
nonconvergent fitting fail the run. An exclusion requires an explicit plan.

The feature is the maximum of exactly 24 hourly temperatures over the event
date's 05:00 UTC to next-day 05:00 UTC window. Integer Fahrenheit brackets use
half-degree continuity corrections and a fixed normal standard deviation of
3 F. This is a simple forecast feature, not a fitted weather-error model or a
claim of exact settlement-measurement equivalence. Six hours after model
initialization is an assumed availability bound, not a historical receipt log.

The two nested L2 logistic models share market log odds, Boston/Philadelphia
indicators and an intercept; one adds forecast log odds. Parameters are fixed:
`C=1`, `lbfgs`, `max_iter=2000`, `tol=1e-10`, seed zero. Only earlier rows whose
final settlements precede the first evaluation decision enter fitting. The
report retains fitted coefficients, every training/evaluation row, predictions,
Brier/log losses, paired event-date clustered intervals and seven-observed-date
lag Newey-West diagnostics. The latter gives equal weight to represented dates;
the primary paired comparison weights contracts. Neither treats correlated
brackets or cities on the same date as independent events.

Each arm's quoted taker policy chooses YES, NO or pass with a 2-cent expected
value hurdle after the upward-cent quadratic fee, one contract per trade.
Integer cash accounting excludes the hurdle from realized P&L: it is a
selection threshold, not an expense. Additional friction is 1 cent per trade.
Zero/one boundary prices are unavailable for execution. Current fee metadata
does not establish historical fees, and quotes do not establish depth or fills.

The original April 3–October 10 protocol failed before fitting: all three
June 11 00 UTC responses contained 72 null temperatures. A separately frozen
amendment excluded June 12 for all cities before performance was calculated.
The original protocol and failure remain part of the evidence; no values were
imputed. Of 3,438 selected contracts, 18 were protocol exclusions and 14 had
stale quotes. The 3,406 usable rows yielded 2,672 training rows, 18 unavailable
training labels, and 716 held-out contracts across September 1–October 10.

| Arm | Held-out Brier (lower is better) | Trades | After-fee P&L | With extra 1 cent per trade |
|---|---:|---:|---:|---:|
| Raw market midpoint | 0.109710 | 0 | $0.00 | $0.00 |
| Raw 00 UTC forecast | 0.146226 | 514 | −$12.28 | −$17.42 |
| Calibrated market | 0.109912 | 37 | $0.91 | $0.54 |
| Market plus 00 UTC forecast | 0.109981 | 54 | −$1.21 | −$1.75 |

**Adding this forecast did not pass the primary candidate gate.** The paired
Brier difference is +0.0000690, with a date-clustered 95% interval of
[−0.0002785, +0.0004165]. Newey-West sensitivity also includes zero. These
are chronological historical results, not a pristine holdout: the same market
dates appeared in the earlier execution study.

### Source-era and forecast-freshness diagnostic

Inspection of native `rules_primary` found that the settlement-source wording
changes on August 14: 2,368 training rows name NWS and 304 name The Weather
Company, while all 716 evaluation rows name The Weather Company. This is a
documented wording difference, not independent proof of a change in the actual
measurement process. The general [Kalshi weather guide](https://help.kalshi.com/en/articles/13823837-weather-markets)
describes NWS for daily markets and The Weather Company for hourly markets;
the contract-specific [GLOBALTEMPERATURE terms](https://assets.kalshi.com/contract_terms/GLOBALTEMPERATURE.pdf),
verified October 12, specify a hierarchy of The Weather Company, NWS, then the
area's national weather service. The first official non-preliminary report
and its full reported precision control resolution. This clarifies current
source priority without dating an actual measurement-process change. The
comparison uses exchange outcomes and preserves native terms rather than
replacing labels with another weather source.

A follow-up diagnostic was specified after the first results: fit August
14–September 20 and evaluate September 21–October 10, comparing explicit 00
and 12 UTC runs with otherwise identical models and policy. The later run is
consistent with the provider's typical 4–6 hour global-model delay before the
20 UTC decision, but actual historical receipt remains unverified. All of
these outcomes had already been analyzed, so this is exploratory.

The 12 UTC acquisition failed on September 14's run, explicitly reported
unavailable by the provider. A second documented amendment excluded September
15 from **both** training panels before any 12 UTC model fitting. The original
00 UTC diagnostic is retained separately. The matched amended comparison has
643 training rows, 18 unavailable training labels, and 359 evaluation contracts
over 20 dates. Contract IDs, quotes and outcomes match between run times.

| Matched diagnostic arm | Brier | Trades | After-fee P&L | Extra-friction P&L |
|---|---:|---:|---:|---:|
| Raw market | 0.106782 | 0 | $0.00 | $0.00 |
| Calibrated market | 0.106781 | 10 | $0.00 | −$0.10 |
| Raw 00 UTC forecast | 0.141816 | 267 | −$6.94 | −$9.61 |
| Raw 12 UTC forecast | 0.139765 | 259 | −$10.33 | −$12.92 |
| Market plus 00 UTC forecast | 0.106839 | 11 | −$0.48 | −$0.59 |
| Market plus 12 UTC forecast | 0.106612 | 7 | $0.97 | $0.90 |

The newer combined forecast's paired Brier difference versus calibrated market
is −0.0001695, with a 95% interval [−0.0006542, +0.0003152]; its candidate
gate also fails. Its positive seven-trade P&L is not evidence of a dependable
edge. The date-clustered economic interval includes zero. Selecting the
best-looking variant after these comparisons would not turn it into a
confirmatory result, and greater forecast accuracy alone did not guarantee
better raw-forecast trading returns.

### Reproduction and implications

From `libs/covenant_ml`, replay a frozen plan with:

```powershell
poetry run python -m scripts.forecast_research `
  --plan <plan.json> --protocol <protocol.json> `
  --markets-archive <integrated-raw> --forecast-archive <forecast-raw> `
  --forecast-mode replay --out <new-report.json>
```

`capture` and `resume` explicitly acquire forecasts; market inputs always
replay the verified market archive. `resume` reuses saved responses and does
not retry failed requests inside core logic. Reports and sidecars refuse
overwrite. Plans are copied into reports and all archive inputs are hashed.
The original amended plan is retained; `forecast-comparison-typed-plan.json`
only makes its existing 00 UTC choice explicit in the final typed schema.

Machine-local evidence remains under `market-edge-20261012/`. The primary
forecast protocol, both amendments, freshness protocol, all plans, 573 older
and 171 newer native forecast responses, development failures, and completed
reports are retained. Final reports and independent replays are byte-identical,
as are their RunRecord sidecars. Every referenced source hash was checked;
cash payouts, fees, aggregate P&L, Brier losses and label chronology were
independently recomputed. `forecast-verification.json` records those checks.

| Final report | SHA-256 |
|---|---|
| `forecast-verified.json` | `47870aafdebc6484bc24b33f76395fa9d3d08dceea37b834daff3597cbf51c65` |
| `forecast-freshness-00-verified.json` | `ac2789e53d80be1cd4aec3fe4cdd8b665de5e18f543e7b53c9f043669ddb5de1` |
| `forecast-freshness-12-verified.json` | `07962971debfbb7e6696cd003aabd62d604c0378bdb8ed193a5c14f9b275ff27` |

This tests a particular source, feature and policy; it does not establish that
software cannot produce a market edge. The useful next evidence is a frozen
prospective comparison with actual forecast receipt and book timestamps,
verified settlement-source alignment and execution costs. Repeatedly tuning
these same historical outcomes cannot supply that missing evidence. No orders
were placed, and no recurring collector was installed.

## Prospective acquisition pilot: October 13 contracts

`scripts/paper_research.py` now captures predictions before outcomes using the
same forecast transforms and saved calibrators as the historical experiment.
`ForecastFeatures` carries only city, prices and weather probability; settled
training observations extend that contract, so prediction needs no fabricated
label or settlement timestamp. `PaperPlan` and `BookTop` have strict immutable
contracts and codecs. The existing archive, HTTP factory, pagination, fee
validation, predictor and RunRecord writer remain the shared implementations.

The collector archives native series metadata, every selected market and full
public book depth, plus the exact explicit-run forecast response. It validates
actual local receipt times in a frozen pre-event window, requires forecast
receipt before each book by at most ten minutes, and seals predictions within
one minute of the final receipt. Duplicate identities, unavailable training
labels, changed model files, invalid chronology and malformed/crossed books
fail explicitly. Unsupported strikes and missing sides have named exclusions.
Capture needs a new archive directory; replay reads and verifies that directory
without HTTP and requires the original prediction contents to match exactly.

The pilot protocol was frozen before acquisition, SHA-256
`6cabb1070c2f5e084af71a0e51d6b8182c4a34508168f58ce29011bba6c18438`.
It retains both 00 and 12 UTC GFS cycles, both fitted calibrators and both raw
references. The matched source-era model reports above supply the coefficients;
nothing is refit or selected using this event. The acquisition window was
October 12 18:42–19:42 UTC for October 13 events. This is an instrumentation
pilot earlier than the historical 16:00 Eastern decision, not a confirmatory
replication of that trading policy.

**Executed:** six runs recorded predictions at 18:43:38–18:44:05 UTC, all before
the target day. The 54 native response envelopes produced 34 forecast/book
rows, each with four probabilities, over 17 unique contracts in three cities
on one event date. Boston's `KXHIGHTBOS-26OCT13-T67` had a missing book side in
both runs and was explicitly excluded. These are not 34 independent events.

| City | 00 UTC maximum F | 12 UTC maximum F | Included contracts per cycle |
|---|---:|---:|---:|
| Boston | 61.88 | 61.34 | 5 |
| Philadelphia | 79.16 | 80.06 | 6 |
| D.C. | 82.76 | 82.58 | 6 |

Every report and RunRecord sidecar reproduced byte-for-byte offline. A separate
PowerShell calculation checked all referenced file hashes, 24-hour forecast
maxima, best bid/opposing ask prices and sizes, fitted-model probabilities and
forecast-before-book chronology. Machine-local evidence is under
`market-edge-20261012/paper-pilot-20261013/`, including six frozen plans, six
capture directories, original/replayed reports and `verification.json`.
The protocol is `market-edge-20261012/paper-pilot-protocol.json`.

| Report prefix | Prediction report SHA-256 |
|---|---|
| `KXHIGHTBOS-00` | `a93cd2f8c31fdba57bef0e35bc044ffc5bbe70eeb3cc9d66efa3e9bf57716b67` |
| `KXHIGHTBOS-12` | `e9749b95f742bdf2a1f2b8497b6e11dda239d744614ad02c527369c58ca38ddb` |
| `KXHIGHPHIL-00` | `0c957a5dccd4a08c607ad55ba5649813e770000f7383d0d451a3bcddea6dd346` |
| `KXHIGHPHIL-12` | `98deb6cee2ebffe9ef22f41d1148de70bd08de6f0e4bbfb65fabdf3dd4363c15` |
| `KXHIGHTDC-00` | `919b97bb7d97931be4954376192fbea3e09c56f831bbd20ff186b63e95809c18` |
| `KXHIGHTDC-12` | `995c709cf8657a9e5be14c4f83da8b15d33368f76e91b3ddca8aee08b7ff7585` |

```powershell
poetry run python -m scripts.paper_research `
  --plan <city-cycle-plan.json> --protocol <paper-pilot-protocol.json> `
  --models <forecast-freshness-cycle-verified.json> `
  --archive <city-cycle-capture> --mode replay --out <new-report.json>
```

Use `--mode capture` only with a fresh archive and a future event's frozen plan.
The pilot used no order endpoint, account credentials, recurring scheduler or money.
The books are snapshots, not reserved liquidity or guaranteed fills; receipts
are local-clock times, not exchange timestamps or independent notarization.
Hourly grid forecasts and settlement measurements remain distinct. Outcomes
are pending: future evaluation must join exchange payouts, retain scalar
resolutions, and freeze any execution/fee assumptions before measuring returns.
This pilot establishes reproducible pre-outcome recording, not an edge.

## Prospective settlement reconciliation

`scripts/paper_settlements.py` joins each saved prediction to an archived public
market response. It verifies the original report, RunRecord payload digest,
every source hash and the sealed prediction body before using its probabilities.
It never refits a model or changes the original evidence. Settlement snapshots
use fresh archives and reproduce their reports and sidecars byte-for-byte offline.

Only the exchange's terminal `finalized` status supplies an outcome. Earlier
states, including `determined`, `disputed` and `amended`, remain pending even
when a provisional result is present; see the official
[market lifecycle](https://docs.kalshi.com/getting_started/market_lifecycle).
Binary and scalar payouts are distinct, including scalar payouts at zero or one.
Changed primary rules are flagged. Brier and log-loss scores use only unchanged
binary contracts and remain absent while any selected contract is pending.
Counts expose every exclusion; these probability scores do not measure fills,
fees or returns. A completed snapshot with no eligible binary outcomes has no score.

The first real reconciliation of all six October 13 prediction reports captured
34 public market responses. All 34 rows across 17 unique contracts were pending,
with no changed primary rules, no finalized labels and `scores: null`. All six
reports and RunRecord sidecars replayed byte-for-byte. Independent PowerShell
checks verified every referenced source hash, pending-state fields and replay
hashes. Evidence is under
`market-edge-20261012/paper-settlements-20261013T0340/`; `verification.json`
records report digests and actual receipt times. The directory suffix is a batch
label; source receipts supply the timestamps.

```powershell
poetry run python -m scripts.paper_settlements `
  --predictions <original-city-cycle-predictions.json> `
  --archive <new-settlement-archive> --mode capture --out <settlements.json>
poetry run python -m scripts.paper_settlements `
  --predictions <original-city-cycle-predictions.json> `
  --archive <same-settlement-archive> --mode replay --out <replayed.json>
```

Run a later capture into a new directory after settlement. Original prediction
replay requires its capture implementation revision because the sealed body
includes that implementation's digest. Settlement reconciliation instead verifies
the original sealed evidence and records its own implementation digest, allowing
the scorer to advance without altering historical predictions. This initial
reconciliation submitted no orders; its scheduled follow-up is described below.

## Frozen daily observation campaign

`scripts/paper_campaign.py` prepares immutable plans and executes one scheduled
tick through the existing prediction and settlement CLIs. The manifest binds
the collector implementation, protocol, both fitted-model reports and every
plan. Shared report verification was lifted into `reporting.py`, so settlement
and campaign checks use one source-hash and RunRecord implementation.

The frozen study covers **60 event dates, October 14 through December 12, 2026**:
360 city/cycle prediction captures and 360 later settlement snapshots. Every
prior day's acquisition window is **16:00-16:15 America/New_York**; the timezone
conversion explicitly follows the November daylight-saving transition. Each
settlement window is event date plus two days, **15:00-16:00 UTC**. A completed
settlement acquisition can still contain pending outcomes: acquisition completion
never means a payout was finalized. The first prediction window starts October
13 at 20:00 UTC (13:00 Pacific). All study observations were still future when
the schedule was installed.

Both forecast cycles and all four probability arms are retained without refits.
The declared primary comparison is the 12 UTC augmented-minus-calibrated-market
Brier loss, averaged within event date before averaging dates. Raw references
and the entire 00 UTC panel remain secondary comparisons. The fixed 60-date
window is an observation plan, not a power guarantee. Autumn-to-winter drift,
missing observations and uncertainty remain material. Probability skill alone
does not establish executable returns; the campaign submits no orders.

Launched attempts have an exclusive started marker and subprocess log; terminal
records also cover omitted attempts. Nonzero exits are persisted and propagated. A subsequent tick records
an interrupted start explicitly; it never repeats that attempt. Missed windows
and settlements without successful predictions remain named omissions. Earlier
failures do not become reconstructed observations. An exclusive SQLite transaction
prevents concurrent ticks; process termination releases the lock automatically.
Each CLI child has a five-minute limit. Tick start/result records also make an
unsuccessful scheduler invocation visible.

**Deployment verified:** the runtime is an isolated copy of 640 source/configuration
files across the package and its local dependencies, with a fresh locked Poetry
environment. Source receipts were checked against the tested checkout. The
existing `tools/fleet/scripts/FleetSchedule.ps1` registration helper installs
`API-MarketPaper-20261014-20261212`: Poetry runs the tick every three minutes and
at boot under the existing limited S4U account settings, with overlapping starts
ignored. Its action pins manifest SHA-256
`4f350cf553c68f6f3f467b59b6f92a8a8c24cf07ddada0c0527224ace17cc077`.
The initial actual scheduled tick exited zero and reported 720 future acquisitions,
zero completed. The PC and network must be available during the declared windows;
an offline period cannot be repaired by a retrospective capture.

Before installing that study schedule, a separate scheduled public-data probe
recorded five Boston predictions at October 13 04:04:58 UTC, retained one missing
book-side exclusion and placed zero orders. The original report and sidecar
replayed byte-for-byte, and all 13 source hashes were independently verified.
That probe is explicitly excluded from the 60-date study; its temporary scheduled
task was removed and removal verified. Probe report SHA-256:
`f8916860662de1d9916772adecf47c9da0a3a3f8a436dd38073520cb8f8a6e55`.

Machine-local evidence under `market-edge-20261012/`:

- `campaign-protocol-20261014-20261212.json`: protocol, SHA-256
  `bb3e6826545afc2959cc58b3cc838449210435f5e235a870258c3165bcc098c8`.
- `campaign-20261014-20261212/`: manifest, 360 plans, attempt/tick records,
  scheduler XML and `deployment-verification.json`; future reports/archives land here.
- `campaign-runtime/`: isolated implementation, locked environment and source receipts.
- `campaign-deployment-probe/`: actual scheduled acquisition, replay and verification.

Run `poetry run python -m scripts.paper_campaign prepare --help` for new studies;
`tick --root <campaign-directory> --digest <original-manifest-sha256>` executes
the current frozen study without modifying its plans. Use the isolated runtime's
package directory. To remove this schedule, run
`Unregister-ScheduledTask -TaskName 'API-MarketPaper-20261014-20261212' -Confirm:$false`
(the `FleetSchedule.ps1` helper that registered it was deleted with the fleet
tick registrations in MCPs board task b27bd34a). Removing the task
preserves every evidence file. The scheduler continues checking evidence after
the finite study window until removed; it does not extend the study implicitly.

**Continuation on Linux (2026-10-17).** austinpc, the Windows host above, is
reimaged to Linux (MCPs board task `e2ffed0d`). The manifest records absolute
Windows paths that every tick re-hashes, so the study cannot run on Linux as
frozen. The operator chose to re-plan it there under a new manifest. The
continuation `frozen-autumn-probability-study-diphtheria` answers for event
dates **October 19 through December 12** (330 plans). The original answers for
**October 14 through 18**. Both are prepared from the same protocol and the same
two fitted-model reports, byte for byte, and run the same collector
implementation, SHA-256
`332f0975d3ef78766c5f1f863ea9034fa37c79c7ebe6d46cd85c9cd5a080fcb7`. The Linux
runtime is a byte-identical copy of `campaign-runtime/` with a fresh locked
Poetry environment, and its `implementation_digest()` was measured equal before
preparation. The continuation's manifest SHA-256 is
`f57a191b3a7a1f289d01e5b9826a9f9d49e6222133a06382e2a92330a122cc85`, under
`/home/corvis/PROJECTS/files/market-edge-20261012/campaign-20261019-20261212/`
on diphtheria. Its first tick (2026-10-17 23:12 UTC) reported 660 future
acquisitions. Since 2026-10-18 the hub runner on diphtheria runs the tick as row
`market-paper-campaign-20261019` of MCPs `scripts/fleet/hub-jobs.json` (added in
MCPs `d34dc2084`): duty control, Linux only, every 180 seconds, with a
3,000-second deadline. Its paths are under `{data}/Corvis/markets`, a symlink to
`/home/corvis/PROJECTS/files`. The runner took the control duty when MCPs
`94319e53b` reached diphtheria (MCPs board task `c39ee056`). The systemd user timer
`covenant-market-paper-20261019-20261212.timer` that ran the tick before was
disabled at 15:37:32 UTC, after its last tick at 15:36:01. The runner's first tick
started at 15:50:18 UTC and carried the same manifest digest. Like every tick
before it, it reported 660 future acquisitions and none missed: the first window
opens October 18 at 20:00 UTC.

The original keeps running on austinpc until the reimage stops it, so that it can
complete the settlement snapshots of its October 14 to 18 captures, the last on
October 20 at 15:00-16:00 UTC. Its task was disabled on 2026-10-18 at 12:03 UTC,
after tick `39c50b23`, which reported 42 complete acquisitions: all 30 captures
and the 12 October 14 and 15 settlements. It was re-enabled at 15:55 UTC, inside
the October 16 settlement window. That tick finished at 15:56:03 with 48 complete,
none missed and none failed. The October 17 and 18 settlement windows (October
19 and 20, 15:00-16:00 UTC) are captured only if austinpc still runs the task
then. A window that passes is not captured later. Whatever the original captures
for October 19 or later is outside its study scope. It is retained as a
same-implementation, cross-host replication of the continuation, not as
observations. The record of the move is
`market-edge-20261012/campaign-relocation-20261017.json`, SHA-256
`c402550295f470a42d8170a991288dede3630308af1efd821bea8710f34f5060`. It names both
manifests, both hosts, each campaign's event dates, and the protocol, model and
implementation identities. An evaluation of the full 60-date window reads each
date from the campaign that answers for it. To remove the continuation's
schedule, delete its row from `scripts/fleet/hub-jobs.json` in MCPs. Removing it
preserves every evidence file.

**Evidence that moved hosts.** Reports and manifests record each source by its
absolute path at writing time, and every reader re-hashes the file there. So
evidence written on Windows could not be verified on any Linux host.
`markets/evidence_location.py` reads a host's declaration,
`~/.config/covenant_ml/evidence-relocations.json`: schema version 1, with a list
of `recorded_prefix` and `current_prefix` pairs. A recorded path under a declared
prefix is read at the same relative place under its current prefix. The first
declared prefix containing the path decides, and a path under none is read
where it was recorded. Report verification (`reporting.py`), the campaign's
frozen-input check (`campaign_inputs.py`), the tick's child commands and the
campaign evaluator all locate through it. The SHA-256 each checks is still the
one recorded, so a relocation can only point at a byte-identical copy. An absent
declaration relocates nothing. A malformed one is refused with
`EVIDENCE-RELOCATION-001`.

The [offline campaign evaluator](CAMPAIGN_EVALUATION.md) now verifies the frozen
inputs and original reports, replays archived settlements, and computes the
declared event-date comparison. It retains every acquisition omission and
outcome exclusion, with separate collector and evaluator implementation hashes.

## Execution rehearsal

The separate [bounded execution rehearsal](EXECUTION_REHEARSAL.md) now exercises
the proposed $20 experiment's durable risk reservations and order reconciliation.
Its fixed limits are $1 per order including reserved fees, $3 combined exposure,
and $5 gross losses plus remaining exposure. It sends no exchange orders and
does not modify the frozen study. The linked contract distinguishes tested local
accounting from the authenticated account integration and demo verification
still required before a live pilot.

The [authenticated demo account preflight](KALSHI_DEMO.md) adds Ed25519/RSA
signatures and strictly validated account reads, with exact microdollar balances.
It refuses production origins, populated or incomplete account evidence and
changes between balance reads. It sends no orders and reports both execution
and live readiness as false. Remote demo verification requires demo credentials;
local socket and cryptography tests are not presented as exchange execution.

The [durable demo order trial](KALSHI_DEMO_ORDER.md) subsequently completed an
actual demo IOC fill: one contract at $0.30 plus $0.0147 fees, leaving $49.6853
of the starting $50 mock cash. A brief order-list visibility delay retained the
reservation; explicit GET-only reconciliation verified order, fills, position
and cash without resubmission. Restarted submission was refused. This verifies
the bounded demo execution path, not a profitable signal or production readiness.

## Frozen research proposal admission

Evaluator-owned journal transitions `select` and `release_final` reserve final
evaluation without adding a second queue or journal. Selection requires a
completed validation attempt, no unsettled admitted work, and no earlier selection.
It binds the candidate, the validation RunRecord's payload digest and the exact
final-manifest digest. Later proposals remain recorded but receive `study_sealed`.
Release must match all three identities and is consumed transactionally once;
reopening the journal does not reset it. The original validation request and
result remain attached to the selected attempt. The stored candidate identifies
the benchmark configuration and feature set, not a separately selected learner.

These transitions are for a trusted evaluator with exclusive journal access;
they do not authenticate a caller. The evaluator issues them through
`markets.research_final`: `research_proposals select --trial T --final-manifest M`
binds the named completed attempt's journal candidate and validation payload to
the SHA-256 of `M`, and `research_trial --request T --out O --ledger L
--final-manifest M` runs `execute_final`. That run requires the journal row to be
selected and `M` to hash to the bound digest, rechecks the policy, protocol,
code and image lineage, and only then commits `release_final`, before any final
relation is read. A failure after release, including a missing data file, leaves
the release consumed; a second run is refused with RESEARCH-009. The candidate,
seed and features come from the journal's own admitted request, which still
names the validation partition: no proposal field authorizes final scoring, and
a proposal naming another partition is refused at admission and can never be
selected. `markets.crsp_final.run_crsp_final` fits the candidate once on every
decision whose exit precedes `holdout_start`, with the declared stopping months
immediately before the boundary, scores every decision from `holdout_start`
through `holdout_end`, and reuses the validation study's holdings, arms and
paired comparisons; its record is labelled `final-<attempt>`. The cases in
`test_research_final.py` and `test_crsp_final.py` exercise each refusal on real
journals and show that changing outcomes after the last final decision changes
no fit or selection. Choosing which completed attempt to select remains the
evaluator's decision.

Proposers reach this package only as data. Their programs run through
`tools/fleet`'s `fleet.cli.research_propose`, which executes evaluator-pinned
program text in a rootless container with no host mount, environment, network
or capabilities, writes the returned object beside the evaluator's private
specification and passes it to `research_proposals propose`, so every proposal
is recorded before decoding. The process boundary covers the proposal program,
not an administrator or an agent session holding the evaluator's account, SSH
or WRDS credentials; such a session is privileged access by definition.

`scripts.research_proposals` records validation proposals under an evaluator-owned
policy. It uses `platform_core.event_journal.EventJournal`, lifted from the
execution rehearsal; those rehearsal callers now use the same persistence code.
The policy pins protocol, dataset, code and image digests, permitted features and
seeds, and candidate, runtime, tree and leaf ceilings. Proposals reuse the existing
benchmark configuration codec and cannot supply a shell command or data path.

From this package, initialize and use a study with:

```sh
poetry run python -m scripts.research_proposals --policy policy.json --ledger attempts.sqlite init
poetry run python -m scripts.research_proposals --policy policy.json --ledger attempts.sqlite propose --proposal proposal.json
poetry run python -m scripts.research_proposals --policy policy.json --ledger attempts.sqlite status
```

Admission returns exit code zero; a policy refusal returns two. Malformed object
proposals are saved as `refused/not_admitted` before decoding raises. Unreadable
files or malformed JSON are input transport errors and never become proposals.
Renaming a hypothesis or reordering feature names cannot reset candidate identity.
An admitted candidate/seed remains consumed after failure, timeout or interruption.
Every admission decision runs inside a SQLite write transaction; a competing
writer is refused rather than allowed to oversubscribe a budget.

The runner-facing lifecycle records `submitting` before an external submission,
then a unique scheduler identity and finally a shared `RunRecord` or explicit
failure. A completion must match the study's data/code/image identities and its
attempt number. Negative observations remain in the record. A crash between
submission and recording the scheduler response leaves `submitting` unresolved;
it does not authorize resubmission.

The same entrypoint now connects an admitted attempt to maintained HPC3 commands:

```sh
poetry run python -m scripts.research_proposals --policy policy.json --ledger attempts.sqlite submit --trial trial.json --cluster cluster.json --directory /absolute/attempt-1
poetry run python -m scripts.research_proposals --policy policy.json --ledger attempts.sqlite reconcile --trial trial.json --cluster cluster.json --directory /absolute/attempt-1
```

`ResearchTrial` contains the frozen policy, exact admitted proposal, evaluation
settings, attempt number and absolute compute-side evaluation manifest path.
`ResearchCluster` is operator-owned: the HPC3 Python executable, workspace,
project, remote attempt root, entrypoint directory, image path, checksum text and
board report task. `python` and `workspace` are absolute paths on the controller's
local filesystem. `root`, `entrypoint` and `image` are absolute POSIX paths on
HPC3, even when the controller runs on Windows. All five reject parent-directory
traversal. The checksums must include the image, evaluation manifest and
`scripts/research_trial.py`; include every source data/query/receipt file and the
entrypoint package initializer as well. The policy's dataset digest is the SHA-256
of the evaluation manifest's exact bytes; that manifest pins the data and SQL.
Its code digest names the external trial entrypoint; the image pins imported code.

Controller portability was verified at API `effe49a2c9baf591547cc66875225bbdcc5d6d13`
on 2026-10-29: fleet job `98aaf40c-680c-4bcd-938b-5ce4042d01d0` on Linux and
[Windows workflow 40816617770](https://github.com/wagner-austin/API/actions/runs/40816617770)
each passed all 4,524 tests with 100% coverage. The tests exercise native controller
paths, relative/traversing path rejection, POSIX-only remote paths and real process
failure reporting. These checks do not replace a native scheduler execution receipt.

Submission freezes a request and recipe in a new directory, then uses `hpc3.cli.stage`
and `hpc3.cli.submit --format json`. The recipe requests one CPU, 4 GiB, no GPU,
the free partition and a rounded allocation deadline. `timeout` enforces the
proposal's exact seconds, with five seconds for forced cleanup. A refused or
already-consumed attempt cannot submit. Transport failures leave durable
`submitting` state; `reconcile` finds exactly one matching attempt in the shared
HPC ledger and never launches another job. If Slurm accepted a job but the HPC
ledger itself was not written, automatic recovery refuses rather than guessing;
the operator must investigate the scheduler before resolving that attempt.

Reconciliation uses `hpc3.cli.execution_inspect`, which reads native accounting,
checks compute-side input hashes and verifies result/RunRecord bytes and write
times inside the allocation. The measuring program posts its evidence to the
board. Only a successful measured allocation may complete; pending jobs remain
running, and terminal failures consume the attempt. Repeated observations preserve
separate logs. These are owner-writable files, not hostile-owner attestation.

`scripts.research_trial` runs the frozen monthly CRSP study described below on
its validation period, using the existing baseline trainers and the shared
`RunRecord` publisher. `ResearchEvaluation` is the decoded frozen design plus
the declared stopping months, fold months and interval confidence; its protocol
hash names `crsp-monthly-holdings-validation-v1` and all of those settings. The
evaluation manifest is the CRSP extract manifest, and the worker refuses to fit
unless the staged protocol bytes decode to exactly the admitted design. A
validation run never constructs a final-period decision; only the evaluator's
`--ledger`/`--final-manifest` mode above does, once. Hash chaining detects
corruption, not deliberate rewriting by someone who can modify the journal
database.

## Frozen monthly CRSP contract

`docs/crsp-monthly-v1.json` is the machine-readable design. Decode it through
`markets.crsp_protocol.decode_crsp_protocol`, then bind its exact digest in the
existing research policy. The codec rejects unknown fields and unsupported
source/valuation conventions; proposals cannot replace the evaluator's policy.

The 2026-10-29 compute probes `40544368` and `40544470` found legacy membership
through 2024-12-31, CIZ returns/calendar through 2025-12-31, and terminal events
through 2025-12-30 with stored return dates through 2025-12-31. Before observing any
expanded outcomes, the design therefore changed to training 2000–2014,
validation 2015–2021 and 36 sealed decisions during 2022–2024. Read coverage
through February 2025 for the final exit. Warm-up begins January 1998.
Table extents do not prove every individual history is complete.

The four ordered `*_columns` arrays in the frozen JSON are decoder-enforced.
Changing, removing, adding or reordering a column is refused. The exact fields are:

- `crsp_a_stock.dsenames`: `permno,namedt,nameendt,shrcd,exchcd`.
  Membership uses inclusive dates and exactly one active record, share codes
  10/11 and exchange codes 1/2/3. Never extend membership beyond its ending date.
- `crsp_a_stock.metaexchangecalendar`: `caldt,exchangegroup,tradingflg`.
  Filter group `CIZ`; retain all calendar days and their `Y`/`N` flags.
  `crsp_windows` checks contiguity and complete months before selecting the
  month's last session and the following two months' first sessions.
- `crsp_a_stock.stkdlysecuritydata`: `permno,dlycaldt,dlyprevdt,dlyret,
  dlyretmissflg,dlyretdurflg,dlydelflg,dlyprc,dlyprcflg,dlyvol`.
  Ordinary execution requires finite positive `TR` closing trades. Universe
  liquidity uses 60 pre-decision sessions; eligibility requires 252 prior
  sessions and decision price at least $5. Future entry prices cannot change
  the ranked selection: an unexecutable selected trade makes the run incomplete.
- `crsp_a_stock.stkdelists`: `permno,delistingdt,deldlydt,delamtdt,delnextdt,
  delret,delretmisstype`. Require a finite return at least -1 with flag `NA`,
  required amount date, and explicit nullable next-price date. The valuation
  date is the latest of stored/amount/present next-price dates. Its multiplier
  reconciles the CIZ terminal return once; do not multiply by both returns.

The declared `dated-value-next-rebalance` convention treats a terminal value
as a receivable until its recorded valuation date, then allows modeled
reinvestment at the first scheduled rebalance strictly after that date.
An unresolved value or one dated after the scored exit makes that outcome
incomplete. This is a revised-history valuation and settlement assumption,
not evidence of historical cash availability or an executable liquidation.

## Monthly holdings and validation study

`markets.crsp_holdings.evaluate_crsp_holdings` accounts for a long-only
equal-weight portfolio across consecutive windows, each entering at the
previous exit. Value starts as one unit of cash. At each rebalance close the
investable value (positions plus terminal values dated strictly before that
close) is split equally across the selection; traded notional is
`sum |target - drifted position|` and the cost, one of the frozen `cost_bps`,
is paid from the portfolio. Positions compound their own ordinary daily returns
over `(entry, exit]`. A position delisted inside the window stops at its
delisting date and becomes a receivable worth its `stkdelists` multiplier,
applied once; the daily delisting-flagged copy is never compounded. The last
window also pays the cost of liquidating its positions at its exit close. Each
period retains its selection, drifted end weights, receivables, turnover, cost
and net return. A missing or non-ordinary held return, a missing or non-`TR`
fill, an unresolved terminal event or a terminal value dated after the exit
refuses the whole evaluation; no partial result is returned.

`markets.crsp_study.run_crsp_validation` considers decisions from `train_start`
to the day before `holdout_start`, then retains only windows whose exit is
strictly before `holdout_start`, before attaching any outcome. A row's label is
whether the security's window
multiplier beats the eligible universe's equal-weight multiplier, available at
the exit close. Validation decisions are scored in consecutive folds of the
declared length; each fold fits on every earlier decision except the declared
stopping months, which supply early stopping, and `purge_label_overlap` drops
every training and stopping label not available strictly before the next
boundary, which is exactly the decision immediately before each boundary.
Candidates select the top `ceil(selection_fraction * eligible)` by predicted
probability, momentum by its lagged return, ties by PERMNO; the equal-weight
universe is reported beside them. Every arm is accounted at every frozen cost,
and at the primary cost each candidate and the universe are compared with
momentum on identical consecutive months by `compare_moving_block` with the
frozen block length, draw count and seed. Labels are binary because the
existing trainers are classifiers; the design's continuous excess return is
ranked through the predicted probability of a positive one. The two validation
months nearest a January final-period start are omitted because their exits
fall in that period: for the committed design, validation decisions end in
October 2021, with the last exit in December 2021. The boundary uses actual
calendar exits, not a fixed number of rows. Regression cases exclude an exit
equal to the boundary and change later returns and prices without changing
validation output. This corrects the earlier implementation that selected
on outcomes extending into the final period. It is a temporal exclusion;
restricted process, file and credential access remains a separate requirement
before any final-period score.

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
The forecast tests additionally fit the real logistic learner, force actual
nonconvergence, flip all held-out labels without changing fitted parameters or
predictions, reconcile optional native strike bounds, reject missing cities,
and run full HTTP capture through byte-identical offline CLI replay.
The completed forecast change passes `make check`: 3,013 tests, strict mypy,
all repository guards, and 100% statement and branch coverage for source and
scripts. The local run took 142 seconds against the existing 300-second budget.
The prospective extension passes all 3,053 tests, strict mypy across 506 files,
all guards, and 100% coverage of 14,534 statements and 2,704 branches. Its full
`make check` took 134 seconds. Real socket failures, subprocess execution,
changed evidence, exact replay and outcome-free prediction parity are tested.
The settlement extension passes all 3,100 tests, strict mypy across 512 files,
all guards, and 100% coverage of 14,731 statements and 2,756 branches. Its full
`make check` took 128 seconds. Tests exercise real HTTP capture and replay,
actual CLI subprocesses, manually calculated proper scores, provisional and
scalar results, changed rules, receipt regressions and corrupted original evidence.
The campaign extension passes all 3,145 tests, strict mypy across 521 files,
all guards, and 100% coverage of 15,026 statements and 2,828 branches. The full
check took 135 seconds. Tests cover daylight-saving windows, real subprocess
and HTTP failures, durable omissions, interrupted starts, actual SQLite lock
contention, original evidence drift, CLI entry points and no-repeat behavior.

API contracts: [historical tiers](https://docs.kalshi.com/getting_started/historical_data),
[live candles](https://docs.kalshi.com/api-reference/market/get-market-candlesticks),
[historical candles](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks),
[public trades](https://docs.kalshi.com/api-reference/market/get-trades).
Prospective liquidity uses the native
[fixed-point order book](https://docs.kalshi.com/api-reference/market/get-market-orderbook).
