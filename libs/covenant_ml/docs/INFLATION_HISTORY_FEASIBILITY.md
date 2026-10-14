# Inflation forecasts: a historical test is feasible

Measured October 14, 2026 UTC, task `69da192b-903c-4633-a4d2-278cd4a80e56`.
This is a data-feasibility result, not evidence of profitable trading.

The official Cleveland Fed chart feed supplies historical daily forecasts
independent of exchange prices. Recent Kalshi inflation contracts also have
historical bid/ask candles. This supports a chronological exploratory comparison
after a strict decoder and frozen decision protocol are implemented. It does
not establish an edge or justify funding.

## Captured sources and coverage

Evidence directory: `C:/Users/Test/PROJECTS/files/inflation-feasibility-20261014`.
The ten original response files are indexed by URL, byte count, SHA-256 and
file-receipt timestamp in `evidence.json`, whose SHA-256 is
`7ee3253306692065d087fda6374ed00a86310b50b858a9594f01db5a6d4f092c`.
These timestamps record this acquisition, not historical publication times.

| Source | Measured coverage |
|---|---|
| [Official yearly chart JSON](https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_year.json?sc_lang=en) | 160 target months, July 2013 through October 2026 |
| Headline CPI forecast series | 4,687 nonempty numeric observations |
| Actual headline CPI series | 156 nonempty numeric observations |
| Missing actual observations | July 2013, October 2025, September and October 2026 |
| Missing forecast series | July 2013 only |
| Chart alignment | All 160 charts have equal date-label, forecast-row and actual-row counts after removing release-marker labels |
| Kalshi historical inventory | 624 contracts, 45 monthly events, terminal empty cursor |
| Kalshi ordinary inventory | 110 contracts, five events, terminal empty cursor |
| Combined inventories | 713 unique contracts, 49 events; 21 overlapping July 2026 contracts |

The 21 overlapping records agree on event, primary rules, strike, strike type,
result and close time. This does not prove every metadata field is identical.
All historical contracts report `greater` strike type and finalized binary
results. Historical event identities include both `CPIYOY` and `KXCPIYOY`:
these are native records returned by the same series query, not a reason to
silently substitute one endpoint or schema for another.

The nowcast file hash is
`bf0edf1132e22c7ba05729efe6ddcd3701cd1912d50ee36f546e13ce9bcf8c46`.
The historical inventory hash is
`aed85acea305dfb269850f7dce86b1a8db893688cad7481ce5ad023e298a8cb1`.

## Timing and settlement alignment

The [official methodology page](https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting)
identifies yearly CPI forecasts as nonseasonally adjusted, says daily updates
normally appear around 10 a.m. Eastern, and describes chart actuals as first
available releases. It also flags the missing October 2025 CPI release.

Chart dates contain month/day without an observation year or publication time;
the target-month caption supplies an anchor, and dates cross year boundaries.
Release-marker labels are extra chart decorations, not data rows. The global
`_comment` says `2026-10-13 00:00` even in old charts: it is not an individual
forecast's receipt timestamp. A decoder must reconstruct ordered calendar dates,
validate each tooltip against its date and value, and distinguish blanks from
numeric zeros. Historical availability remains an exploratory assumption;
prospective captures would supply actual receipt evidence.

The captured [Kalshi series](https://api.elections.kalshi.com/trade-api/v2/series/KXCPIYOY)
names BLS as settlement source. Captured contract rules use a **one-decimal
reported value strictly greater than the strike**. A continuous inflation
forecast is neither that rounded settlement value nor a contract probability.
Any forecast-error distribution must be fitted on earlier releases only, with
the rounding boundary represented explicitly. Do not compare an unrounded
forecast directly with the threshold and call that a probability.

October 2025 is excluded from an initial comparison. For example,
`KXCPIYOY-25OCT-T3.1` reports a Yes outcome and expiration value `3.3`, while
the Cleveland source reports no actual October release. The cause and full
settlement handling have not been established here; substituting the forecast
for an observed inflation release would contaminate training.

## Historical quote probe

For five preselected event months, select the lexicographically first contract
and query the hour ending exactly 24 hours before its recorded close. This
selection checks data presence, not attractive prices or profitable decisions.
The manifest records every exact URL. Requests use Kalshi's documented
[historical candle endpoint](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks)
with one-minute intervals.

| Target month / selected threshold | Returned candles | Last candle age at cutoff |
|---|---:|---:|
| November 2022 / 6.8 | 0 | unavailable |
| December 2023 / 2.8 | 0 | unavailable |
| December 2024 / 2.0 | 1 | 780 seconds |
| December 2025 / 2.4 | 9 | 300 seconds |
| June 2026 / 2.5 | 11 | 0 seconds |

None of these counts establishes executable depth or fills. Sparse candles
require an explicit freshness gate; old values cannot be carried forward
silently. Empty samples do not prove that every contract in those years lacks
history. Metadata coverage is much longer than demonstrated quote coverage.

## Next experiment

Freeze one cutoff per monthly release, safely after the prior business day's
normal forecast publication and before the release. Decode the public forecast
history and fit its error distribution only from already released months.
Score all supported strikes with actual same-cutoff quote coverage, account for
exclusions and fees, and aggregate uncertainty by release date, not contract.
Multiple thresholds on the same inflation release are highly dependent.

Use an earlier calibration period and a later chronological evaluation period
declared before scoring. Publish every eligible release and exclusion, retain
missing quotes as missing, and require independent prospective evidence before
advancing. First implement and test date rollover, marker removal, empty cells,
rounding boundaries and settlement mismatches; then measure complete quote
coverage. No new account, credentials or real money is needed for this test.
