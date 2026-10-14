# Frozen inflation forecast experiment

Declared October 14, 2026 UTC before scoring, task
`d673be48-bc8b-4b85-a086-280b870eec21`. This is exploratory: source coverage,
five quote windows and examples of actual releases have already been inspected.

Evaluate target months January 2024 through August 2026 inclusive. Read all
native records from the captured historical and ordinary `KXCPIYOY` inventories.
Account for every unique contract and exclusion. Overlapping records must agree
on identity, rules, strike, result and close time; reject conflicting records.
Exclude October 2025 because the official forecast feed has no actual release.

Use midnight UTC on the actual release day as the single decision cutoff for
each event. Require that day to match the exchange close date, that the contract
was already open, and that its final binary settlement matches the official
actual rounded to one decimal with half-away-from-zero rounding. Require an
explicit strictly-greater native strike and the one-decimal primary rule.
Document mismatches rather than changing labels. The retrospective release date
is a calendar alignment assumption, not independently archived advance notice.

Use the most recent forecast strictly before that midnight, at most four calendar
days old. For each prediction use only earlier target months whose actual was
released on an earlier day. Select the last 60 available release-day forecast
errors, requiring at least 36. Each error is unrounded actual minus its own last
forecast before release, with the same age rule. Do not pool later outcomes.

Add each earlier error to the current forecast, round that synthetic release to
one decimal, and count results strictly above the contract strike. The fixed
probability is `(successes + 0.5) / (number_of_errors + 1)`. No alternative model,
window search, error rescaling or outcome-driven threshold selection is allowed.

Request the exact midnight one-minute candle through its explicit native partition.
Reuse the existing quote validator: missing or noninterior quotes, crossed
prices, wrong timestamps and spreads above five cents cannot become trades.
Do not carry older quotes forward. Measure coverage independently of forecasts.

Compute Brier scores across every jointly eligible contract, first averaging
within each release so a large threshold ladder cannot dominate the score.
For hypothetical trading select at most one contract per release: largest
absolute difference between forecast and market midpoint, ticker ascending on
ties. Selection uses no outcome. Reuse the existing decision rule: one-cent
adverse price movement, rounded quadratic taker fee coefficient 0.07 and at
least three cents expected margin after costs. Report doubled fees as well.

Estimate a Student-t interval across release dates, including zero-return eligible
releases. This assumes independent months and may understate serial dependence.
Advancement
requires at least 24 eligible release dates, at least 20 hypothetical trades,
lower mean Brier than market, positive lower 95% mean-profit bound and positive
total profit with doubled fees. Passing would support a fresh prospective demo
test only. Historical candles prove neither depth nor fills; no orders or funds
are part of this experiment. Hash every source and protocol, preserve all
exclusions, and reproduce the report in an independent offline process.

## Acquisition correction before scoring

At 2026-10-14 05:00 UTC, acquisition encountered a 404 for an August 2026
contract absent from the historical inventory but present in the ordinary
inventory. No forecast score or profit result had been calculated. Route
contracts present in the captured historical inventory to the historical
endpoint; route ordinary-only contracts to the series candle endpoint with its
native dollar-price fields. Record the selected partition. Never try one route
after the other fails. This corrects endpoint selection without changing dates,
probabilities, costs, quote eligibility or advancement gates. Preserve the
completed historical captures and the acquisition failure in the evidence log.
