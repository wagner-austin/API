# Inflation history decoded with explicit cutoff dates

Task `e4442aef-93f2-4425-812e-9ba9a0be3932`, October 14, 2026 UTC.
The native yearly headline feed now passes the production decoder. This is a
validated model input, not a fitted probability model or trading result.

`inflation_history.decode_inflation_history` reads parsed JSON through the
existing internal validators. It checks the chart's yearly identity, anchors
month/day labels to target years, separates release markers from observations,
and validates both required headline series. Duplicate series and target months,
reversed dates, mismatched row counts and tooltip identities, malformed numeric
strings, nonfinite values and forecasts on or after release all fail explicitly.
Blank observations remain missing. No alternate schema or substitute series is
tried when validation fails.

`inflation_records` supplies immutable typed records and both directions of each
codec. `forecast_before(month, cutoff_day, max_age_days)` selects only observations
strictly before the given day's midnight and rejects excessive age or a cutoff
after the actual release day. The midnight boundary deliberately excludes all
same-day forecasts; this is not a claim about intraday publication timestamps.
Pure decoding and selection need no I/O hooks; consumers use the market package's
existing direct dependency seams.

## Complete captured feed

Input: `C:/Users/Test/PROJECTS/files/inflation-feasibility-20261014/nowcast_year.json`,
SHA-256 `bf0edf1132e22c7ba05729efe6ddcd3701cd1912d50ee36f546e13ce9bcf8c46`.
The [feasibility record](INFLATION_HISTORY_FEASIBILITY.md) binds its official URL
and acquisition timestamp and explains unresolved retrospective provenance.

All 160 target months decode and round-trip through the normalized codecs:
4,687 forecasts and 156 actual releases. Missing releases remain explicit for
2013-07, 2025-10, 2026-09 and 2026-10. The full feed is accounted for; no source
record was silently skipped to obtain these counts.

Two separate offline processes produced `normalized-first.json` and
`normalized-replay.json` in the evidence directory with identical bytes,
SHA-256 `7d3c21c6f70cd808caf1bbf1ab6b5c25031ea9b0849e41ff9f756bcfc931401d`.
Each artifact includes the source and implementation digests, every normalized
month and a cutoff selection at midnight on each of the 156 observed release
days, with a maximum forecast age of four calendar days. The artifact contains
156 eligible selections and no missing selection at those declared cutoffs.
It contains both labels and forecasts for auditing; model callers must restrict training
labels to those already released at their own evaluation cutoff.

## Captured integration fixture

`tests/markets/fixtures/inflation_year_2014_01.json` is the January 2014 native
chart from that exact source capture, retaining its chart metadata, date axis
and two headline series. Other inflation series were removed explicitly;
whitespace was serialized by the shared JSON encoder. It is a source-derived
fixture, not a complete raw HTTP response.

The production decoder finds exactly 33 forecasts and the first-release value
1.5789473684 on February 20, 2014. The cutoff selector returns February 19's
forecast, 1.4688104404. Unit tests additionally exercise year rollover, missing
releases, strict age limits, dates and all named decoder failures using the real
codecs and selector.

The full package `make check` passed: 3,568 tests, strict typing across 592
files, guards, and 100% coverage of 16,932 statements and 3,364 branches.
The complete check took 102 seconds against its 300-second budget.

The next step is a frozen chronological forecast-error model and full
same-cutoff exchange quote coverage, with rounded settlement boundaries and
release-level uncertainty. No funding is needed to run that comparison.
