# Frozen prospective stock selection

Board task: 077c3e57-8770-4028-9320-d7e32e49cb58.
The [protocol](EQUITY_PILOT_PROTOCOL.md) was committed and pushed before
rankings were computed, at revision
`31363f3683b6de092b1c396478e30f5726b87f3f`.
Protocol SHA-256:
`1c3204c2b4120d9e573a180ee0b8b8946e84292067f80252db2c3a27c2f99530`.

The actual-source calculation completed at **2026-10-14T06:03:01.814265Z**,
after all source receipts and before the declared 13:00 UTC decision.
No October 14 entry close or November 11 exit close was yet observable.
All outcomes remain explicit nulls, with zero orders and funded promotion false.

## Fixed paper selections

| Order | Symbol | Liquidity rank | Profitability rank | Score | Selected |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | NVDA | 17 | 17 | 34 | yes |
| 2 | GOOGL | 16 | 14 | 30 | yes |
| 3 | META | 15 | 12 | 27 | yes |
| 4 | MSFT | 13 | 13 | 26 | yes |
| 5 | AAPL | 8 | 16 | 24 | yes |
| 6 | JNJ | 11 | 11 | 22 | yes |
| 7 | COST | 10 | 9 | 19 | no |
| 8 | ADBE | 2 | 15 | 17 | no |
| 9 | AMZN | 9 | 7 | 16 | no |
| 10 | INTC | 14 | 0 | 14 | no |
| 11 | ORCL | 12 | 2 | 14 | no |
| 12 | CSCO | 5 | 8 | 13 | no |
| 13 | PEP | 6 | 5 | 11 | no |
| 14 | PG | 0 | 10 | 10 | no |
| 15 | TGT | 7 | 1 | 8 | no |
| 16 | CRM | 4 | 3 | 7 | no |
| 17 | IBM | 3 | 4 | 7 | no |
| 18 | WMT | 1 | 6 | 7 | no |

The six selected names have equal one-sixth weights; the benchmark holds all
eighteen equally. The resulting selection is concentrated in large technology
and communication businesses. That observation does not authorize changing
the rule after seeing the ranking. This is one declared hypothesis and one
future portfolio window, not evidence of successful stock picking.

The [machine-readable projection](equity_pilot_20261014.json) preserves every
ratio, rank, selection, weight and null outcome. It identifies the complete
evidence report rather than pretending the projection contains all source files.

## Verification and later scoring

Two independent processes wrote `pilot-first.json` and `pilot-replay.json` in
`C:/Users/Test/PROJECTS/files/stock-panel-20261014`. Their bytes are identical:
`780f599ecda6416ca52680a80808ef493d1a9240b0fcfd29a55bc2680d593d9c`.
The report verifier checked 87 paths: the report and sidecar plus 85 source
paths including the prior coverage report, its source captures, the frozen
protocol and the recorded first-computation timestamp. Replay preserves that
original timestamp; it does not mislabel the replay time as the first decision.

`equity_features.py` binds each financial pair to its issuer, accession, form,
filing date, USD unit and exact period. It checks observed receipt before
decision, annual duration, positive denominators and finite arithmetic.
`equity_signals.py` provides immutable codecs and the deterministic ranking.
Actual Apple and Microsoft observations exercise these functions in tests;
malformed pairings, invalid chronology, bad arithmetic and rank ties are tested
through the real implementation. Existing archive and HTTP boundaries are reused.

The complete `make check` passed: **3,687 tests**, strict mypy over source,
tests and scripts, all guards, and 100% of 17,427 statements and 3,520 branches.
The check took 112 seconds of its 300-second budget. Its full preserved log
is `make-check-pilot.log` in the evidence directory; PowerShell printed the
requested final 100 lines and propagated the actual exit status.

The [official exchange calendar](https://www.nyse.com/trade/hours-calendars)
lists no planned exchange holiday in the October 14–November 11 interval.
The frozen rule requires all 21 expected weekday bars and exact entry/exit
dates. The regular close changes its UTC offset with daylight saving time;
use session dates and the actual native timestamps, not a fixed UTC close hour.
Unexpected closure, missing data, suspension or delisting must be reported
as incomplete under this protocol rather than replaced or discarded.

Later scoring must preserve all eighteen outcomes, apply the fixed 0.002
round-trip cost and 0.005 sensitivity, and compare with the gross all-cohort
benchmark. The cost figures are assumptions, not measured execution costs.
No paper outcome or live-trading readiness is established today. No new
scheduled collector was installed by this snapshot; actual future acquisition
and scoring remain explicit work, with the frozen records already available.
