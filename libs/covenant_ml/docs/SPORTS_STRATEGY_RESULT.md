# Sports temperature strategy: reject advancement

The [frozen protocol](SPORTS_STRATEGY.md) produced no selected trades. Training
favored the identity exponent, 1.0, so the fitted forecast equals the market
midpoint. Paying the ask, fees and adverse movement cannot create an edge from
that forecast. Do not advance this strategy to a funded or demo trading trial.

This is a tested negative result for a specific probability calibration rule,
not evidence that sports forecasting or trading generally cannot be profitable.
It adds no team, injury, lineup, rest, travel or independent sportsbook data.
Future hypotheses require a separately frozen experiment; these evaluation
dates have now been examined and cannot become a new untouched holdout.

## Frozen choices and observed results

The protocol was written before requesting the evaluation inventory or prices.
Its exact UTF-8 SHA-256 was
`98d14721d1ddea6d3eb2bd6f6c6292e488f8bdf040cc799a175c7b045d5851cc`.
The preliminary schema probe inspected June 14 contracts, outside the study.
The first candle probe was a January 1 training contract.

| Measurement | Training | Evaluation |
| --- | ---: | ---: |
| Dates specified | Jan 1-Feb 28, 2026 | Mar 1-Apr 15, 2026 |
| Eligible observations | 233 | 128 |
| Dates represented | 40 | 39 |
| Missing exact-cutoff candles | 70 | 233 |
| Market midpoint Brier | 0.2412741416309013 | 0.242362890625 |
| Selected exponent | 1.0 | Frozen from training |
| Adjusted Brier | 0.2412741416309013 | 0.242362890625 |
| Selected hypothetical trades | Not evaluated for selection | 0 |

The archive inventory contained 3,076 contracts. Selecting one contract per
event in the frozen window yielded 676 games. One had a non-binary settlement;
eleven training outcomes settled after the frozen training deadline. Of 664
remaining requests, 361 returned eligible quotes and 303 returned no candle
ending at the required instant. No alternate time or trade price replaced them.
The large and unequal missingness means the evaluated subset is not a census
and may be unrepresentative. No missing-at-random assumption is asserted.

All six candidate training Brier scores are retained in the report. Both sample
count gates passed; the trade-count, improved-Brier and positive-profit-bound
gates failed. Zero trades produce zero hypothetical profit and an identically
zero interval for this abstaining policy, not a confidence interval proving
that all trading opportunities have zero value. Doubling assumed fees leaves
the zero-trade result unchanged.

## Reproduction

Implementation: `markets/sports_history.py`, `markets/sports_strategy.py` and
`scripts/sports_strategy.py` in `libs/covenant_ml`. Existing native candle
decoding, exact integer fee arithmetic, interval estimators, response archives,
dependency hooks and RunRecord reporting are reused. No execution API is used.

Evidence root:
`C:/Users/Test/PROJECTS/files/sports-strategy-20261013`.
`inventory/` holds four complete paginated responses; `candles/` holds 664
individual requests. Every envelope binds request URL, receipt time and body
hash. The report's source list binds all 668 envelopes. The separate `probe/`
directory records the cutoff and out-of-window schema probes.

`initial-result.json` and `verified-result.json` were produced in separate
processes and match byte for byte, SHA-256
`4842afcb8d71edd00fa92a375c405319d93b4a4136b42a08068f9cc28fdd8336`.
Both have RunRecord sidecars. This proves deterministic replay of the captured
evidence, not completeness of the exchange's historical data.

Local `make check` passed 3,460 tests, strict typing and guards, with 100%
coverage of 16,447 statements and 3,190 branches. The full check took 206 of
300 budgeted seconds; its log is `make-check.log` in the evidence root.
The 46 added tests exercise chronological separation, fee arithmetic,
abstention accounting, real HTTP acquisition and failure, offline replay,
and successful and failed subprocess execution. Earlier remote weather-check
runtime failures remain separate and are not repaired by this local pass.

From `libs/covenant_ml`:

```powershell
poetry run python -m scripts.sports_strategy --archive C:/Users/Test/PROJECTS/files/sports-strategy-20261013 --market-url https://api.elections.kalshi.com/trade-api/v2 --mode replay --out C:/Users/Test/PROJECTS/files/sports-strategy-20261013/reproduced.json
```

For fresh acquisition, use `--mode capture` and a new archive directory. Existing
requests and output files are refused. The CLI has no retries or implicit
recovery. HTTP failure or corrupt archive evidence aborts the run. Offline
replay requires every exact request; it cannot fetch missing evidence.

The cost screen assumes the declared fee coefficient rather than reconstructing
historical fee overrides. Native candles have no historical executable size.
Daily intervals assume independent dates and may understate serial dependence.
The inventory and contract metadata are retrospective. These limitations remain
even with a correct implementation and complete software test coverage.
