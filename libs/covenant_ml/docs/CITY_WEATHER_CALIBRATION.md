# City weather calibration experiment

This exploratory experiment tests whether city-specific forecast bias and uncertainty
improve the existing fixed normal distribution with standard deviation 3°F. It does
not alter the frozen October 14–December 12 prospective campaign.

The design was declared before running this calibration on the saved observations:

- Primary input: `forecast-freshness-12-verified.json`; 00 UTC is secondary.
- Preserve the original chronological partition: August 14–September 20 training,
  September 21–October 10 evaluation, excluding September 15 as originally specified.
  Labels unavailable before the first evaluation decision remain excluded.
- Fit each city's normal mean offset on −6 to +6°F in 0.5°F steps, and standard
  deviation on 1 to 8°F in 0.5°F steps: 375 candidates per city.
- Minimize mean daily binary-contract Brier loss, weighting represented city dates
  equally. Require at least ten training dates for every city. Ties choose smaller
  bias, then smaller standard deviation. No fallback model or extra search.
- Use native rounded Fahrenheit bracket boundaries and binary settlement labels;
  do not treat a winning bracket midpoint as the observed temperature.
- Compare fixed weather, calibrated weather, market midpoint, calibrated market,
  and market plus weather. Refit each logistic model only on earlier dates using
  the existing evaluator. Report all arms, including unfavorable results.
- Keep the existing hypothetical one-contract quoted taker policy, 700 basis-point
  quadratic whole-cent-rounded fees, two-cent expected-value hurdle, and additional
  one-cent adverse execution diagnostic. These are assumptions, not observed fills.
- Report date-clustered paired Brier intervals. These later dates were inspected in
  prior experiments, so nominal intervals do not correct for repeated exploration
  and cannot establish a confirmed trading edge.

`scripts.forecast_calibration` verifies the original report, RunRecord and every
source hash. Inventory requests must be part of that original evidence. It reconciles
each observation's identity, settlement, timing, rule and fixed-model probability
against archived native market metadata. The new report retains all evidence hashes,
the implementation digest, declared grid, fitted parameters and both full evaluations.

Run from `libs/covenant_ml` using the actual source report and archive:

```text
poetry run python -m scripts.forecast_calibration --report <original-report.json> --markets-archive <integrated-raw> --market-url https://external-api.kalshi.com/trade-api/v2 --out <new-calibration-report.json>
```

The command is offline and refuses existing output paths. Tests execute the real
collector against a local HTTP server, verify offline repeatability, run the CLI in
a subprocess, reject changed and unbound evidence, and prove held-out labels cannot
change fitted parameters or predictions.

## Measured results, October 13, 2026

Both cycles used 643 training contracts across 36 dates per city and 359 evaluation
contracts across 20 later dates. Positive bias means adding degrees to the forecast.

| City | 12 UTC bias / sigma (°F) | 00 UTC bias / sigma (°F) |
| --- | ---: | ---: |
| Boston | +2.0 / 1.5 | +2.0 / 2.5 |
| Philadelphia | 0.0 / 2.0 | 0.0 / 2.5 |
| Washington | −1.0 / 2.5 | −1.0 / 2.5 |

Lower Brier is better. Dollar amounts below are aggregate hypothetical P&L after
quoted spreads and assumed fees, not returns on a funded portfolio or actual fills.
The final column additionally charges one cent per executed contract.

| Cycle | Arm | Brier | Trades | P&L | Extra-cent P&L |
| --- | --- | ---: | ---: | ---: | ---: |
| Both | Market midpoint | 0.106782 | 0 | $0.00 | $0.00 |
| Both | Calibrated market | 0.106781 | 10 | $0.00 | −$0.10 |
| 12 UTC | Fixed weather | 0.139765 | 259 | −$10.33 | −$12.92 |
| 12 UTC | City-calibrated weather | 0.137083 | 213 | −$6.70 | −$8.83 |
| 12 UTC | Market + fixed weather | 0.106612 | 7 | +$0.97 | +$0.90 |
| 12 UTC | Market + city-calibrated weather | 0.107072 | 11 | −$0.76 | −$0.87 |
| 00 UTC | Fixed weather | 0.141816 | 267 | −$6.94 | −$9.61 |
| 00 UTC | City-calibrated weather | 0.140190 | 252 | −$9.10 | −$11.62 |
| 00 UTC | Market + fixed weather | 0.106839 | 11 | −$0.48 | −$0.59 |
| 00 UTC | Market + city-calibrated weather | 0.106847 | 13 | −$0.51 | −$0.64 |

For the primary 12 UTC combined model, calibrated minus fixed Brier was +0.000460,
with a date-clustered 95% interval of [−0.000679, +0.001599]. Relative to the calibrated
market baseline, the new combined model's difference was +0.000290, interval
[−0.001071, +0.001652]. Neither establishes improvement. The secondary combined
model likewise failed the existing improvement diagnostic.

**Result: reject promotion of city calibration on this evidence.** Weather-only
scores improved slightly, but remained much worse than the market baseline; the
combined predictions deteriorated. The positive seven-trade fixed-model result
is exploratory and does not establish a profitable strategy. Keep the original
prospective study frozen, and do not expand the grid to chase these evaluation dates.

Reports and adjacent RunRecord sidecars are under
`C:/Users/Test/PROJECTS/files/market-edge-20261012/`:

- `city-calibration-12-20261013-verified.json`, SHA-256
  `e62ff3e7880c512a6cd756c68de0d2cb26cfa47215d58fc8e27655940427baf3`.
- `city-calibration-00-20261013-verified.json`, SHA-256
  `6333dfa3cda45590a1a5233d08489fd44d4a4da891be7951bbdb643bc8e570e5`.

Both final-code runs reproduced every fitted parameter, prediction, score,
paired interval and economic diagnostic from their initial runs exactly.

Validation: `make check` passed all 3,375 tests, strict mypy and repository guards,
with 100% coverage of 16,070 statements and 3,080 branches in source and scripts.
The full check took 213 seconds within the existing 300-second budget. Its log is
`calibration-final-make-check.log` in the same evidence directory.
