# Logistic baseline uncertainty audit

Board task: ec1bece1-99b5-41b6-b0cf-ff5be25634b7.

## Measured result

The saved historical logistic advantage over the raw market is smaller than its
uncertainty. Replaying the exact 359 evaluation contracts on 20 dates gives a
logistic-minus-market equal-date Brier difference of -0.0000418221418453.
Negative favors logistic. The date-clustered 95% interval is
[-0.001214149115, +0.001130504831]; the seven-lag Newey-West sensitivity is
[-0.001211134070, +0.001127489786]. Neither separates an improvement from zero.

The first interval has half-width 0.001172326973, about 28 times the observed
advantage. This is a precision diagnostic, not an 80%-power minimum detectable
effect. No prospective sample-size guarantee is inferred. The sensitivity's
`n_units=3` is block metadata; there are 20 observed event dates, not three.

The logistic model's four simulated trades occurred on only three dates:
September 26, 27 and 30. Their total stays +$0.67 after the original assumed
fees and +$0.63 after an extra cent per trade. Raw market makes zero trades;
fixed boosting makes 285 for -$6.94, or -$9.79 with the extra-cent stress.
No dates, thresholds, fees, models or trades were selected by this audit.

## What was verified

The reader verifies the original report against its shared RunRecord and every
referenced source digest. Each saved fold's evaluation rows are decoded, dates
and recorded training-settlement cutoffs checked, duplicate contracts rejected,
market probabilities rebuilt from quotes and logistic probabilities replayed
from saved coefficients. No learner is fitted. The original scorer recomputes
all three arms' predictions, losses, actions, fees and cash payoffs; every fold
and the pooled scores must agree exactly with the original saved report.

Full `make check` passed: 4,092 tests, strict mypy across 677 files, all guards,
100% of 18,874 statements and 3,924 branches, in 103 seconds. Log:
`C:/Users/Test/PROJECTS/files/baseline-audit-check-20261015.log`.

The added contrast uses the same shared equal-date and paired-interval functions
as the booster comparison. Independent mean, sample-standard-error and
19-degree-of-freedom interval arithmetic also matched. Tests exercise actual
fitted models, mutated scores and probabilities, duplicate contracts, changed
chronology, real file provenance and successful/failing child processes.

## Reproduce

From `libs/covenant_ml`, with a new output filename:

```powershell
poetry run python -m scripts.forecast_baseline_audit --report C:/Users/Test/PROJECTS/files/forecast-boosting-20261014/first.json --source-sha256 53de96b6c3dcdccb1cc22ce4d26cd748be137deb205fbf7a6d1fbb197375fb74 --out C:/Users/Test/PROJECTS/files/baseline-audit-20261015/new.json
```

The actual `first.json` and `replay.json` runs in that output directory are
byte-identical, SHA256
`ca9e060d603cee5944be6c8f28eb4e171cc051303b2fd457a3d5dc333444e067`.
Each report and RunRecord replay verifies 6,088 paths, including the new report
and sidecar. The new result itself references 6,086 original evidence paths.

## Interpretation limits

These dates and model results were already examined. This is an exploratory
evidence audit, not a new holdout or multiplicity-adjusted confirmation. The
date-clustered interval assumes independent dates; the serial alternative is
only a sensitivity with a small sample. Three traded dates cannot establish
repeatable earnings. Historical quoted taker fills do not verify available
depth, latency, actual fee schedules or a collateral-constrained $20 portfolio.
The raw midpoint mechanically abstains under a spread-plus-fee hurdle; its
zero profit is not a competing executable strategy's measured return.

The earlier reports' arithmetic survives this audit. The claim that their
small positive logistic return establishes an edge does not. The frozen
prospective study remains unchanged and requires its planned outcomes.

## Independent test correction

The first arithmetic test had the same contract count on every date, so its
row mean could not distinguish equal-date weighting from contract weighting.
The independent audit correctly identified that test gap. A new regression
uses 35 synthetic held-out observations across 20 dates: five dates have one
contract and fifteen have two. Its equal-date and per-contract mean differences
differ by more than 0.01. The test independently computes each daily difference,
its standard error and its Student-t interval, and checks the preserved policy
accounting for every arm. These are test inputs, not additional market results.

The new test passed with the real implementation. Temporarily replacing daily
aggregation with per-contract input made that same test fail, reporting 35
values where 20 date means were required. The implementation was restored;
no scoring or model code change remains. Negative-control output is preserved
in `files/baseline-unequal-weight-mutation-20261015.log`. The complete final
check is recorded in `files/baseline-unequal-weight-check-20261015.log`.
That check passed 4,193 tests, strict typing and all guards, with 100% statement
and branch coverage. The earlier historical scores and source artifacts remain
unchanged; this correction strengthens verification, not measured model skill.
