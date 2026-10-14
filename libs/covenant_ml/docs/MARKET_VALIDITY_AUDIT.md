# Market research validity audit, October 14, 2026

This is a targeted audit of the market study entry points, shared chronological
splitter, and general optimizer feature path. It is not certification of every
model, data provider, deployed service or historical optimization run.

## Confirmed defects and repairs

1. `evaluation/splits.py`, `ChronologicalGroupedSplits`: sorting rows by time
   did not keep equal time keys together at partition boundaries. A 15-row
   reproduction placed date 4 in both training and validation. Even twenty
   rows with one identical time key produced three nominal partitions.
   Three regression cases failed before repair. The factory now partitions
   distinct sorted keys, expands masks to entire groups, and embargoes whole
   keys. Fractions and embargo are measured in keys, not rows or elapsed time.
   Label availability still requires correctly chosen keys and an adequate
   embargo; grouping alone does not prove it. The lower-level row-mask utility
   retains its explicitly row-based contract.
2. `features.py`: capped ratios/products selected columns by variance over the
   supplied matrix. General optimizer objectives engineered that matrix before
   splitting it. Appending future rows changed unchanged past ratios from
   `a/b, b/a` to `d/c, c/d`, and products from `a*b` to `d*c`. No future labels
   were necessary: held-out feature distributions influenced the representation.
   Two new regression cases failed before repair. Capped pair generation now
   follows declared column order, independent of all row values. Past-value
   positive controls still change the resulting numeric features.

The second repair deliberately removes learned variance selection; it does not
claim fixed-order pairs are the most predictive pairs. An explicit training-only
learned selector would be a separately evaluated model component. Models using
the old variance-selected capped pairs must be retrained with their exact new
feature names/order; old artifacts and optimization scores are not upgraded by
changing the code. Existing uncapped
pair order and row-local arithmetic are retained.

Call-site review found no use of `ChronologicalGroupedSplits` or the general
`engineer_features` path in `markets/`. These bugs therefore do not explain or
invalidate the current market results by themselves. The wider optimizer's
random stratified partitions are also unsuitable for a forward market claim:
keeping companies separate, or preserving class balance, is not time separation.
For example, `optimizer/objectives/lightgbm_objective.py` fixes a random 70/15/15
partition and optimizes validation AUC. That validation score is a selection
score, not untouched final-test performance.

The reviewed pre-split feature-engineering callers are the classifier objectives
for LightGBM, XGBoost, ClearGBM, logistic regression and random forests, plus
the LightGBM and XGBoost regressor objectives. The leak requires the old
variance-selection branch to be active; a positive cap that exceeds the number
of pairs, `none`, or the row-local log-only preset does not trigger that branch.
This audit does not quantify how much a particular old optimization score moved.

The separate `validation/strategies/time_series.py` strategy splits by input
row position and declares `supports_groups=False`. Its caller must establish
chronology; it receives no timestamps or label-availability times. Its broad
docstring about preventing leakage is not a guarantee for repeated-date panels
or overlapping outcome horizons. Do not substitute that strategy for the
explicitly grouped and purged design required here.

## Reviewed market paths and residual risks

| Path | Protection checked in code | Remaining limitation |
|---|---|---|
| Weather logistic comparison | `forecast_model.chronological_split` keeps whole event dates apart and requires training settlement before first test decision; fitting receives only training rows | Dates already inspected in maker study; later cycle and city-calibration variants reuse evaluation data |
| City calibration | `calibration_evaluation` fits city bias/scale only on the earlier partition | 375 bias/scale combinations per city; calibration and logistic stage share training labels, rather than out-of-fold stacking; outer chronological test remains separate |
| Sports calibration and ratings | `split_sports` checks settlement deadlines; exponent/blend selection uses training games; ratings update only on settlements preceding prediction | Finalized binary inventory and eligible candles define a selected subset; repeated later-date use is exploratory |
| Inflation | `earlier_errors` excludes same/later target months and releases at/after cutoff; one selected contract per release, release-weighted scores | Retrospective chart values and calendars do not prove actual publication receipt or unrevised vintages |
| Maker replay | Prior quote selection and strictly later trade-through prints; duplicate identities rejected; date-cluster and calendar serial intervals | Trade-through is a fill proxy, candle-end time is not book freshness, historical fees are assumed, partial fills and capital competition absent |
| Equity paper pilot | Filing acceptance and observed receipt precede decision; frozen cohort retained; all declared outcome dates required | Selected modern cohort is not a historical survivorship-free universe; prospective return has not matured |

No direct future-label fitting was found in these inspected market paths.
That is narrower than saying the whole research process is free of leakage.
Tests that change future labels cannot establish historical provider availability,
untouched human model selection, or realistic execution.

## Claims that must remain qualified

- Weather `candidate_gate` tests a clustered Brier interval, not profitability;
  its daily serial diagnostic is separate. It is not a live-trading approval.
- Maker `candidate_gate` uses the per-fill date-clustered lower bound alone;
  it does not require positive extra-friction profit or the calendar serial
  lower bound. Read those diagnostics together; the flag is only a screen.
- Sports and inflation profit intervals assume independent represented dates.
  Their reports disclose this, but serial dependence can make uncertainty larger.
- Group/arm comparisons are exploratory unless declared before independent
  evaluation. The seven maker groups have no multiplicity correction.
- A SHA-256 and reproducible replay establish identity and repeatability, not
  that a source was available at the simulated decision or a strategy generalizes.
- The historical inflation experiment had **30 selected release dates before
  quote filtering but only 14 eligible dates**, with 41 contracts and 12 trades.
  Calling its evaluated sample 30 releases would overstate the evidence.
- A passing test suite and full coverage do not prove statistical validity.
  Both defects above existed alongside previously passing tests.

## Models actually used

The market maker is a fixed rule. Weather uses L2 logistic regression; city
calibration uses a fixed grid; sports uses probability-temperature calibration
and sequential Elo-style ratings; inflation uses an empirical earlier-error
distribution; equities use declared financial-ratio ranks.

The broader library and Covenant Radar expose LightGBM, XGBoost and ClearGBM.
Their presence is not evidence that a boosted model produced the market results.
Boosting is a reasonable future challenger, not a prerequisite. Compare a
bounded, preregistered model against the same baseline, features, event groups,
cost assumptions and untouched outer test. Tune and early-stop only inside
earlier training/validation folds; fit preprocessing there too. Reject random
row splits for this use. Preserve every tried model and score, not just winners.

## Historical holdout eligibility

Reading only ticker identities from verified existing maker inventory envelopes
found 2,190 Philadelphia contracts dated 2025 and 252 dated 2024. The 2026
inventory has 1,698 Philadelphia, 1,488 Boston and 1,602 Washington contracts.
These are inventory counts, not counts with complete quotes or trades.

The 2026 Jan 1-Oct 10 maker outcomes, weather evaluation dates, and sports
Mar 1-Apr 15 outcomes have already been inspected. They cannot be relabeled
untouched by splitting or hiding their labels now. The older Philadelphia
inventory was captured previously, including native outcomes; absence of a
score in the reviewed reports is not proof nobody inspected those outcomes.

A 2025 Philadelphia evaluation of a hypothesis discovered in 2026 is a
**retrospective temporal transfer test**, not forward out-of-sample confirmation.
It is still useful for falsification and stability. Public-paper overlap also
needs to be retained as a limitation. A later independent holdout requires its
own exposure record and no model changes after opening it. The current frozen
prospective campaign supplies a separate future window and remains unchanged.

For subsequent walk-forward model comparisons, freeze date boundaries, event
groups, label availability, purge/embargo rules, features, search budget, costs,
missingness treatment and primary metric before evaluating. Use nested earlier
validation for tuning, evaluate each outer fold once, and retain failed folds.
Rolling refits may use outcomes of earlier folds only once actually available.

## Verification and scope of the delivery

Full `make check` passed 3,847 tests, strict mypy over 639 files and all guards,
with 100% of 17,897 statements and 3,654 branches in 133/300 seconds.
Five new regression cases failed on the original implementations. One obsolete
test for a removed indexing helper was deleted; numerical feature tests remain.
The initial full check caught strict test-array typing errors, which were fixed
before this final passing run. Log: `market-edge-20261012/leakage-audit-final-make-check-20261014.log`
under `C:/Users/Test/PROJECTS/files`.

The changed source files have 240 and 355 lines. The changed/new test files
have 117, 554 and 34 lines; the new causality tests are a separate concern.
No new network client, model trainer, compatibility path or testing harness
was introduced. Existing generic feature sanitation behavior was not redesigned
by this targeted leakage repair.

An actual offline replay of the original 2026 archive after both repairs
preserved all `results` and `pre_entry_regimes` exactly. Its report is
`market-edge-20261012/audit-replay-20261014.json`, with 5,910 paths verified and
SHA-256 `d0316ec9f43dbaab2de841c5289196c7fa407d5e63d0bd81627bf5a6f86102da`.
The new implementation digest is
`61732c1206c2c0c3cb9f3cd2edc4c0388c48f3feed2c01aeee4ad9cfc7a50e02`.

The [2025 transfer protocol](MAKER_TEMPORAL_TRANSFER_PROTOCOL.md) was committed
before new price/trade acquisition. It tests a fixed diagnostic rather than
searching for a better-looking year or relabeling the existing sample a holdout.
The [completed transfer result](MAKER_TEMPORAL_TRANSFER_RESULT.md) processed
2,190 contracts and independently replayed identical report and RunRecord bytes.
Its declared high-variation/falling group lost $5.61, or $7.00 with additional
friction, and did not pass either uncertainty check.
