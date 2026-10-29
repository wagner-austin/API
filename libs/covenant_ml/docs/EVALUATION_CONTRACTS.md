# Evaluation instruments and their limits

The shared evaluation package owns these instruments. The CRSP monthly study
must integrate them with its frozen holdings and data rules; passing their
regressions does not establish portfolio performance or holdout isolation.

| Instrument | Implemented contract | Independent check |
|---|---|---|
| Paired comparisons | Subtract observations on shared rows. Cluster means retain the row-weighted estimand with cluster-robust errors; time-series means can use Bartlett Newey-West errors. | `test_numerical_contracts.py` unequal-cluster hand calculation; paired and AUC tests compare independent formulas. |
| Confidence intervals | Validate finite estimates, nonnegative finite errors, confidence and sample size before constructing intervals. | Analytic Cauchy quantile at one degree of freedom; zero-error and malformed-input refusals. |
| Ranking metrics | One shared midrank kernel gives tied positive/negative pairs half credit. Average precision groups distinct score thresholds. AUC needs both classes; average precision needs positive cases. | Pairwise enumeration, row permutations, constant-score AUC 0.5 and AP equal to prevalence, mixed-tie AP 7/12. |
| Feature ablation | Drop exactly the named feature columns from each partition, refit the real learner with paired seeds and unchanged labels, and compare to the full arm. | Real LightGBM on planted signal/noise data: removing signal reduces AUC; removing noise has bounded effect. Unknown, empty and all-feature removals fail. |
| Learning curves | Each seed uses nested random prefixes of its training fold; validation and scoring rows stay fixed. | Exact prefix inclusion and unchanged held-out arrays, plus real fits at each fraction. This is sample-size behavior, not an expanding-history curve. |
| Chronological splits | Partition distinct time keys and purge training labels unavailable before stopping decisions and stopping labels unavailable before scoring decisions. Embargo is measured in distinct keys. | Actual label endpoints cross both boundaries in `test_temporal.py`; reordering observations preserves their availability. The WRDS pilot supplies next-observation dates. |
| Moving-block uncertainty | Sample uniformly from overlapping, noncircular blocks; truncate only the final draw to retain exactly n months. Both arms share weights and must have identical consecutive month identities. | Enumerate all small-sample block combinations, hand-derived percentile endpoints and variance, paired market-movement cancellation, and exact-count repeatability. |
| Provenance | The WRDS adapter verifies query and CSV bytes; shared staging binds exact byte lengths and hashes; RunRecord preserves numerical observations and runtime/configuration identities. | Changed query/data/manifest/code/image refusals, real written-bundle readback, round trips and incompatible-record comparisons. |

`tests/evaluation/`, `tests/test_metrics.py`, `tests/test_ranking.py`,
`tests/markets/test_research_trial.py` and `test_research_execution.py` contain
the executable checks. `libs/platform_core/tests/test_run_record.py` verifies
shared observation and comparison records. Fleet `make check` includes the
whole affected package, strict typing, guard rules and statement/branch coverage.

## Interpretation limits

- DeLong assumes independent observations. Cluster methods need meaningful
  independent clusters; correlated market dates do not become independent by
  assigning names to them. Newey-West and moving blocks encode particular
  dependence assumptions, not a universal correction.
- The moving-block percentile interval uses the observed empirical series.
  End observations appear in fewer noncircular blocks. The 12-month choice
  in the CRSP protocol is fixed before outcome inspection. Thirty-six final
  months provide limited information; `n_units` counts complete block lengths
  fitting in the history, not independent market histories.
- The shared percentile record's `p_value` is an uncentered bootstrap sign-tail
  diagnostic. It is not a calibrated null-bootstrap significance test. Do not
  use it as a formal rejection rule or confuse draw count with sample size.
- Across-seed intervals apply a Nadeau-Bengio-style variance correction and a
  within-seed-error floor. This is a declared heuristic for repeated fits;
  seeds do not supply independent historical samples. Monthly inference must
  operate on aligned monthly outcomes, not treat seeds as additional months.
- Ablation estimates the effect of removing features under the tested fitting
  procedure. It does not identify causal importance. Learning curves are
  conditional on the chosen learner, settings and sampled training prefixes.
- A RunRecord binds declared evidence and comparisons. It does not itself
  reread every source file, prove historical vendor availability or protect
  against an administrator who can rewrite the evidence. Actual-byte checks
  belong to the extraction, adapter and staging boundaries. Filesystem/tool
  isolation of research proposers is a separate requirement.

Historical reports produced by the earlier tied-score implementations are
not silently rewritten. A rerun must retain its new code and metric identities;
unchanged data alone cannot establish equivalence to those older scores.
