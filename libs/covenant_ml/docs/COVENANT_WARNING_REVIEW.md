# Fixed-budget covenant warning comparisons

## Implemented instrument

`evaluation/warning_review.py` scores a single common decision batch. The caller
provides validated `WarningRow` evidence and named model score maps. Every model
must score exactly the same borrowers, each borrower appears once, and all rows
share one prediction cutoff and horizon. At least two models are required.

`select_warning_reviews` receives only borrower identifiers, scores and the fixed
budget. It has no outcome input. Larger scores rank first; ties use ascending
borrower identifiers. Scores need not be probabilities, so a ratio rule can share
the ranking comparison. NaN and infinite scores are refused. The budget must be
positive; undersized batches select every eligible borrower.

`compare_warning_reviews` requires mature horizons, available label evidence and
noncensored labels, without filtering unfavorable or unknown outcomes. A breach
at the horizon boundary counts; already-breached borrowers are excluded by the
existing evidence decoder. It returns ranked selected borrowers, eligible and
breach counts, the requested budget, and each detected breach's lead time in
milliseconds, in detected-alert order. The result has strict encode/decode
functions and immutable fields. Impossible confusion counts are rejected.

`warning_review_rates` derives precision using actual alert count, breach recall,
and false-alert count. Recall is explicitly `None` when there were no breaches.
No redundant stored rates can drift from the underlying counts.

## Interpretation limits

This is a batch instrument, not a completed model study. The caller must freeze
eligibility and budget before outcomes and retain evidence that predictions were
available at the cutoff. The scorer cannot establish those facts from a score map.
It does not fit models, choose budgets, create cross-validation folds, calibrate
scores or open a holdout. Existing warning split validation enforces chronological
borrower exclusion; an integrated tuning and holdout workflow is still outstanding.

Lead times apply only to detected breaches. Missed breaches remain in the recall
denominator. Comparing mean lead time alone across different detected subsets is
not a paired comparison of earlier warning. Repeated batches need a predeclared
alert-suppression/event-matching policy and borrower/calendar dependence handling;
summing these batch recalls does not produce unique-event recall. False-alert
counts are not false alarms per borrower-time without an exposure denominator.
Calibration and uncertainty remain separate required study outputs.

## Validation

The 36 focused tests run production codecs, selection and comparison, including
JSON file replay. They check hand-computed results, outcome mutation, input-order
invariance, deterministic ties, unequal cohorts, invalid scores and budgets,
censoring, immature horizons, delayed evidence, horizon-boundary breaches,
undersized batches, no-breach recall and impossible serialized counts.

```powershell
.venv/Scripts/python.exe -m pytest tests/evaluation/test_warning_review.py -n 0 -q
```

These constructed-data tests establish instrument behavior, not predictive skill
on real borrowers. A verified disclosure/breach cohort is still required.
