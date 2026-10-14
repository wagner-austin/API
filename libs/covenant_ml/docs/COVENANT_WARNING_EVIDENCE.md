# Covenant warning evidence boundary

This implements the dated-data prerequisite from
`EVIDENCE_REQUIREMENTS_AUDIT_20261014.md`. It does not repair the existing
repository trainer by pretending its undated records are historical vintages.
It does not fit a model or establish useful warning. The new code is in
`covenant_ml.evaluation.warning_records` and `warning_panel`.

## Evidence contract

`WarningDisclosure` retains borrower identity, financial period end, actual
availability timestamp, source SHA256 and three finite ratios: debt/EBITDA,
interest coverage and current ratio. Availability must not precede period end.
The caller must archive the source bytes and establish availability from evidence;
a digest field alone does not prove either assertion. Negative earnings ratios
are permitted; missing and nonfinite ratios are rejected, never imputed.

`select_warning_disclosure` considers only the requested borrower and vintages
available at the cutoff. It chooses the latest financial period and then that
period's latest available revision. A newer revision of an older period cannot
replace the latest period. Ambiguous selected vintages are rejected. Adding or
changing future vintages cannot alter the earlier selection.

`WarningFollowup` separately retains a complete surveillance interval, the first
contractual breach within that interval, when this outcome evidence was available,
and its source SHA256. Contractual breach time is not the later disclosure time.
The source must establish complete coverage, contractual definitions and borrower
identity; silence in an incomplete filing collection is not a negative label.
The study must declare the at-risk origin and incident-event population. These
codecs do not prove absence of breaches before the asserted coverage start.

`WarningRow` combines the two evidence records with a prediction cutoff and
horizon end. Its decoder rejects future predictors, mismatched borrowers,
follow-up that does not establish at-risk coverage at the cutoff, and breaches
at or before the cutoff. `warning_label` returns one only for a breach in
(cutoff, horizon end], zero only for a fully observed breach-free horizon, and
None for incomplete follow-up. A positive event can be labeled before horizon
end, but the split validator still requires the planned horizon to mature.

`require_warning_split` rejects empty partitions, repeated borrower/cutoff
opportunities, borrowers shared between training and evaluation, different
forecast horizons, censored labels, and unavailable outcome evidence. Every
training horizon and its label evidence must precede the first evaluation
cutoff strictly. Evaluation horizons and evidence must be available by the
explicit analysis timestamp. No rejected row is silently removed.

All three TypedDict records use ReadOnly fields and explicit encode/decode
functions. This layer is deterministic and performs no network, database,
clock or model access. Callers supply evidence and timestamps directly; no
dependency is replaced by a test fake. It is part of the existing strict
package harness, not a separate untyped experiment.

## Native data gap

The inspected domain `Measurement` and `CovenantResult` types have period dates
but no availability or vintage fields. The PostgreSQL measurement repository
inserts measurements and rejects duplicates; the result repository upserts
computed results, overwriting prior values/statuses. This corrects the earlier
conversational statement that both repositories overwrite. There is no justified
adapter from these records to point-in-time warning evidence. The old trainer's
ever-recorded-breach target remains unsuitable for an advance-warning claim.

Before a real cohort can use this boundary, acquisition must retain disclosure
vintages, contract definitions/effective dates, borrower links, dated breach
events and complete follow-up evidence. WRDS access and entitlements remain
unverified; even an available dataset must be checked for these fields. No
historical availability timestamp was invented to make a cohort pass.

## Validation design and top-k

Code validation and scientific validation answer different questions. The
regression tests execute actual codecs, selection, labeling and partition
validation on constructed examples. An integration test writes and reloads
serialized evidence and rejects a damaged record. These tests verify the
instrument, not predictive performance on real borrowers.

A real model study should predeclare a development period and untouched final
later-period cohort. Tuning uses expanding-time folds within development, with
labels matured before each next fold and borrower identities excluded from its
training. The final cohort's borrower identities must be excluded from all
development and tuning, not merely the final fit. Calibration, imputation,
feature selection and thresholds must be fitted inside those same boundaries.
Overlapping horizons must be purged at each fold boundary. Random row splits
cannot answer the requested later-period, unseen-borrower question.

Top-k is an operational metric, not a validation split. Freeze the analyst's
review budget per decision batch before outcomes: rank each arm's same eligible
borrowers, use a declared outcome-independent tie rule, and report precision@k,
breach recall, realized alert count and warning lead time. If fewer than k
borrowers qualify, report the actual denominator rather than inventing reviews.
Also retain calibration, false alarms per borrower-time and missed breaches.
Define repeated-alert suppression and event matching before scoring so many
warnings about one borrower cannot inflate success. Compare ratio/headroom
rules, logistic and any justified challenger at the same budget and dates.

Do not choose k, horizon, fold boundaries or alarm threshold by maximizing the
final cohort's result. Multiple development folds are correlated observations,
not extra borrowers; uncertainty must respect borrower and calendar dependence.
Choose the minimum worthwhile improvement and sample requirement before the
confirmatory cohort is inspected. The actual horizon, review budget and cohort
size are deliberately not invented here; these are proposed study requirements,
not an already frozen protocol or implemented top-k scorer.

## Reproduction

From `libs/covenant_ml`, run the real regression tests with:

```powershell
.venv/Scripts/python.exe -m pytest tests/evaluation/test_warning_records.py tests/evaluation/test_warning_panel.py -q --no-cov -o addopts=''
```

Run the full package `make check` for strict typing, guards and source/script
statement and branch coverage. The synthetic examples establish no real borrower
performance, lead-time benefit, false-alarm rate or trading edge.

On October 14, all 59 focused tests passed. Full `make check` then passed
3,927 tests, strict mypy and all guards, with 100% coverage of 18,261 statements
and 3,748 branches in 127 seconds. The real check log is
`C:/Users/Test/PROJECTS/files/warning-boundary-check-20261014.log`.
