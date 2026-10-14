# Warning history consistency audit

Measured 2026-10-14 against the chronological warning experiment. The prior
published implementation at a8af2e00176527891f96e1e7e74889bdad7d02a0 passed
4,029 tests locally and on serendipity. That did not establish completeness
of its validation rules.

## Reproduced gap

Two observations of one fitting borrower could declare different first-breach
dates and still pass `require_warning_fold`. Dates, label availability and
borrower-disjoint stage boundaries were individually valid. The existing
longitudinal alert scorer rejected contradictory histories, but candidate
fitting and development scoring did not perform that check.

Seven regression cases failed before the correction: different first-breach
dates and event-free surveillance contradicting a known breach, each in fitting,
calibration and validation, plus conflicting evidence reused between otherwise
individually valid development folds. These cases executed the real study;
the expected refusal did not occur. No historical data result was involved.

## Correction

`evaluation/warning_history.py:require_warning_history` now owns the shared
evidence consistency rule lifted from alert scoring. Fold validation checks
all three partitions; study validation also checks evidence across development
folds and final fitting/calibration/holdout partitions before invoking a learner.
Alert scoring uses the same implementation. Contradictions propagate WARN-024.

An earlier observation with no observed breach is valid when its surveillance
ends before the later known breach. A positive regression case verifies this,
including reversed row order and identical repeated evidence. Duplicate
opportunities inside a partition remain the separate partition validator's job.

Conflicting source revisions require adjudication before constructing the
study. The code does not silently select a preferred revision. This check
validates consistency, not the accuracy or completeness of the source records.

## Verification

`make check` passed: 4,037 tests, 18,712 statements and 3,890 branches,
zero missed and 100% coverage of src and scripts; lint, guards and strict mypy
cover src/tests/scripts. Total time was 103 seconds. The local log is
`PROJECTS/files/warning-history-check-20261014.log`.

The new tests are `tests/evaluation/test_warning_history.py`; existing study,
alert, calibration and replay tests remain in the full suite. Coverage measures
executed paths, not all scientifically necessary assumptions. Historical
cohort acquisition, frozen model comparison and dependence-aware uncertainty
remain separate evidence requirements; this correction establishes no edge.
