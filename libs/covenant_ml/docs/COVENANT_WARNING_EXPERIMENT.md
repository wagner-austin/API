# Chronological warning experiment and falsification record

As of 2026-10-14. This is an implementation validation record, not measured
predictive performance on a historical borrower cohort.

## Implemented experiment

`evaluation/warning_experiment.py:evaluate_warning_experiment` combines explicit
development folds, candidate selection, later calibration and repeated-alert
evaluation. `warning_folds.py` validates each fit/calibration/validation triple:
borrowers are disjoint across stages, earlier labels and horizons precede the
next stage, and final calibration and holdout borrowers are excluded from every
development stage. Development validation windows are strictly ordered.

Candidates use identical declared folds. Selection minimizes pooled calibrated
Brier loss over development validation rows, with equal weight per row and
lexical candidate-name tie breaking. It does not select on final outcomes.
Every candidate is then fitted on final fitting data, calibrated on final
calibration data and reported on the same final holdout opportunities.

The implemented factories are the existing real logistic learner, with explicit
regularization and no hidden shuffled split, and a fixed three-ratio rule.
Ratio thresholds and logistic settings must be declared before evaluation.
The ratio rule's raw zero/one output encodes a decision, not an estimated risk.
Platt calibration uses a separate later partition containing both classes;
one-class data and exhausted logistic iterations fail explicitly.

## Repeated alerts and calibration

All arms receive the same opportunity rows, review capacity, cooldown and
reliability bins. Selection receives scores and earlier alert history only.
Cooldown can yield different eligible subsets and unused capacity by arm;
equal capacity does not imply equal actual reviews. Actual counts are retained.

Repeated successful alerts count toward alert precision but do not multiply
unique detected first breaches. Recall's incident denominator includes breaches
inside at least one supplied prediction horizon. Earliest qualifying alerts
determine lead times. Exposure unions overlapping borrower windows and stops
at first breach; its unit is borrower milliseconds. False-alert counts are
alerts minus true alerts. This is not a recurrent-breach or recovery model.

Raw and calibrated Brier loss, equal-width reliability-bin counts, probability
sums, positive counts and longitudinal outcomes are retained side by side.
`warning_calibration_error` computes descriptive fixed-bin calibration error.
Lead-time summaries concern detected incidents only, not a paired earlier-warning
claim across different detected subsets.

## Falsification and validation

The full package `make check` passed on 2026-10-14: 4,029 tests, 18,697 covered
statements and 3,886 covered branches, zero missed, 100% coverage of src and
scripts, lint/guards and strict mypy across src/tests/scripts. The run took 120
seconds. Log: `PROJECTS/files/warning-experiment-check-20261014.log`.

Actual logistic fits, calibrator transforms and file codecs are exercised.
Changing final outcomes leaves candidate selection, development losses, raw
predictions and calibration state unchanged. Reversing rankings reduces incident
detection. Misleading final calibration labels worsen a perfect synthetic ratio
rule's Brier score from zero to greater than 0.9 and eliminate its detections.
Calibration is therefore not presumed beneficial. Hand-calculated event counts
and unioned exposure are checked independently; Brier loss also agrees with
the real sklearn metric. Overlapping identities, immature labels, contradictory
breach histories, malformed probabilities and corrupted calibration replay fail.

## Reproduction and remaining limits

Fold codecs preserve partition assignments and source-pinned rows. Forecast
codecs preserve raw probabilities and fitted calibration state and verify exact
calibration replay. Evaluation codecs cross-check cohort counts and exposure.
These records alone do not serialize learner weights or freeze arbitrary
callable factories. Full reproduction must also retain source bytes, declared
candidate settings, review policy, as-of time, bins, code commit and environment.
Archive decoding checks consistency; it does not authenticate source truth.

No verified historical borrower cohort was evaluated in this implementation.
No effect-size power requirement, clustered confidence interval or statistical
significance claim is supplied. Repeated rows remain dependent even though
selection uses row-weighted loss. Candidate choices and bins must be frozen
before a real holdout is opened, and inspecting a holdout spends that evidence.
The legacy service trainer is not replaced by this library workflow. Production
warning claims require verified disclosure availability and breach surveillance,
later borrower-disjoint evaluation and predeclared uncertainty analysis.
