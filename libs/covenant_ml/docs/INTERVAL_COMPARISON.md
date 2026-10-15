# Fixed crypto development comparison

This comparison asks whether simple recalibration improves the prior market
probability on later dates. It is exploratory historical development work.
The inventory and dates have already been inspected; they are not an untouched
holdout, and this command does not activate a prospective study or place orders.

## Inputs and boundaries

The command verifies the cohort and quote report RunRecords, every referenced
source hash, the quote-to-cohort link and the two hashes declared in the plan.
An existing report or sidecar is refused. Every declared contract must have one
quote attachment, with matching identity and decision timestamp. The shared
interval collector validates the nominal fifteen-minute grid.

The declared clock is opening plus five minutes, with prior quotes aged one to
120 seconds. Missing quotes, boundary prices and nonbinary/unresolved labels
remain explicit exclusions. Every arm uses the same resulting observations.
Settlement labels and timestamps are capture-time snapshots; historical
revisions or differences from the first available settlement are unverified.

The fixed training close-time boundary is midnight September 15, 2026 UTC.
The nominal first evaluation contract could close at that boundary, so its
decision is ten minutes earlier. Training labels must settle strictly before
that fixed instant, even if the earliest evaluation slot or label is missing.
This prevents evaluation missingness from changing training availability.

## Frozen arms

- Raw market midpoint, used for probability scoring.
- Per-asset earlier base rate, smoothed as `(yes_count + 0.5) / (count + 1)`.
- Logistic recalibration using only prior market log odds and an Ether indicator.

The logistic learner reuses the existing typed backend factory, with L2,
`C=1`, `lbfgs`, tolerance `1e-10`, seed zero and at most 2,000 iterations.
No class weighting, feature selection, hyperparameter search or additional
calibration set is used. A valid stationary initial solution can use zero
iterations; exhausted iterations are an error. Saved coefficients, base rates,
training identities and every later prediction are included in the report.

## Scores, dependence and costs

The primary contrast is logistic minus raw-market Brier loss. Negative values
favor logistic. Brier/log-loss totals weight opportunities equally; paired
intervals first average within UTC close dates, pooling both assets, then use
the existing seven-lag Newey-West estimator with equal date weights. Fewer than
nine consecutive evaluation dates or a missing whole date is an error.
This estimator does not make adjacent observations independent.

Trading diagnostics buy one YES at the ask or one NO at one minus the bid,
or pass. The shared payoff-table machinery selects the largest expected net
payoff; ties pass. Selection includes a fixed two-cent expected-value hurdle.
The hurdle is not charged again in realized payoffs.

All cost scenarios assume a 0.07 quadratic fee, rounded upward to a cent per
one-contract order, plus zero, one or two cents of extra friction. Each scenario
recomputes decisions under its own costs. Per-row integer payoffs, their integer
sum, actions and fees are retained. These are hypothetical assumptions, not
verified historical fee schedules, available depth, latency or actual fills.

## Reproduction

From the locked `libs/covenant_ml` environment:

```powershell
.venv/Scripts/python.exe -m scripts.interval_comparison --cohort <cohort/results.json> --quotes <interval-quotes/results.json> --plan <comparison/plan.json> --out <new-results.json>
```

Repeat with a new output filename to verify deterministic replay. Relocation
requires reconstructing the source reports with the existing capture archives
and binding a separately documented relocated plan to those new report hashes;
never rewrite the original plan or claim its bytes stayed identical.

The October 15 declaration at 22:00:07 UTC precedes scoring. Its plan hash is
`dea2e2db959a30371c10aae1b69b240c78bbb7bfd430cbe9aa0485fc76a5c47b`.
Training close dates are August 16 through September 14; evaluation dates are
September 15 through October 14. No performance-based revision of that split
is permitted within this declaration.

## Falsifiers

The suite executes real fits and child-process reports, hand-calculated YES/NO
payoffs, actual nonconvergence and corrupted-source failures. Changing later
outcomes cannot change coefficients, predictions or actions; changing a later
quote cannot change earlier predictions; removing an early evaluation label
cannot move the training cutoff. Duplicate identities, mismatched source hashes,
non-midnight partitions and invalid prior quote times fail explicitly.

## October 15 measured result

The frozen run trained on 5,694 contracts and evaluated 5,696 later contracts
over thirty close dates. No observations were excluded or training labels
purged. Market Brier loss was 0.19547673235604213; the earlier base rate scored
0.2499897362923675; logistic recalibration scored 0.19540278410749512.
The primary equal-date logistic-minus-market difference was -0.000074629934,
with a 95% seven-lag Newey-West interval [-0.000293984389, 0.000144724521].
The interval includes no improvement. Its reported four units are bookkeeping
blocks of at most eight dates, not four trades or four independent observations.

Market and logistic arms made zero hypothetical purchases under all three
declared cost scenarios. The base-rate arm lost $151.698 on 4,938 purchases
with zero extra friction, $200.708 on 4,793 with one cent extra friction, and
$245.938 on 4,643 with two cents. These assume one contract per selected quote,
the declared fee formula and fills at the archived sides; no live fills occurred.

The original report and independent CLI replay are byte-identical at SHA-256
`985e3dd51ee3b9c7f7dd2fd6d4ee571f78e58fef01bbe80505b29c0f75147f94`.
All 559 referenced paths verify. The native reconstruction test now re-fits
this actual panel and compares every result field to the pinned report, with
1e-10 absolute/relative float tolerance across platforms and exact discrete
values. It independently recomputes Brier losses, daily serial intervals,
every action, per-order integer fee and integer cash flow. It reuses the
existing raw archive extraction, without duplicating the evidence or selection.

This result does not establish a profitable edge. The same evaluation period
cannot become a fresh holdout after changing features or thresholds. Execution
evidence and prospective forecasting remain separate requirements.
