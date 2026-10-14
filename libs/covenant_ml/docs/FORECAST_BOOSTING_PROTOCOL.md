# Fixed boosting development comparison, October 14, 2026

This protocol is frozen before fitting the challenger. It compares whether a
small boosted model extracts more useful predictions from the existing weather
features than the existing logistic model. It is a development experiment on
previously inspected historical dates, not untouched holdout confirmation.

## Input and exposure

Use only the verified report
`C:/Users/Test/PROJECTS/files/market-edge-20261012/forecast-freshness-12-verified.json`,
SHA-256 `07962971debfbb7e6696cd003aabd62d604c0378bdb8ed193a5c14f9b275ff27`.
Verify its RunRecord and every source before decoding rows. Its existing
training partition has 643 contracts on 36 dates, August 14-September 19;
its evaluation partition has 359 contracts on 20 dates, September 21-October 10.
These are eligible observations, not the entire inventory. September 15 was
excluded by the source protocol. September 20 training labels were unavailable
at its first evaluation decision; do not silently reconstruct those rows.

Use the union of these two saved row sets, with unique contract identities and
their original timestamps, outcomes, prices and features. All dates have been
used in previous research. Selecting this source's 12:00 run is also informed
by prior work; this protocol does not undo that exposure. Forecast initialization
plus six hours is an availability assumption, not a historical receipt record.

## Fixed arms and chronological folds

Keep the raw market midpoint as a reference. Fit the existing L2 logistic model
with forecast augmentation and one LightGBM challenger to exactly the same
earlier rows. Both use the four existing WEATHER_FEATURES in their declared
order: market logit, Boston indicator, Philadelphia indicator, forecast logit.
No new transformations, variance selection, class weighting or city calibration.
Existing fixed probability clipping applies to the logits.

Use two expanding-window folds:

| Fold | Candidate training dates | Evaluation dates |
|---|---|---|
| 1 | August 14-September 20 | September 21-30 |
| 2 | August 14-September 30 | October 1-10 |

Every training row must have settlement strictly before the earliest decision
in that fold. Exclude and count unavailable labels. Keep complete event dates
together, retain every eligible evaluation row once, and report actual distinct
dates and row counts. Earlier fold outcomes may enter the later fit only after
settlement. Do not refit within a fold or change boundaries after seeing scores.

LightGBM configuration: binary objective, binary log-loss metric, CPU with one
thread, seed 0, 20 estimators, learning rate 0.05, four leaves, maximum depth 2,
minimum child samples 50, L1 penalty 0, L2 penalty 10, full row and column
sampling, no class weights, no early stopping. This is one declared candidate,
not an optimizer search. No tuning set is needed because no parameter is
selected from the data. Fitting must receive no evaluation rows or labels.
The logistic model keeps the existing fixed configuration, including C=1.

## Metrics, costs and interpretation

Primary contrast is boosted minus forecast-augmented logistic squared error,
averaged within each represented date and then equally across dates. Negative
means improvement. Report both fold results and the pooled 20-date result;
retain every prediction and per-row loss. The raw-market contrast and log loss
are secondary. Retain a date-clustered interval and a seven-lag Newey-West
interval on ordered daily means. State that twenty dates provide weak evidence;
neither interval corrects for the preceding human search or a new future regime.

Run the existing quoted taker policy unchanged: one contract per opportunity,
YES at ask or NO at one minus bid, quadratic taker fee coefficient 700 with
its existing cent rounding, and a two-cent expected-value hurdle. Report total
P&L, trade count and an additional one-cent-per-trade stress for every arm.
These are hypothetical fills without verified queue/depth/latency, historical
fee reconstruction, or a shared collateral budget. This is not a $20 portfolio.

A development lead requires lower primary mean loss in both folds, both pooled
upper uncertainty bounds below zero, and positive extra-cent simulated P&L.
Report failure of any criterion explicitly. A passing development screen still
does not establish an edge or permit funded execution. Do not select a different
feature, model, cycle, threshold or subgroup from these results. A further
candidate is a separately recorded experiment, including this one's failure.

## Execution and evidence requirements

Reuse the existing strict learner factories, row codecs, chronological label
filtering, metrics, cost policy and RunRecord writer. Add focused modules rather
than routing these observations through the generic random-split optimizer.
Test that changing later labels cannot change earlier fits or predictions, that
later features cannot affect an earlier fit, and that changing past information
does affect it. Run real learner and CLI integration tests, without mocked fits.

Record the exact learner version, implementation digest, protocol digest,
training identities, exclusion counts, hyperparameters and all predictions.
Preserve native model artifacts where supported. Record a fresh offline replay;
never overwrite an existing result. Use absolute evidence paths and a frozen
protocol copy in the experiment directory, with the same bytes as this file,
so later documentation work cannot silently change its source evidence.
The running prospective campaign remains frozen and receives no challenger.
