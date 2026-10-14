# Forecast capacity and input diagnosis, October 14, 2026

This is a new exploratory experiment after the fixed boosted model failed.
Its observed probability range was approximately 0.09 to 0.36; underfitting
is a hypothesis, not an established cause. All outer dates have already been
examined. Nothing here becomes an untouched holdout or trading authorization.

Use the identical verified source, folds, settlement availability filtering,
cost policy and raw-market/logistic references from FORECAST_BOOSTING_PROTOCOL.md.
Source report SHA-256 is
`07962971debfbb7e6696cd003aabd62d604c0378bdb8ed193a5c14f9b275ff27`.
Outer evaluation periods remain September 21-30 and October 1-10, 2026.

Within each outer training set only, reserve its last ten represented event
dates for validation. All earlier dates form inner training, purged again so
every label settles strictly before the first validation decision. Require at
least twenty represented outer training dates. Outer evaluation labels and
features must not enter candidate training or selection.

Fit the Cartesian product of 20/100/400 iterations, 4/8 leaves and absent/present
weather feature: twelve candidates per fold. Keep learning rate 0.05, minimum
child size 50, L2 penalty 10 and all other settings unchanged. Maximum tree
depth is log2(leaves). Candidate order is rounds, leaves, weather False/True;
ties select the first candidate. Select minimum equal-date validation Brier,
not profit, on these earlier ten dates. Refit that candidate on all eligible
outer training rows and evaluate once on the outer period.

Retain every candidate's native model, settings, inner-training and validation
predictions, date-mean losses, training IDs and label availability boundaries.
Also record market-only and weather-augmented logistic validation losses to
distinguish input contribution from model capacity. Do not inspect outer scores
for unselected candidates. The two outer selected models use the unchanged
three-arm scorer and cost assumptions. Report all results including failure;
no post-result expansion of this grid. No significance claim from selecting
the best of twelve candidates on ten dates. This design is a diagnostic, not
a declaration of sufficient statistical power.

Expected work: 24 candidate fits, two selected refits, four logistic validation
fits and two outer logistic fits. Persist protocol and native models with the
shared RunRecord; verify independent offline replay and future-mutation tests.
Existing frozen prospective campaigns remain unchanged.
