# Evidence requirements audit, October 14

This is an audit of evidence, not a new model or a declaration of an edge.
The inspected implementation is API commit
`9651bd88ce24e666cf734d096bc40647c1322103`. Existing prospective protocols
and collectors were not changed. Board task:
`1ca6f19b-3701-4f57-a280-24fd129c0875`.

## Prospective status

The real `scripts.paper_campaign_report` evaluator completed at
2026-10-14T20:06:08.705673+00:00. Its report is
`files/market-edge-20261012/campaign-evaluation-20261014-audit.json` under
`C:/Users/Test/PROJECTS/`. The evaluator verified original plans, prediction
seals and collector identity. A separate SHA256 pass verified all 520 entries
in its `sources` array. It recorded 12 complete acquisitions and 708 future
acquisitions; both forecast cycles have zero eligible settled dates and
contracts, and null aggregate scores. Twelve captures cover two event dates,
three cities and two forecast cycles, not twelve independent outcomes.

The original protocol fixes 60 event dates, October 14 through December 12,
with settlement snapshots two days later; the final scheduled window is
December 14. It freezes raw_market, raw_forecast, market_calibrated and
market_plus_forecast arms. The primary comparison is the 12 UTC augmented
model minus calibrated-market Brier score, averaged within event date and
then equally across dates. Missing or failed windows remain omissions.

The protocol explicitly describes an instrumentation pilot and probability
forecasting only. It does not promise statistical significance, profitability,
fills, or a detectable effect at the planned sample size. Calling it a powered
trading trial would misstate its frozen design. Boosting is not a prospective
arm; it cannot be inserted retrospectively. Finish and report the entire
window, including missingness, without selecting a favorable stopping date.

Reproduce with the command in `CAMPAIGN_EVALUATION.md`, the original manifest
digest `4f350cf553c68f6f3f467b59b6f92a8a8c24cf07ddada0c0527224ace17cc077`,
and a new output path. A later observation will have different acquisition
dispositions; that is not a byte-identical replay of this timestamp.

## Comparable baselines and independent observations

The historical fixed and selected boosting reports compare raw market,
market-plus-forecast logistic and boosting on identical later rows, cost
formula, one-contract sizing and decision rule. The selection study retains
all earlier validation candidates. Neither boosting study passed its gates.
These reused historical dates are exploratory, not another independent test.
See `FORECAST_BOOSTING_RESULTS.md` and `FORECAST_SELECTION_RESULTS.md`.

The prospective pilot includes a raw-forecast reference as well as both market
references. Its score comparison does not measure trading returns. A future
economic protocol must explicitly include the simple forecast-only model,
market baseline and any justified challenger with the same opportunity set,
fees, bankroll, position limits and execution rule. Opportunity counts and
abstentions must be retained for every arm, not just its selected trades.

There is no established power-based sample requirement for the current pilot.
Before a separate confirmatory study: declare the smallest worthwhile paired
improvement, primary metric, error rate and desired power; estimate variability
and serial dependence using development dates only; calculate required event
clusters under conservative dependence/missingness assumptions; freeze those
choices before observing that study's outcomes. Cities, forecast cycles and
contracts sharing an event must not inflate the sample. More parameter runs
do not increase the number of independent dates. An effect threshold chosen
after seeing the prospective result would itself be exploratory.

## Execution evidence

The campaign retains books and sizes but represents no execution. Historical
boosting uses a stated fee assumption and extra-price stress, not demonstrated
historical depth, actual fees or fills. The demo order establishes order and
reconciliation mechanics, not production liquidity or live expected returns.

A separate execution study must bind each decision to the exact available
quote, side, size, exchange/receipt/decision/send/ack timestamps, fee schedule,
order constraints, fills, partial fills, rejects and cancellations. It must
report unfilled opportunities and observed latency, adverse movement and
capital occupancy. A later candle touch cannot certify an earlier fill.
Neither positive Brier skill nor a simulated profit clears this requirement.

## Covenant Radar warning validity

Concrete defect in the warning interpretation: in
`services/covenant-radar-api/src/covenant_radar_api/worker/train_job.py`,
`_build_training_data` creates one row per deal, labels it positive if ANY
result has status BREACH, and passes ALL results to feature extraction.
`_extract_features_for_deal` uses the latest sorted measurement periods.
There is no prediction cutoff or future outcome horizon in that construction.
Consequently a breach may precede or coincide with its supposed predictors.
This target is ever-recorded breach classification, not future warning.

The same job calls `train_model_with_validation` without borrower groups.
`libs/covenant_ml/src/covenant_ml/trainer_fit.py` delegates to
`trainer.stratified_split`; absent groups, that implementation shuffles rows
within label classes. This is not a later-period, unseen-borrower evaluation.
The domain helper `_count_near_breaches` in
`libs/covenant_domain/src/covenant_domain/features.py` slices the first four
results, assuming ordering; it does not enforce four distinct periods.
These are source-level findings, not newly measured predictive losses.

Current AUC/accuracy cannot certify lead time, future breach discrimination or
an acceptable false-alarm rate. Existing default/bankruptcy datasets do not
automatically supply covenant-specific breach dates and terms. Do not relabel
those scores as warning validation. A replacement study needs borrower IDs,
contract definitions and effective dates, original disclosures and availability
timestamps, dated breach outcomes, and explicit right-censoring rules.

Construct each borrower-period snapshot using only information available at
its cutoff. Place outcomes strictly after the cutoff within a frozen horizon;
exclude already-breached cases according to a declared incident-event rule.
Keep borrowers disjoint and evaluation dates later than training, with label
maturity enforced before fitting. Freeze ratio/headroom and logistic baselines,
alarm thresholds and repeated-alert policy on development data. Measure warning
lead time, event recall, precision and false alarms per borrower-time at the
same alert budget; retain calibration and borrower-cluster uncertainty.
Future-data mutation tests and native-source integration tests must accompany
that construction. No such future-warning result is established by this audit.

## Provenance coverage and remaining portability gaps

Recent market studies retain protocols, hashes, raw-response archives, input
receipts, implementation fingerprints, fitted native models, exclusions and
RunRecords. Fixed/selected boosting archived replays reproduced their measured
outputs, and the campaign report re-verifies bound evidence. Wiki pages pin
copied sources and cite locators. These are concrete strengths, not proof that
every older research artifact has been inventoried or independently rerun.

Some manifests contain absolute workstation paths. A wiki copy of a report
does not itself include every transitive raw input or native model. A clean
machine must receive the full dependency closure and environment, not merely
the Markdown page. Local clocks are not independent timestamp notarization.
Hash equality proves byte identity, not that inputs were available historically
or that a study avoided selection bias. A green wiki audit checks structural
citations and pins; claim support still requires reading the cited source.

Universal reproducibility is NOT certified. Remaining work is a per-study
inventory and portable dependency export, documented environment and exact
replay commands, plus a clean-machine offline replay with expected numerical
tolerances and failure tests for missing/tampered evidence. Preserve negative
results and every inspected variant in the experiment ledger.

## WRDS request

The operator reports requesting UCI access; approval and actual subscriptions
have not been verified. UCI's official guide says available subscriptions must
be checked after login and access is for scholarly, non-commercial research:
https://guides.lib.uci.edu/merage-resources/wrds (checked October 14).
Access alone does not establish a covenant-outcome panel, historical disclosure
availability, or permission to use licensed inputs in a trading product.
First inspect entitlements and data dictionaries, then inventory identifiers,
coverage, vintage fields, delistings and dated covenant events before designing
an extract. No WRDS credentials or data were accessed in this audit.

## Verification scope

The prior implementation passed independent fleet `make check` on serendipity:
job `ba71d3c8-478d-410c-b4fb-36374f161b1c`, exit 0, 3,868 tests,
coverage 100.00%, 267 seconds, ended 2026-10-14T19:36:30Z.
This audit changes documentation only; those tests do not validate an economic
edge or repair the Radar target. The new campaign evaluator execution and
520-source hash check are the additional verification performed here.
