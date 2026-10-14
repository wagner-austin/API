# Causal maker diagnostics: no advancement

The [fixed regime protocol](MAKER_REGIME_PROTOCOL.md) was committed in
`9a8ed8f563e753eefcebfc612b0ec6bd93ec957c` before computing the archived panel's
regime results. Its local file SHA-256 is
`49bccb222f38fa44d4948c4999c2fd0068fa9c46537e5b28dd9e80655286e7b2`.
This remains exploratory analysis of previously examined historical outcomes.

The diagnostic classifies 4,549 of 4,788 contracts; 239 lack all six required
two-sided hourly closes. Every contract stays in the report. All three original
execution assumptions retain exactly the same orders, exclusions, pooled and
city summaries, monthly totals and advancement gates. The regime labels never
select an order or change its price.

## Primary two-cent-through result

The panel covers Boston, Philadelphia and Washington daily-high contracts from
January 1 through October 10, 2026. It is the existing three-city maker study,
not a newly acquired New York sample or an equity-volatility study.

The original 652 one-contract fill proxies produce $4.31 after the assumed
maker fees, or -$2.21 after the additional one-cent-per-contract execution
stress. These are simulated payoffs, not fills we executed. The regime partition
reconciles exactly to both amounts.

| Pre-entry movement group | Contracts examined | Fill proxies | Filled event dates | Hypothetical P&L | With extra 1-cent cost |
|---|---:|---:|---:|---:|---:|
| Low variation, falling | 733 | 135 | 98 | $1.36 | $0.01 |
| Low variation, flat | 223 | 28 | 27 | -$0.28 | -$0.56 |
| Low variation, rising | 386 | 47 | 41 | $0.29 | -$0.18 |
| High variation, falling | 1,507 | 286 | 183 | $3.72 | $0.86 |
| High variation, flat | 148 | 15 | 13 | $0.46 | $0.31 |
| High variation, rising | 1,552 | 108 | 87 | -$1.24 | -$2.32 |
| Unclassified | 239 | 33 | 28 | $0.00 | -$0.33 |
| Total | 4,788 | 652 | 251 unique dates | $4.31 | -$2.21 |

Dates overlap across groups and must not be added as independent observations.
All seven date-clustered mean-P&L intervals include zero, as do the calendar-day
serially robust intervals. There is no corrected multiple-comparison test here.
The larger positive falling-group total does not establish a profitable
falling-only rule. No threshold, sample or trading gate was tuned to this table.

## Implementation and tests

`regime_records.py` owns the immutable measurement record and strict codecs.
`regimes.py` computes six exact closed-candle midpoints, five hourly changes,
their population standard deviation and final-minus-initial direction.
Variation at or above 100 internal units (one cent) is high; lower is low.
There is no stock-return annualization, forward fill or missing-price substitute.

`experiment.evaluate_panel` computes these features once, adds the definition
and all contract records under `pre_entry_regimes`, and writes `by_regime` for
each existing execution arm. The existing `summarize_replay` provides monetary
accounting and clustered/serial uncertainty. Every original order retains its
exclusion reason and can be joined by ticker to its regime record.

The added tests exercise the actual code, including:

- Future quote insertion, trade changes, settlement changes, removal and
  reordering leave pre-entry features and maker offer unchanged. Expected future
  fills and profits are allowed to change and their exact changed values are checked.
- A past quote control changes the measured trend, variability and offer.
- Future weather evaluation-label changes preserve the training rows, actual
  fitted logistic model and predictions. Flipping past training labels changes
  the actual fitted model and predictions.
- Missing timestamps, missing sides and off-grid timestamps remain unclassified.
- Boundary classifications, strict codec rejection and complete group accounting
  have explicit expected results.
- Real socket acquisition, native candle decoding, report persistence and offline
  replay carry both classified and unclassified contracts through the existing CLI.

Full `make check` passed **3,843 tests**, strict typing and all guards, with
**100% of 17,923 statements and 3,662 branches**, in **112/300 seconds**.
The change adds 37 test cases and strengthens the existing real-HTTP integration.
Pure diagnostic modules use the package's existing dependency architecture;
they introduce no new external effects or alternate networking implementation.

These tests protect the exercised input paths. They do not prove that all
historical information was publicly available at a claimed time, eliminate
retrospective inventory selection, or validate counterfactual execution.

## Archived reproduction

Evidence root: `C:/Users/Test/PROJECTS/files/market-edge-20261012`.

`regimes-first-20261014.json` and `regimes-replay-20261014.json` were produced
by separate CLI processes using only the existing `integrated-raw` archive.
Both report bytes and both RunRecord bytes are identical. Report SHA-256:

`f1c9b7dca35af80c382798d368de56a6c9f3a983993a0a79f25abc1f35d91817`

Implementation digest:
`329df6d0635c8d1728fbf6bd922bdf2a138875b62487208ed2e00b5a09f532fe`.
Each report verification checks 5,910 paths including its report and sidecar.
The existing `integrated-results.json` was separately provenance-verified;
every original arm field compares exactly after removing only the newly added
`by_regime` field. Group counts and monetary totals also reconcile for all arms.

Reproduce with `scripts.market_research --mode replay --workers 3`, using
`integrated-plan.json` as `--plan`, **`plan.json` as `--protocol`**, and
`integrated-raw` as `--archive`; choose a new `--out` path. The original report's
source manifest identifies that protocol file. An initial invocation with
`integrated-protocol-addendum.json` was correctly rejected as `MARKET-031`
before a report was written. The frozen plan and protocol were not modified.
The full check log is `make-check-regimes-final-20261014.log`.

## Origin of the review

The operator's quoted comparison was found in local Claude session
`d3b0e0ab-92e6-4cfa-9fb2-bfdfac3db85b`, message dated
`2026-10-14T09:21:44.296Z`. The reviewed local clone was
`ananya-sampat/historical-bermudan-asian-policies` at
`9120af06e9bf4c137261d940e6d9ee37a0324d29`.

Inspection and direct execution showed that its test changes future `prc`,
while volatility and the normalized future path use `ret`-derived data. That
edit left both features and path unchanged. Changing future `ret` changed the
path while preserving past volatility and trailing return; changing past `ret`
changed both features. Its unchanged volatility was 0.14616629512941293, and
doubling prior returns changed it to 0.29233259025882624. This motivated positive
controls and mutations of inputs the tested implementation actually consumes.
No third-party implementation was copied.

Our sports-rating future-outcome invariance test and maker prior-quote selection
test already existed and passed when rerun during review. This work broadens
those checks and adds the previously absent regime diagnostic. Frozen prospective
campaigns, stock and CPI runtimes, and existing schedules remain unchanged.
